import torch
import numpy as np
import cv2
import os
import sys
from PIL import Image
from termcolor import cprint
import re

# --- Mount GPC dependency paths ---
sys.path.append("/home/zxn/gpc_code")
sys.path.append("/home/zxn/gpc_code/world_model_train_phase_two")
sys.path.append("/home/zxn/gpc_code/gpc_opt_evaluation")

try:
    from diffusion.denoiser import Denoiser, DenoiserConfig, InnerModelConfig
    from diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    cprint(f"[Error] Failed to import GPC dependencies: {e}", "red")

TASK_PROMPTS = {
    "meat_on_grill": "The robot needs to place the piece of meat onto the grill.",
    "move_hanger": "The robot needs to move the clothes hanger to the specified location.",
    "open_drawer": "The robot needs to successfully pull the drawer open.",
    "place_wine_at_rack_location": "The robot needs to grasp the wine bottle and place it securely on the rack.",
    "put_money_in_safe": "The robot needs to place the stack of money inside the safe.",
    "scoop_with_spatula": "The robot needs to use the spatula to scoop up the target object.",
    "setup_chess": "The robot needs to stand the fallen chess piece upright on the board.",
    "turn_oven_on": "The robot needs to grasp and turn the oven dial to switch it on."
}

class GPCEngine:
    """
    Generative Predictive Control (GPC) Engine Wrapper.
    Integrates a video diffusion world model for trajectory rollout and a 
    Vision-Language Model (Qwen2.5-VL) as a judge to rank candidate actions.
    """
    def __init__(self, task_name, wm_ckpt, device="cuda:0"):
        self.device = device
        self.task_name = task_name
        
        cprint(f"[Init] Loading GPC World Model for task: {task_name}...", "cyan")
        inner_cfg = InnerModelConfig(img_channels=3, num_steps_conditioning=4, cond_channels=256, depths=[2,2,2,2], channels=[96, 96, 96, 96], attn_depths=[0, 0, 1, 1], num_actions=8, is_upsampler=None)
        denoiser_cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.1, noise_previous_obs=True)
        world_model = Denoiser(denoiser_cfg).to(self.device)
        world_model.setup_training(None)
        world_model.load_state_dict(torch.load(wm_ckpt, map_location=self.device))
        world_model.eval()
        self.world_model_sampler = DiffusionSampler(world_model, DiffusionSamplerConfig(num_steps_denoising=3))
        
        cprint("[Init] Loading Qwen2.5-VL Judge (This may be slow and consume ~15GB VRAM)...", "cyan")
        local_model_path = "/data/colosseum_dataset/hf_cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model_path, torch_dtype=torch.bfloat16, device_map=self.device, local_files_only=True).eval()
        self.qwen_processor = AutoProcessor.from_pretrained(local_model_path, local_files_only=True)
        cprint("[Success] GPC initialization complete.", "green")

    @torch.no_grad()
    def steer(self, obs, candidates, instruction):
        """
        Execute the Propose-and-Rank pipeline: Generate visual futures using the World Model,
        then utilize the VLM Judge to rank and select the optimal action candidate.
        """
        num_trial = len(candidates)
        
        # 1. Image Preprocessing (Process into 4 identical consecutive frames at 96x96 resolution)
        img_np = obs["cam_front_view_image"] # RGB
        img_resized = cv2.resize(img_np, (96, 96))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 127.5 - 1.0
        input_imgs = torch.stack([img_tensor] * 4).unsqueeze(0).repeat(num_trial, 1, 1, 1, 1).to(self.device)
        
        # 2. Action Candidate Pool Preprocessing
        target_horizon = 4
        processed_candidates = []
        for cand in candidates:
            cand_np = np.array(cand)
            if len(cand_np.shape) == 1: cand_np = cand_np.reshape(1, -1)
            curr_len = cand_np.shape[0]
            if curr_len < target_horizon:
                padding = np.tile(cand_np[-1:], (target_horizon - curr_len, 1))
                cand_np = np.vstack([cand_np, padding])
            else:
                cand_np = cand_np[:target_horizon]
            if cand_np.shape[-1] > 8: cand_np = cand_np[:, :8]
            processed_candidates.append(cand_np)
        naction = torch.tensor(np.array(processed_candidates), dtype=torch.float32).to(self.device)
        
        # 3. World Model Rollout
        pred_videos, _ = self.world_model_sampler.sample(input_imgs, naction)
        last_pred_frames = pred_videos[:, -1]
        
        # 4. Qwen VLM Judge Evaluation
        task_prompt = TASK_PROMPTS.get(self.task_name, instruction)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a robotic task evaluator. Output ONLY a single integer."}]},
            {"role": "user", "content": [{"type": "text", "text": f"Goal: {task_prompt}. Look at these predicted future states."}]}
        ]
         
        for i in range(num_trial):
            single_frame = last_pred_frames[i]
            
            # 1. Ensure tensor is moved to CPU, converted to Numpy, and handle potential 4D shape (B, C, H, W)
            f_np = single_frame.detach().cpu().float().numpy()
            if f_np.ndim == 4: f_np = f_np[0] 

            # 2. Automatically identify and align dimensions to (H, W, C)
            if f_np.shape[0] == 3: # If shape is (C, H, W)
                f_np = f_np.transpose(1, 2, 0)
            elif f_np.shape[-1] != 3: # If the shape doesn't match expected image dimensions, force reconstruction
                # Force reshape to 96x96x3 to prevent (1, 1, 1) collapse
                try:
                    f_np = f_np.reshape(96, 96, 3)
                except ValueError:
                    # If data volume is severely mismatched (e.g., model output is empty), pad with a black image to prevent crash
                    f_np = np.zeros((96, 96, 3), dtype=np.float32)

            # 3. Denormalization and type conversion (Map GPC's [-1, 1] range back to [0, 255])
            tmp_img = ((f_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            
            # 4. Final safety check: Prevent PIL from failing on malformed images like (1, 1, 3)
            if tmp_img.shape[0] < 10: 
                tmp_img = cv2.resize(tmp_img, (96, 96), interpolation=cv2.INTER_NEAREST)

            pil_img = Image.fromarray(tmp_img)
            
            # 5. Maintain original author's 'append all images' logic 
            # (Note: This rapidly consumes VRAM as candidates scale)
            img_item = {"type": "image", "image": pil_img}
            text_item = {"type": "text", "text": f"Index: {i}"}
            messages[0]["content"].extend([img_item, text_item])
            messages[1]["content"].extend([img_item, text_item])
            
        prompt_suffix = f"Which Index (0 to {num_trial-1}) shows the most successful completion of the task? Output only the integer."
        messages[1]["content"].append({"type": "text", "text": prompt_suffix})
        messages[0]["content"].append({"type": "text", "text": prompt_suffix})
        
        # Execute Large Language Model Inference
        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.qwen_processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)
        
        generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=5, temperature=0.2)
        output_text = self.qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        match = re.search(r'\d+', output_text.split()[-1])
        best_idx = int(match.group()) if match else 0
        best_idx = min(max(best_idx, 0), num_trial - 1)
        
        return best_idx, output_text
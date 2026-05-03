import re
import os
import json
import torch
import base64
import time
import numpy as np
from PIL import Image
from io import BytesIO
from peft import PeftModel
from safetensors.torch import load_file
from flask import Flask, request, jsonify 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

app = Flask(__name__)

class VLMCorrector:
    """
    Vision-Language Model (VLM) Corrector Engine.
    
    This class handles the initialization, weight loading (supporting both standard SFT 
    and LoRA PEFT formats), memory management, and step-by-step inference for the 
    Chain-of-Correction (CoC) framework. It includes dynamic prompt generation for 
    various ablation studies.
    """
    def __init__(self, model_path, base_model_path="/home/zxn/MomaGraph-R1", device="cuda:0", ablation="none"):
        print(f"Loading Error-Correction VLM: {model_path} | Ablation Mode: {ablation} ...")
        self.device = device
        self.ablation = ablation
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            base_model_path, 
            trust_remote_code=True
        )

        print("Loading base model architecture...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,           # Load base model structure first
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  
        )
 
        # ==========================================
        #  Smart Weight Loading: SFT vs RL LoRA
        # ==========================================
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            self.is_lora = True
            # Branch A: PEFT LoRA Weights (RL tuned)
            print("Detected LoRA Adapter. Mounting PEFT weights...")
            self.model = PeftModel.from_pretrained(self.model, model_path)
            
            # CRITICAL SPEEDUP: Merge LoRA weights into base model
            # print("Merging LoRA weights into base model for max inference speed...")
            self.model = self.model.merge_and_unload()
            
            print("LoRA weights successfully merged!")
            
        else:
            self.is_lora = False
            # Branch B: Standard SFT Full Weights
            print("No LoRA format detected. Using standard full-weight loading...")
            checkpoint_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(checkpoint_path):
                state_dict = load_file(checkpoint_path)
            else:
                checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
                state_dict = torch.load(checkpoint_path, map_location="cpu")

            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = self._remap_key(k)
                new_state_dict[new_key] = v

            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
        # ==========================================

        self.model.eval()
        self.history_messages = [] # Initialize as empty list
        print("VLM Server initialized successfully. Waiting for requests...")

    @staticmethod
    def _remap_key(k): 
        """Remap state dict keys to match the target Qwen model structure."""
        # 1. Handle lm_head independently
        if "lm_head" in k:
            return "lm_head.weight" if "weight" in k else "lm_head.bias"
        # 2. Visual Encoder module
        if "visual." in k:
            # Split by "visual." and retain the trailing structure
            return "model.visual." + k.split("visual.", 1)[1]
        # 3. Language Model backbone (Layers, Embeddings, Norm)
        for keyword in ["layers.", "embed_tokens.", "norm."]:
            if keyword in k:
                return "model.language_model." + keyword + k.split(keyword, 1)[1]
        return k 

    def _convert_to_pil(self, img_array):
        """Convert a numpy array to a PIL Image."""
        if isinstance(img_array, Image.Image):
            return img_array
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            img_array = (img_array * 255).astype(np.uint8)
        if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3, 4]:
            img_array = np.transpose(img_array, (1, 2, 0))
        return Image.fromarray(img_array)

    def _convert_to_qwen_messages(self, prompt_messages):
        """Align block conversion logic with the training dataset format."""
        converted = []
        for msg in prompt_messages:
            role, content = msg["role"], msg["content"]
            images = msg.get("images", [])
            if role == "user" and images:
                blocks = [{"type": "image", "image": img} for img in images]
                blocks.append({"type": "text", "text": content})
                converted.append({"role": role, "content": blocks})
            else:
                converted.append({"role": role, "content": content})
        return converted

    def _get_current_text(self, instruction, base_proposal_list, time_step):
        """Dynamically generate prompts with specific ablation constraints."""
        
        # 1. w/o Progress ablation
        if self.ablation == "wo_progress":
            if time_step == 0:
                return (
                    f"Task instruction: {instruction}\n"
                    f"[Time Step: 0] Task initiation. Analyze the multi-view images to extract the initial scene graph. "
                    f"Predict the resulting spatial changes if the Action Proposal {base_proposal_list} is applied. Do NOT compute or evaluate global task progress."
                )
            else:
                return (
                    f"[Time Step: {time_step}] Previous action executed. Given the new observations and the Action Proposal: {base_proposal_list}, "
                    f"perform a structured physical dual-check:\n"
                    f"1. Backward check: Compare History vs. Current to quantify geometric distance changes and identify unintended deviations.\n"
                    f"2. Forward check: Predict how the Current Scene Graph will change if the Proposal is executed. Skip task progress evaluation.\n"
                    f"Synthesize these checks to objectively determine if a correction is needed. Output the structured evaluation and the corrected action."
                )
                
        # 2. w/o Prediction ablation
        elif self.ablation == "wo_prediction":
            if time_step == 0:
                return (
                    f"Task instruction: {instruction}\n"
                    f"[Time Step: 0] Task initiation. Analyze the multi-view images to extract the initial scene graph. "
                    f"Evaluate the initial task progress. Skip future spatial prediction and directly evaluate if the Action Proposal {base_proposal_list} establishes a valid initial kinematic state."
                )
            else:
                return (
                    f"[Time Step: {time_step}] Previous action executed. Given the new observations and the Action Proposal: {base_proposal_list}, "
                    f"perform a structured physical dual-check:\n"
                    f"1. Backward check: Compare History vs. Current to quantify geometric distance changes and identify unintended deviations.\n"
                    f"2. Forward check: Skip future scene graph prediction. Directly evaluate the structural soundness of the Action Proposal.\n"
                    f"Synthesize these checks to objectively determine if a correction is needed."
                )
                
        # 3. w/o Geometry ablation
        elif self.ablation == "wo_geometry":
            if time_step == 0:
                return (
                    f"Task instruction: {instruction}\n"
                    f"[Time Step: 0] Task initiation. Analyze the multi-view images to extract the initial scene graph WITHOUT precise geometric attributes. "
                    f"Evaluate the initial task progress, and predict the resulting topological relations if the Action Proposal {base_proposal_list} is applied."
                )
            else:
                return (
                    f"[Time Step: {time_step}] Previous action executed. Given the new observations and the Action Proposal: {base_proposal_list}, "
                    f"perform a structured physical dual-check:\n"
                    f"1. Backward check: Compare History vs. Current to identify physical stasis. Do NOT quantify exact geometric distance changes.\n"
                    f"2. Forward check: Predict how the Current Scene Graph (spatial relations only, no geometry) and task progress will change.\n"
                    f"Synthesize these checks to objectively determine if a correction is needed."
                )

        # 5. w/o History ablation - Forcefully strip backward check
        elif self.ablation == "wo_history":
            if time_step == 0:
                return (
                    f"Task instruction: {instruction}\n"
                    f"[Time Step: 0] Task initiation. Analyze the single current multi-view image to extract the initial scene graph. "
                    f"Evaluate the initial task progress, and predict the resulting spatial changes if the Action Proposal {base_proposal_list} is applied."
                )
            else:
                return (
                    f"[Time Step: {time_step}] Previous action executed. You ONLY have access to the current single-frame observation. Given the Action Proposal: {base_proposal_list}, "
                    f"perform a physical check based SOLELY on the current state:\n"
                    f"1. Forward check: Predict how the Current Scene Graph and task progress will change if the Proposal is executed. (Backward check is DISABLED since history is unavailable).\n"
                    f"Evaluate if this proposed action causes collisions or kinematic violations purely from the current visual context. "
                    f"Determine if a correction is needed. Output the structured evaluation and the corrected action."
                )
                
        # 4. Default Full Model
        else:
            if time_step == 0:
                return (
                    f"Task instruction: {instruction}\n"
                    f"[Time Step: 0] Task initiation. Analyze the multi-view images to extract the initial scene graph. "
                    f"Evaluate the initial task progress, and predict the resulting spatial changes if the Action Proposal {base_proposal_list} is applied."
                )
            else:
                return (
                    f"[Time Step: {time_step}] Previous action executed. Given the new observations and the Action Proposal: {base_proposal_list}, "
                    f"perform a structured physical dual-check:\n"
                    f"1. Backward check: Compare History vs. Current to quantify geometric distance changes and identify any physical stasis or unintended deviations.\n"
                    f"2. Forward check: Predict how the Current Scene Graph (specifically edge distances and spatial relations) and task progress will change if the Proposal is executed.\n"
                    f"Synthesize these checks to objectively determine if a correction is needed. Output the structured evaluation and the corrected action."
                )

    def reset_memory(self):
        """Clear the observation and interaction history memory."""
        self.history_messages = []

    def step(self, current_images, base_policy_proposal, time_step, instruction):
        """
        Execute one step of inference and return the corrected action.
        
        Args:
            current_images (list): List of numpy arrays (C, H, W) or (H, W, C) representing visual inputs.
            base_policy_proposal (list): Action output from the base policy (length <= 8).
            time_step (int): Current time step index.
            instruction (str): The language instruction for the task.
            
        Returns:
            tuple: A tuple containing the base proposal list and the generated response text.
        """

        base_proposal_list = [float(v) for v in base_policy_proposal[:8]]
        current_text = self._get_current_text(instruction, base_proposal_list, time_step)

        current_history = [] if self.ablation == "wo_history" else self.history_messages

        if self.is_lora: 
            history_elements = [msg["content"] for msg in current_history]
            history_elements.append(current_text)
             
            full_problem_text = "\n\n".join(history_elements)
             
            raw_messages = [{"role": "user", "content": full_problem_text, "images": current_images}]
        else:
            # 1. Concatenate history memory (text) with current observation (images)
            raw_messages = current_history + [{"role": "user", "content": current_text, "images": current_images}]

        # 2. Inference
        qwen_msgs = self._convert_to_qwen_messages(raw_messages)

        inputs_text = self.processor.apply_chat_template(
            qwen_msgs, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_msgs)

        inputs = self.processor(
            text=[inputs_text],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        if time_step == 0:
            print(f"\n[{'RL-LoRA' if self.is_lora else 'SFT'} Mode Prompt]:\n{inputs_text}\n")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        response_text = self.processor.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
 
        # 3. Core alignment: Reconstruct memory. Extract JSON and construct plaintext history for the next frame
        # Extract scene graph and task progress
        graph_match = re.search(r'Current Scene Graph \(t=\d+\):\s*(\{.*?\})\n', response_text)
        prog_match = re.search(r'Global Task Progress:\s*([0-9.]+)', response_text)
        
        scene_graph = {}
        if graph_match:
            graph_str = graph_match.group(1)
            try:
                # Attempt standard strict JSON parsing
                scene_graph = json.loads(graph_str)
            except Exception:
                try:
                    # Salvage LLM hallucinations: fallback to literal_eval for Python dict formats
                    import ast
                    scene_graph = ast.literal_eval(graph_str)
                except Exception as e:
                    print(f"[Warning] Scene graph parsing failed completely, safely falling back to empty dict. Faulty text: {graph_str}")
                    scene_graph = {}
        progress = float(prog_match.group(1)) if prog_match else 0.0
        
        # Construct JSON matching the T_PREV format of the training set
        if self.ablation == "wo_geometry" and "edges" in scene_graph:
            for edge in scene_graph["edges"]:
                edge["geometric_attributes"] = {}

        if self.ablation == "wo_progress":
            history_json_dict = {"scene_graph": scene_graph}
            past_user_text = f"Task instruction: {instruction}\n[Time Step: {time_step}] Analyze the historical observations to extract the past scene graph."
        else:
            history_json_dict = {
                "scene_graph": scene_graph,
                "progress": progress
            }
            past_user_text = f"Task instruction: {instruction}\n[Time Step: {time_step}] Analyze the historical observations to extract the past scene graph and task progress."
        
        clean_json = f"```json\n{json.dumps(history_json_dict, indent=2)}\n```"
        past_assistant_text = f"{clean_json}\nAction Executed: {base_proposal_list}"

        # Sliding window: clear and store only the converted text memory of the current frame
        self.history_messages = [
            {"role": "user", "content": past_user_text},
            {"role": "assistant", "content": past_assistant_text}
        ]
        return base_proposal_list, response_text

    def _parse_action(self, response_text, fallback_action):
        """Extract the JSON formatted corrected_action from the response text."""
        try:
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group(1))
                if "corrected_action" in result_dict:
                    return result_dict["corrected_action"][:8]
        except Exception:
            pass
        return fallback_action
 

# ==========================================
# Flask Web API Routing
# ==========================================

@app.route('/step', methods=['POST'])
def handle_step():
    start_time = time.time()
    data = request.json
    pil_images = [Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB") for img_b64 in data['images_b64']]
    _, cot_text = engine.step(pil_images, data['base_policy_proposal'], data['time_step'], data['instruction'])
    
    # Extract corrected_action to return to the client
    corrected_action = data['base_policy_proposal']
    correction_needed = False  
    try:
        json_match = re.search(r'```json\n(.*?)\n```', cot_text, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group(1))
            corrected_action = parsed_json.get("corrected_action", corrected_action)[:8]
            correction_needed = parsed_json.get("correction_needed", False)
    except: pass

    end_time = time.time() 
    cost_time = round(end_time - start_time, 3)   
    print(f"[Profiling] Time Step: {data.get('time_step', 'N/A')} | CoC inference latency: {cost_time:.4f} seconds")

    return jsonify({
        "corrected_action": corrected_action, 
        "cot": cot_text, 
        "correction_needed": correction_needed,
        "cost_time_sec": cost_time   
    })


@app.route('/reset', methods=['POST'])
def handle_reset():
    engine.reset_memory()
    return jsonify({"status": "success"})

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to VLM checkpoint")
    parser.add_argument("--port", type=int, default=5050) 
    parser.add_argument("--gpu", type=str, default="cuda:0")
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "wo_progress", "wo_prediction", "wo_geometry", "wo_history"])
    args = parser.parse_args()

    engine = VLMCorrector(model_path=args.ckpt, device=args.gpu, ablation=args.ablation)

    print(f"Starting Flask server, listening on port {args.port}...")
    app.run(host='0.0.0.0', port=args.port, threaded=False)



# --- Server command: conda activate cot_sft ---
# nohup python3 vlm_corrector_server.py \
    # --ckpt /data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1 \
    # --gpu cuda:4 \
    # --port 5050 > logs/vlm_server_ep1_5050.log 2>&1 &

# nohup python3 vlm_corrector_server.py \
#     --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_1_step_110/actor/lora_adapter \
#     --gpu cuda:3 \
#     --port 5054 > logs/vlm_server_lora_5054.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=2 nohup python3 vlm_corrector_server.py \
#     --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_1_step_110/actor/lora_adapter \
#     --gpu cuda:0 \
#     --port 5052 > logs/vlm_server_lora_gpu2_5052.log 2>&1 &
# Allow 1-2 minutes for initialization. Use 'tail -f logs/vlm_server_lora_gpu2.log' to monitor load status.
# CUDA_VISIBLE_DEVICES=2 VLM_PORT=5052 nohup bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_worker.sh \
#     "lora_test_worker2" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 2 \
#     "move_hanger" > /home/zxn/Chain-of-Correction/logs/eval_lora_gpu2.log 2>&1 &


# --- Evaluation command: conda activate bridgevla ---
# CUDA_VISIBLE_DEVICES=4 VLM_PORT=5050 nohup bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_worker.sh \
#     "ep1_resume_test" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 4 \
#     "setup_chess scoop_with_spatula stack_cups" > /home/zxn/Chain-of-Correction/logs/eval_ep1_remaining.log 2>&1 &
# "put_money_in_safe place_wine_at_rack_location meat_on_grill move_hanger open_drawer turn_oven_on setup_chess scoop_with_spatula stack_cups" 
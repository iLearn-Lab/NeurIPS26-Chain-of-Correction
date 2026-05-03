import json
import os
import gc
import sys
import torch
from tqdm import tqdm
# from qwen_vl_utils import process_vision_info # close this when evaluating Llama-3.2-11B or VLM-3R
import PIL.Image

VLM3R_CODE_PATH = "/home/zxn/Chain-of-Correction/baselines/VLM-3R"
sys.path.append(VLM3R_CODE_PATH)
sys.path.append(os.path.join(VLM3R_CODE_PATH, "CUT3R"))

# ================= Security Override =================
import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
# =========================================================

# ===================== Configuration =====================
VAL_SET_PATH = "/data/colosseum_dataset/ablations/offline_val_set_200.json"
OUTPUT_DIR = "/data/colosseum_dataset/ablations/eval_results"
DEVICE = "cuda:0"
NUM_TRIALS = 3
MODELS_TO_RUN = {
    # --- Original Qwen Family ---
    # "/home/zxn/MomaGraph-R1": "MomaGraph-R1-Base-Qwen",
    # "/data/colosseum_dataset/hf_cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5": "Qwen2.5-Vanilla",
    # "/data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1": "CoC-Full-V1-Qwen", 

    # "/data/colosseum_dataset/checkpoints/sft/v1_wo_prediction/checkpoint-epoch-1": "CoC-wo-prediction",
    # "/data/colosseum_dataset/checkpoints/sft/v1_wo_progress/checkpoint-epoch-1": "CoC-wo-progress",
    # "/data/colosseum_dataset/checkpoints/sft/v1_wo_geometric/checkpoint-epoch-1": "CoC-wo-geometric",

    # --- Open-Source VLM Family ---
    # "/data/colosseum_dataset/hf_cache/huggingface/hub/LLaVA-1.5-7B": "LLaVA-1.5-7B",
    # "/data/colosseum_dataset/hf_cache/huggingface/hub/unsloth/Llama-3___2-11B-Vision-Instruct": "Llama-3.2-11B",

    # --- LoRA Patched Models ---
    "/data/colosseum_dataset/hf_cache/huggingface/hub/VLM-3R": "VLM_3R_Model",
    
    # --- Closed-Source APIs ---
    # "gpt-5.4": "GPT-5.4",
    # "gemini-3.1-pro-preview": "Gemini-3.1-Pro" # Gemini-3.1-Pro-preview
}
# ===================================================

def extract_image_paths_from_messages(messages):
    """
    Extract physical image paths from the unified message format.
    Strips the 'file://' prefix to ensure compatibility with PIL reading.
    """
    image_paths = []
    for msg in messages:
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    path = item.get("image", "")
                    if path.startswith("file://"): path = path[7:]
                    image_paths.append(path)
    return image_paths

def get_pure_text_prompt(messages, extra_instruction):
    """
    Construct a pure text prompt for models that do not support complex multimodal dictionary inputs.
    """
    prompt = ""
    for msg in messages:
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item.get("type") == "text":
                    prompt += item.get("text", "") + "\n"
        elif isinstance(msg["content"], str):
            prompt += msg["content"] + "\n"
    prompt += extra_instruction
    return prompt.strip()

def run_model_inference(model_path, model_name, trial_idx):
    """
    Execute the core inference loop for global progress estimation across varying model architectures.
    """
    print(f"\n[Launch] Progress Evaluation | Model: {model_name} | Trial: {trial_idx+1}/{NUM_TRIALS}")

    model, processor = None, None
    is_api_model = "Gemini" in model_name or "GPT" in model_name

    # ================= 1. Dynamic Model & Processor Loading =================
    if not is_api_model:
        if model_name == "VLM_3R_Model":
            print("   -> Initiating manual mount mode for VLM-3R...")
            from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
            from transformers import AutoTokenizer
            from peft import PeftModel
            # from llava.mm_utils import process_images
            BASE_PATH = "/data/colosseum_dataset/hf_cache/huggingface/hub/LLaVA-NeXT-Video-7B-Qwen2"
            CUT3R_WEIGHTS = "/home/zxn/Chain-of-Correction/baselines/VLM-3R/CUT3R/src/cut3r_512_dpt_4_64.pth"

            processor = AutoTokenizer.from_pretrained(BASE_PATH, use_fast=False)  
            
            base_model = LlavaQwenForCausalLM.from_pretrained(
                BASE_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": DEVICE}
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            
            print("   -> Activating vision tower and CUT3R module...")
            vision_tower = model.get_model().get_vision_tower()
            LOCAL_SIGLIP_PATH = "/data/colosseum_dataset/hf_cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3" 
            model.get_model().config.mm_vision_tower = LOCAL_SIGLIP_PATH
            vision_tower.vision_tower_name = LOCAL_SIGLIP_PATH
            if not vision_tower.is_loaded: vision_tower.load_model()
            vision_tower.to(device=DEVICE, dtype=torch.bfloat16)
            model.image_processor = vision_tower.image_processor  

            if hasattr(model.get_model(), 'spatial_encoder'):
                model.get_model().spatial_encoder.load_weights(CUT3R_WEIGHTS)
                model.get_model().spatial_encoder.to(device=DEVICE, dtype=torch.bfloat16)
 
        elif "Qwen" in model_name or "CoC" in model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            if "checkpoint" in model_path:
                print("   -> Mounting full SFT Checkpoint for Qwen...")
                base_model_path = "/home/zxn/MomaGraph-R1"
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_path, torch_dtype="auto", device_map={"": DEVICE})
                processor = AutoProcessor.from_pretrained(base_model_path)
                state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
                 
                clean_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
                    new_key = new_key.replace("model.", "", 1) if new_key.startswith("model.") else new_key
                    clean_state_dict[new_key] = v
        
                missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
                if len(missing) > 0:
                    print(f"   [Warning] {len(missing)} weights remain unmatched!")
                else:
                    print("   [Success] SFT weights mounted perfectly!")
                
                del state_dict, clean_state_dict; gc.collect()
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map={"": DEVICE})
                processor = AutoProcessor.from_pretrained(model_path)
        
        elif "LLaVA" in model_name:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map={"": DEVICE})
            processor = AutoProcessor.from_pretrained(model_path)
            
        elif "Llama" in model_name:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            model = MllamaForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                device_map={"": DEVICE},
                attn_implementation="sdpa"   
            )
            processor = AutoProcessor.from_pretrained(model_path)
            
        else:
            raise ValueError(f"Unrecognized model type, please check alias: {model_name}")

    else:
        from openai import OpenAI 
        client = OpenAI(
            api_key="sk-4o5CciFQHMSsuG3cNoK3PJRekuvoCtCEhrEIt8CudgLx6YHL",  
            base_url="https://bitexingai.com/v1"
        )
        model = client   
    # ==========================================================

    with open(VAL_SET_PATH, 'r') as f:
        val_data = json.load(f)

    results = []
    output_filename = os.path.join(OUTPUT_DIR, f"progress_results_{model_name}_trial_{trial_idx+1}.json")

    # Inference Loop
    for item in tqdm(val_data, desc=f"Evaluating Progress Perception: {model_name}"):
        messages = item["messages"][:-1]
        ground_truth_raw = item["messages"][-1]["content"]

        # Enhanced instruction specifically targeting progress estimation
        extra_instruction = (
            "\n\nFollow the format: [Thinking Process] then final JSON block. "
            "Based on the visual observation, estimate the completion progress of the task. "
            "End your response with a JSON block containing 'global_progress' (a float between 0.0 and 1.0)."
        )

        output_text = ""

        # --- Branch A: API Inference Route ---
        if is_api_model:
            import base64
            image_paths = extract_image_paths_from_messages(messages)
            prompt = get_pure_text_prompt(messages, extra_instruction)
            
            messages_content = [{"type": "text", "text": prompt}]
            
            for p in image_paths:
                try:
                    with open(p, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    messages_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                except Exception as img_e:
                    pass  

            try:
                # Pass the actual model_path as the requested model name
                response = model.chat.completions.create(
                    model=model_path, 
                    messages=[{"role": "user", "content": messages_content}],
                    temperature=0.0
                )
                output_text = response.choices[0].message.content
            except Exception as e:
                output_text = f"API Request Failed: {str(e)}"

        # --- Branch B: Local Open-Source Inference Route ---
        else:
            if model_name == "VLM_3R_Model":
                from llava.conversation import conv_templates
                
                image_paths = extract_image_paths_from_messages(messages)
                raw_images = [PIL.Image.open(p).convert('RGB') for p in image_paths if os.path.exists(p)]
                
                prompt_text = get_pure_text_prompt(messages, extra_instruction)
                
                # VLM-3R requires the Qwen1.5 conversation template format
                conv = conv_templates["qwen_1_5"].copy()
                conv.append_message(conv.roles[0], prompt_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                input_ids = processor(prompt, return_tensors='pt').input_ids.to(DEVICE)
                
                image_tensor, spatial_features, image_sizes = None, None, None
                if len(raw_images) > 0:
                    try:
                        from llava.mm_utils import process_images
                        image_tensor = process_images(raw_images, model.image_processor, model.config)
                        image_tensor = [t.to(DEVICE, dtype=torch.bfloat16) for t in image_tensor] if isinstance(image_tensor, list) else image_tensor.to(DEVICE, dtype=torch.bfloat16)
                        image_sizes = [img.size for img in raw_images]
                    except:
                        image_tensor = model.image_processor.preprocess(raw_images, return_tensors='pt')['pixel_values'].to(DEVICE, dtype=torch.bfloat16)
                        image_sizes = [img.size for img in raw_images]

                    if hasattr(model.get_model(), 'spatial_encoder'):
                        spatial_features_list = [model.get_model().spatial_encoder(img) for img in raw_images]
                        spatial_features = torch.stack(spatial_features_list).to(device=DEVICE, dtype=torch.bfloat16)
                
                try:
                    with torch.inference_mode():
                        output_ids = model.generate(
                            inputs=input_ids, images=image_tensor, image_sizes=image_sizes, spatial_features=spatial_features,
                            max_new_tokens=1280, do_sample=False, use_cache=True
                        )
                    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                except Exception as e:
                    output_text = f"Inference Failed Error: {str(e)}"

            elif "Qwen" in model_name or "CoC" in model_name:
                from qwen_vl_utils import process_vision_info
                # Inject progress evaluation instructions
                last_content = messages[-1]["content"]
                if isinstance(last_content, list):
                    for part in last_content:
                        if part.get("type") == "text":
                            part["text"] += extra_instruction
                            break
                elif isinstance(last_content, str):
                    messages[-1]["content"] += extra_instruction

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1280, do_sample=False)
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

            # LLaVA / Llama inference pipeline
            elif "LLaVA" in model_name or "Llama" in model_name:
                import copy
                image_paths = extract_image_paths_from_messages(messages)
                imgs = [PIL.Image.open(p).convert("RGB") for p in image_paths]
                processed_messages = copy.deepcopy(messages)
                
                img_token = "<|image|>" if "Llama" in model_name else "<image>"
                for msg in processed_messages:
                    if isinstance(msg["content"], list):
                        text_content = ""
                        for item in msg["content"]:
                            if item.get("type") == "image": text_content += img_token + "\n"
                            elif item.get("type") == "text": text_content += item.get("text", "") + "\n"
                        msg["content"] = text_content.strip()

                last_user_idx = -1
                for idx, msg in enumerate(processed_messages):
                    if msg["role"] == "user": last_user_idx = idx
                if last_user_idx != -1:
                    processed_messages[last_user_idx]["content"] += extra_instruction

                formatted_prompt = processor.apply_chat_template(processed_messages, tokenize=False, add_generation_prompt=True)
                
                if len(imgs) > 0:
                    inputs = processor(text=formatted_prompt, images=imgs, padding=True, return_tensors="pt").to(DEVICE)
                else:
                    inputs = processor(text=formatted_prompt, padding=True, return_tensors="pt").to(DEVICE)
                    
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1280, do_sample=False)
                    output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        # --------------------------------------------------
        results.append({
            "id": item.get("id", "unknown_id"),
            "prediction": output_text,
            "ground_truth": ground_truth_raw
        })

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"[Success] Trial {trial_idx+1} completed. Results saved to {output_filename}")

    if not is_api_model:
        del model
        if processor is not None: del processor
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Success] Trial {trial_idx+1} concluded. VRAM completely released!")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    for path, name in MODELS_TO_RUN.items():
        for i in range(NUM_TRIALS):
            run_model_inference(path, name, i)

    print("\n\n[All progress evaluation tasks have been successfully completed!]")

if __name__ == "__main__":
    main()
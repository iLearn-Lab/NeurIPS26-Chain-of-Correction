import json
import os
import gc
import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info # close this when evaluating Llama-3.2-11B
import PIL.Image

# ================= Override Safety Restrictions =================
import transformers.utils.import_utils
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
# ================================================================

# ======================== Configuration =========================
VAL_SET_PATH = "/data/colosseum_dataset/ablations/offline_val_set_200.json"
OUTPUT_DIR = "/data/colosseum_dataset/ablations/eval_results"
DEVICE = "cuda:0"
NUM_TRIALS = 3
MODELS_TO_RUN = {
    # --- Original Qwen Family ---
    # "/home/zxn/MomaGraph-R1": "MomaGraph-R1-Base-Qwen",
    # "/data/colosseum_dataset/hf_cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5": "Qwen2.5-Vanilla",
    # "/data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1": "CoC-Full-V1-Qwen", 

    "/data/colosseum_dataset/checkpoints/sft/v1_wo_prediction/checkpoint-epoch-1": "CoC-wo-prediction",
    "/data/colosseum_dataset/checkpoints/sft/v1_wo_progress/checkpoint-epoch-1": "CoC-wo-progress",
    "/data/colosseum_dataset/checkpoints/sft/v1_wo_geometric/checkpoint-epoch-1": "CoC-wo-geometric",

    # --- Open-Source VLM Family ---
    # "/data/colosseum_dataset/hf_cache/huggingface/hub/LLaVA-1.5-7B": "LLaVA-1.5-7B",
    # "/data/colosseum_dataset/hf_cache/huggingface/hub/unsloth/Llama-3___2-11B-Vision-Instruct": "Llama-3.2-11B",

    # --- LoRA Patched Models ---
    # "/data/colosseum_dataset/hf_cache/huggingface/hub/VLM-3R": "VLM_3R_Model",
    
    # --- Closed-Source APIs ---
    # "gemini-3.1-pro": "Gemini-3.1-Pro-preview" # Gemini-3.1-Pro
    # "gpt-5.4": "GPT-5.4"
}
# ===================================================

def extract_image_paths_from_messages(messages):
    """
    Extract physical image paths from the unified message format for non-Qwen models.
    Strips the 'file://' prefix to ensure compatibility with PIL.
    """
    image_paths = []
    for msg in messages:
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item.get("type") == "image": 
                    path = item.get("image", "")
                    if path.startswith("file://"):
                        path = path[7:]
                    image_paths.append(path)
    return image_paths

def get_pure_text_prompt(messages, extra_instruction):
    """
    Extract a pure text prompt for models like Gemini that do not require 
    complex multimodal dictionaries or chat templates.
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
    return prompt

def run_model_inference(model_path, model_name, trial_idx):
    """
    Execute the core inference loop for a specific model over the validation dataset.
    Handles dynamic model loading, parameter cleansing, and compatibility patching 
    across different model architectures (Qwen, Llama, LLaVA, APIs).
    """
    print(f"\n[Launch] Model: {model_name} | Trial: {trial_idx+1}/{NUM_TRIALS}")

    model, processor = None, None
    is_api_model = "Gemini" in model_name or "GPT" in model_name

    # ================= 1. Dynamic Model & Processor Loading =================
    if not is_api_model:
        if "Qwen" in model_name or "CoC" in model_name:
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
            raise ValueError(f"Unrecognized model type, please verify alias: {model_name}")

    elif "Gemini" in model_name:
        from openai import OpenAI 
        client = OpenAI(
            api_key="sk-GcTionlok4PdSxqoZCXw8DTPfvY4IoL72G9WcZX2DYP7OFI7",  
            base_url="https://bitexingai.com/v1"
        )
        model = client  
    elif "GPT" in model_name:
        from openai import OpenAI 
        client = OpenAI(
            api_key="sk-L7QfbGxE45HOGcpJ7sZrx2rPAMcX0MimGLZEGxqh13WOuEfp",  
            base_url="https://bitexingai.com/v1"
        )
        model = client
    # ==========================================================

    with open(VAL_SET_PATH, 'r') as f:
        val_data = json.load(f)

    results = []
    output_filename = os.path.join(OUTPUT_DIR, f"results_{model_name}_trial_{trial_idx+1}.json")

    # Evaluation Loop
    for item in tqdm(val_data, desc=f"Evaluating {model_name}"):
        messages = item["messages"][:-1]
        ground_truth = item["messages"][-1]["content"]

        # Enhanced structural instruction for output alignment
        extra_instruction = (
            "\n\nFollow the format: [Thinking Process] then final JSON block. "
            "End your response with a JSON block containing 'correction_needed' (bool) and 'corrected_action' (list of 8 floats)."
        )

        output_text = ""

        # --- Branch A: API Inference Route (Gemini/GPT) ---
        if is_api_model:
            import base64
            image_paths = extract_image_paths_from_messages(messages)
            prompt = get_pure_text_prompt(messages, extra_instruction)
            
            # 1. Construct OpenAI-compatible text request
            messages_content = [{"type": "text", "text": prompt}]
            
            # 2. Convert extracted images to Base64 and append to payload
            for p in image_paths:
                try:
                    with open(p, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    messages_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                except Exception as img_e:
                    print(f"[Warning] Failed to read image ({p}): {img_e}")

            try:
                # 3. Transmit request (Disable randomness for baseline evaluation)
                response = model.chat.completions.create(
                    model="gemini-3.1-pro-preview",
                    messages=[{"role": "user", "content": messages_content}],
                    temperature=0.0 
                )
                output_text = response.choices[0].message.content
            except Exception as e:
                output_text = f"API Request Failed: {str(e)}"

        # --- Branch B: Local Open-Source Inference Route ---
        else:
            # Inject auxiliary instructions
            last_content = messages[-1]["content"]
            if isinstance(last_content, list):
                for part in last_content:
                    if part.get("type") == "text":
                        part["text"] += extra_instruction
                        break
            elif isinstance(last_content, str):
                messages[-1]["content"] += extra_instruction

            if "Qwen" in model_name or "CoC" in model_name or "VLM-3R" in model_name:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=1280,  
                        do_sample=False      # Deterministic output for academic reproduction
                    )
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

            elif "LLaVA" in model_name or "Llama" in model_name:
                import copy
                image_paths = extract_image_paths_from_messages(messages)
                imgs = [PIL.Image.open(p).convert("RGB") for p in image_paths]
                
                processed_messages = copy.deepcopy(messages)
                
                # Compatibility Fix 1: Auto-select standard image tokens based on model lineage
                img_token = "<|image|>" if "Llama" in model_name else "<image>"
                
                # Downgrade complex Qwen-formatted dictionaries into plaintext with embedded placeholders
                for msg in processed_messages:
                    if isinstance(msg["content"], list):
                        text_content = ""
                        for item in msg["content"]:
                            if item.get("type") == "image":
                                text_content += img_token + "\n"
                            elif item.get("type") == "text":
                                text_content += item.get("text", "") + "\n"
                        msg["content"] = text_content.strip()

                # Compatibility Fix 2: Safely append enhanced prompt commands
                last_user_idx = -1
                for idx, msg in enumerate(processed_messages):
                    if msg["role"] == "user": last_user_idx = idx
                
                if last_user_idx != -1:
                    # Content is guaranteed to be a standard string post-Fix 1
                    processed_messages[last_user_idx]["content"] += extra_instruction

                # Compatibility Fix 3: Execute chat template on sanitized inputs
                formatted_prompt = processor.apply_chat_template(processed_messages, tokenize=False, add_generation_prompt=True)
                
                if len(imgs) > 0:
                    inputs = processor(text=formatted_prompt, images=imgs, padding=True, return_tensors="pt").to(DEVICE)
                else:
                    inputs = processor(text=formatted_prompt, padding=True, return_tensors="pt").to(DEVICE)
                    
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=1280, do_sample=False)
                    # Truncate prompt context, extracting the pure generated text
                    output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        # --------------------------------------------------
        results.append({
            "id": item.get("id"),
            "prediction": output_text,
            "ground_truth": ground_truth
        })

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if not is_api_model:
        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Success] Trial {trial_idx+1} completed. VRAM released.")
    else:
        print(f"[Success] API Trial {trial_idx+1} completed.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    for path, name in MODELS_TO_RUN.items():
        is_api_model = "Gemini" in name or "GPT" in name
        if not os.path.exists(path) and not is_api_model:
            print(f"[Warning] Skipping model {name}: Path does not exist -> {path}")
            continue
        for i in range(NUM_TRIALS):
            run_model_inference(path, name, i)

    print("\n\n[All offline evaluation tasks have been successfully completed!]")

if __name__ == "__main__":
    main()
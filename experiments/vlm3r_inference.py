import json
import os
import gc
import torch
from tqdm import tqdm
import PIL.Image
import sys

# ================= System Path Configuration =================
# Append VLM-3R and CUT3R source paths to the system environment 
# to ensure local modules can be imported correctly.
VLM3R_CODE_PATH = "/home/zxn/Chain-of-Correction/baselines/VLM-3R"
sys.path.append(VLM3R_CODE_PATH)
sys.path.append(os.path.join(VLM3R_CODE_PATH, "CUT3R"))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
# =======================================================================

# ===================== Configuration =====================
VAL_SET_PATH = "/data/colosseum_dataset/ablations/offline_val_set_200.json"
OUTPUT_DIR = "/data/colosseum_dataset/ablations/eval_results"
DEVICE = "cuda:0"  
 
LORA_PATH = "/data/colosseum_dataset/hf_cache/huggingface/hub/VLM-3R"
BASE_PATH = "/data/colosseum_dataset/hf_cache/huggingface/hub/LLaVA-NeXT-Video-7B-Qwen2"
CUT3R_WEIGHTS = "/home/zxn/Chain-of-Correction/baselines/VLM-3R/CUT3R/src/cut3r_512_dpt_4_64.pth"
# ===================================================

def extract_image_paths(messages):
    """
    Extract physical image paths from the structured message format.
    
    Args:
        messages (list): List of conversation messages.
    Returns:
        list: Physical paths to image files.
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

def get_pure_text_prompt(messages):
    """
    Concatenate all text components from multimodal messages into a single string.
    
    Args:
        messages (list): List of conversation messages.
    Returns:
        str: Flattened text prompt.
    """
    prompt = ""
    for msg in messages:
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item.get("type") == "text":
                    prompt += item.get("text", "") + "\n"
        elif isinstance(msg["content"], str):
            prompt += msg["content"] + "\n"
    return prompt.strip()

def run_vlm3r():
    """
    Main evaluation routine for VLM-3R. 
    Implements manual model loading to bypass builder defects, injects LoRA 
    weights, and activates the CUT3R spatial encoder.
    """
    print(f"\n[Task Start] Evaluating model: VLM-3R-LoRA")

    # ================= Core Model Loading Fix =================
    # Manually mounting components to bypass official builder issues.
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    from transformers import AutoTokenizer
    from peft import PeftModel
    from llava.mm_utils import process_images

    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, use_fast=False)
     
    base_model = LlavaQwenForCausalLM.from_pretrained(
        BASE_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": DEVICE}
    )
    
    # Inject VLM-3R LoRA weights
    print("   -> Injecting VLM-3R 3D spatial core weights...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    # Manually activate the vision tower
    print("   -> Activating vision tower...")
    vision_tower = model.get_model().get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=DEVICE, dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor
    # ========================================================

    # Load CUT3R Spatial Encoder weights
    if hasattr(model.get_model(), 'spatial_encoder'):
        print("   -> Spatial Encoder (CUT3R) detected. Loading pretrained weights...")
        model.get_model().spatial_encoder.load_weights(CUT3R_WEIGHTS)
        model.get_model().spatial_encoder.to(device=DEVICE, dtype=torch.bfloat16)

    with open(VAL_SET_PATH, 'r') as f:
        val_data = json.load(f)

    results = []
    output_filename = os.path.join(OUTPUT_DIR, f"results_VLM-3R_trial_3.json")

    extra_instruction = (
        "\n\nFollow the format: [Thinking Process] then final JSON block. "
        "End your response with a JSON block containing 'correction_needed' (bool) and 'corrected_action' (list of 8 floats)."
    )

    # Evaluation Loop
    for item in tqdm(val_data, desc="Evaluating VLM-3R"):
        messages = item["messages"][:-1]
        ground_truth = item["messages"][-1]["content"]

        # 1. Image loading
        image_paths = extract_image_paths(messages)
        raw_images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                raw_images.append(PIL.Image.open(img_path).convert('RGB'))
            else:
                print(f"   [Warning] Image file not found: {img_path}")

        # 2. Prompt Construction
        # Utilizing standard chat templates to ensure prompt formatting consistency.
        prompt_text = get_pure_text_prompt(messages) + extra_instruction
         
        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize using original tokenizer to ensure no negative token IDs (e.g., -200) occur.
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)

        # 3. Multimodal Feature Extraction
        image_tensor = None
        spatial_features = None
        image_sizes = None

        if len(raw_images) > 0:
            # A. Vision Tower processing (LLaVA-NeXT AnyRes)
            try:
                from llava.mm_utils import process_images
                image_tensor = process_images(raw_images, image_processor, model.config)
                if isinstance(image_tensor, list):
                    image_tensor = [t.to(DEVICE, dtype=torch.bfloat16) for t in image_tensor]
                else:
                    image_tensor = image_tensor.to(DEVICE, dtype=torch.bfloat16)
                image_sizes = [img.size for img in raw_images]
            except Exception as e:
                print(f"   [Warning] Vision Tower processing failed: {e}")
                image_tensor = image_processor.preprocess(raw_images, return_tensors='pt')['pixel_values'].to(DEVICE, dtype=torch.bfloat16)
                image_sizes = [img.size for img in raw_images]

            # B. 3D Spatial Feature processing (CUT3R)
            try:
                spatial_features_list = []
                for img in raw_images: 
                    feat = model.get_model().spatial_encoder(img)
                    spatial_features_list.append(feat)
                
                if len(spatial_features_list) > 0:
                    spatial_features = torch.stack(spatial_features_list).to(device=DEVICE, dtype=torch.bfloat16)
            except Exception as e:
                print(f"   [Warning] CUT3R feature extraction failed: {e}")
                spatial_features = None

        # 4. Inference Execution
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    spatial_features=spatial_features,
                    max_new_tokens=1280,
                    do_sample=False,
                    use_cache=True
                )
            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            output_text = f"Inference Failed Error: {str(e)}"

        results.append({
            "id": item.get("id"),
            "prediction": output_text,
            "ground_truth": ground_truth
        })

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"[Success] VLM-3R evaluation completed.")

if __name__ == "__main__":
    run_vlm3r()
import torch
from safetensors.torch import save_file
from transformers import Qwen2_5_VLForConditionalGeneration
import os

# ================= Configuration =================
# Define paths for the conversion process
ckpt_dir = "/data/colosseum_dataset/checkpoints/sft/checkpoint-final"
bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
save_path = os.path.join(ckpt_dir, "model.safetensors")

print(f"[Process] Converting weight format: {input_bin_path} -> {output_safetensors_path}")

# ================= Loading Weights =================
# Load the PyTorch state_dict. 
# We map to 'cpu' to avoid unnecessary VRAM consumption during serialization.
state_dict = torch.load(bin_path, map_location="cpu")

# ================= Safetensors Serialization =================
# Convert the state_dict to SafeTensors format.
# This format is faster to load, safer (prevents arbitrary code execution), 
# and is the modern standard for Hugging Face models like Qwen2.5-VL.
save_file(state_dict, save_path)

print("[Success] Conversion complete! 'model.safetensors' has been generated.")
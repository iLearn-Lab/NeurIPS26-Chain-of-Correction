import torch
from safetensors.torch import save_file
from transformers import Qwen2_5_VLForConditionalGeneration
import os

# 1. 设定路径
ckpt_dir = "/data/colosseum_dataset/checkpoints/sft/checkpoint-final"
bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
save_path = os.path.join(ckpt_dir, "model.safetensors")

print(f"🔄 正在转换权重格式: {bin_path} -> {save_path}")

# 2. 加载旧的权重 (使用 cpu 即可，避免占用显存)
state_dict = torch.load(bin_path, map_location="cpu")

# 3. 转换为 safetensors 格式
# 注意：Qwen2.5-VL 可能会有一些非张量的 metadata，save_file 会自动处理
save_file(state_dict, save_path)

print("✅ 转换完成！model.safetensors 已生成。")
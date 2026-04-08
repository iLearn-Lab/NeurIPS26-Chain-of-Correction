import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

# ================= 配置路径 =================
base_model_path = "/home/zxn/MomaGraph-R1" # 你的基础模型
sft_bin_path = "/data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1/pytorch_model.bin"
save_path = "/data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1-hf" # 洗白后的新目录

def _remap_key(k): 
    if "lm_head" in k: return "lm_head.weight" if "weight" in k else "lm_head.bias"
    if "visual." in k: return "model.visual." + k.split("visual.", 1)[1]
    for keyword in ["layers.", "embed_tokens.", "norm."]:
        if keyword in k: return "model.language_model." + keyword + k.split(keyword, 1)[1]
    return k 

print("1. 正在加载基座配置与结构...")
processor = Qwen2_5_VLProcessor.from_pretrained(base_model_path, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu", 
    trust_remote_code=True
)

print("2. 正在加载并映射 SFT 权重...")
state_dict = torch.load(sft_bin_path, map_location="cpu")
new_state_dict = {_remap_key(k): v for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

print(f"3. 正在保存为标准 HuggingFace 格式至: {save_path} ...")
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print("✅ 转换完成！")
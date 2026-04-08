#!/bin/bash
echo "正在启动 VLM 纠错..."

# 1. 加载 Conda 环境 (路径请根据你的实际 anaconda/miniconda 安装位置修改)
source ~/anaconda3/etc/profile.d/conda.sh

# 2. 激活高版本 transformers 的专属环境
conda activate cot_sft

# 3. 启动 Flask 服务端 
python3 /home/zxn/Chain-of-Correction/vlm_corrector_server.py --ckpt /data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1 --gpu cuda:4
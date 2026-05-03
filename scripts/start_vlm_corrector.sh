#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate cot_sft

python3 /home/zxn/Chain-of-Correction/vlm_corrector_server.py --ckpt /data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1 --gpu cuda:4
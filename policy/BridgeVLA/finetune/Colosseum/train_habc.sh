#!/bin/bash
 
cd /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune 

export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0

cd Colosseum
 
port=29601
GPUS_PER_NODE=4 
NNODES=1
 
export CUDA_VISIBLE_DEVICES=1,2,4,5

echo " GPU $CUDA_VISIBLE_DEVICES for training HA-BC (BridgeVLA) ..."

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$port \
    train_habc.py \
    --exp_cfg_path configs/colosseum_config.yaml \
    --exp_note "HA-BC_Finetune" \
    --freeze_vision_tower \
    --log_dir /data/colosseum_dataset/logs/habc \
    --load_pretrain \
    --pretrain_path /home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/pretrain \
    --epochs 81 \
    --resume_ckpt /data/colosseum_dataset/logs/habc/train/debug/HA-BC_Finetune/04_22_10_24/model_last.pth
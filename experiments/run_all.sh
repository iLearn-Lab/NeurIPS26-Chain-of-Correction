#!/bin/bash

# ==========================================
# Unified logging directory
# ==========================================
LOG_DIR="/home/zxn/Chain-of-Correction/logs"
mkdir -p "$LOG_DIR"
NUM_TRIALS=3

echo "========================================================="
echo "[Orchestrator] Initiating highly concurrent rolling evaluation tasks (Total $NUM_TRIALS trials) | Time: $(date)"
echo "========================================================="

source $(conda info --base)/etc/profile.d/conda.sh

# ==========================================
# Step 1: Launch VLM engines (Persistent across all evaluation trials)
# ==========================================
echo -e "\n[1/4] Activating 'cot_sft' environment, allocating GPUs, and launching VLM servers..."
conda activate cot_sft

# --- GPU 0 ---
CUDA_VISIBLE_DEVICES=0 nohup python3 vlm_corrector_server.py --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_2_step_442/actor/lora_adapter --gpu cuda:0 --port 5070 > $LOG_DIR/server_5070.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python3 vlm_corrector_server.py --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_2_step_442/actor/lora_adapter --gpu cuda:0 --port 5071 > $LOG_DIR/server_5071.log 2>&1 &

# --- GPU 1 ---
CUDA_VISIBLE_DEVICES=1 nohup python3 vlm_corrector_server.py --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_2_step_442/actor/lora_adapter --gpu cuda:0 --port 5072 > $LOG_DIR/server_5072.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python3 vlm_corrector_server.py --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_2_step_442/actor/lora_adapter --gpu cuda:0 --port 5073 > $LOG_DIR/server_5073.log 2>&1 &

# --- GPU 2 ---
CUDA_VISIBLE_DEVICES=2 nohup python3 vlm_corrector_server.py --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_2_step_442/actor/lora_adapter --gpu cuda:0 --port 5074 > $LOG_DIR/server_5074.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 vlm_corrector_server.py --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_2_step_442/actor/lora_adapter --gpu cuda:0 --port 5075 > $LOG_DIR/server_5075.log 2>&1 &

# --- GPU 3 ---
CUDA_VISIBLE_DEVICES=3 nohup python3 vlm_corrector_server.py --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_2_step_442/actor/lora_adapter --gpu cuda:0 --port 5076 > $LOG_DIR/server_5076.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 vlm_corrector_server.py --ckpt /data/colosseum_dataset/easyr1_ckpt/checkpoints/easy_r1/momagraph_dapo_multitask_baseline/epoch_archive/epoch_3_step_663/actor/lora_adapter --gpu cuda:0 --port 5077 > $LOG_DIR/server_5077.log 2>&1 &

echo -e "\n[2/4] Initiating 120-second warmup sequence to ensure models are fully loaded into VRAM..."
for i in {1..6}; do sleep 20; echo "   ... Health Check: Waited $((i*20)) seconds"; done
echo "[Status] Warmup complete. All VLM ports should be ready!"


# ==========================================
# Step 2: Define the independent serial evaluation pipeline function
# ==========================================
echo -e "\n[3/4] Activating 'bridgevla' environment and launching independent pipelines..."
conda activate bridgevla

# Pipeline function: Executes multiple trials sequentially within a single background thread
run_eval_pipeline() {
    local gpu_id=$1
    local port=$2
    local exp_name_prefix=$3
    local epoch=$4
    local ckpt_dir=$5
    local sim_gpu=$6
    local task_name=$7

    for trial in $(seq 1 $NUM_TRIALS); do
        echo "[Pipeline | GPU $gpu_id | Port $port] Starting: $task_name (Trial $trial / $NUM_TRIALS)"
        
        # Critical: 'nohup' and background '&' are intentionally omitted here to ensure serial execution per port.
        # This prevents multiple trials from colliding over a single VLM port constraint.
        CUDA_VISIBLE_DEVICES=$gpu_id VLM_PORT=$port bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_worker.sh \
            "${exp_name_prefix}_trial${trial}" $epoch "$ckpt_dir" $sim_gpu "$task_name" \
            > "$LOG_DIR/eval_${exp_name_prefix}_${task_name}_trial${trial}.log" 2>&1
            
    done
    echo "[Pipeline Complete] $task_name concluded $NUM_TRIALS trials. Releasing GPU $gpu_id resources."
}

# ==========================================
# Step 3: Dispatch all 8 pipelines into the background concurrently.
# (The trailing '&' executes the function block asynchronously)
# ==========================================

# --- GPU 0 ---
run_eval_pipeline 0 5070 "rl_ep2" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 0 "meat_on_grill" &
run_eval_pipeline 0 5071 "rl_ep2" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 0 "move_hanger" &

# --- GPU 1 ---
run_eval_pipeline 1 5072 "rl_ep2" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 1 "open_drawer" &
run_eval_pipeline 1 5073 "rl_ep2" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 1 "place_wine_at_rack_location" &

# --- GPU 2 ---
run_eval_pipeline 2 5074 "rl_ep2" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 2 "put_money_in_safe" &
run_eval_pipeline 2 5075 "rl_ep2" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 2 "turn_oven_on" &

# --- GPU 3 ---
run_eval_pipeline 3 5076 "rl_ep2" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 3 "setup_chess" &
run_eval_pipeline 3 5077 "rl_ep3" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 3 "meat_on_grill" &


echo -e "\n[4/4] All 8 evaluation pipelines have been dispatched to the background!"
echo "[Status] Main orchestrator process suspended. Awaiting completion of all background tasks..."

# Final synchronization block: Wait for all background pipeline functions to terminate.
wait

echo "[Success] All 24 evaluations across 8 tasks successfully concluded!"
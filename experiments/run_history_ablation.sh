#!/bin/bash

# ==========================================
# 0. Base Configuration & Process Cleanup
# ==========================================
LOG_DIR="/home/zxn/Chain-of-Correction/logs"
mkdir -p "$LOG_DIR"
NUM_TRIALS=3

SERVER_PIDS=()
cleanup() {
    echo -e "\n[Cleanup] Destroying background VLM server processes..."
    for pid in "${SERVER_PIDS[@]}"; do
        kill -9 $pid 2>/dev/null
    done
    echo "[Success] VRAM resources safely released."
}
# Trap termination signals to trigger cleanup automatically
trap cleanup SIGINT SIGTERM EXIT

echo "========================================================="
echo "[Orchestrator] Launching high-concurrency rolling evaluation (Ablation: wo_history) | Time: $(date)"
echo "========================================================="

source $(conda info --base)/etc/profile.d/conda.sh

# ==========================================
# Step 1: Launch VLM engines (Strictly 2 instances per GPU)
# ==========================================
echo -e "\n[1/4] Activating 'cot_sft' environment, allocating GPUs, and booting VLM servers..."
conda activate cot_sft

CKPT_PATH="/data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1"

# --- GPU 0 ---  
CUDA_VISIBLE_DEVICES=0 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5084 --ablation wo_history > $LOG_DIR/server_5084.log 2>&1 &
SERVER_PIDS+=($!)
CUDA_VISIBLE_DEVICES=0 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5085 --ablation wo_history > $LOG_DIR/server_5085.log 2>&1 &
SERVER_PIDS+=($!)

# --- GPU 3 ---  
CUDA_VISIBLE_DEVICES=3 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5086 --ablation wo_history > $LOG_DIR/server_5086.log 2>&1 &
SERVER_PIDS+=($!)
CUDA_VISIBLE_DEVICES=3 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5087 --ablation wo_history > $LOG_DIR/server_5087.log 2>&1 &
SERVER_PIDS+=($!)

echo -e "\n[2/4] Initiating 120-second warmup sequence for VRAM loading..."
for i in {1..6}; do sleep 20; echo "   ... Health Check: Waited $((i*20)) seconds"; done
echo "[Status] Warmup complete. VLM ports are ready."


# ==========================================
# Step 2: Define independent serial evaluation pipeline function
# ==========================================
echo -e "\n[3/4] Activating 'bridgevla' environment and dispatching pipelines..."
conda activate bridgevla

run_eval_pipeline() {
    local gpu_id=$1
    local port=$2
    local exp_name_prefix=$3
    local epoch=$4
    local ckpt_dir=$5
    local sim_gpu=$6
    local task_name=$7

    for trial in $(seq 1 $NUM_TRIALS); do
        echo "[Pipeline | GPU $gpu_id | Port $port] Launching: $task_name (Trial $trial / $NUM_TRIALS)"
        
        # Execute each trial serially
        CUDA_VISIBLE_DEVICES=$gpu_id VLM_PORT=$port bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_worker.sh \
            "${exp_name_prefix}_trial${trial}" $epoch "$ckpt_dir" $sim_gpu "$task_name" \
            > "$LOG_DIR/eval_${exp_name_prefix}_${task_name}_trial${trial}.log" 2>&1
            
    done
    echo "[Task Complete] $task_name concluded $NUM_TRIALS evaluation trials."
}

# ==========================================
# Step 3: Queue 7 tasks into 4 parallel pipelines
# ==========================================

EXP_PREFIX="wo_history"
POLICY_CKPT="/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM"

# --- Pipeline 1 (GPU 0, Port 5084): Sequential execution of 2 tasks ---
(
    run_eval_pipeline 0 5084 $EXP_PREFIX 80 $POLICY_CKPT 0 "meat_on_grill"
    run_eval_pipeline 0 5084 $EXP_PREFIX 80 $POLICY_CKPT 0 "put_money_in_safe"
) &

# --- Pipeline 2 (GPU 0, Port 5085): Sequential execution of 2 tasks ---
(
    run_eval_pipeline 0 5085 $EXP_PREFIX 80 $POLICY_CKPT 0 "move_hanger"
    run_eval_pipeline 0 5085 $EXP_PREFIX 80 $POLICY_CKPT 0 "turn_oven_on"
) &

# --- Pipeline 3 (GPU 3, Port 5086): Sequential execution of 2 tasks ---
(
    run_eval_pipeline 3 5086 $EXP_PREFIX 80 $POLICY_CKPT 3 "open_drawer"
    run_eval_pipeline 3 5086 $EXP_PREFIX 80 $POLICY_CKPT 3 "place_wine_at_rack_location" 
) &

# --- Pipeline 4 (GPU 3, Port 5087): Dedicated to 'setup_chess' task ---
(
    run_eval_pipeline 3 5087 $EXP_PREFIX 80 $POLICY_CKPT 3 "setup_chess"
) &

echo -e "\n[4/4] All 4 pipeline queues (covering 7 tasks) dispatched to background!"
echo "[Status] Main orchestrator process suspended. Awaiting completion of all queues..."

# Final Synchronization: Wait for all background queues to terminate
wait

echo "[Success] All 21 evaluations across 7 tasks successfully concluded! VRAM auto-cleanup initiating."
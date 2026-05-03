#!/bin/bash

# ==========================================
# 0. Base Configuration & Cross-Process Cleanup
# ==========================================
LOG_DIR="/home/zxn/Chain-of-Correction/logs"
mkdir -p "$LOG_DIR"
NUM_TRIALS=3

# Target PIDs to monitor for resource release (Currently bypassed/commented out below)
WAIT_PID_GPU1=1702307  # Task currently occupying resources intended for Thread 1
WAIT_PID_GPU0=1917969  # Task currently occupying resources intended for Thread 2

# Utilize a temporary file to track newly launched VLM PIDs across sub-processes
PID_FILE="$LOG_DIR/vlm_pids_geo_$$.txt"
> "$PID_FILE" 

cleanup() {
    echo -e "\n[Cleanup Triggered] Executing cross-process termination of all background VLM servers..."
    if [ -f "$PID_FILE" ]; then
        while read pid; do
            kill -9 $pid 2>/dev/null
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
    echo "[Success] VRAM resources safely released!"
}
trap cleanup SIGINT SIGTERM EXIT

echo "========================================================="
echo "[Orchestrator] Dual-Track Asynchronous Concurrent Launch (Ablation: wo_geometric) | Time: $(date)"
echo "========================================================="

source $(conda info --base)/etc/profile.d/conda.sh

CKPT_PATH="/data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1"
EXP_PREFIX="wo_geometric"
POLICY_CKPT="/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM"

# ==========================================
# 1. Define Evaluation Pipeline Dispatch Function
# ==========================================
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
        CUDA_VISIBLE_DEVICES=$gpu_id VLM_PORT=$port bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_worker.sh \
            "${exp_name_prefix}_trial${trial}" $epoch "$ckpt_dir" $sim_gpu "$task_name" \
            > "$LOG_DIR/eval_${exp_name_prefix}_${task_name}_trial${trial}.log" 2>&1
    done
    echo "[Task Complete] $task_name concluded $NUM_TRIALS evaluation trials."
}


# =====================================================================
# Daemon Thread 1: Resource Monitor and Auto-Takeover (Targeting GPU 3)
# =====================================================================
(
    # echo -e "\n[Daemon Standby] Monitoring prerequisite task (PID: $WAIT_PID_GPU1)..."
    # while kill -0 $WAIT_PID_GPU1 2>/dev/null; do
    #     sleep 60
    # done
    # echo -e "\n[Takeover Successful] Prerequisite task terminated. Awaiting 10 seconds for VRAM flush..."
    # sleep 10

    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate cot_sft

    echo "[GPU 3] Booting VLM servers (Ports 5090, 5091)..."
    CUDA_VISIBLE_DEVICES=3 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5090 --ablation wo_geometry > $LOG_DIR/server_5090.log 2>&1 &
    echo $! >> "$PID_FILE"
    CUDA_VISIBLE_DEVICES=3 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5091 --ablation wo_geometry > $LOG_DIR/server_5091.log 2>&1 &
    echo $! >> "$PID_FILE"

    echo "[GPU 3] Initiating 120-second model loading sequence..."
    sleep 120
    conda activate bridgevla

    # GPU 3 handles the distribution of 4 tasks
    (
        run_eval_pipeline 3 5090 $EXP_PREFIX 80 $POLICY_CKPT 3 "meat_on_grill"
        run_eval_pipeline 3 5090 $EXP_PREFIX 80 $POLICY_CKPT 3 "put_money_in_safe"
    ) &
    (
        run_eval_pipeline 3 5091 $EXP_PREFIX 80 $POLICY_CKPT 3 "move_hanger"
        run_eval_pipeline 3 5091 $EXP_PREFIX 80 $POLICY_CKPT 3 "turn_oven_on"
    ) &
    wait  
) &


# =====================================================================
# Daemon Thread 2: Resource Monitor and Auto-Takeover (Targeting GPU 0)
# =====================================================================
(
    # echo -e "\n[Daemon Standby] Monitoring prerequisite task (PID: $WAIT_PID_GPU0)..."
    # while kill -0 $WAIT_PID_GPU0 2>/dev/null; do
    #     sleep 60
    # done
    # echo -e "\n[Takeover Successful] Prerequisite task terminated. Awaiting 10 seconds for VRAM flush..."
    # sleep 10

    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate cot_sft

    echo "[GPU 0] Booting VLM servers (Ports 5092, 5093)..."
    CUDA_VISIBLE_DEVICES=0 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5092 --ablation wo_geometry > $LOG_DIR/server_5092.log 2>&1 &
    echo $! >> "$PID_FILE"
    CUDA_VISIBLE_DEVICES=0 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5093 --ablation wo_geometry > $LOG_DIR/server_5093.log 2>&1 &
    echo $! >> "$PID_FILE"

    echo "[GPU 0] Initiating 120-second model loading sequence..."
    sleep 120
    conda activate bridgevla

    # GPU 0 handles the distribution of the remaining 3 tasks
    (
        run_eval_pipeline 0 5092 $EXP_PREFIX 80 $POLICY_CKPT 0 "open_drawer"
        run_eval_pipeline 0 5092 $EXP_PREFIX 80 $POLICY_CKPT 0 "place_wine_at_rack_location" 
    ) &
    (
        run_eval_pipeline 0 5093 $EXP_PREFIX 80 $POLICY_CKPT 0 "setup_chess"
    ) &
    wait  
) &

# ==========================================
# Final Synchronization Block: Suspend main thread until all daemon subshells finish
# ==========================================
echo -e "\n[Status] Master process suspended. Dual-track daemon processes operating in the background..."
wait

echo "[Success] 'wo_geometric' ablation evaluation successfully concluded! VRAM cleanup sequence initiating."
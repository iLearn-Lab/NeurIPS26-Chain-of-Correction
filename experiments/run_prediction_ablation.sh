#!/bin/bash

LOG_DIR="/home/zxn/Chain-of-Correction/logs"
mkdir -p "$LOG_DIR"

SERVER_PIDS=()
cleanup() {
    echo -e "\n[Cleanup Triggered] Destroying 16 background VLM server processes..."
    for pid in "${SERVER_PIDS[@]}"; do kill -9 $pid 2>/dev/null; done
    echo "[Success] VRAM resources completely and safely released!"
}
trap cleanup SIGINT SIGTERM EXIT

echo "========================================================="
echo "[Orchestrator] 16-Pipeline Super Array Initiated | Time: $(date)"
echo "========================================================="

source $(conda info --base)/etc/profile.d/conda.sh

# ==========================================
# 1. Launch 16 VLM Engines (2 instances per GPU)
# ==========================================
echo -e "\n[1/3] Activating 'cot_sft' environment, allocating 16 server nodes..."
conda activate cot_sft

CKPT_PRED="/data/colosseum_dataset/checkpoints/sft/v1_wo_prediction/checkpoint-epoch-1"
CKPT_HIST="/data/colosseum_dataset/checkpoints/sft/v1/checkpoint-epoch-1"

# --- GPU 0-3 (8 Nodes): Allocated to wo_history ablation ---
CUDA_VISIBLE_DEVICES=0 nohup python3 vlm_corrector_server.py --ckpt $CKPT_HIST --gpu cuda:0 --port 5100 --ablation wo_history > $LOG_DIR/turbo_5100.log 2>&1 & SERVER_PIDS+=($!)
CUDA_VISIBLE_DEVICES=0 nohup python3 vlm_corrector_server.py --ckpt $CKPT_HIST --gpu cuda:0 --port 5101 --ablation wo_history > $LOG_DIR/turbo_5101.log 2>&1 & SERVER_PIDS+=($!)

CUDA_VISIBLE_DEVICES=1 nohup python3 vlm_corrector_server.py --ckpt $CKPT_HIST --gpu cuda:0 --port 5102 --ablation wo_history > $LOG_DIR/turbo_5102.log 2>&1 & SERVER_PIDS+=($!)
CUDA_VISIBLE_DEVICES=1 nohup python3 vlm_corrector_server.py --ckpt $CKPT_HIST --gpu cuda:0 --port 5103 --ablation wo_history > $LOG_DIR/turbo_5103.log 2>&1 & SERVER_PIDS+=($!)

CUDA_VISIBLE_DEVICES=2 nohup python3 vlm_corrector_server.py --ckpt $CKPT_HIST --gpu cuda:0 --port 5104 --ablation wo_history > $LOG_DIR/turbo_5104.log 2>&1 & SERVER_PIDS+=($!)
CUDA_VISIBLE_DEVICES=2 nohup python3 vlm_corrector_server.py --ckpt $CKPT_HIST --gpu cuda:0 --port 5105 --ablation wo_history > $LOG_DIR/turbo_5105.log 2>&1 & SERVER_PIDS+=($!)

CUDA_VISIBLE_DEVICES=3 nohup python3 vlm_corrector_server.py --ckpt $CKPT_HIST --gpu cuda:0 --port 5106 --ablation wo_history > $LOG_DIR/turbo_5106.log 2>&1 & SERVER_PIDS+=($!)
CUDA_VISIBLE_DEVICES=3 nohup python3 vlm_corrector_server.py --ckpt $CKPT_HIST --gpu cuda:0 --port 5107 --ablation wo_history > $LOG_DIR/turbo_5107.log 2>&1 & SERVER_PIDS+=($!)

# --- GPU 4-7 (8 Nodes): Allocated to wo_prediction ablation ---
CUDA_VISIBLE_DEVICES=4 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PRED --gpu cuda:0 --port 5108 --ablation wo_prediction > $LOG_DIR/turbo_5108.log 2>&1 & SERVER_PIDS+=($!)
CUDA_VISIBLE_DEVICES=4 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PRED --gpu cuda:0 --port 5109 --ablation wo_prediction > $LOG_DIR/turbo_5109.log 2>&1 & SERVER_PIDS+=($!)

CUDA_VISIBLE_DEVICES=5 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PRED --gpu cuda:0 --port 5110 --ablation wo_prediction > $LOG_DIR/turbo_5110.log 2>&1 & SERVER_PIDS+=($!)
CUDA_VISIBLE_DEVICES=5 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PRED --gpu cuda:0 --port 5111 --ablation wo_prediction > $LOG_DIR/turbo_5111.log 2>&1 & SERVER_PIDS+=($!)

CUDA_VISIBLE_DEVICES=6 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PRED --gpu cuda:0 --port 5112 --ablation wo_prediction > $LOG_DIR/turbo_5112.log 2>&1 & SERVER_PIDS+=($!)
CUDA_VISIBLE_DEVICES=6 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PRED --gpu cuda:0 --port 5113 --ablation wo_prediction > $LOG_DIR/turbo_5113.log 2>&1 & SERVER_PIDS+=($!)

CUDA_VISIBLE_DEVICES=7 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PRED --gpu cuda:0 --port 5114 --ablation wo_prediction > $LOG_DIR/turbo_5114.log 2>&1 & SERVER_PIDS+=($!)
CUDA_VISIBLE_DEVICES=7 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PRED --gpu cuda:0 --port 5115 --ablation wo_prediction > $LOG_DIR/turbo_5115.log 2>&1 & SERVER_PIDS+=($!)

echo "[Status] Initiating 120-second wait sequence for all 16 models to load into VRAM..."
sleep 120

# ==========================================
# 2. Define Independent Pipeline Dispatcher
# ==========================================
echo -e "\n[2/3] Activating 'bridgevla' environment, dispatching 16 concurrent pipelines..."
conda activate bridgevla
POLICY_CKPT="/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM"

run_specific_trials() {
    local gpu_id=$1; local port=$2; local exp_prefix=$3; local task=$4; shift 4; local trials=("$@")
    for trial in "${trials[@]}"; do
        echo "[Pipeline | GPU $gpu_id | $exp_prefix | Port $port] Launching: $task (Trial $trial)"
        CUDA_VISIBLE_DEVICES=$gpu_id VLM_PORT=$port bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_worker.sh \
            "${exp_prefix}_trial${trial}" 80 "$POLICY_CKPT" $gpu_id "$task" \
            > "$LOG_DIR/turbo_${exp_prefix}_${task}_trial${trial}.log" 2>&1
    done
}

# ==========================================
# 3. Deploy 16-Pipeline Array (Extreme Partitioning to Eliminate Queues)
# ==========================================

# --- Zone A: wo_history (Occupying 8 pipelines on GPU 0-3) ---
( run_specific_trials 0 5100 "wo_history" "setup_chess" 1 2 ) &
( run_specific_trials 0 5101 "wo_history" "setup_chess" 3; run_specific_trials 0 5101 "wo_history" "open_drawer" 2 ) &
( run_specific_trials 1 5102 "wo_history" "place_wine_at_rack_location" 1 2 ) &
( run_specific_trials 1 5103 "wo_history" "place_wine_at_rack_location" 3; run_specific_trials 1 5103 "wo_history" "open_drawer" 3 ) &
( run_specific_trials 2 5104 "wo_history" "put_money_in_safe" 1 2 ) &
( run_specific_trials 2 5105 "wo_history" "put_money_in_safe" 3; run_specific_trials 2 5105 "wo_history" "meat_on_grill" 2 ) &
( run_specific_trials 3 5106 "wo_history" "turn_oven_on" 1 2 ) &
( run_specific_trials 3 5107 "wo_history" "turn_oven_on" 3; run_specific_trials 3 5107 "wo_history" "move_hanger" 2 3; run_specific_trials 3 5107 "wo_history" "meat_on_grill" 3 ) &

# --- Zone B: wo_prediction (Occupying 8 pipelines on GPU 4-7) ---
( run_specific_trials 4 5108 "wo_prediction" "setup_chess" 2 ) &
( run_specific_trials 4 5109 "wo_prediction" "setup_chess" 3 ) &
( run_specific_trials 5 5110 "wo_prediction" "place_wine_at_rack_location" 1 2 ) &
( run_specific_trials 5 5111 "wo_prediction" "place_wine_at_rack_location" 3 ) &
( run_specific_trials 6 5112 "wo_prediction" "put_money_in_safe" 1 2 ) &
( run_specific_trials 6 5113 "wo_prediction" "put_money_in_safe" 3 ) &
( run_specific_trials 7 5114 "wo_prediction" "open_drawer" 3 ) &
( run_specific_trials 7 5115 "wo_prediction" "meat_on_grill" 3 ) &

echo -e "\n[3/3] All 16 super-pipelines dispatched to background! Master process suspending..."
wait
echo "[Success] Ultimate evaluation array concluded! All checkpoint tasks successfully completed!"

# #!/bin/bash

# # Unified logging directory
# LOG_DIR="/home/zxn/Chain-of-Correction/logs"
# mkdir -p "$LOG_DIR"
# NUM_TRIALS=3

# SERVER_PIDS=()
# cleanup() {
#     echo -e "\n[Cleanup] Destroying background VLM server processes..."
#     for pid in "${SERVER_PIDS[@]}"; do
#         kill -9 $pid 2>/dev/null
#     done
#     echo "[Success] VRAM resources safely released."
# }
# # Trap termination signals to trigger cleanup automatically
# trap cleanup SIGINT SIGTERM EXIT

# echo "========================================================="
# echo "[Orchestrator] Launching high-concurrency rolling evaluation (Ablation: wo_prediction) | Time: $(date)"
# echo "========================================================="

# source $(conda info --base)/etc/profile.d/conda.sh

# # ==========================================
# # Step 1: Launch VLM engines (Strictly 2 instances per GPU)
# # ==========================================
# echo -e "\n[1/4] Activating 'cot_sft' environment, allocating GPUs, and booting VLM servers..."
# conda activate cot_sft

# # If using a specifically fine-tuned model, keep this path; if using the unified model, change to v1/checkpoint-epoch-1
# CKPT_PATH="/data/colosseum_dataset/checkpoints/sft/v1_wo_prediction/checkpoint-epoch-1"

# # --- GPU 6 --- (Ports 5080, 5081) Note: Added --ablation wo_prediction here
# CUDA_VISIBLE_DEVICES=6 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5080 --ablation wo_prediction > $LOG_DIR/server_5080.log 2>&1 &
# SERVER_PIDS+=($!)
# CUDA_VISIBLE_DEVICES=6 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5081 --ablation wo_prediction > $LOG_DIR/server_5081.log 2>&1 &
# SERVER_PIDS+=($!)

# # --- GPU 7 --- (Ports 5082, 5083) Note: Added --ablation wo_prediction here
# CUDA_VISIBLE_DEVICES=7 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5082 --ablation wo_prediction > $LOG_DIR/server_5082.log 2>&1 &
# SERVER_PIDS+=($!)
# CUDA_VISIBLE_DEVICES=7 nohup python3 vlm_corrector_server.py --ckpt $CKPT_PATH --gpu cuda:0 --port 5083 --ablation wo_prediction > $LOG_DIR/server_5083.log 2>&1 &
# SERVER_PIDS+=($!)

# echo -e "\n[2/4] Initiating 120-second warmup sequence for VRAM loading..."
# for i in {1..6}; do sleep 20; echo "   ... Health Check: Waited $((i*20)) seconds"; done
# echo "[Status] Warmup complete. VLM ports are ready."


# # ==========================================
# # Step 2: Define independent serial evaluation pipeline function
# # ==========================================
# echo -e "\n[3/4] Activating 'bridgevla' environment and dispatching pipelines..."
# conda activate bridgevla

# run_eval_pipeline() {
#     local gpu_id=$1
#     local port=$2
#     local exp_name_prefix=$3
#     local epoch=$4
#     local ckpt_dir=$5
#     local sim_gpu=$6
#     local task_name=$7

#     for trial in $(seq 1 $NUM_TRIALS); do
#         echo "[Pipeline | GPU $gpu_id | Port $port] Launching: $task_name (Trial $trial / $NUM_TRIALS)"
        
#         # Execute each trial serially
#         CUDA_VISIBLE_DEVICES=$gpu_id VLM_PORT=$port bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_worker.sh \
#             "${exp_name_prefix}_trial${trial}" $epoch "$ckpt_dir" $sim_gpu "$task_name" \
#             > "$LOG_DIR/eval_${exp_name_prefix}_${task_name}_trial${trial}.log" 2>&1
            
#     done
#     echo "[Task Complete] $task_name concluded $NUM_TRIALS evaluation trials."
# }

# # ==========================================
# # Step 3: Queue 8 tasks into 4 parallel pipelines
# # ==========================================

# EXP_PREFIX="wo_prediction"
# POLICY_CKPT="/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM"

# # --- Pipeline 1 (GPU 6, Port 5080): Sequential execution of 2 tasks ---
# (
#     run_eval_pipeline 6 5080 $EXP_PREFIX 80 $POLICY_CKPT 6 "meat_on_grill"
#     run_eval_pipeline 6 5080 $EXP_PREFIX 80 $POLICY_CKPT 6 "put_money_in_safe"
# ) &

# # --- Pipeline 2 (GPU 6, Port 5081): Sequential execution of 2 tasks ---
# (
#     run_eval_pipeline 6 5081 $EXP_PREFIX 80 $POLICY_CKPT 6 "move_hanger"
#     run_eval_pipeline 6 5081 $EXP_PREFIX 80 $POLICY_CKPT 6 "turn_oven_on"
# ) &

# # --- Pipeline 3 (GPU 7, Port 5082): Sequential execution of 2 tasks ---
# (
#     run_eval_pipeline 7 5082 $EXP_PREFIX 80 $POLICY_CKPT 7 "open_drawer"
#     run_eval_pipeline 7 5082 $EXP_PREFIX 80 $POLICY_CKPT 7 "place_wine_at_rack_location" 
# ) &

# # --- Pipeline 4 (GPU 7, Port 5083): Dedicated to 'setup_chess' task ---
# (
#     run_eval_pipeline 7 5083 $EXP_PREFIX 80 $POLICY_CKPT 7 "setup_chess"
# ) &

# echo -e "\n[4/4] All 4 pipeline queues (covering 7 tasks) dispatched to background!"
# echo "[Status] Main orchestrator process suspended. Awaiting completion of all queues..."

# # Final Synchronization: Wait for all background queues to terminate
# wait

# echo "[Success] All 21 evaluations across 7 tasks successfully concluded! VRAM auto-cleanup initiating."
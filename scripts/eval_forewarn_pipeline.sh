#!/bin/bash

# ==============================================================================
# Helper Function: Resolve Checkpoint Paths
# Maps task names to their respective LoRA and World Model (WM) weights.
# ==============================================================================
get_checkpoint_paths() {
    local TASK=$1
    export LORA_CKPT="/data/colosseum_dataset/wm_data/finetuned_models/vlm_${TASK}/peft_checkpoint_5"
    
    # Conditional timestamp matching: specific tasks utilize the 153050 window, others use 102201.
    if [[ "$TASK" == "setup_chess" || "$TASK" == "turn_oven_on" || "$TASK" == "scoop_with_spatula" ]]; then
        export WM_CKPT="/home/zxn/Forewarn/logs/dreamer_cont/${TASK}/0408/153050/best_pretrain_joint_0_00.pt"
    else
        export WM_CKPT="/home/zxn/Forewarn/logs/dreamer_cont/${TASK}/0408/102201/best_pretrain_joint_0_00.pt"
    fi
}

# ==============================================================================
# Pipeline Execution Function
# Workflow: Resource Cleanup -> Server Boot -> VRAM Warmup -> Client Evaluation
# ==============================================================================
execute_eval_pipeline() {
    local TASK_NAME=$1
    local LORA_CKPT=$2
    local WM_CKPT=$3
    local GPU_ID=$4
    local PORT=$5
    local SCREEN_NUM=$6

    echo "----------------------------------------------------------"
    echo "[Dispatch] Task: $TASK_NAME assigned to GPU: $GPU_ID (Port: $PORT, Display: X$SCREEN_NUM)"
    echo "----------------------------------------------------------"

    # 1. Environment Cleanup
    echo "[Cleanup] Terminating processes on Port $PORT and resetting X-display $SCREEN_NUM..."
    fuser -k $PORT/tcp 2>/dev/null
    rm -f /tmp/.X${SCREEN_NUM}-lock
    rm -f /tmp/.X11-unix/X${SCREEN_NUM}
    sleep 5

    # 2. Launch VLM Backend Server
    env CUDA_VISIBLE_DEVICES=$GPU_ID /home/zxn/anaconda3/envs/dreamer/bin/python3 -u /home/zxn/Forewarn/vlm/llama-recipes/vlm_forewarn_server.py \
        --task "$TASK_NAME" \
        --ckpt "$LORA_CKPT" \
        --wm_ckpt "$WM_CKPT" \
        --port $PORT \
        --gpu cuda:0 > "/home/zxn/Chain-of-Correction/logs/server_new_${TASK_NAME}_gpu${GPU_ID}.log" 2>&1 &

    SERVER_PID=$!

    # 3. Model Weight Loading Period
    echo "[Status] Warmup: Waiting 90 seconds for weight injection into GPU $GPU_ID..."
    sleep 90 

    # 4. Launch Policy Client (Headless rendering via Xvfb)
    echo "[Client] Initializing policy evaluation for $TASK_NAME..."
    xvfb-run -a --server-args='-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
        env CUDA_VISIBLE_DEVICES=$GPU_ID VLM_PORT=$PORT \
        bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_forewarn_worker.sh \
        "forewarn_new_${TASK_NAME}" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" "$GPU_ID" "$TASK_NAME" "/data/colosseum_dataset/test" \
        > "/home/zxn/Chain-of-Correction/logs/client_new_${TASK_NAME}_gpu${GPU_ID}.log" 2>&1

    # 5. Resource Release
    echo "[Teardown] Task $TASK_NAME complete. Releasing GPU $GPU_ID and Port $PORT..."
    kill -9 $SERVER_PID 2>/dev/null
    fuser -k $PORT/tcp 2>/dev/null
    sleep 10
}

# ==============================================================================
# Dynamic Task Queue Controller
# Manages parallel execution across 3 designated GPU slots.
# ==============================================================================

# Prerequisite Tasks
TASKS=("meat_on_grill" "setup_chess" "turn_oven_on" "put_money_in_safe" "place_wine_at_rack_location" "open_drawer" "move_hanger")

# Resource mapping for slots 0, 1, and 2
GPUS=(5 6 7)
PORTS=(6065 6066 6067)
SCREENS=(405 406 407)

# Track PIDs of active background processes (0 indicates an available slot)
PIDS=(0 0 0)

echo "[Initialization] Starting dynamic queue for core tasks..."

for TASK in "${TASKS[@]}"; do
    get_checkpoint_paths "$TASK"
    assigned=false
    
    # Polling loop: Wait for an available hardware slot
    while [ "$assigned" = false ]; do
        for i in 0 1 2; do
            # Check if slot is unused or if the previously assigned process has terminated
            if [ "${PIDS[$i]}" -eq 0 ] || ! kill -0 ${PIDS[$i]} 2>/dev/null; then
                
                # Dispatch task to background pipeline
                execute_eval_pipeline "$TASK" "$LORA_CKPT" "$WM_CKPT" "${GPUS[$i]}" "${PORTS[$i]}" "${SCREENS[$i]}" &
                
                # Record background PID and update assignment status
                PIDS[$i]=$!
                assigned=true
                break
            fi
        done
        
        # All slots busy: suspend thread before next polling cycle
        if [ "$assigned" = false ]; then
            sleep 10
        fi
    done
done

echo "[Status] All primary tasks dispatched. Awaiting final synchronization..."
wait 

echo "[Success] Primary task block complete."

# ==============================================================================
# Final High-Memory Task: Scoop with Spatula
# Executed after primary queue completion to maximize resource availability.
# ==============================================================================
echo "[Process] Launching final task: scoop_with_spatula..."
get_checkpoint_paths "scoop_with_spatula"

# Assign to Slot 0 (GPU 5) as all resources are now idle
execute_eval_pipeline "scoop_with_spatula" "$LORA_CKPT" "$WM_CKPT" 5 6065 405 &

wait
echo "[Final] All 8 FOREWARN evaluation pipelines successfully concluded."
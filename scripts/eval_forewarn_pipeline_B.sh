#!/bin/bash

# ==============================================================================
# Automated Evaluation Scheduler
# Workflow: Initialize Server -> Warmup (Weight Loading) -> Launch Client -> Cleanup
# ==============================================================================

run_eval_pipeline() {
    local TASK_NAME=$1
    local LORA_CKPT=$2
    local WM_CKPT=$3
    local SERVER_GPU=$4
    local CLIENT_GPU=$5
    local PORT=$6

    echo "----------------------------------------------------------"
    echo "[STEP 1] Launching FOREWARN Server for $TASK_NAME"
    echo "         GPU: $SERVER_GPU | Port: $PORT"
    echo "----------------------------------------------------------"
    
    # Launch VLM/FOREWARN backend in the background
    env CUDA_VISIBLE_DEVICES=$SERVER_GPU /home/zxn/anaconda3/envs/dreamer/bin/python3 -u /home/zxn/Forewarn/vlm/llama-recipes/vlm_forewarn_server.py \
        --task "$TASK_NAME" \
        --ckpt "$LORA_CKPT" \
        --wm_ckpt "$WM_CKPT" \
        --port $PORT \
        --gpu cuda:0 > "/home/zxn/Chain-of-Correction/logs/server_${TASK_NAME}_gpu${SERVER_GPU}.log" 2>&1 &

    SERVER_PID=$!
    
    # Required delay for loading large Llama and World Model weights into VRAM
    echo "[Status] Warmup: Waiting 90 seconds for model loading..."
    sleep 90 

    echo "----------------------------------------------------------"
    echo "[STEP 2] Launching Policy Client for $TASK_NAME"
    echo "         GPU: $CLIENT_GPU | Port: $PORT"
    echo "----------------------------------------------------------"
    
    # Client script executes 3 evaluation trials internally
    env CUDA_VISIBLE_DEVICES=$CLIENT_GPU VLM_PORT=$PORT bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_forewarn_worker.sh \
        "forewarn_${TASK_NAME}" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 0 "$TASK_NAME" "/data/colosseum_dataset/test" \
        > "/home/zxn/Chain-of-Correction/logs/client_${TASK_NAME}_gpu${CLIENT_GPU}.log" 2>&1

    echo "----------------------------------------------------------"
    echo "[STEP 3] Evaluation complete. Releasing VRAM and cleaning up..."
    echo "----------------------------------------------------------"
    
    # Force termination of the backend server
    kill -9 $SERVER_PID
    fuser -k $PORT/tcp 2>/dev/null
    
    # Cool-down period to ensure driver-level VRAM release
    sleep 15
}

# ==============================================================================
# Task Queue Scheduling
# ==============================================================================

# --- Task 1: Place Wine at Rack ---
run_eval_pipeline "place_wine_at_rack_location" \
    "/data/colosseum_dataset/wm_data/finetuned_models/vlm_place_wine_at_rack_location/peft_checkpoint_5" \
    "/home/zxn/Forewarn/logs/dreamer_cont/place_wine_at_rack_location/0408/102201/best_pretrain_joint_0_00.pt" \
    2 1 6061

# --- Task 2: Open Drawer ---
run_eval_pipeline "open_drawer" \
    "/data/colosseum_dataset/wm_data/finetuned_models/vlm_open_drawer/peft_checkpoint_5" \
    "/home/zxn/Forewarn/logs/dreamer_cont/open_drawer/0408/102201/best_pretrain_joint_0_00.pt" \
    2 1 6061

# --- Task 3: Scoop with Spatula ---
run_eval_pipeline "scoop_with_spatula" \
    "/data/colosseum_dataset/wm_data/finetuned_models/vlm_scoop_with_spatula/peft_checkpoint_5" \
    "/home/zxn/Forewarn/logs/dreamer_cont/scoop_with_spatula/0408/153050/best_pretrain_joint_0_00.pt" \
    2 1 6061

# --- Task 4: Move Hanger ---
run_eval_pipeline "move_hanger" \
    "/data/colosseum_dataset/wm_data/finetuned_models/vlm_move_hanger/peft_checkpoint_5" \
    "/home/zxn/Forewarn/logs/dreamer_cont/move_hanger/0408/102201/best_pretrain_joint_0_00.pt" \
    2 1 6061

echo "[Complete] All scheduled tasks in Pipeline B have concluded."
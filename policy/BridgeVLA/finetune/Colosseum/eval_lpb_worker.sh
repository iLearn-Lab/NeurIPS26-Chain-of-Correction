#!/bin/bash
# Example Usage: 
# bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_lpb_worker.sh "debug_lpb" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 4 "meat_on_grill move_hanger open_drawer put_money_in_safe place_wine_at_rack_location turn_oven_on setup_chess"

cd /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune

# ==============================================================================
# 1. Base Environment Configuration
# ==============================================================================
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT

export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT/platforms
export QT_PLUGIN_PATH=$COPPELIASIM_ROOT

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export GALLIUM_DRIVER=softpipe
export PYTHONPATH=/home/zxn/Chain-of-Correction:$PYTHONPATH

# ==============================================================================
# 2. Receive External Parameters
# ==============================================================================
export trial_prefix=$1 
export epoch=$2 
export model_folder=$3 
export gpu_id=$4       
export task_list_str=$5 

export eval_episodes=25
export DEMO_PATH_ROOT=/data/colosseum_dataset/test
export EVAL_ROOT=/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM/opensource_colosseum_lpb

# ==============================================================================
# 3. Core Evaluation Loop (3 Rounds x Tasks -> Variations)
# ==============================================================================
for round_idx in 1 2 3; do
    echo "----------------------------------------------------------"
    echo "[Init] Round $round_idx / 3: Starting LPB evaluation loop..."
    echo "----------------------------------------------------------"
    
    for root_task in $task_list_str; do
        export DEMO_PATH=$DEMO_PATH_ROOT/$root_task
        
        for current_var in {0..14}; do
            export task_variant="${root_task}_${current_var}"
            export DATA_PATH="$DEMO_PATH/$task_variant"

            # Check if variation directory exists; bypass if missing
            if [ ! -d "$DATA_PATH" ]; then
                echo "[Skip] GPU $gpu_id | Round $round_idx: Task $root_task Variation $current_var not found."
                continue
            fi    

            # ==========================================================
            # Dynamic Directory Repair Mechanism 
            # Detects and fixes missing 'variation0' hierarchies
            # ==========================================================
            if [ ! -d "$DATA_PATH/variation0" ] && [ -d "$DATA_PATH/episodes" ]; then
                echo "[Repair] Invalid directory structure detected for $task_variant. Dynamically fixing..."
                
                # 1. Create missing variation0 folder
                mkdir -p "$DATA_PATH/variation0"
                
                # 2. Move episodes folder inside variation0
                mv "$DATA_PATH/episodes" "$DATA_PATH/variation0/"
                
                # 3. Extract variation_descriptions.pkl to variation0 root
                pkl_file=$(find "$DATA_PATH/variation0/episodes" -name "variation_descriptions.pkl" | head -n 1)
                if [ -n "$pkl_file" ]; then
                    cp "$pkl_file" "$DATA_PATH/variation0/"
                    echo "[Success] Repair complete: 'variation0' hierarchy restored and pkl file extracted."
                else
                    echo "[Warning] 'variation_descriptions.pkl' not found. Evaluation metadata might be incomplete."
                fi
            fi
            # ==========================================================
            
            # Construct dedicated output directory for this evaluation run
            export current_log_dir="$EVAL_ROOT/epoch_${epoch}_${trial_prefix}/round_${round_idx}/$root_task/$current_var"
            
            # ==========================================================
            # Anti-Overwrite Safety Mechanism
            # Check if terminal_output.log exists to prevent redundant runs
            # ==========================================================
            if [ -f "$current_log_dir/terminal_output.log" ]; then
                echo "[Safety Skip] Round $round_idx: Results for $task_variant already exist. Bypassing."
                continue
            fi
            
            mkdir -p "$current_log_dir"

            echo "[Eval] GPU $gpu_id | Round $round_idx: Evaluating $task_variant ($eval_episodes Episodes)"
            
            # Execute Evaluation Headless via Xvfb
            CUDA_VISIBLE_DEVICES=$gpu_id xvfb-run --server-num=20$gpu_id --auto-servernum --server-args='-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
                python3 Colosseum/eval_lpb.py \
                --model-folder "$model_folder" \
                --eval-datafolder "$DEMO_PATH" \
                --tasks "$task_variant" \
                --eval-episodes $eval_episodes \
                --log-name "$current_log_dir" \
                --device 0 \
                --headless \
                --model-name "model_$epoch.pth" > "$current_log_dir/terminal_output.log" 2>&1

            echo "[Success] Task $task_variant complete. Logs saved to $current_log_dir"
            
        done
    done
done

echo "[Finished] GPU $gpu_id: All tasks and variations completed across 3 rounds!"
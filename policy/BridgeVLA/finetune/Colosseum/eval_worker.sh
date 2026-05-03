#!/bin/bash
# ==============================================================================
# Script: eval_worker.sh
# Purpose: Launches a headless evaluation worker for the Chain-of-Correction (CoC)
#          framework within the CoppeliaSim/Colosseum environment.
# 
# Usage:
#   bash eval_worker.sh <trial> <epoch> <model_folder> <gpu_id> <task_list_str>
#
# Arguments:
#   trial         : Identifier for the current trial run (e.g., "trial1", "vlm_debug")
#   epoch         : The epoch number of the model checkpoint being evaluated.
#   model_folder  : Root directory containing the trained model checkpoints.
#   gpu_id        : Target GPU ID to allocate for this specific worker.
#   task_list_str : A space-separated string of root tasks to evaluate 
#                   (e.g., "setup_chess scoop_with_spatula").
# ==============================================================================
cd /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune

# ==========================================
# 1. Base Environment & Rendering Patches
# ==========================================
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT

export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT/platforms
export QT_PLUGIN_PATH=$COPPELIASIM_ROOT

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export GALLIUM_DRIVER=softpipe

# ==========================================
# 2. Argument Parsing & Paths Definition
# ==========================================
export DEMO_PATH_ROOT=/data/colosseum_dataset/test
export eval_root_dir=opensource_colosseum
export eval_episodes=25
export trial=$1  
export epoch=$2  
export model_folder=$3  
export gpu_id=$4       
export task_list_str=$5 
 
export CUDA_VISIBLE_DEVICES=$gpu_id

cd Colosseum

# ==========================================
# 3. Core Evaluation Loop (Tasks & Variations)
# ==========================================
for root_task in $task_list_str; do
    export DEMO_PATH=$DEMO_PATH_ROOT/$root_task
    
    ALL_TASKS=""
    # for current_var in 13; do 
    for current_var in {0..14}; do
        export task_list=${root_task}_${current_var}
        export eval_dir=$eval_root_dir/epoch_${epoch}_trial_${trial}/${current_var}/$root_task/  
        export DATA_PATH=$DEMO_PATH/$task_list/

        if [ ! -d "$DATA_PATH" ]; then
            echo "[GPU $gpu_id] Variation $current_var for task $root_task does not exist. Skipping."
            continue
        fi    

        # ==========================================================
        # Dynamic Directory Structure Patch
        # ==========================================================
        # Fixes missing 'variation0' hierarchies in legacy dataset formats
        if [ ! -d "$DATA_PATH/variation0" ] && [ -d "$DATA_PATH/episodes" ]; then
            echo "[Notice] Detected anomalous directory structure ($task_list is missing variation0). Applying dynamic patch..."
            
            # 1. Create the missing variation0 directory
            mkdir -p "$DATA_PATH/variation0"
            
            # 2. Move the entire episodes folder inside variation0
            mv "$DATA_PATH/episodes" "$DATA_PATH/variation0/"
            
            # 3. Extract variation_descriptions.pkl to the variation0 root (to match standard format)
            # Search for variation_descriptions.pkl in any episode and copy it out
            pkl_file=$(find "$DATA_PATH/variation0/episodes" -name "variation_descriptions.pkl" | head -n 1)
            if [ -n "$pkl_file" ]; then
                cp "$pkl_file" "$DATA_PATH/variation0/"
                echo "[Success] Patch complete. Reconstructed variation0 hierarchy and extracted pkl file."
            else
                echo "[Warning] variation_descriptions.pkl not found in episode. This may affect evaluation."
            fi
        fi
        # ==========================================================

        ALL_TASKS="$ALL_TASKS ${root_task}_${current_var}"
    done
    
    export eval_dir=$eval_root_dir/epoch_${epoch}_trial_${trial}/all_vars/$root_task/

    echo "=========================================================="
    echo "[GPU $gpu_id] Initiating simulation rollout for: $ALL_TASKS" 
    echo "=========================================================="
    export PYTHONPATH=/home/zxn/Chain-of-Correction:$PYTHONPATH
 
    xvfb-run --server-num=20$gpu_id --auto-servernum --server-args='-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
        python3 eval.py \
        --model-folder $model_folder  \
        --eval-datafolder $DEMO_PATH \
        --tasks $ALL_TASKS \
        --eval-episodes $eval_episodes \
        --log-name $eval_dir \
        --device 0 \
        --headless \
        --model-name model_$epoch.pth \
        # --save-video
    # done
done 
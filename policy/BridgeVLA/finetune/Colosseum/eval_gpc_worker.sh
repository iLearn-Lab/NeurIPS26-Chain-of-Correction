#!/bin/bash
 
cd /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune
 
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT/platforms
export QT_PLUGIN_PATH=$COPPELIASIM_ROOT
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export GALLIUM_DRIVER=softpipe
export PYTHONPATH=/home/zxn/Chain-of-Correction:$PYTHONPATH
 
trial_prefix=$1
epoch=$2
model_folder=$3
gpu_id=$4
task_list_str=$5

 
EVAL_ROOT="/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM/opensource_colosseum_gpc"
DEMO_PATH_ROOT="/data/colosseum_dataset/test"

 
for round_idx in 1 2 3; do
    echo "Starting [Round $round_idx] ..."
    for root_task in $task_list_str; do
        for var_idx in {0..14}; do
            
            task_variant="${root_task}_${var_idx}"
            data_path="$DEMO_PATH_ROOT/$root_task/$task_variant/"
             
            if [ ! -d "$data_path" ]; then
                continue
            fi
 
            current_log_dir="$EVAL_ROOT/epoch_${epoch}_${trial_prefix}/round_${round_idx}/$root_task/$var_idx"
            mkdir -p "$current_log_dir"

            echo " [GPU $gpu_id][Round $round_idx] evaluating: $task_variant (25 Episodes)"
             
            CUDA_VISIBLE_DEVICES=$gpu_id xvfb-run --server-num=80$gpu_id --auto-servernum --server-args='-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
                python3 Colosseum/eval_gpc.py \
                --model-folder "$model_folder" \
                --eval-datafolder "$DEMO_PATH_ROOT/$root_task/" \
                --tasks "$task_variant" \
                --eval-episodes 25 \
                --log-name "$current_log_dir" \
                --device 0 \
                --headless \
                --model-name "model_$epoch.pth" > "$current_log_dir/terminal_output.log" 2>&1
 
            echo "Task $task_variant finished."
        done
    done
done
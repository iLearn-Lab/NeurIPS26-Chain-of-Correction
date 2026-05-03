#!/bin/bash
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Error: CUDA_VISIBLE_DEVICES is not set! Aborting to protect GPU 0."
    exit 1
fi
 
LOG_NAME=$1       
EPOCH=$2         #  80
MODEL_FOLDER=$3   
DEVICE_ID=$4     # GPU ID
TASKS=$5         #  "meat_on_grill"  
EVAL_DATAFOLDER=$6 #  /data/colosseum_dataset/test
 
cd /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune
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
export PYTHONPATH=/home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum:$PYTHONPATH
export PYTHONPATH=/home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/bridgevla/libs/PyRep:$PYTHONPATH
export PYTHONPATH=/home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/bridgevla/libs/RLBench:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=$DEVICE_ID

echo " Launching FOREWARN Body on GPU $DEVICE_ID"
echo " Base Tasks: $TASKS"
echo " Dataset Root: $EVAL_DATAFOLDER"

cd Colosseum
 
for raw_task in $TASKS; do 
    export root_task=$(echo "$raw_task" | sed 's/_[0-9]*$//')
     
    if [[ "$EVAL_DATAFOLDER" == *"$root_task" ]]; then
        export EVAL_DATAFOLDER=$(dirname "$EVAL_DATAFOLDER")
    fi

    export DEMO_PATH=$EVAL_DATAFOLDER/$root_task
    ALL_TASKS=""
     
    for current_var in {0..14}; do
        export task_list=${root_task}_${current_var}
        export DATA_PATH=$DEMO_PATH/$task_list/
 
        if [ ! -d "$DATA_PATH" ]; then
            echo " Skipping [GPU $DEVICE_ID] Task $task_list."
            continue
        fi    
 
        if [ ! -d "$DATA_PATH/variation0" ] && [ -d "$DATA_PATH/episodes" ]; then 
            mkdir -p "$DATA_PATH/variation0"
            mv "$DATA_PATH/episodes" "$DATA_PATH/variation0/"
            
            pkl_file=$(find "$DATA_PATH/variation0/episodes" -name "variation_descriptions.pkl" | head -n 1)
            if [ -n "$pkl_file" ]; then
                cp "$pkl_file" "$DATA_PATH/variation0/"
                echo "Fixed."
            fi
        fi
        # ==========================================================
 
        ALL_TASKS="$ALL_TASKS $task_list"
    done
    
    ALL_TASKS=$(echo $ALL_TASKS | xargs) 
    if [ -z "$ALL_TASKS" ]; then
        echo "=========================================================="
        echo "Check $DEMO_PATH !"
        echo "=========================================================="
        exit 1
    fi

    echo "=========================================================="
    echo " Evaluating $ALL_TASKS"
    echo "=========================================================="
 
    for i in {1..3}; do
        echo "=========================================================="
        echo " Starting Evaluation Run $i for $root_task ..."
        echo "=========================================================="
         
        xvfb-run --server-num=30$gpu_id --auto-servernum --server-args='-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
            python3 /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_forewarn.py \
            --model-folder "$MODEL_FOLDER" \
            --model-name "model_$EPOCH.pth" \
            --tasks $ALL_TASKS \
            --eval-datafolder "$DEMO_PATH" \
            --eval-episodes 25 \
            --device 0 \
            --headless \
            --log-name "forewarn_eval/${LOG_NAME}_run${i}/$root_task"
            
        echo " Run $i finished."
        sleep 5  
    done 
done
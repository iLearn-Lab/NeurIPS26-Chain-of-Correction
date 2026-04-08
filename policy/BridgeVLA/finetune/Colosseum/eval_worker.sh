#!/bin/bash
# bash /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/Colosseum/eval_worker.sh "vlm_debug" 80 "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM" 4 "setup_chess"
cd /home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune

# ==========================================
# 1. 基础环境补丁 (最纯净版)
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
# 2. 接收外部参数
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
# 3. 核心评估循环 (任务 -> 变体)
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
            echo "⏭️  [GPU $gpu_id] 任务 $root_task 的 Variation $current_var 不存在，自动跳过。"
            continue
        fi    

        # ==========================================================
        if [ ! -d "$DATA_PATH/variation0" ] && [ -d "$DATA_PATH/episodes" ]; then
            echo "🔧 检测到异常目录结构 ($task_list 缺少 variation0)，正在动态修复..."
            
            # 1. 创建缺失的 variation0 文件夹
            mkdir -p "$DATA_PATH/variation0"
            
            # 2. 把 episodes 整个文件夹移进去
            mv "$DATA_PATH/episodes" "$DATA_PATH/variation0/"
            
            # 3. 提取 variation_descriptions.pkl 到 variation0 根目录（匹配标准格式）
            # 寻找任意一个 episode 里的 variation_descriptions.pkl 并复制出来
            pkl_file=$(find "$DATA_PATH/variation0/episodes" -name "variation_descriptions.pkl" | head -n 1)
            if [ -n "$pkl_file" ]; then
                cp "$pkl_file" "$DATA_PATH/variation0/"
                echo "✅ 修复完成！已补齐 variation0 层级并提取 pkl 文件。"
            else
                echo "⚠️ 警告：未在 episode 中找到 variation_descriptions.pkl，可能会影响评测。"
            fi
        fi
        # ==========================================================

        ALL_TASKS="$ALL_TASKS ${root_task}_${current_var}"
    done
    
    export eval_dir=$eval_root_dir/epoch_${epoch}_trial_${trial}/all_vars/$root_task/

    echo "=========================================================="
    echo "🚀 [GPU $gpu_id] 开始测试: $ALL_TASKS" # $root_task | Variation: $current_var
    echo "=========================================================="
    export PYTHONPATH=/home/zxn/Chain-of-Correction:$PYTHONPATH
 
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
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
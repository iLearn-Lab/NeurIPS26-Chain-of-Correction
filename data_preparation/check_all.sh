#!/bin/bash

# Configure environment variables for the CoppeliaSim simulator
export COPPELIASIM_ROOT=/home/zxn/Chain-of-Correction/policy/BridgeVLA/finetune/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT 
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT/platforms
export QT_QPA_PLATFORM=offscreen

# Define core paths
BASE_EXPERT_DIR="/data/colosseum_dataset/expert_v1_1"
BASE_FAIL_DIR="/data/colosseum_dataset/failure_v1_1"
FINAL_JSON="/data/colosseum_dataset/mixed_sft_v1_0.2_cleaned.json"

# Setup logging
LOG_FILE="/data/colosseum_dataset/pipeline_full_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# List of tasks to process (Add your remaining tasks here)
TASKS=(
    "basketball_in_hoop"     
    "close_box"               
    "close_laptop_lid"    
    "empty_dishwasher"                
    "get_ice_from_fridge"     
    "hockey"              
    "insert_onto_square_peg"
    "meat_on_grill"           
    "move_hanger"             
    "open_drawer"         
    "place_wine_at_rack_location"     
    "put_money_in_safe"       
    "reach_and_drag"      
    "scoop_with_spatula"
    "setup_chess"               
    "slide_block_to_target"   
    "stack_cups"          
    "straighten_rope"                 
    "turn_oven_on"
    "wipe_desk"
)

echo "================= [Phase 1: Data Extraction] ================="
for TASK_NAME in "${TASKS[@]}"
do
    echo "[$(date +%H:%M:%S)] Processing task: $TASK_NAME"
    
    # Extract expert trajectories
    echo "  -> Extracting successful rollouts..."
    xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' python data_preparation/build_dataset.py --task $TASK_NAME --episodes 100 --output_dir $BASE_EXPERT_DIR --mode success
    
    EXPERT_JSON_FILE="$BASE_EXPERT_DIR/${TASK_NAME}.json"
    if [ -f "$EXPERT_JSON_FILE" ]; then
        python data_preparation/check_task_abc.py "$EXPERT_JSON_FILE" --tolerance 0.02
    else
        echo "[Error] $TASK_NAME generates no JSON ($EXPERT_JSON_FILE)"
    fi

    # Extract failed trajectories
    echo "  -> Extracting failed rollouts..."
    xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' python data_preparation/build_dataset.py --task $TASK_NAME --output_dir $BASE_FAIL_DIR --mode fail
    
    FAIL_JSON_FILE="$BASE_FAIL_DIR/${TASK_NAME}.json"
    if [ -f "$FAIL_JSON_FILE" ]; then
        python data_preparation/check_task_abc.py "$FAIL_JSON_FILE" --tolerance 0.02
    else
        echo "[Error] $TASK_NAME generates no JSON ($FAIL_JSON_FILE)"
    fi

    # Kill leftover simulator processes to free up memory and ports
    ps -ef | grep coppeliaSim | grep -v grep | awk '{print $2}' | xargs -r kill -9
    sleep 2
done

echo ""
echo "================= [Phase 2: Data Balancing & Verification] ================="
python data_preparation/build_balanced_dataset.py \
    --expert_dir $BASE_EXPERT_DIR \
    --fail_dir $BASE_FAIL_DIR \
    --output_path $FINAL_JSON \
    --target_ratio 0.20

echo ""
echo "Pipeline execution completed successfully."
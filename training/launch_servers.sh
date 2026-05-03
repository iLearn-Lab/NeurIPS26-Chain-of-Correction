#!/bin/bash

# ==============================================================================
# Multi-Task Reward Server Orchestrator
# This script initializes multiple simulation-based reward servers in headless 
# mode using Xvfb. Each task is assigned a unique network port and log file.
# ==============================================================================

# 1. Define the suite of robotic manipulation tasks
TASKS=(
    "hockey" 
    "wipe_desk" 
    "setup_chess" 
    "stack_cups" 
    "empty_dishwasher" 
    "straighten_rope" 
    "turn_oven_on" 
    "put_money_in_safe" 
    "move_hanger" 
    "get_ice_from_fridge" 
    "insert_onto_square_peg" 
    "place_wine_at_rack_location" 
    "scoop_with_spatula" 
    "meat_on_grill" 
    "open_drawer"
)

# 2. Initialize log directory for telemetry
LOG_DIR="logs/reward_servers"
mkdir -p "$LOG_DIR"

# 3. Execution Loop
# Iterate through the task list and launch background processes
START_PORT=8001
CURRENT_PORT=$START_PORT

for TASK in "${TASKS[@]}"; do
    echo "[Deployment] Launching Reward Server: $TASK on Port $CURRENT_PORT"
    
    # Run server in background with virtual frame buffer (Xvfb) for headless rendering
    # -a: auto-servernum | -s: server-args for screen resolution and depth
    nohup xvfb-run -a -s "-screen 0 1024x768x24" \
        python reward_server.py --port "$CURRENT_PORT" --task "$TASK" \
        > "${LOG_DIR}/${TASK}.log" 2>&1 &
    
    # Increment port and throttle initialization to prevent driver race conditions
    ((CURRENT_PORT++))
    sleep 2
done

# 4. Final Status Report
echo "----------------------------------------------------------------"
echo "[Success] All Reward Servers have been initialized (Ports $START_PORT to $((CURRENT_PORT-1)))."
echo "[Monitor] Run 'netstat -tpln | grep 800' to verify active listeners."
echo "[Logs] Check '$LOG_DIR' for individual task output."
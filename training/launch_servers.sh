#!/bin/bash

# 定义任务列表
TASKS=("hockey" "wipe_desk" "setup_chess" "stack_cups" "empty_dishwasher" "straighten_rope" "turn_oven_on" "put_money_in_safe" "move_hanger" "get_ice_from_fridge" "insert_onto_square_peg" "place_wine_at_rack_location" "scoop_with_spatula" "meat_on_grill" "open_drawer")

# 创建日志文件夹
mkdir -p logs/reward_servers

PORT=8001
for TASK in "${TASKS[@]}"; do
    echo "Starting Reward Server for $TASK at port $PORT"
    nohup xvfb-run -a -s "-screen 0 1024x768x24" python reward_server.py --port $PORT --task $TASK > "logs/reward_servers/${TASK}.log" 2>&1 &
    let PORT+=1
    sleep 2
done

echo "✅ 所有任务的 Reward Server 已在后台启动 (Ports 8001-8015)"
echo "💡 通过 'netstat -tpln | grep 800' 检查端口占用情况。"
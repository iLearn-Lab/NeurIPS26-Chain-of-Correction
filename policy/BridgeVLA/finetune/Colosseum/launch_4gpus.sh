#!/bin/bash
 
TRIAL="trial_2"
EPOCH="80"
MODEL_DIR="/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM"
 
rm -f logs/log_gpu*.txt
 
 
# GPU 0
nohup bash eval_worker.sh $TRIAL $EPOCH $MODEL_DIR 0 "basketball_in_hoop close_box empty_dishwasher" > logs/log_gpu0.txt 2>&1 &
# GPU 1
nohup bash eval_worker.sh $TRIAL $EPOCH $MODEL_DIR 1 "get_ice_from_fridge hockey meat_on_grill" > logs/log_gpu1.txt 2>&1 &
# GPU 2
nohup bash eval_worker.sh $TRIAL $EPOCH $MODEL_DIR 2 "move_hanger wipe_desk open_drawer" > logs/log_gpu2.txt 2>&1 &
# GPU 3
nohup bash eval_worker.sh $TRIAL $EPOCH $MODEL_DIR 3 "slide_block_to_target reach_and_drag put_money_in_safe" > logs/log_gpu3.txt 2>&1 &

# GPU 4
nohup bash eval_worker.sh $TRIAL $EPOCH $MODEL_DIR 4 "place_wine_at_rack_location insert_onto_square_peg" > logs/log_gpu4.txt 2>&1 &
# GPU 5
nohup bash eval_worker.sh $TRIAL $EPOCH $MODEL_DIR 5 "turn_oven_on straighten_rope" > logs/log_gpu5.txt 2>&1 &
# GPU 6
nohup bash eval_worker.sh $TRIAL $EPOCH $MODEL_DIR 6 "setup_chess scoop_with_spatula" > logs/log_gpu6.txt 2>&1 &
# GPU 7
nohup bash eval_worker.sh $TRIAL $EPOCH $MODEL_DIR 7 "close_laptop_lid stack_cups" > logs/log_gpu7.txt 2>&1 &

echo "'watch -n 1 nvidia-smi'"
echo "'tail -f logs/log_gpu0.txt'"
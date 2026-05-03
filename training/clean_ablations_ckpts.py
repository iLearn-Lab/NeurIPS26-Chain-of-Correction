import os
import time
import shutil

CHECKPOINT_DIRS = [
    # "/data/colosseum_dataset/checkpoints/sft/v1_wo_progress",
    # "/data/colosseum_dataset/checkpoints/sft/v1_wo_geometric",
    # "/data/colosseum_dataset/checkpoints/sft/v1_wo_prediction",
    "/data/colosseum_dataset/checkpoints/sft/v1_ratio_0.25"
]
KEEP_LATEST_N = 2 

while True:
    for ckpt_dir in CHECKPOINT_DIRS:
        if os.path.exists(ckpt_dir):
            step_ckpts = []
            for name in os.listdir(ckpt_dir):
                if name.startswith("checkpoint-") and name.split("-")[1].isdigit():
                    step_ckpts.append(name)
            
            step_ckpts.sort(key=lambda x: int(x.split("-")[1]))
            
            if len(step_ckpts) > KEEP_LATEST_N:
                to_delete = step_ckpts[:-KEEP_LATEST_N]
                for ckpt in to_delete:
                    ckpt_path = os.path.join(ckpt_dir, ckpt)
                    try:
                        shutil.rmtree(ckpt_path)
                    except Exception as e:
                        print(f"{e}")
                     
    time.sleep(300)
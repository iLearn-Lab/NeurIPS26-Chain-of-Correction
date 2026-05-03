import os
import time
import shutil

# ==========================================
# Configuration
# ==========================================
# Path to the directory where training checkpoints are stored
CHECKPOINT_DIR = "/data/colosseum_dataset/checkpoints/sft/v1_oversample"

# Number of most recent step-based checkpoints to preserve
KEEP_LATEST_N = 2

# Interval between cleanup cycles (1800 seconds = 30 minutes)
CLEANUP_INTERVAL = 1800

print(f"[Monitor] Starting background checkpoint cleanup for: {CHECKPOINT_DIR}")

while True:
    if os.path.exists(CHECKPOINT_DIR):
        # 1. Identify valid step-based checkpoints
        # Filters for folders matching 'checkpoint-<step>' while ignoring 'epoch-' or 'final' folders
        step_ckpts = []
        for name in os.listdir(CHECKPOINT_DIR):
            parts = name.split("-")
            if name.startswith("checkpoint-") and len(parts) > 1 and parts[1].isdigit():
                step_ckpts.append(name)
        
        # 2. Sort checkpoints by step number in ascending order
        step_ckpts.sort(key=lambda x: int(x.split("-")[1]))
        
        # 3. Prune older checkpoints if count exceeds the retention limit
        if len(step_ckpts) > KEEP_LATEST_N:
            # Select all but the latest N checkpoints for deletion
            to_delete = step_ckpts[:-KEEP_LATEST_N]
            
            for ckpt in to_delete:
                ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
                try:
                    shutil.rmtree(ckpt_path)
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] Successfully removed stale checkpoint: {ckpt}")
                except Exception as e:
                    print(f"[Error] Failed to delete {ckpt}: {e}")
                    
    # Suspend execution until the next cleanup cycle
    time.sleep(CLEANUP_INTERVAL)
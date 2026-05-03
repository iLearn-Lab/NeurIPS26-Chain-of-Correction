import os
import shutil
import time
import re

# ==========================================
# Configuration
# ==========================================
# Root directory for raw training checkpoints
CHECKPOINT_DIR = "/home/zxn/EasyR1-main/checkpoints/easy_r1/momagraph_dapo_multitask_baseline"

# Target directory for storing lightweight epoch milestones
ARCHIVE_DIR = os.path.join(CHECKPOINT_DIR, "epoch_archive")

# Frequency of archival (Total steps per epoch)
STEPS_PER_EPOCH = 110 

# Monitoring interval in seconds
SCAN_INTERVAL = 60
# ==========================================

os.makedirs(ARCHIVE_DIR, exist_ok=True)
print(f"[Info] Lightweight Checkpoint Archiver started.")
print(f"[Info] Monitoring directory: {CHECKPOINT_DIR}")

# Tracking set to prevent redundant archival of the same step
archived_steps = set()

while True:
    try:
        # Scan for directories starting with "global_step_"
        folders = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("global_step_")]
        
        for folder in folders:
            # Extract the numeric step value from the folder name
            match = re.search(r'\d+', folder)
            if not match:
                continue
            
            step_num = int(match.group())
            
            # Archive only if the step is an epoch milestone and hasn't been processed
            if step_num % STEPS_PER_EPOCH == 0 and step_num not in archived_steps:
                
                # Logic: Locate only the LoRA adapter to discard heavy optimizer states/gradients
                src_lora_path = os.path.join(CHECKPOINT_DIR, folder, "actor", "lora_adapter")
                
                epoch_idx = step_num // STEPS_PER_EPOCH
                archive_folder_name = f"epoch_{epoch_idx}_step_{step_num}"
                dst_lora_path = os.path.join(ARCHIVE_DIR, archive_folder_name, "actor", "lora_adapter")
                
                # Verify that the lora_adapter has been fully generated
                if os.path.exists(src_lora_path):
                    print(f"[Process] Epoch milestone detected: Step {step_num}. Extracting LoRA weights...")
                    
                    # Perform recursive copy of the adapter directory
                    shutil.copytree(src_lora_path, dst_lora_path, dirs_exist_ok=True)
                    
                    archived_steps.add(step_num)
                    print(f"[Success] Weights archived to: {dst_lora_path}")
                    
    except Exception as e:
        print(f"[Error] An exception occurred during the scan cycle: {e}")
        
    time.sleep(SCAN_INTERVAL)
import os
import re
import shutil
import time

# ==============================================================================
# Configuration
# ==============================================================================
# Source directory containing training checkpoints
CHECKPOINT_DIR = "/home/zxn/EasyR1-main/checkpoints/easy_r1/momagraph_dapo_multitask_wo_edge"

# Target directory for archived epoch milestones
ARCHIVE_DIR = os.path.join(CHECKPOINT_DIR, "epoch_archive")

# Number of steps representing a single epoch
STEPS_PER_EPOCH = 221

# Frequency of directory scanning (seconds)
SCAN_INTERVAL = 60
# ==============================================================================

def run_checkpoint_archiver():
    """
    Monitors a training directory and archives LoRA adapters at epoch milestones.
    This provides a lightweight alternative to saving full model states.
    """
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    print(f"[Info] Checkpoint Archiver initialized.")
    print(f"[Info] Monitoring: {CHECKPOINT_DIR}")

    archived_steps = set()

    while True:
        try:
            # Filter for global step directories (e.g., "global_step_442")
            step_folders = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("global_step_")]
            
            for folder in step_folders:
                # Extract the numeric step value from folder name
                match = re.search(r'\d+', folder)
                if not match:
                    continue
                    
                step_num = int(match.group())
                
                # Check if step is an epoch milestone and hasn't been archived yet
                if step_num % STEPS_PER_EPOCH == 0 and step_num not in archived_steps:
                    # Define source and destination for the lightweight LoRA weights
                    src_lora_path = os.path.join(CHECKPOINT_DIR, folder, "actor", "lora_adapter")
                    
                    epoch_idx = step_num // STEPS_PER_EPOCH
                    dst_folder = f"epoch_{epoch_idx}_step_{step_num}"
                    dst_lora_path = os.path.join(ARCHIVE_DIR, dst_folder, "actor", "lora_adapter")
                     
                    if os.path.exists(src_lora_path):
                        print(f"[Archive] Milestone reached: Step {step_num} (Epoch {epoch_idx})")
                        print(f"[Archive] Copying LoRA adapter weights...")
                        
                        # Copy only the LoRA adapter to save storage space
                        shutil.copytree(src_lora_path, dst_lora_path, dirs_exist_ok=True)
                        archived_steps.add(step_num)
                        
                        print(f"[Success] Adapter archived to: {dst_lora_path}")
                        
        except Exception as e:
            print(f"[Warning] Error during scan cycle: {e}")
            
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    run_checkpoint_archiver()
import json
import os
import re
import shutil
import glob
# ================= Configuration =================
BASELINE_JSON = "/data/colosseum_dataset/ablations/eval_results/results_GPC_trial_1.json"
OURS_JSON = "/data/colosseum_dataset/ablations/eval_results/results_CoC-Full-V1-Qwen_trial_1.json"
VAL_SET_PATH = "/data/colosseum_dataset/ablations/offline_val_set_200.json"
OUTPUT_DIR = "/data/colosseum_dataset/ablations/eval_results"
   
TARGET_IDS = [
    # --- Scenario A: LPB Critical Misses (False Negatives) ---
    "setup_chess_0", 
    "setup_chess_19",
    
    # --- Scenario B: Over-steering / False Alarms (False Positives) ---
    "place_wine_at_rack_location_0", # FOREWARN,GPC
    "place_wine_at_rack_location_1", # all
    "place_wine_at_rack_location_3",  
    "place_wine_at_rack_location_4", 
    "place_wine_at_rack_location_5", 
    "place_wine_at_rack_location_7", 
    "place_wine_at_rack_location_8", 
    "setup_chess_2", 
    "setup_chess_7", 
    "setup_chess_17",
    "put_money_in_safe_1", 
    "put_money_in_safe_3",
    "meat_on_grill_0", 
    "meat_on_grill_1", 
    "meat_on_grill_2",
    "meat_on_grill_3", 
    "meat_on_grill_4", 
    "meat_on_grill_5",
    "turn_oven_on_2", 
    "turn_oven_on_3", 
    "turn_oven_on_7",
    "move_hanger_2", 
    "move_hanger_9",
    "open_drawer_2"
]

COLOSSEUM_TASKS = [
    "meat_on_grill", "move_hanger", "open_drawer", 
    "place_wine_at_rack_location", "put_money_in_safe", 
    "setup_chess", "turn_oven_on"
]
# ==========================================
 
def extract_bool_from_text(text):
    """
    Robust text parser designed specifically to extract the boolean 
    'correction_needed' flag from the model's raw string output.
    """
    if not isinstance(text, str):
        return False
        
    # Attempt to extract JSON block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if 'correction_needed' in data: return data['correction_needed']
        except: pass
    
    text_lower = text.lower()
    if '"correction_needed": true' in text_lower: return True
    if '"correction_needed": false' in text_lower: return False
    if "no correction needed" in text_lower: return False
    if "correction required" in text_lower or "correction needed" in text_lower: return True
    return False

def parse_task(sample):
    """Generic task classification logic for datasets."""
    # 1. Attempt to extract from the image path
    if "image_paths" in sample and len(sample["image_paths"]) > 0:
        path_str = sample["image_paths"][0].lower()
        for task in COLOSSEUM_TASKS:
            if task in path_str: return task
            
    # 2. Attempt to extract from the text prompt
    text = str(sample['messages']).lower()
    for task in COLOSSEUM_TASKS: 
        search_key = task.replace('_', ' ')
        if search_key in text or task in text: return task 
        if task == "place_wine_at_rack_location" and "wine" in text: return task
        
    return "unknown"



def mainn():
    """
    Data Mining Module: Compares Baseline predictions against Ours to identify
    False Negatives and False Positives.
    """
    print("[Init] Launching Multimodal Zip-Matching & Smart Text Parsing Engine...")
    
    with open(BASELINE_JSON, 'r', encoding='utf-8') as f: base_data = json.load(f)
    with open(OURS_JSON, 'r', encoding='utf-8') as f: ours_data = json.load(f)

    # --- 1. Group Baseline samples by Task ---
    base_grouped = {}
    for item in base_data:
        base_grouped.setdefault(item.get('task', 'unknown'), []).append(item)
        
    # The 'Ours' data is a continuous 200-length list. We manually partition it 
    # based on the baseline task distribution for proper alignment.
    ours_grouped = {}
    ours_index = 0
    for task, b_list in base_grouped.items():
        count = len(b_list)
        # Slice the corresponding number of samples for 'Ours'
        ours_grouped[task] = ours_data[ours_index : ours_index + count]
        ours_index += count

    golden_missed = []
    golden_oversteer = []
    total_matched = 0

    # --- 2. Sequentially pair samples within the same task and dynamically parse text ---
    for task in base_grouped:
        base_list = base_grouped[task]
        ours_list = ours_grouped.get(task, [])
        
        # Ensure lengths match before zipping
        if len(base_list) != len(ours_list):
            print(f"[Warning] Task {task} count mismatch (Base:{len(base_list)}, Ours:{len(ours_list)}). Skipping task.")
            continue

        for idx, (b_item, o_item) in enumerate(zip(base_list, ours_list)):
            total_matched += 1
             
            gt = b_item['ground_truth']['correction_needed']
            base_pred = b_item['prediction']['correction_needed']
             
            ours_pred_text = o_item.get('prediction', '')
            ours_pred = extract_bool_from_text(ours_pred_text)
            
            pseudo_id = f"{task}_{idx}"
            
            # Scenario A: Critical Misses (Baseline failed to catch, Ours caught it)
            if (gt is True) and (base_pred is False) and (ours_pred is True):
                golden_missed.append({'id': pseudo_id, 'task': task})
                
            # Scenario B: Over-steering / False Alarms (Baseline panicked, Ours remained calm)
            if (gt is False) and (base_pred is True) and (ours_pred is False):
                golden_oversteer.append({'id': pseudo_id, 'task': task})

    print(f"[Success] Successfully parsed and matched {total_matched} samples!\n")

    # --- 3. Display Results ---
    print("================ Mining Results ================")
    print(f"[Scenario A: Baseline Critical Misses (FN)] Found: {len(golden_missed)}")
    for item in golden_missed:  
        print(f"   └── Task: {item['task']} | {item['id']}")

    print(f"\n[Scenario B: Baseline Over-steering (FP)] Found: {len(golden_oversteer)}")
    for item in golden_oversteer: 
        print(f"   └── Task: {item['task']} | {item['id']}")
    print("================================================")

def main():
    """
    Asset Extraction Module: Extracts entire physical frame sequences along with 
    the current keyframe and final failure frame for targeted visualization.
    """
    print(f"[Init] Preparing to extract full frame sequences and key contrast frames for {len(TARGET_IDS)} golden samples...")
    
    OUTPUT_DIR_FRAMES = os.path.join(OUTPUT_DIR, "frames_analysis")
    
    if not os.path.exists(VAL_SET_PATH):
        print(f"[Error] Validation set file not found: {VAL_SET_PATH}")
        return
        
    with open(VAL_SET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    grouped_data = {}
    for item in data:
        task_name = parse_task(item)
        grouped_data.setdefault(task_name, []).append(item)

    os.makedirs(OUTPUT_DIR_FRAMES, exist_ok=True)
    success_count = 0

    for target_id in TARGET_IDS:
        try:
            task_part, idx_part = target_id.rsplit('_', 1)
            idx = int(idx_part)
        except ValueError:
            print(f"[Warning] Unable to parse ID format: {target_id}")
            continue
            
        if task_part not in grouped_data or idx >= len(grouped_data[task_part]):
            print(f"[Warning] Sample not found: {target_id}")
            continue
            
        sample = grouped_data[task_part][idx]
        img_paths = sample.get("image_paths", [])
        if not img_paths:
            continue
            
        source_img_path = img_paths[0]
        sample_folder = os.path.join(OUTPUT_DIR_FRAMES, target_id)
        os.makedirs(sample_folder, exist_ok=True)
        
        try:
            episode_dir = os.path.dirname(source_img_path)
            target_file_name = os.path.basename(source_img_path)
            
            all_frames = sorted(glob.glob(os.path.join(episode_dir, "*.png")))
            
            if not all_frames:
                print(f"[Warning] No PNG images found in directory for {target_id}!")
                continue
                
            final_frame_path = all_frames[-1]
            final_file_name = os.path.basename(final_frame_path)

            # 1. Copy the full sequence for step-by-step cinematic analysis
            for frame_path in all_frames:
                frame_name = os.path.basename(frame_path)
                dest_frame_path = os.path.join(sample_folder, frame_name)
                shutil.copy(frame_path, dest_frame_path)
                
            # 2. Highlight: Extract the 'Current Decision Keyframe' and pin it to the top
            highlight_curr_path = os.path.join(sample_folder, f"00_CURRENT_KEYFRAME_{target_file_name}")
            shutil.copy(source_img_path, highlight_curr_path)
            
            # 3. Highlight: Extract the 'Final Failure Frame' and pin it to the bottom
            highlight_final_path = os.path.join(sample_folder, f"99_FINAL_FAILURE_{final_file_name}")
            shutil.copy(final_frame_path, highlight_final_path)

            print(f"[Success] Exported [ {target_id} ] containing {len(all_frames)} frames + keyframes -> {sample_folder}")
            success_count += 1
            
        except Exception as e:
            print(f"[Error] Failed to extract sequence for {target_id}: {e}")

    print(f"\n[Complete] Batch extraction finished! Successfully extracted {success_count}/{len(TARGET_IDS)} samples.")
    print(f"[Info] Please check {OUTPUT_DIR_FRAMES}. The first and last frames in each folder are optimized for paper figures!")

if __name__ == "__main__":
    main()
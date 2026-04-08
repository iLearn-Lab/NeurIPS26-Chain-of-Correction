import os
import json
import random
import argparse

def verify_and_clean_in_memory(dataset):
    """
    Integrates historical image tag cleaning and alignment verification logic.
    Validates that the number of <image> tags matches the provided image paths.
    """
    cleaned_dataset = []
    error_count = 0
    
    for idx, item in enumerate(dataset):
        messages = item.get("messages", [])
        images_in_paths = len(item.get("image_paths", []))
        
        # 1. Forcibly clean historical image tags (Defensive programming)
        last_user_idx = next((i for i in range(len(messages)-1, -1, -1) if messages[i].get("role") == "user"), -1)
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and i != last_user_idx:
                msg["content"] = msg["content"].replace("<image>\n", "").replace("<image>", "").lstrip()
        
        # 2. Strict verification check
        tags_in_messages = sum(msg.get("content", "").count("<image>") for msg in messages if msg.get("role") == "user")
        if tags_in_messages != images_in_paths:
            print(f"  [Drop] Invalid sample {idx}: <image> tags ({tags_in_messages}) != image paths ({images_in_paths})")
            error_count += 1
            continue
            
        cleaned_dataset.append(item)
        
    return cleaned_dataset, error_count

def get_correction_flag(item):
    """
    Extracts the correction flag and validates the 8-dimensional action format.
    Returns True/False if valid, or None if the data is corrupted.
    """
    try:
        content = item["messages"][-1]["content"]
        json_str = content.split("```json\n")[1].split("\n```")[0]
        parsed = json.loads(json_str)
        
        # Strict quality boundary: action must have exactly 8 dimensions
        if len(parsed.get("corrected_action", [])) != 8:
            return None
            
        return parsed.get("correction_needed", False)
    except Exception:
        return None

def build_balanced_dataset(expert_dir, fail_dir, output_path, target_true_ratio):
    print("Starting high-precision data mixing and oversampling engine...")
    expert_data, failure_data = [], []
    
    # 1. Load data
    for directory, data_list in [(expert_dir, expert_data), (fail_dir, failure_data)]:
        for file in os.listdir(directory):
            if file.endswith(".json"):
                with open(os.path.join(directory, file), 'r') as f:
                    data_list.extend(json.load(f))

    # 2. Refined classification and quality filtering
    false_pool = []
    true_pool = []
    
    print("Executing data quality filtering...")
    dropped_count = 0
    for item in expert_data + failure_data:
        is_needed = get_correction_flag(item)
        if is_needed is None:
            dropped_count += 1
            continue
            
        if not is_needed:
            false_pool.append(item)
        else:
            true_pool.append(item)

    # 3. Core Math: Calculate global ratio and multiplier
    total_false = len(false_pool)
    original_total_true = len(true_pool)
    
    if original_total_true == 0:
        print("Error: No valid True (correction) samples found!")
        return

    target_total_true = int(total_false * target_true_ratio / (1.0 - target_true_ratio))
    global_multiplier = max(1, target_total_true // original_total_true)
    
    print("\n--- Global Data Distribution & Oversampling ---")
    print(f"Dropped invalid/corrupted samples: {dropped_count}")
    print(f"Healthy False samples: {total_false}")
    print(f"Original True samples: {original_total_true}")
    print(f"Target True count: {target_total_true} ({target_true_ratio*100}% of total)")
    print(f"Global uniform multiplier: {global_multiplier}x")

    # 4. Execute proportional oversampling
    final_true_pool = true_pool * global_multiplier
    
    # 5. Merge and shuffle
    final_dataset = false_pool + final_true_pool
    random.shuffle(final_dataset)
    
    # 6. Memory-level cleaning and verification
    print("\nExecuting final memory-level verification...")
    final_dataset, err_cnt = verify_and_clean_in_memory(final_dataset)
    if err_cnt == 0:
        print("Verification passed! Training set is perfectly aligned.")
    
    print("\n--- Final Dataset Statistics ---")
    total_size = len(final_dataset)
    print(f"False ratio: {len(false_pool) / total_size * 100:.1f}%")
    print(f"True ratio:  {len(final_true_pool) / total_size * 100:.1f}%")
    print(f"Total size:  {total_size} samples")

    # 7. Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    print(f"\nBalanced dataset successfully saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_dir", type=str, required=True)
    parser.add_argument("--fail_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--target_ratio", type=float, default=0.20)
    args = parser.parse_args()
    
    build_balanced_dataset(args.expert_dir, args.fail_dir, args.output_path, args.target_ratio)
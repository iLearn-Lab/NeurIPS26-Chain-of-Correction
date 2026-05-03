import json
import random
import os

# Paths configuration
INPUT_FILE = "/data/colosseum_dataset/mixed_sft_v1_0.2_cleaned.json"
OUTPUT_FILE = "/data/colosseum_dataset/ablations/offline_val_set_200.json"
NUM_TRUE = 100   # Target number of correction-needed (True) samples
NUM_FALSE = 100  # Target number of non-correction (False) samples

def extract_correction_flag(item):
    """
    Extract the 'correction_needed' boolean flag from the Assistant's JSON response block.
    
    Args:
        item (dict): A single data instance from the JSON dataset containing conversation messages.
        
    Returns:
        bool or None: The value of 'correction_needed' if successfully parsed, otherwise None.
    """
    try:
        content = item["messages"][-1]["content"]
        json_str = content.split("```json\n")[1].split("\n```")[0]
        parsed = json.loads(json_str)
        return parsed.get("correction_needed")
    except:
        return None

def build():
    """
    Build a balanced offline validation dataset for ablation studies.
    
    This function reads the full cleaned dataset, pools samples based on their 
    'correction_needed' flag, and randomly samples a defined number of True and False 
    instances. The resulting balanced set is shuffled and saved to disk to serve as 
    a standardized evaluation benchmark.
    """
    print(f"[Info] Reading original dataset: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    true_pool = []
    false_pool = []

    for item in data:
        flag = extract_correction_flag(item)
        if flag is True:
            true_pool.append(item)
        elif flag is False:
            false_pool.append(item)

    print(f"[Info] Pooling complete. True samples: {len(true_pool)}, False samples: {len(false_pool)}")

    if len(true_pool) < NUM_TRUE or len(false_pool) < NUM_FALSE:
        print("[Error] Insufficient samples to construct a balanced validation set!")
        return

    # Randomly sample with a fixed seed to ensure experimental reproducibility
    random.seed(42)
    sampled_true = random.sample(true_pool, NUM_TRUE)
    sampled_false = random.sample(false_pool, NUM_FALSE)

    exam_set = sampled_true + sampled_false
    random.shuffle(exam_set)  

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(exam_set, f, indent=2, ensure_ascii=False)

    print(f"[Success] Validation set constructed successfully! Saved to: {OUTPUT_FILE}")
    print("[Info] This benchmark dataset will serve as the unified offline evaluation baseline for all ablation studies.")

if __name__ == "__main__":
    build()
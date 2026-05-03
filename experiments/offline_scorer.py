import json, os, re
import numpy as np

EVAL_DIR = "/data/colosseum_dataset/ablations/eval_results"

def extract_bool_from_text(text):
    """
    Robust text parser designed specifically to extract the boolean 
    'correction_needed' flag from the model's raw string output.
    """
    if isinstance(text, dict):
        return text.get('correction_needed')
        
    # Strategy 1: Search for a standard JSON block
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return data.get('correction_needed')
        except: pass
    
    # Strategy 2: If truncated, search for the last occurrence of the boolean flag
    if '"correction_needed": true' in text.lower(): return True
    if '"correction_needed": false' in text.lower(): return False
    
    # Strategy 3: Blind guess from detailed reasoning text
    if "no correction needed" in text.lower(): return False
    if "correction required" in text.lower() or "correction needed" in text.lower(): return True
    
    return None

def extract_action_from_text(text):
    """
    Robustly extract an 8-dimensional action array, bypassing malformed JSON structures.
    """
    if isinstance(text, dict):
        action = text.get('corrected_action')
        return np.array(action, dtype=float) if action else None
    # Strategy 1: Standard parsing
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if 'corrected_action' in data and len(data['corrected_action']) == 8:
                return np.array(data['corrected_action'], dtype=float)
        except: pass

    # Strategy 2: Brute-force regex extraction of the array content
    # Matches "corrected_action": [ followed by numbers and commas ]
    action_match = re.search(r'"corrected_action"\s*:\s*\[(.*?)\]', text)
    if action_match:
        try:
            # Split extracted string by comma and convert to floats
            num_strs = action_match.group(1).split(',')
            nums = [float(n.strip()) for n in num_strs if n.strip()]
            if len(nums) >= 8:
                return np.array(nums[:8], dtype=float)
        except: pass
        
    return None

def calculate_metrics_for_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tp, fp, fn, tn = 0, 0, 0, 0
    valid_samples_count = 0  # Tracks the number of valid samples
    failed_samples_count = 0 # Tracks discarded samples due to API crashes

    trans_errs, rot_errs, grip_errs = [], [], []
    
    for item in data:
        pred_text = item.get('prediction', "")
        
        # Core Interceptor: Only blocks data with total API failures or empty strings.
        # Open-source models will not trigger this branch, ensuring their results remain unaffected.
        if isinstance(pred_text, str):
            if "API Request Failed" in pred_text or pred_text.strip() == "":
                failed_samples_count += 1
                continue # Skip this data entirely, excluding it from TP/FP/FN/TN statistics
        elif pred_text is None:
            failed_samples_count += 1
            continue
            
        valid_samples_count += 1 # If no error occurred, count as a valid sample for evaluation

        # Parse Ground Truth (Standard format)
        gt_text = item.get('ground_truth', "")
        gt_corr = extract_bool_from_text(gt_text)
        
        # Parse Prediction (May be unstructured or truncated)
        pred_corr = extract_bool_from_text(pred_text)

        if gt_corr is None or pred_corr is None:
            # If parsing completely fails, record as a False Negative (missed)
            if gt_corr is True: fn += 1
            else: tn += 1
            continue

        if gt_corr is True and pred_corr is True: tp += 1
        elif gt_corr is False and pred_corr is True: fp += 1
        elif gt_corr is True and pred_corr is False: fn += 1
        else: tn += 1

        # ====== 2. Calculate Action Deviation (Only for samples truly requiring correction) ======
        if gt_corr is True:
            gt_action = extract_action_from_text(gt_text)
            pred_action = extract_action_from_text(pred_text)
            
            # Only calculate error if both GT and Pred successfully extracted an 8D action
            if gt_action is not None and pred_action is not None:
                # Translation Error (L2 norm of first 3 dims)
                trans_errs.append(np.linalg.norm(pred_action[:3] - gt_action[:3]))
                # Rotation Error (L2 norm of next 4 dims)
                rot_errs.append(np.linalg.norm(pred_action[3:7] - gt_action[3:7]))
                # Gripper Error (Binarized comparison, assuming >0 is open)
                gt_grip = 1 if gt_action[7] > 0 else 0
                pred_grip = 1 if pred_action[7] > 0 else 0
                grip_errs.append(1.0 if gt_grip != pred_grip else 0.0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    avg_trans = np.mean(trans_errs) if trans_errs else 0.0
    avg_rot = np.mean(rot_errs) if rot_errs else 0.0
    avg_grip = np.mean(grip_errs) * 100 if grip_errs else 0.0 # %

    return f1, precision, recall, tp, fp, fn, valid_samples_count, failed_samples_count, avg_trans, avg_rot, avg_grip


def main():
    print("=================== [Metrics] Evaluation Scorecard ===================")
    # Add table header for quick comparison
    print(f"{'Model Name':<25} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'Trans':<6} | {'Rot':<6} | {'Grip%':<6}")
    print("-" * 85)
    
    for filename in sorted(os.listdir(EVAL_DIR)): 
        if filename.endswith(".json") and not filename.startswith("progress_"):
            filepath = os.path.join(EVAL_DIR, filename)
            f1, p, r, tp, fp, fn, valid_cnt, failed_cnt, trans, rot, grip = calculate_metrics_for_file(filepath)
            
            model_name = filename.replace('results_', '').replace('.json', '')
            print(f"[Processing] Model: {model_name}")
            
            # Highlight filtered invalid data if any; otherwise, omit
            if failed_cnt > 0:
                print(f"   [Warning] Excluded {failed_cnt} API failure logs. Scoring based on {valid_cnt} valid samples.")
            else:
                print(f"   [Info] Valid Sample Size: {valid_cnt} / 200")
                
            if valid_cnt == 0:
                print(f"{model_name[:25]:<25} | {'API FAIL':<35}")
                continue
                
            print(f"{model_name[:25]:<25} | {f1:.4f} | {p:.4f} | {r:.4f} | {trans:.4f} | {rot:.4f} | {grip:>5.1f}%")
            print(f"   [Stats] F1-Score:  {f1:.4f}  (P: {p:.4f}, R: {r:.4f})")
            print(f"   [Stats] Granular: TP={tp}, FP={fp}, FN={fn}")
            print("-" * 50)


if __name__ == "__main__":
    main()

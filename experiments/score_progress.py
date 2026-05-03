import json
import os
import re
import numpy as np

EVAL_DIR = "/data/colosseum_dataset/ablations/eval_results"

def extract_progress_from_text(text):
    """
    Robustly extract global_progress (float between 0.0 and 1.0) from text.
    
    Strategies include:
    1. Direct dictionary access (if input is a dict).
    2. Regex matching for "Global Task Progress: 0.xxx" patterns (GT format).
    3. JSON block parsing (Standard Assistant output).
    4. Brute-force regex key-value pair matching.
    
    Args:
        text (str or dict): The raw output from the model or ground truth data.
        
    Returns:
        float or None: Extracted progress value, or None if extraction fails.
    """
    if isinstance(text, dict):
        prog = text.get('global_progress', text.get('progress'))
        return float(prog) if prog is not None else None
        
    text_str = str(text)
    
    # Strategy 1: Targeted capture for Ground Truth "Global Task Progress: 0.xxx"
    gt_match = re.search(r'Global Task Progress:\s*([0-9]*\.?[0-9]+)', text_str, re.IGNORECASE)
    if gt_match:
        try:
            return float(gt_match.group(1))
        except: pass

    # Strategy 2: Standard JSON block extraction
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_str, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if 'global_progress' in data: return float(data['global_progress'])
            if 'progress' in data: return float(data['progress'])
        except: pass
        
    # Strategy 3: Brute-force regex matching for JSON-style keys
    match = re.search(r'"(?:global_)?progress"\s*:\s*([0-9]*\.?[0-9]+)', text_str)
    if match:
        try:
            return float(match.group(1))
        except: pass
        
    return None

def calculate_progress_metrics(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sq_errs = []   # Squared Errors
    abs_errs = []  # Absolute Errors
    valid_samples_count = 0
    failed_samples_count = 0 
    
    for item in data:
        pred_text = item.get('prediction', "")
        gt_text = item.get('ground_truth', "")
        
        # Filter API failures or empty outputs
        if isinstance(pred_text, str):
            if "API Request Failed" in pred_text or pred_text.strip() == "":
                failed_samples_count += 1
                continue
        elif pred_text is None:
            failed_samples_count += 1
            continue

        # Extract Ground Truth and Prediction values
        gt_prog = extract_progress_from_text(gt_text)
        pred_prog = extract_progress_from_text(pred_text)

        # Safety Check: Skip if Ground Truth is missing
        if gt_prog is None:
            continue
            
        # Error Check: Mark as failure if model failed to output a valid numerical progress
        if pred_prog is None:
            failed_samples_count += 1
            continue
            
        # Bound enforcement [0.0, 1.0]
        pred_prog = max(0.0, min(1.0, pred_prog))
        gt_prog = max(0.0, min(1.0, gt_prog))

        sq_errs.append((gt_prog - pred_prog) ** 2)
        abs_errs.append(abs(gt_prog - pred_prog))
        valid_samples_count += 1

    # Statistical computation
    mse = np.mean(sq_errs) if sq_errs else 0.0
    rmse = np.sqrt(mse) # RMSE calculation
    mae = np.mean(abs_errs) if abs_errs else 0.0

    return rmse, mae, valid_samples_count, failed_samples_count


def main():
    print("=================== [Metrics] Progress Estimation Scorecard ===================")
    print(f"{'Model Name':<30} | {'RMSE (↓)':<8} | {'MAE (↓)':<8} | {'Valid'}")
    print("-" * 75)
    
    for filename in sorted(os.listdir(EVAL_DIR)):
        if filename.startswith("progress_results_") and filename.endswith(".json"):
            filepath = os.path.join(EVAL_DIR, filename)
            
            rmse, mae, valid_cnt, failed_cnt = calculate_progress_metrics(filepath)
            
            model_name = filename.replace('progress_results_', '').replace('.json', '')
            
            if valid_cnt == 0:
                print(f"{model_name[:30]:<30} | {'API FAIL':<19} | {0}")
                continue
                
            print(f"{model_name[:30]:<30} | {rmse:.4f}   | {mae:.4f}   | {valid_cnt}")
            
            if failed_cnt > 0:
                print(f"   [Warning] Excluded {failed_cnt} samples due to API errors or parsing failures.")
                
    print("=========================================================")

if __name__ == "__main__":
    main()
import os
import json
import re
import numpy as np
import torch
import cv2
import sys
from PIL import Image
from termcolor import cprint 

CURRENT_MODEL = "FOREWARN"
# CURRENT_MODEL = "GPC" 

VAL_SET_PATH = "/data/colosseum_dataset/ablations/offline_val_set_200.json"
OUTPUT_DIR = "/data/colosseum_dataset/ablations/eval_results"
NUM_CANDIDATES = 8
NUM_TRIALS = 3

if CURRENT_MODEL == "LPB":
    sys.path.append("/home/zxn/lpb")
    from lpb_wrapper import LPBWorldModelWrapper
elif CURRENT_MODEL == "FOREWARN":
    sys.path.append("/home/zxn/Forewarn/vlm/llama-recipes")
    from vlm_forewarn_server import ForewarnEngine 
elif CURRENT_MODEL == "GPC":
    try:
        from gpc_wrapper import GPCEngine
    except ImportError:
        cprint("[Warning] Failed to import gpc_wrapper", "red")

COLOSSEUM_TASKS = [
    "meat_on_grill", "move_hanger", "open_drawer", 
    "place_wine_at_rack_location", "put_money_in_safe", 
    "setup_chess", "turn_oven_on"
]

TASK_PROMPTS = {
    "meat_on_grill": "The robot needs to place the piece of meat onto the grill.",
    "move_hanger": "The robot needs to move the clothes hanger to the specified location.",
    "open_drawer": "The robot needs to successfully pull the drawer open.",
    "place_wine_at_rack_location": "The robot needs to grasp the wine bottle and place it securely on the rack.",
    "put_money_in_safe": "The robot needs to place the stack of money inside the safe.",
    "scoop_with_spatula": "The robot needs to use the spatula to scoop up the target object.",
    "setup_chess": "The robot needs to stand the fallen chess piece upright on the board.",
    "turn_oven_on": "The robot needs to grasp and turn the oven dial to switch it on."
}

def parse_task_from_sample(sample):
    """
    Robust task name extractor. 
    Prioritizes deterministic extraction from the physical image path to bypass language instruction variance.
    """
    # Strategy 1: Extract from image path (Highly robust, 100% accurate)
    if "image_paths" in sample and len(sample["image_paths"]) > 0:
        path_str = sample["image_paths"][0].lower()
        for task in COLOSSEUM_TASKS:
            if task in path_str:
                return task
                
    # Strategy 2: Fallback to hard-matching within the prompt text
    text = str(sample['messages']).lower()
    for task in COLOSSEUM_TASKS:
        # Replace underscores with spaces for natural language matching
        if task.replace('_', ' ') in text or task in text:
            return task
            
    return "unknown"
 
def extract_bool_from_text(text):
    """Safely extract the boolean 'correction_needed' flag from LLM text output."""
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

def extract_action_from_text(text):
    """Safely extract the 8-dimensional action array from LLM text output."""
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if 'corrected_action' in data and len(data['corrected_action']) == 8:
                return np.array(data['corrected_action'], dtype=float)
        except: pass
    action_match = re.search(r'"corrected_action"\s*:\s*\[(.*?)\]', text)
    if action_match:
        try:
            num_strs = action_match.group(1).split(',')
            nums = [float(n.strip()) for n in num_strs if n.strip()]
            if len(nums) >= 8: return np.array(nums[:8], dtype=float)
        except: pass
    return np.zeros(8)

# ================= 2. Core: Controlled Candidate Pool Generator =================
def generate_mock_candidates(gt_action, gt_trigger, num_candidates=8):
    """
    Injects physics-based controlled noise to simulate a realistic Propose-and-Rank scenario.
    """
    candidates = np.zeros((num_candidates, 8))
    # Generate baseline Gaussian noise for sub-optimal action simulation
    noise = np.random.normal(0, 0.05, (num_candidates, 8))
    
    if gt_trigger:
        # Scenario A: Correction is needed
        # Candidate 0 (Base Policy's original plan): Inject severe destructive noise to simulate impending failure
        bad_noise = np.random.uniform(0.15, 0.3, 8) * np.random.choice([-1, 1], 8)
        candidates[0] = gt_action + bad_noise
        # Candidate 1: Insert the perfect expert action as the recovery solution
        candidates[1] = gt_action
        # Candidates 2-7: Insert sub-optimal solutions with minor noise
        candidates[2:] = gt_action + noise[2:]
    else:
        # Scenario B: No correction needed
        # Candidate 0: Already flawless, insert ground truth directly
        candidates[0] = gt_action
        # Candidates 1-7: Distractors with minor noise
        candidates[1:] = gt_action + noise[1:]
        
    # Actions are typically normalized within [-1, 1], apply clipping to prevent bounds violation
    return np.clip(candidates, -1.0, 1.0)

# ================= 3. Dataset Loading and Grouping =================
def load_and_group_dataset(json_path):
    with open(json_path, 'r') as f: data = json.load(f)
    grouped_data = {}
    
    for sample in data:
        task_name = parse_task_from_sample(sample)
        gt_text = sample['messages'][-1]['content']
        
        gt_trigger = extract_bool_from_text(gt_text)
        gt_action = extract_action_from_text(gt_text)
        
        # Dynamically generate controlled candidate pool
        candidates = generate_mock_candidates(gt_action, gt_trigger, NUM_CANDIDATES)
        
        item = {
            "image_paths": sample.get('image_paths', []),
            "gt_trigger": gt_trigger,
            "gt_action": gt_action,
            "candidates": candidates,
            "proprio": np.zeros(8) # Most visual rankers do not strictly require true proprioceptive states
        }
        
        if task_name not in grouped_data: grouped_data[task_name] = []
        grouped_data[task_name].append(item)
        
    return grouped_data

# ================= 4. Evaluation Pipeline =================
def evaluate_baselines(trial_idx):
    grouped_data = load_and_group_dataset(VAL_SET_PATH)
    total_loaded = sum([len(v) for v in grouped_data.values()])
    cprint(f"[Info] Total JSON samples loaded: {total_loaded} / 200", "magenta", attrs=['bold'])
    json_output_data = []
    
    results = {
        "y_true": [], "y_pred": [], "trans_errs": [], "rot_errs": [], "grip_errs": []
    }
    evaluated_tasks_count = 0
    skipped_samples_count = 0
    for task_name, samples in grouped_data.items():
        if task_name == "unknown": 
            skipped_samples_count += len(samples)
            continue 
            
        # ================= 1. Dynamic path verification and skip logic =================
        should_skip = False
        if CURRENT_MODEL == "LPB":
            ckpt_path = f"/data/colosseum_dataset/lpb_data/outputs/ckpts/{task_name}/model_100.pth" 
            if not os.path.exists(ckpt_path):
                should_skip = True
        elif CURRENT_MODEL == "FOREWARN":
            peft_path = f"/data/colosseum_dataset/wm_data/finetuned_models/vlm_{task_name}/peft_checkpoint_5"
            if not os.path.exists(peft_path):
                should_skip = True
        elif CURRENT_MODEL == "GPC":
            wm_ckpt = f"/data/colosseum_dataset/gpc_data/world_model_ckpt/{task_name}/phase_two/checkpoint_epoch_20/denoiser.pth"
            if not os.path.exists(wm_ckpt):
                should_skip = True
                
        if should_skip:
            cprint(f"[Skip] Skipping untrained/missing weights for task: {task_name} ({len(samples)} samples)", "yellow")
            skipped_samples_count += len(samples)
            continue

        cprint(f"\n[Launch] [Trial {trial_idx+1}] Evaluating Task Bucket: {task_name} | Sample Volume: {len(samples)}", "cyan", attrs=['bold'])
        evaluated_tasks_count += 1

        # ================= 2. Dynamic model instantiation =================
        try:
            if CURRENT_MODEL == "LPB":
                model_inst = LPBWorldModelWrapper(model_ckpt=ckpt_path, task_name=task_name)
            elif CURRENT_MODEL == "FOREWARN":
                base_model_path = "/data/colosseum_dataset/wm_data/mllama_base/Llama-3.2-11B-Vision-Instruct/custom"
                 
                possible_timestamps = ["153050", "102201"]
                wm_ckpt_path = None
                for ts in possible_timestamps:
                    test_path = f"/home/zxn/Forewarn/logs/dreamer_cont/{task_name}/0408/{ts}/best_pretrain_joint_0_00.pt"
                    if os.path.exists(test_path):
                        wm_ckpt_path = test_path
                        cprint(f"    [Path Match] Located World Model weights: {ts}", "green")
                        break
                
                if wm_ckpt_path is None:
                    cprint(f"[Error] Warning: No valid .pt file found for task {task_name}!", "red")
                    evaluated_tasks_count -= 1
                    skipped_samples_count += len(samples)
                    continue 
                
                model_inst = ForewarnEngine(
                    model_name=base_model_path,
                    peft_path=peft_path,
                    task_name=task_name,
                    wm_ckpt=wm_ckpt_path,
                    device="cuda:0"
                )
            elif CURRENT_MODEL == "GPC":
                model_inst = GPCEngine(task_name=task_name, wm_ckpt=wm_ckpt, device="cuda:0")

        except Exception as e:
            cprint(f"[Error] Failed to load {CURRENT_MODEL}: {e}", "red")
            evaluated_tasks_count -= 1
            skipped_samples_count += len(samples)
            continue
        
        # ================= 3. Inference Execution =================
        for idx, sample in enumerate(samples):
            gt_trigger = sample["gt_trigger"]
            gt_action = sample["gt_action"]
            candidates = sample["candidates"]
            
            # Image preprocessing
            img_path = sample["image_paths"][0]
            front_img = cv2.imread(img_path)
            if front_img is not None:
                front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
            else:
                front_img = np.zeros((224, 224, 3), dtype=np.uint8)

            mock_obs = {"cam_front_view_image": front_img, "proprio": sample["proprio"]}
            
            # front_img_pil = Image.fromarray(front_img)
            tmp_img = front_img
            if hasattr(tmp_img, 'detach'):
                tmp_img = tmp_img.detach().cpu().numpy()
            else:
                tmp_img = np.array(tmp_img)

            # 1. Automatically handle dimension collapse or redundant dimensions (H, W, C)
            # If formatted as (1, C, H, W) -> (C, H, W)
            if tmp_img.ndim == 4 and tmp_img.shape[0] == 1:
                tmp_img = tmp_img[0]
                
            # If formatted as (C, H, W) and C=3 -> (H, W, C)
            if tmp_img.ndim == 3 and tmp_img.shape[0] == 3:
                tmp_img = tmp_img.transpose(1, 2, 0)
                
            # 2. Handle numerical ranges (Critical Step)
            # If formatted as float (0-1), convert to uint8 (0-255)
            if tmp_img.dtype != np.uint8:
                if tmp_img.max() <= 1.1: 
                    tmp_img = (tmp_img * 255).astype(np.uint8)
                else:
                    tmp_img = tmp_img.astype(np.uint8)

            # 3. Final failsafe: If dimensions are still invalid, force check
            if tmp_img.shape == (1, 1, 1) or tmp_img.size < 100:
                print(f"\n[CRITICAL] Anomalous image generated by model! Current Shape: {tmp_img.shape}")
                raise ValueError("Model generated an empty or 1x1 image frame.")

            try:
                front_img_pil = Image.fromarray(tmp_img)
            except Exception as e:
                print(f"[Error] PIL Conversion failed. Shape: {tmp_img.shape}, Dtype: {tmp_img.dtype}")
                raise e
            instruction = sample.get('instruction', TASK_PROMPTS.get(task_name, "Perform the robotic task."))
            
            # Execute Inference
            if CURRENT_MODEL == "FOREWARN":
                best_idx, raw_response = model_inst.rank_step(
                    images_pil=[front_img_pil], 
                    state_np=None, 
                    candidates_np=candidates, 
                    instruction=instruction
                )
                best_action = candidates[best_idx]
                # FOREWARN optimal logic: Trigger correction if returned index is strictly greater than 0
                pred_trigger = bool(best_idx > 0)
            
            elif CURRENT_MODEL == "GPC":
                # GPC interface aligns with FOREWARN returning index and text
                best_idx, raw_response = model_inst.steer(
                    obs=mock_obs, 
                    candidates=candidates, 
                    instruction=instruction
                )
                best_action = candidates[best_idx]
                pred_trigger = bool(best_idx > 0)
                
            elif CURRENT_MODEL == "LPB":
                best_action = model_inst.steer(mock_obs, candidates)
                raw_response = "N/A"
                # LPB optimal logic: Compare floats 
                is_same_as_base = np.allclose(best_action, candidates[0], atol=1e-2)
                pred_trigger = not is_same_as_base

            # Append to JSON compilation list
            json_output_data.append({
                "id": sample.get("id", f"{task_name}_{idx}"),
                "task": task_name,
                "trial": trial_idx + 1,
                "model": CURRENT_MODEL,
                "prediction": {
                    "correction_needed": pred_trigger,  
                    "corrected_action": best_action.tolist(),
                    "raw_thought": raw_response
                },
                "ground_truth": {
                    "correction_needed": bool(gt_trigger),
                    "corrected_action": gt_action.tolist()
                }
            })
                
            results["y_true"].append(gt_trigger)
            results["y_pred"].append(pred_trigger)
            
            if gt_trigger:
                results["trans_errs"].append(np.linalg.norm(best_action[:3] - gt_action[:3]))
                results["rot_errs"].append(np.linalg.norm(best_action[3:7] - gt_action[3:7]))
                gt_grip = 1 if gt_action[7] > 0 else 0
                pred_grip = 1 if best_action[7] > 0 else 0
                results["grip_errs"].append(1.0 if gt_grip != pred_grip else 0.0)
                
            if idx % 5 == 0: # Reduce print frequency
                print(f"   └ Progress: [{idx+1}/{len(samples)}] | GT: {gt_trigger} | Pred: {pred_trigger}")

        # Free VRAM
        del model_inst
        torch.cuda.empty_cache() 
            
    # Calculate and print trial statistics
    valid_total = len(results["y_true"])
    cprint(f"\n=================== [Metrics] {CURRENT_MODEL} Scorecard (Trial {trial_idx+1}) ===================", "green", attrs=['bold'])
    print(f"[Info] Active Tasks: {evaluated_tasks_count} | Total Evaluated Samples: {valid_total}")

    if valid_total > 0:
        y_true = np.array(results["y_true"], dtype=bool)
        y_pred = np.array(results["y_pred"], dtype=bool)
        tp = np.sum((y_true == True) & (y_pred == True))
        fp = np.sum((y_true == False) & (y_pred == True))
        fn = np.sum((y_true == True) & (y_pred == False))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        avg_trans = np.mean(results["trans_errs"]) if results["trans_errs"] else 0.0
        avg_rot = np.mean(results["rot_errs"]) if results["rot_errs"] else 0.0
        avg_grip = np.mean(results["grip_errs"]) * 100 if results["grip_errs"] else 0.0
        
        print(f"F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Trans: {avg_trans:.4f} | Rot: {avg_rot:.4f} | Grip%: {avg_grip:>5.1f}%")
    
    return json_output_data

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cprint(f"[Init] Current Evaluation Model: {CURRENT_MODEL}", "white", "on_magenta", attrs=['bold'])
    
    for i in range(NUM_TRIALS):
        cprint(f"\n[Status] Initiating Evaluation Trial {i+1}", "white", "on_blue", attrs=['bold'])
        trial_data = evaluate_baselines(trial_idx=i)
        
        output_filename = os.path.join(OUTPUT_DIR, f"results_{CURRENT_MODEL}_trial_{i+1}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(trial_data, f, indent=2, ensure_ascii=False)
            
        cprint(f"[Success] Results securely saved to: {output_filename}", "green")

if __name__ == "__main__":
    main()
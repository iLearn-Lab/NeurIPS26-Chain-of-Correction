import re
import json
import requests
import numpy as np

# ==============================================================================
# 1. Task Server Configuration
# Map specific robotic tasks to their corresponding simulator ports.
# ==============================================================================
TASK_PORT_MAP = {
    "hockey": 8001,
    "wipe_desk": 8002,
    "setup_chess": 8003,
    "stack_cups": 8004,
    "empty_dishwasher": 8005,
    "straighten_rope": 8006,
    "turn_oven_on": 8007,
    "put_money_in_safe": 8008,
    "move_hanger": 8009,
    "get_ice_from_fridge": 8010,
    "insert_onto_square_peg": 8011,
    "place_wine_at_rack_location": 8012,
    "scoop_with_spatula": 8013,
    "meat_on_grill": 8014,
    "open_drawer": 8015
}




REWARD_SERVER_URL = "http://127.0.0.1:8080/evaluate"


# ==============================================================================
# 2. Offline RL Hyperparameter Matrix (w/o Edge Ablation Variant)
# These weights define the reward landscape for the Reasoning-Policy.
# ==============================================================================
W_FORMAT      = 2.0   # Structural constraint for valid JSON output
W_INTENT      = 15.0  # Core Intent Classification weight (Intervention logic)
W_SUCCESS     = 10.0  # (Deprecated: maintained for format compatibility)

# Cognitive Reasoning Weights
W_COG_EDGE    = 1.0   # Topological edge identification (Focus Edges)
W_COG_CURR_P  = 1.0   # Precise perception of current task progress
W_COG_CORRECT = 1.0   # Correctness of the binary correction flag
# ====================================================

def extract_action_from_response(response_str):
    """
    Robustly extracts the action vector and correction flag from the VLM response.
    Supports standard Markdown code blocks or raw JSON strings.
    """
    try:
        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, str(response_str), re.DOTALL)
        if not match:
            json_pattern = r"\{.*?\}"
            match = re.search(json_pattern, str(response_str), re.DOTALL)
            if not match:
                return None
        
        data = json.loads(match.group(1) if "```json" in str(response_str) else match.group(0))
        action = data.get("corrected_action", [])
        correction_needed = data.get("correction_needed", False)
        
        if len(action) == 8:
            return {"action": action, "correction_needed": correction_needed}
        else:
            return None
    except Exception:
        return None


def extract_response_components(response_str):
    """Separates the Chain-of-Thought (CoT) and the final JSON action."""
    think_content = ""
    think_match = re.search(r"<think>(.*?)</think>", str(response_str), re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        
    action_data = None
    try:
        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, str(response_str), re.DOTALL)
        if not match:
            json_pattern = r"\{.*?\}"
            match = re.search(json_pattern, str(response_str), re.DOTALL)
            
        if match:
            raw_json = json.loads(match.group(1) if "```json" in str(response_str) else match.group(0))
            action = raw_json.get("corrected_action", [])
            correction_needed = raw_json.get("correction_needed", False)
            if len(action) == 8:
                action_data = {"action": action, "correction_needed": correction_needed}
    except Exception:
        pass
        
    return think_content, action_data



def compute_score(reward_inputs, **kwargs):
    """
    Evaluates the model's reasoning and decision-making performance.
    
    The total reward is calculated as:
    $$R_{total} = W_{fmt} + (W_{edge} \cdot M_{edge}) + (W_{prog} \cdot M_{prog}) + (W_{corr} \cdot M_{corr}) + M_{intent}$$
    """
    scores = []
    
    for item in reward_inputs:
        solution_str = str(item["response"])
         
        # Logging raw output for debugging
        print("-" * 30)
        print(f"[RAW OUTPUT PREVIEW]: {solution_str[:150]}...")

        # Extract metadata
        extra_info_dict = item.get('extra_info', {})
        info = extra_info_dict.get('extra_info', extra_info_dict) if isinstance(extra_info_dict, dict) else {}

        real_task_name = info.get('task', 'unknown')
        baseline_p = float(info.get('baseline_progress', 0.0))
        gt_correction = info.get('is_correction_sample', False) 
        gt_focus_edges = info.get('focus_edges', [])
        if not gt_focus_edges:
            task_thresholds = info.get('task_thresholds', {})
            gt_focus_edges = task_thresholds.get('focus_edges', []) if isinstance(task_thresholds, dict) else []

        # Parse Chain-of-Thought
        think_content = ""
        think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL | re.IGNORECASE)
        if think_match: think_content = think_match.group(1).lower()
        else:
            split_text = solution_str.split("```json")
            if len(split_text) > 0:
                think_content = split_text[0].lower()

        parsed = extract_action_from_response(solution_str) 
        if not parsed: 
            print(f"[WARN] {task_name}: Format error. Penalty applied: {-W_FORMAT}")
            scores.append({"overall": float(-W_FORMAT)})
            continue

        print(f"[INFO] Evaluating {task_name} | Correction Flag: {parsed['correction_needed']}")

        # ----------------------------------------------------------------------
        # COGNITIVE METRIC 1: Topological Scene Graph Reasoning
        # Verifies if the model identifies key object relationships in the CoT.
        # ----------------------------------------------------------------------
        metric_edge = 0.0
        if gt_focus_edges:
            matched_edges = 0
            for edge in gt_focus_edges:
                node1, node2 = str(edge[0]).replace('_', ' ').lower(), str(edge[1]).replace('_', ' ').lower()
                # Check if core entities co-occur in the reasoning chain
                if node1 in think_content and node2 in think_content:
                    matched_edges += 1
            metric_edge = matched_edges / len(gt_focus_edges) 
            print(f" |- [COG] Edge Hit: {matched_edges}/{len(gt_focus_edges)} | Score: {metric_edge:.2f}")
        else: 
            print(" |- [COG] Edge: Ground Truth missing, skipping.")

        # ----------------------------------------------------------------------
        # COGNITIVE METRIC 2: Precise Progress Perception
        # Evaluates the accuracy of the model's estimated task completion.
        # ----------------------------------------------------------------------
        metric_curr_p = 0.0 
        curr_p_match = re.search(r"progress[^0-9]*([0-9]+\.[0-9]+)", think_content)

        if curr_p_match:
            pred_curr_p = float(curr_p_match.group(1))
            error_curr = abs(pred_curr_p - baseline_p)
            metric_curr_p = max(0.0, 1.0 - (error_curr / 0.1)) 
            print(f" |- [COG] Progress: Pred={pred_curr_p}, GT={baseline_p:.3f} | Score: {metric_curr_p:.2f}")
        else:
            print(" |- [COG] Progress: Numeric progress not found in text.")

        # ----------------------------------------------------------------------
        # INTENT METRIC: Correction Logic
        # High-weight reward for identifying when to intervene vs when to stay idle.
        # ----------------------------------------------------------------------
        metric_intent = 0.0
        if not parsed['correction_needed']: 
            if not gt_correction:
                metric_intent = W_INTENT # Correctly identified stable state
                print(f" |- [INTENT] Stability Bonus: Correctly identified no correction needed.")
            else:
                metric_intent = -5.0      # Missed required correction
                print(f" |- [INTENT] Missed Detection: Failed to flag required correction.")
        else:
            if gt_correction:
                metric_intent = W_INTENT  # Correctly identified required correction
                print(f" |- [INTENT] Detection Bonus: Correctly flagged required correction.")
            else:
                metric_intent = -5.0      # False positive correction
                print(f" |- [INTENT] False Positive: Flagged unnecessary correction.")
                
        # ----------------------------------------------------------------------
        # Final Reward Synthesis
        # ----------------------------------------------------------------------
        metric_correct = 1.0 if (parsed['correction_needed'] == gt_correction) else -1.0

        total_reward = (
            W_FORMAT * 1.0 + 
            W_COG_EDGE * metric_edge + 
            W_COG_CURR_P * metric_curr_p + 
            W_COG_CORRECT * metric_correct +
            metric_intent 
        )
        
        # Clipping to ensure numerical stability in RL training
        final_score = float(np.clip(total_reward, -5.0, 20.0))
        print(f"[REWARD] Final Score: {final_score:.2f}\n")
        scores.append({"overall": final_score})
        
    return scores
import json 
import os
import re
import argparse 
import collections
 
def check_all_tasks(json_path, tolerance=0.02):
    print(f"\n[QA] Inspecting file: {json_path}")
    print("-" * 60)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    task_a, task_b, task_c = [], [], []
    
    # Filter logic: Parse unique keys within the Answer JSON block
    for item in dataset:
        ans_dict = json.loads(item["messages"]["Answer"])
        if "progress" in ans_dict and "scene_graph" in ans_dict:
            task_a.append(item["messages"])
        elif "predicted_progress" in ans_dict and "future_scene_graph" in ans_dict:
            task_b.append(item["messages"])
        elif "thought_process" in ans_dict and "corrected_action" in ans_dict:
            task_c.append(item["messages"])
            
    print(f"[INFO] Scanned {len(dataset)} entries. Task distribution:")
    print(f" |- Task A (Current Perception): {len(task_a)} entries ({len(task_a)/len(dataset):.1%})")
    print(f" |- Task B (Future Prediction): {len(task_b)} entries ({len(task_b)/len(dataset):.1%})")
    print(f" |- Task C (Correction Control): {len(task_c)} entries ({len(task_c)/len(dataset):.1%})")
    
    # ================= Task A Spot Check =================
    print("\n" + "="*20 + " [Task A Spot Check (Entry 1)] " + "="*20)
    if task_a:
        instruction_a = task_a[0]['Problem'].split('Task instruction:')[-1].splitlines()[0].strip()
        print(f"Instruction: {instruction_a}")
        ans_a = json.loads(task_a[0]['Answer'])
        print(f"Extracted Progress: {ans_a.get('progress')}")
        print(f"Extracted Nodes: {ans_a.get('scene_graph', {}).get('nodes')}")

    # ================= Task B Spot Check =================
    print("\n" + "="*20 + " [Task B Spot Check (Entry 1)] " + "="*20)
    if task_b:
        instruction_b = task_b[0]['Problem'].split('Task instruction:')[-1].splitlines()[0].strip()
        print(f"Instruction: {instruction_b}")
        ans_b = json.loads(task_b[0]['Answer'])
        print(f"Predicted Progress: {ans_b.get('predicted_progress')}")
        print(f"Future Nodes: {ans_b.get('future_scene_graph', {}).get('nodes')}")

    # ================= Task C Spot Check =================
    print("\n" + "="*20 + " [Task C Spot Check (Entry 1)] " + "="*20)
    if task_c:
        instruction_c = task_c[0]['Problem'].split('Task instruction:')[-1].splitlines()[0].strip()
        print(f"Instruction: {instruction_c}")
        ans_c = json.loads(task_c[0]['Answer'])
        print(f"Correction Needed: {ans_c.get('correction_needed')}")
        print(f"Corrected Action: {ans_c.get('corrected_action')}")
        print(f"Thought Process: {ans_c.get('thought_process')}")

    # ================= Temporal Logic Verification =================
    print(f"\n" + "="*20 + f" [Task A Expert Trajectory Temporal Logic (Tolerance: {tolerance})] " + "="*20)
    if task_a:
        progress_list = [json.loads(s["Answer"]).get("progress", 0.0) for s in task_a]
        
        abnormal_drops = 0
        max_drop = 0.0
        
        for i in range(1, len(progress_list)):
            prev_p = progress_list[i-1]
            curr_p = progress_list[i]
            
            # Episode Boundary Detection: Drops > 0.5 indicate transition to a new episode
            if prev_p - curr_p > 0.5:
                continue
                
            # Calculate regression
            if curr_p < prev_p:
                drop = prev_p - curr_p 
                if drop > tolerance: 
                    abnormal_drops += 1
                    max_drop = max(max_drop, drop)

        print(f"Start Frame Progress: {progress_list[0]} | End Frame Progress: {progress_list[-1]}")
        
        if abnormal_drops == 0:
            print(f"[SUCCESS] All {len(task_a)} entries verified. No abnormal progress regressions detected.")
        else:
            print(f"[ERROR] Found {abnormal_drops} abnormal progress regressions in the dataset.")
            print(f" |- Maximum single-step regression: {max_drop:.3f}")
            print("[ADVICE] Check for expert trajectory failures or bugs in the ProgressEstimator.")

def check_task_b_dynamics(json_path, tolerance=0.02):
    print(f"\n[QA] Launching Task B (Forward Dynamics) Specific Inspection: {json_path}")
    print("-" * 60)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    task_b_samples = []
    for item in dataset:
        ans_dict = json.loads(item["messages"]["Answer"])
        if "predicted_progress" in ans_dict and "future_scene_graph" in ans_dict:
            task_b_samples.append(item["messages"])
            
    print(f"[INFO] Located {len(task_b_samples)} entries for Task B.")
    
    abnormal_dynamics = 0
    max_regression = 0.0
    action_format_errors = 0

    for idx, sample in enumerate(task_b_samples):
        problem = sample["Problem"]
        answer = json.loads(sample["Answer"])
 
        # 1. Extract progress at time t from Problem string
        try:
            split_key = "\nAction Proposal (t):" if "\nAction Proposal (t):" in problem else "\nExecuted Action (t):"
            graph_str = problem.split("Current Scene Graph (t):\n")[1].split(split_key)[0]
            curr_graph = json.loads(graph_str)
            
            thresholds = curr_graph.get("task_thresholds", {})
             
            curr_prog = 0.0
            found_prog = False
            
            # Prioritize keys containing "progress"
            for k, v in thresholds.items():
                if "progress" in k.lower():
                    try:
                        curr_prog = float(v)
                        found_prog = True
                        break
                    except: continue
            
            # Fallback to any float-convertible value
            if not found_prog:
                for v in thresholds.values():
                    try:
                        curr_prog = float(v)
                        break
                    except: continue
                    
        except Exception as e: 
            print(f"[ERROR] Parsing failed for entry {idx}: {e}")
            continue 
            
        # 2. Extract predicted progress at t+1 from Answer
        next_prog = answer.get("predicted_progress", 0.0)
        
        # 3. Dynamics Check: Predicted future progress should not drop significantly
        delta = next_prog - curr_prog
        if delta < -tolerance:
            abnormal_dynamics += 1
            max_regression = min(max_regression, delta)
            
        # 4. Action Format Validation
        try:
            action_str = problem.split(split_key + " ")[1].split("\n")[0]
            action_list = json.loads(action_str)
            if len(action_list) != 8 or not all(isinstance(x, float) for x in action_list):
                action_format_errors += 1
        except:
            action_format_errors += 1

    print("\n" + "="*20 + f" [Task B Dynamics QA Report (Tolerance: {tolerance})] " + "="*20)
    if abnormal_dynamics == 0:
        print("[SUCCESS] Physical logic consistent. All t -> t+1 predictions align with monotonic expert progress.")
    else:
        print(f"[WARN] Found {abnormal_dynamics} instances of physics violations (t+1 progress << t progress).")
        print(f" |- Maximum single-step regression: {max_regression:.3f}")
        
    if action_format_errors == 0:
        print("[SUCCESS] Action space format validated (8D continuous float vectors).")
    else:
        print(f"[ERROR] Found {action_format_errors} entries with malformed action vectors.")


def check_task_c_branches(json_path):
    print(f"\n[QA] Launching Task C (Correction & Replanning) Branch Scan: {json_path}")
    print("-" * 70)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    task_c_samples = []
    for item in dataset:
        try:
            ans_dict = json.loads(item["messages"]["Answer"])
            if "thought_process" in ans_dict and "correction_needed" in ans_dict:
                task_c_samples.append((item["messages"]["Problem"], ans_dict))
        except:
            continue
            
    print(f"[INFO] Located {len(task_c_samples)} entries for Task C.\n")
    
    # Feature keywords for branch classification (aligned with CoT design)
    branch_keywords = {
        "Branch 1 (Topological Drop)": "Catastrophic topological transition",
        "Branch 2 (Severe Regression)": "Macroscopic",
        "Branch 3A (Stasis/Stuck)": "stuck",
        "Branch 3B (Directional Deviation)": "Directional deviation",
        "Branch 4 (Pre-emptive Prevention)": "Pre-emptive error prevention",
        "Branch 5 (Perfect Expert)": "No correction needed"
    }
    
    branch_counts = collections.defaultdict(int)
    branch_examples = {}

    # Scanning and classification
    for problem, ans_dict in task_c_samples:
        thought = ans_dict.get("thought_process", "")
        
        assigned_branch = "Unknown (Format Exception)"
        for branch_name, keyword in branch_keywords.items():
            if keyword in thought:
                assigned_branch = branch_name
                break
                
        branch_counts[assigned_branch] += 1 
        if assigned_branch not in branch_examples:
            branch_examples[assigned_branch] = (problem, ans_dict)

    print("--- [Task C Branch Statistics] ---")
    for branch_name in branch_keywords.keys():
        count = branch_counts[branch_name]
        percentage = (count / len(task_c_samples)) * 100 if len(task_c_samples) > 0 else 0
        print(f" |- {branch_name}: {count} entries ({percentage:.1f}%)")
    print("=========================================================\n")

    print("--- [Branch Spot Check Details] ---")
    for branch_name in branch_keywords.keys():
        if branch_name not in branch_examples:
            print(f"[WARN] {branch_name} has no triggers (not present in expert trajectories).\n")
            continue
            
        problem, ans_dict = branch_examples[branch_name]
         
        problem_lines = problem.split('\n')
        instruction = [line for line in problem_lines if "Task instruction:" in line][0]
        action_prop = [line for line in problem_lines if "Action Proposal (t):" in line][0]
        
        print(f"[BRANCH] {branch_name} Example:")
        print(f" -> Input Instruction: {instruction}")
        print(f" -> Proposal Action: {action_prop}")
        print(f" -> Correction Needed: {ans_dict['correction_needed']}")
        print(f" -> Output Action: {ans_dict['corrected_action']}")
        print(f" -> CoT Thought Process:")
        for sentence in ans_dict['thought_process'].split('] '):
            if sentence.strip():
                print(f"       ] {sentence.strip()}" if sentence.startswith("[") else f"       [{sentence.strip()}")
        print("-" * 70)

  


def check_unified_dataset(json_path, tolerance=0.02):
    print(f"\n[QA] Executing In-depth Logic Panoramic Inspection on Unified Dataset: {json_path}")
    print("-" * 80)
    
    if not os.path.exists(json_path):
        print(f"[ERROR] File not found: '{json_path}'")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    stats = {
        "total": len(dataset),
        "t0_frames": 0,
        "tn_frames": 0,
        "perception_checked": 0,
        "abnormal_drops": 0,
        "max_drop": 0.0,            
        "first_prog": None,         
        "last_prog": None,          
        "action_format_errors": 0,
        "correction_checked": 0,
        "format_errors": 0
    }

    branch_keywords = {
        "Branch 1 (Topological Drop)": "Invalid topological transition detected",
        "Branch 2 (Severe Regression)": "Severe regression detected",
        "Branch 3A (Physical Stasis)": "Physical stasis detected",
        "Branch 3B (Directional Deviation)": "Directional deviation detected",
        "Branch 4 (Pre-emptive Analysis)": "Pre-emptive analysis shows",
        "Expert (Valid Expert Action)": "No correction needed"
    }
    
    branch_counts = collections.defaultdict(int)
    branch_examples = {}
    task_a_drops_dist = []

    for idx, item in enumerate(dataset):
        msgs = item.get("messages", [])
        
        # 1. Extract Initial Frame Progress (t=0)
        if len(msgs) == 2:
            stats["t0_frames"] += 1
            if stats["first_prog"] is None:
                try:
                    curr_content = msgs[1]['content']
                    think_content = curr_content.split("<think>")[1].split("</think>")[0]
                    match_t0 = re.search(r'Global Task Progress:\s*([\d.]+)', think_content)
                    if match_t0: stats["first_prog"] = float(match_t0.group(1))
                except: pass
            continue
            
        if len(msgs) != 4:
            stats["format_errors"] += 1
            continue
            
        stats["tn_frames"] += 1
        
        try:
            prev_ans_raw = msgs[1]['content']
            prev_ans_json = json.loads(prev_ans_raw.split("```json\n")[-1].split("\n```")[0])
            prev_prog = prev_ans_json.get("progress", 0.0)

            user_msg_t = msgs[2]['content']
            prop_action_match = re.search(r'Action Proposal:\s*(\[.*?\])', user_msg_t)
            proposed_action = prop_action_match.group(1) if prop_action_match else "[]"
            
            curr_assistant_msg = msgs[3]['content']
            think_content = curr_assistant_msg.split("<think>")[1].split("</think>")[0]
            final_json = json.loads(curr_assistant_msg.split("```json\n")[-1].split("\n```")[0])
            
            # --- Task A Check: Perception Consistency ---
            match_curr = re.search(r'Global Task Progress:\s*([\d.]+)', think_content)

            if match_curr:
                curr_prog = float(match_curr.group(1))
                stats["perception_checked"] += 1
                stats["last_prog"] = curr_prog 
                
                if curr_prog < prev_prog:
                    drop = prev_prog - curr_prog
                    if drop > tolerance:
                        stats["abnormal_drops"] += 1
                        stats["max_drop"] = max(stats["max_drop"], drop) 
                        task_a_drops_dist.append(drop)
                        
            # Action Format Check
            corrected_action = final_json.get("corrected_action", [])
            if len(corrected_action) != 8 or not all(isinstance(x, (float, int)) for x in corrected_action):
                stats["action_format_errors"] += 1

            # --- Task C Check: Correction Logic ---
            stats["correction_checked"] += 1
            assigned_branch = "Unknown (Unmatched Keyword)"
            
            for branch_name, keyword in branch_keywords.items():
                if keyword in think_content:
                    assigned_branch = branch_name
                    break
                    
            branch_counts[assigned_branch] += 1
            if assigned_branch not in branch_examples:
                instruction_match = re.search(r'Task(?: instruction)?:\s*(.*?)\n', msgs[0]['content'])
                instruction = instruction_match.group(1) if instruction_match else "Unknown Task"
                branch_examples[assigned_branch] = {
                    "instruction": instruction.strip(),
                    "proposed_action": proposed_action,
                    "correction_needed": final_json.get("correction_needed"),
                    "corrected_action": corrected_action,
                    "thought_process": think_content.strip()
                }
            
        except Exception as e:
            stats["format_errors"] += 1
            continue

    print(f"[SUCCESS] Scan complete. Total samples: {stats['total']}")
    print(f" |- Frame Distribution: [t=0 Init]: {stats['t0_frames']} | [t>0 Loop]: {stats['tn_frames']}")
    
    print(f"\n" + "="*20 + f" [Task A Verification (Tolerance: {tolerance})] " + "="*20)
    print(f"Start Progress: {stats['first_prog']} | End Progress: {stats['last_prog']}")
    
    if stats["abnormal_drops"] == 0:
        print(f"[SUCCESS] All {stats['perception_checked']} entries verified. No unhandled abnormal regressions found.")
    else:
        print(f"[WARN] Detected {stats['abnormal_drops']} regressions (physical resets, expected to be caught by Branch 2).")
        print(f" |- Maximum single drop: {stats['max_drop']:.3f}")

    if stats["action_format_errors"] == 0:
        print("[SUCCESS] Action format validated (8D float vectors).")
    else:
        print(f"[ERROR] Found {stats['action_format_errors']} entries with invalid action formats.")
         
    print(f"\n[INFO] Launching Branch Scanning for Decision Correction: {json_path}")
    print("-" * 70)
    print(f"[INFO] Successfully isolated {stats['correction_checked']} analytical entries.\n")
    
    print("--- [Branch Distribution Statistics] ---")
    for branch_name in branch_keywords.keys():
        count = branch_counts.get(branch_name, 0)
        percentage = (count / max(1, stats['tn_frames'])) * 100
        print(f" |- {branch_name}: {count} entries ({percentage:.1f}%)")
        
    for branch_name, count in branch_counts.items():
        if branch_name not in branch_keywords.keys():
            percentage = (count / max(1, stats['tn_frames'])) * 100
            print(f" [?] {branch_name}: {count} entries ({percentage:.1f}%)")
    print("="*60 + "\n")

    print("--- [Branch CoT Spot Check Details] ---")
    for branch_name in branch_keywords.keys():
        if branch_name not in branch_examples:
            print(f"[WARN] {branch_name} not triggered in expert trajectories.\n")
            continue
            
        ex = branch_examples[branch_name]
        print(f"[BRANCH] {branch_name} Example:")
        print(f" -> Input Instruction: {ex['instruction']}")
        print(f" -> Proposal Action: {ex['proposed_action']}")
        print(f" -> Correction Needed: {ex['correction_needed']}")
        print(f" -> Output Action: {ex['corrected_action']}")
        print(f" -> CoT Thought Process:\n")
        
        thought_text = ex['thought_process'].strip()
        for line in thought_text.split('\n'):
            line = line.strip()
            if not line: continue
            if line.startswith('['):
                print(f"      {line}")
            else:
                print(f"         {line}")
        print("-" * 70) 

    if stats["format_errors"] > 0:
        print(f"\n[WARN] {stats['format_errors']} entries failed format validation and were excluded from physical checks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
 
    if not os.path.exists(args.json_path):
        print(f"[ERROR] File not found: '{args.json_path}'")
    else:
        check_unified_dataset(args.json_path, args.tolerance)
        print(f"\n" + "*" * 10 + " Quality Assurance Process Terminated " + "*" * 10)

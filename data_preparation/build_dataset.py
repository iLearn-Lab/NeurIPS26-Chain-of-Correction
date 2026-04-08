import os
import json
import copy
import random
import pickle
import glob
import argparse
import numpy as np

from rlbench.environment import Environment
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from pyrep.const import JointType
import rlbench.tasks
from pyrep.objects.shape import Shape
import colosseum.rlbench.tasks

from progress_estimator import ProgressEstimator
from extract_momagraph import extract_momagraph_frame
from utils import EndEffectorPoseViaPlanning2
from main_annotation import TASK_REGISTRY  

class UnifiedMultiTurnCoTFormatter:
    def __init__(self):
        """
        Unified Multi-Turn Chain-of-Thought Extractor:
        0. Outputs a complete logical chain for every frame.
        1. t = 0 (2 messages): Initial perception CoT.
        2. t > 0 (4 messages): 
            - Historical comparison (explicit real timestep injection, evaluating strategic regressions).
            - Future prediction (geometric trend evaluation).
            - Action correction (CoT deduces relative displacement; JSON outputs corrected absolute action).
        """
        self.system_prompt = (
            "You are an advanced Embodied AI agent specialized in robotic manipulation, functioning as a precise action-correction module. "
            "You must analyze task instruction, visual observations and temporal history to evaluate action consequences, including geometric trends and overall task progress. "
            "You achieve this by structuring the environment as a Scene Graph (Nodes = objects, Edges = spatial/functional relations). "
            "CRITICAL RULES:\n"  
            "1. Identify the currently focused edge(s) within the Scene Graph and check their key spatial relations to reason about the task progress.\n"
            "2. Judge the validity of actions based on a synthesis of geometric distance trends, step-by-step progress scalars, and relative spatial deviations.\n"
            "3. Strategic regressions (e.g., temporary distance increases to bypass obstacles) are valid and require no correction.\n" 
            "4. If the backward check reveals physical stasis (e.g., negligible change in edge distances), you MUST output a diverging correction to escape the local minima.\n" 
            "5. Output reasoning strictly within structured <think>...</think> tags using these explicit phases:\n"
            "   - [Phase 1: State Representation]: Explicitly extract the Current Scene Graph (nodes, focused edges, spatial distances) and the Global Task Progress.\n"
            "   - [Phase 2: Future Prediction]: Predict ONLY the resulting spatial and geometric changes to the focused edges if the Action Proposal is executed.\n"
            "   - [Phase 3: Trajectory Evaluation]: Synthesize Phase 1 & 2 to conduct Backward/Forward checks and objectively determine if a trajectory correction is required.\n"
            "6. Final decisions must be in a ```json\n...\n``` block using STRICTLY these keys: {\"correction_needed\": boolean, \"corrected_action\": [list of high-precision floats]}."
        )  

        self.PROMPTS = {
            "T0": [
                "Task instruction: {instruction}\n"
                "[Time Step: 0] Task initiation. Analyze the multi-view images to extract the initial scene graph. "
                "Evaluate the initial task progress, and predict the resulting spatial changes if the Action Proposal {proposed_action} is applied." 
            ],
            "T_PREV": [
                "Task instruction: {instruction}\n"
                "[Time Step: {t_prev}] Analyze the historical observations to extract the past scene graph and task progress."
            ],  
            "T_CURR": [
                "[Time Step: {t_curr}] Previous action executed. Given the new observations and the Action Proposal: {proposed_action}, "
                "perform a structured physical dual-check:\n"
                "1. Backward check: Compare History vs. Current to quantify geometric distance changes and identify any physical stasis or unintended deviations.\n"
                "2. Forward check: Predict how the Current Scene Graph (specifically edge distances and spatial relations) and task progress will change if the Proposal is executed.\n"
                "Synthesize these checks to objectively determine if a correction is needed. Output the structured evaluation and the corrected action."
            ] 
        }     
 

    def generate_dataset(self, episodes_data, extents_dict, output_path="qwen_unified_cot_sft.json"):
        """
        Process the list of episodes and generate the final SFT JSON file.
        """
        dataset = []
        for episode in episodes_data:
            # Reset physical state tracker before each episode
            self.stagnation_counter = 0
            for t in range(len(episode) - 1): 
                result = self._build_thread(episode, t, extents_dict)
                
                if result:
                    message_thread, img_paths = result
                    dataset.append({
                        "messages": message_thread, 
                        "image_paths": img_paths
                    })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully generated {len(dataset)} unified CoT training samples. Saved to {output_path}.")
        return dataset


    def _build_thread(self, episode, t, extents_dict):
        messages = []
        if t + 1 >= len(episode): return None
        
        curr_frame = episode[t]
        next_frame = episode[t+1]
        images_paths = curr_frame["images"]
        instructions = curr_frame["momagraph"]["task_instruction"]
        if isinstance(instructions, list):
            instruction = random.choice(instructions)
        else:
            instruction = instructions
        gt_action = curr_frame["action"]

        real_flawed_action = curr_frame.get("policy_action", None)
        curr_graph = self._simplify_graph(curr_frame["momagraph"])
        next_graph = self._simplify_graph(next_frame["momagraph"])
        curr_prog = self._get_prog(curr_frame["momagraph"])
        next_prog = self._get_prog(next_frame["momagraph"])
 
        curr_image_tokens = "<image>\n" * len(images_paths)
 
        focus_edges_list = curr_frame["momagraph"].get("focus_edges", [])  
        if focus_edges_list:
            formatted_edges = []
            for e in focus_edges_list: 
                if isinstance(e, dict):
                    u, v = e.get("object1", "entity"), e.get("object2", "entity") 
                elif isinstance(e, (list, tuple)) and len(e) >= 2:
                    u, v = e[0], e[1]
                else:
                    u, v = "entity", "entity"
                formatted_edges.append(f"({u}, {v})")
            focus_str = " and ".join(formatted_edges)
        else: 
            all_nodes = curr_frame["momagraph"].get("nodes", [])
            target_nodes = [n for n in all_nodes if n != "gripper"]
            if target_nodes:
                focus_str = f"the interaction between gripper and {', '.join(target_nodes)}"
            else:
                focus_str = "core objects" 

        # ==========================================
        # 1. Initial Step Processing (t=0)
        # ==========================================
        if t == 0: 
            curr_dist = self._get_focus_dist(curr_graph)
            next_dist = self._get_focus_dist(next_graph)
            prog_change = next_prog - curr_prog

            if curr_dist is not None and next_dist is not None:
                dist_change = next_dist - curr_dist
                abs_dist_change = abs(dist_change)
                if prog_change > 0:
                    if dist_change < -0.0001:
                        phase_2_pred = f"advance task progress to {next_prog:.6f} and reduce the geometric distance for edge(s) {focus_str} by {abs_dist_change:.4f}."
                    elif dist_change > 0.0001:
                        phase_2_pred = f"advance task progress to {next_prog:.6f} while increasing the geometric distance for edge(s) {focus_str} by {dist_change:.4f}."
                    else:
                        phase_2_pred = f"advance task progress to {next_prog:.6f} while maintaining distance at ~{curr_dist:.4f} for edge(s) {focus_str}."
                elif prog_change < 0: 
                    if dist_change < -0.0001:
                        phase_2_pred = f"cause a regression in task progress ({next_prog:.6f}) despite reducing distance by {abs_dist_change:.4f} for edge(s) {focus_str}."
                    elif dist_change > 0.0001:
                        phase_2_pred = f"cause a regression in task progress ({next_prog:.6f}) and increase distance by {dist_change:.4f}, diverging from the target spatial alignment for edge(s) {focus_str}."
                    else:
                        phase_2_pred = f"cause a regression in task progress ({next_prog:.6f}) while maintaining distance for edge(s) {focus_str}."
                else:
                    if dist_change < -0.0001:
                        phase_2_pred = f"maintain current progress ({next_prog:.6f}) but reduce distance for edge(s) {focus_str} by {abs_dist_change:.4f}."
                    elif dist_change > 0.0001:
                        phase_2_pred = f"maintain current progress ({next_prog:.6f}) but increase distance for edge(s) {focus_str} by {dist_change:.4f}."
                    else:
                        phase_2_pred = f"maintain current progress ({next_prog:.6f}) and geometric distance (~{curr_dist:.4f}), performing in-place posture adjustments for edge(s) {focus_str}."

                # Dual-check correction trigger: Progress drop OR (Significant distance increase AND no progress improvement)
                is_diverging = prog_change < 0 or (dist_change > 0.005 and prog_change <= 0)
            else:
                # Fallback: 1D evaluation when distance field is missing
                if prog_change > 0:
                    phase_2_pred = f"advance task progress to {next_prog:.6f}, initiating geometric convergence for edge(s) {focus_str}."
                elif prog_change < 0:
                    phase_2_pred = f"cause a regression in task progress ({next_prog:.6f}), diverging from the target spatial alignment for edge(s) {focus_str}."
                else:
                    phase_2_pred = f"maintain current progress ({next_prog:.6f}), performing in-place posture adjustments for edge(s) {focus_str}."
                
                is_diverging = prog_change < 0

            if real_flawed_action is not None and is_diverging: 
                prompt = random.choice(self.PROMPTS["T0"]).format(instruction=instruction, proposed_action=real_flawed_action)
                corrected_action = self._synthesize_expert_action(
                    curr_graph, 
                    curr_frame["gripper_pose"], 
                    real_flawed_action, 
                    "PREVENTIVE_TRACKING", 
                    extents_dict,
                    t=t, 
                    curr_prog=curr_prog
                )
                ans_json = {"correction_needed": True, "corrected_action": corrected_action} 
                forward_check = "2. Forward check: Proposal induces an invalid geometric divergence from the target state." 
                conclusion = f"3. Conclusion: Action Proposal diverges from the target initial trajectory. Applying correction to establish valid spatial alignment for edge(s) {focus_str}." 
            else:
                prompt = random.choice(self.PROMPTS["T0"]).format(instruction=instruction, proposed_action=gt_action)
                ans_json = {"correction_needed": False, "corrected_action": gt_action}
                forward_check = "2. Forward check: Proposal establishes the valid initial kinematic state as predicted."
                conclusion = "3. Conclusion: Action Proposal executes valid initial kinematic movement. No correction needed." 

            content = (
                f"<think>\n"
                f"[Phase 1: State Representation]\n"
                f"Current Scene Graph (t=0): {json.dumps(curr_graph, separators=(',', ':'))}\n"
                f"Global Task Progress: {curr_prog:.6f}\n\n"
                f"[Phase 2: Future Prediction]\n"
                f"If the Action Proposal is executed, the resulting spatial changes will {phase_2_pred}\n\n"
                f"[Phase 3: Trajectory Evaluation]\n"
                f"1. Backward check: Task just initialized. No historical data to compare.\n" 
                f"{forward_check}\n"
                f"{conclusion}\n"
                f"</think>\n\n"
                f"```json\n{json.dumps(ans_json, indent=2)}\n```"
            )
            messages.append({"role": "user", "content": f"{curr_image_tokens}{prompt}"})
            messages.append({"role": "assistant", "content": content})
            return messages, list(images_paths.values())
 
        # ==========================================
        # 2. Standard Step Processing (t>0): Construct 4-turn dialogue
        # ==========================================
        prev_frame = episode[t-1]
        prev_graph = self._simplify_graph(prev_frame["momagraph"])
        prev_prog = self._get_prog(prev_frame["momagraph"])
        prev_image_tokens = "<image>\n" * len(prev_frame["images"])

        gripper_holding_edge = None
        for edge in prev_graph.get("edges", []):
            obj1, obj2 = edge.get("object1", ""), edge.get("object2", "")
            if (obj1 == "gripper" or obj2 == "gripper") and "HOLDING" in edge.get("spatial_relations", []):
                gripper_holding_edge = f"({obj1}, {obj2})"
                break
        
        # --- Turns 1 & 2: Establish Historical Anchor --- 
        prompt_prev = random.choice(self.PROMPTS["T_PREV"]).format(instruction=instruction, t_prev=t-1) 
        # Historical query uses text only (no image tokens)
        messages.append({"role": "user", "content": prompt_prev}) 
        messages.append({
            "role": "assistant", 
            "content": (
                f"```json\n"
                f"{json.dumps({'scene_graph': prev_graph, 'progress': round(prev_prog, 6)}, indent=2)}\n"
                f"```\n"
                f"Action Executed: {prev_frame['action']}"
            )
        }) 
 
        # --- Turns 3 & 4: Current Perception & Correction ---
        success_radius = extents_dict.get("success_radius", 0.05)
        SEVERE_DROP_THRESHOLD = 0.15  # Progress drop > 15% is considered a severe physical cliff
        STAGNATION_THRESHOLD = 0.02   # Progress drop between 2%~15% is considered local stagnation/deviation

        # Physical Diagnostics
        delta_prog = prev_prog - curr_prog 
        if abs(delta_prog) <= 1e-4: self.stagnation_counter += 1
        else: self.stagnation_counter = 0

        curr_dist = self._get_focus_dist(curr_graph)
        prev_dist = self._get_focus_dist(prev_graph) 
        is_severe_divergence = False
        dist_str = ""
        dist_change = 0
        if curr_dist is not None and prev_dist is not None:
            dist_change = curr_dist - prev_dist # >0 means moving away
            
            # Dynamic divergence evaluation: Relative ratio + Absolute physical limit
            is_dist_surge = (dist_change / max(prev_dist, 0.01) >= 0.20)
            is_beyond_noise = (dist_change >= success_radius * 0.5)
            is_severe_divergence = is_dist_surge and is_beyond_noise and (delta_prog >= -1e-4)
 
        def get_bad_proposal(expert_action):
            if real_flawed_action is not None:
                return real_flawed_action
            else:
                return self._add_noise(expert_action)
    
        def is_holding(graph):
            for edge in graph.get("edges", []):
                obj1, obj2 = edge.get("object1", ""), edge.get("object2", "")
                if (obj1 == "gripper" or obj2 == "gripper") and "HOLDING" in edge.get("spatial_relations", []):
                    return True
            return False 

        dropped_holding = is_holding(prev_graph) and (real_flawed_action is not None) and not is_holding(curr_graph)
 
        phase_2_pred = ""
        phase_3_eval = ""

        # Branch 1: Dropped object recovery
        if dropped_holding:
            proposed_action = get_bad_proposal(gt_action) 
            correction_needed = True 
            edge_mention = f"edge {gripper_holding_edge}" if gripper_holding_edge else f"edges {focus_str}"
            # Dynamically reconstruct graph focus
            if gripper_holding_edge:
                dropped_obj = gripper_holding_edge.split(',')[1].strip(' )') 
                curr_graph["focus_edges"] = [(dropped_obj, "gripper")]
            phase_2_pred = f"Action Proposal maintains the detached kinematic state, failing to recover physical contact for {edge_mention}."
            phase_3_eval = (
                f"1. Backward check: Invalid topological transition detected (loss of HOLDING relation).\n"
                f"2. Forward check: Proposal fails to recover physical contact.\n"
                f"3. Conclusion: Topology modification required. Rejecting proposal to initiate drop-recovery sub-goal."
            )
            corrected_action = self._synthesize_expert_action(curr_graph, curr_frame["gripper_pose"], proposed_action, "DROP_RECOVERY", extents_dict, t=t, curr_prog=curr_prog, target_node=dropped_obj)

        # Branch 2: Severe physical regression
        elif delta_prog >= SEVERE_DROP_THRESHOLD or is_severe_divergence:
            proposed_action = get_bad_proposal(gt_action) 
            dist_str = f", distance increased by {dist_change:.4f}" if dist_change > 0 else f", distance changed by {dist_change:.4f}"
            if (real_flawed_action is not None):
                correction_needed = True
                phase_2_pred = f"Action Proposal reinforces the divergent vector; spatial error for edge(s) {focus_str} will propagate."
                phase_3_eval = (
                    f"1. Backward check: Severe regression detected (Progress drop: {delta_prog:.6f}{dist_str}). Critical spatial deviation for {focus_str}.\n"
                    f"2. Forward check: Executing proposal fails to arrest the deviation accumulation.\n"
                    f"3. Conclusion: Rejecting proposal. Macroscopic correction required to re-establish valid spatial alignment."
                )
                corrected_action = self._synthesize_expert_action(curr_graph, curr_frame["gripper_pose"], proposed_action, "SEVERE_DROP", extents_dict, t=t, curr_prog=curr_prog)
            else:
                corrected_action = gt_action
                correction_needed = False
                phase_2_pred = f"Action Proposal will continue the non-monotonic trajectory adjustment."
                phase_3_eval = (
                    f"1. Backward check: Non-monotonic regression (Progress drop: {delta_prog:.6f}{dist_str}) observed in historical trajectory.\n"
                    f"2. Forward check: Proposal strictly follows the required geometric adjustment for the subsequent phase.\n"
                    f"3. Conclusion: Regression is kinematically valid. No correction needed."
                ) 
    
        # Branch 3: Local Stagnation
        elif delta_prog >= STAGNATION_THRESHOLD or self.stagnation_counter >= 5: 
            proposed_action = get_bad_proposal(gt_action) 
            # Distance change < 1mm is considered absolute deadlock; otherwise wandering
            if abs(dist_change) < 0.001:
                dist_str = f" and geometric distance frozen at ~{curr_dist:.4f}"
            else:
                dist_str = f", distance shifting by {dist_change:.4f}"
 
            is_stuck = self.stagnation_counter >= 5
            if (real_flawed_action is not None):
                correction_needed = True
                if is_stuck:
                    b_type = "STAGNATION_STUCK" 
                    phase_2_pred = f"Action Proposal fails to command a diverging trajectory; local spatial stasis will persist for edge(s) {focus_str}."
                    phase_3_eval = ( 
                        f"1. Backward check: Physical stasis detected. Progress stuck at {curr_prog:.6f}{dist_str} for {self.stagnation_counter} frames.\n" 
                        f"2. Forward check: Proposal yields negligible corrective momentum, failing to escape the local minimum.\n"
                        f"3. Conclusion: Rejecting proposal. Macroscopic spatial translation required to break stasis."
                    )
                else:
                    b_type = "STAGNATION_DEVIATION" 
                    phase_2_pred = f"Action Proposal maintains the erroneous spatial vector, failing to correct the directional drift."
                    phase_3_eval = (
                        f"1. Backward check: Directional deviation detected. Progress locally regressed by {delta_prog:.6f}{dist_str} for edge(s) {focus_str}.\n" 
                        f"2. Forward check: Proposal will propagate the spatial misalignment rather than converging.\n"
                        f"3. Conclusion: Rejecting proposal. Corrective trajectory adjustment required."
                    )
                corrected_action = self._synthesize_expert_action(curr_graph, curr_frame["gripper_pose"], proposed_action, b_type, extents_dict, t=t, curr_prog=curr_prog)
            
            else:
                correction_needed = False 
                next_dist = self._get_focus_dist(next_graph)
                expert_dist_str = ""
                if curr_dist is not None and next_dist is not None:
                    expert_dist_change = next_dist - curr_dist
                    expert_dist_str = f" by altering distance by {expert_dist_change:.4f}"

                if is_stuck:
                    phase_2_pred = f"Action Proposal introduces a spatial differential{expert_dist_str}, terminating the static phase."
                    phase_3_eval = (
                        f"1. Backward check: Progress unchanged at {curr_prog:.6f} for {self.stagnation_counter} frames.\n" 
                        f"2. Forward check: Proposal generates valid kinematic movement to resolve the stagnation.\n"
                        f"3. Conclusion: Legitimate expert maneuver. No correction needed."
                    )
                else:
                    phase_2_pred = f"Action Proposal executes a valid non-monotonic trajectory adjustment{expert_dist_str}."
                    phase_3_eval = (
                        f"1. Backward check: Minor local regression (Progress drop: {delta_prog:.6f}{dist_str}) observed.\n" 
                        f"2. Forward check: Proposal aligns with the required intermediate geometric state and maintains kinematic viability.\n"
                        f"3. Conclusion: Regression is a valid intermediate step. No correction needed."
                    )
                corrected_action = gt_action

        # Branches 4 & 5: Normal progression and Pre-emptive correction 
        else:  
            historical_dist_change = curr_dist - prev_dist if (curr_dist is not None and prev_dist is not None) else 0.0
            if historical_dist_change < -0.0001:
                history_eval = f"Previous action successfully reduced distance by {abs(historical_dist_change):.4f} (Progress: {curr_prog:.6f})."
            elif historical_dist_change > 0.0001:
                history_eval = f"Previous action caused minor spatial divergence ({historical_dist_change:.4f}) but maintained task viability."
            else:
                history_eval = f"Previous action maintained spatial relations (Progress: {curr_prog:.6f})."

            # Predict future trend to use as a rule, without leaking exact values to the VLM
            next_dist = self._get_focus_dist(next_graph)
            future_dist_change = (next_dist - curr_dist) if (next_dist is not None and curr_dist is not None) else 0.0
            future_delta = curr_prog - next_prog # >0 means future drop

            # Escalated trigger: Future progress drop >= 1% or distance increase >= 2cm
            is_future_divergent = (future_delta >= 0.01) or (future_dist_change >= 0.02)

            if (real_flawed_action is not None) and is_future_divergent:
                proposed_action = real_flawed_action 
                correction_needed = True 
                 
                phase_2_pred = f"Action Proposal misaligns with the geometric convergence path, predicting an imminent regression in spatial alignment for edge(s) {focus_str}."
                phase_3_eval = ( 
                    f"1. Backward check: {history_eval}\n" 
                    f"2. Forward check: Pre-emptive analysis shows the proposal diverges from the target topological trajectory.\n" 
                    f"3. Conclusion: Rejecting proposal to prevent impending spatial divergence. Corrective tracking required."
                )
                corrected_action = self._synthesize_expert_action(curr_graph, curr_frame["gripper_pose"], proposed_action, "PREVENTIVE_TRACKING",extents_dict,t=t,curr_prog=curr_prog) 
            
            else:
                # Branch 5: Safe and valid proposal
                proposed_action = real_flawed_action if (real_flawed_action is not None and not is_future_divergent) else gt_action 
                correction_needed = False 
                
                phase_2_pred = f"Action Proposal maintains the required kinematic states for stable spatial convergence on edge(s) {focus_str}."
                phase_3_eval = (
                    f"1. Backward check: {history_eval}\n" 
                    f"2. Forward check: Proposal consistently advances or maintains the valid geometric state towards the goal.\n"
                    f"3. Conclusion: Proposal is structurally sound. No correction needed."
                )
                corrected_action = proposed_action
             
        # Assemble output content
        prompt_curr = random.choice(self.PROMPTS["T_CURR"]).format(proposed_action=proposed_action, t_curr=t)
        messages.append({"role": "user", "content": f"{curr_image_tokens}{prompt_curr}"})
        ans_json = {"correction_needed": correction_needed, "corrected_action": corrected_action}
 
        content = (
            f"<think>\n"
            f"[Phase 1: State Representation]\n"
            f"Current Scene Graph (t={t}): {json.dumps(curr_graph, separators=(',', ':'))}\n"
            f"Global Task Progress: {curr_prog:.6f}\n\n"
            f"[Phase 2: Future Prediction]\n"
            f"{phase_2_pred}\n\n"
            f"[Phase 3: Trajectory Evaluation]\n"
            f"{phase_3_eval}\n"
            f"</think>\n\n"
            f"```json\n{json.dumps(ans_json, indent=2)}\n```"
        ) 
        messages.append({"role": "assistant", "content": content})
 
        return messages, list(images_paths.values())

 
    def _get_focus_dist(self, graph):
        """Extracts the geometric distance for the focused edge."""
        focus_edges = graph.get("focus_edges", [])
        if not focus_edges: return None
        target = focus_edges[0]
        u, v = target[0], target[1] if isinstance(target, (list, tuple)) and len(target) >= 2 else (None, None)
        for edge in graph.get("edges", []):
            eu, ev = edge.get("object1"), edge.get("object2")
            if (u == eu and v == ev) or (u == ev and v == eu):
                geo = edge.get("geometric_attributes", {})
                if "dist" in geo: return geo["dist"]
        return None

    def _synthesize_expert_action(self, curr_graph, gripper_pose, policy_action, b_type, extents_dict, t, curr_prog, target_node=None, _counts={}):
        """Synthesizes a valid corrective action based on Scene Graph geometric fields and kinematic constraints."""
        if b_type == "NORMAL": return policy_action
        
        curr_pos, curr_quat = np.array(gripper_pose[:3]), np.array(gripper_pose[3:7])
        
        # 1. Lock onto target edge
        focus_edges = curr_graph.get("focus_edges", [])
        if focus_edges:
            target_edge_info = focus_edges[0]
        elif target_node:
            target_edge_info = ("gripper", target_node) 
        else:
            return policy_action

        target_edge = next((e for e in curr_graph.get("edges", []) 
                            if (e["object1"] == target_edge_info[0] and e["object2"] == target_edge_info[1]) or 
                               (e["object2"] == target_edge_info[0] and e["object1"] == target_edge_info[1])), None)
        
        if not target_edge or "geometric_attributes" not in target_edge: 
            return policy_action

        geo = target_edge["geometric_attributes"]
        delta_3d = np.array(geo.get("delta_3d", [0,0,0]))
        # Direction correction: Invert delta_3d if the edge is (obj, gripper) instead of (gripper, obj)
        if target_edge["object1"] != "gripper":
            delta_3d = -delta_3d

        dist = geo.get("dist", 0.0)
        direction = delta_3d / np.linalg.norm(delta_3d) if dist > 1e-4 else np.zeros(3)
        
        # Dynamic step size control: Ensure correction does not overshoot
        max_safe_step = extents_dict.get("success_radius", 0.05) * 0.8
        step_size = min(max_safe_step, dist * 0.5)  
        corrected_pos = curr_pos + direction * step_size
        corrected_quat = curr_quat 
        corrected_gripper = policy_action[7] 

        # 2. Branch-specific Kinematic Adjustments
        if b_type == "DROP_RECOVERY" or b_type == "SEVERE_DROP":
            # Gripper timing fix: Open gripper if distance > 2cm, close if < 2cm
            if dist > 0.02:
                corrected_gripper = 1.0  
            else:
                corrected_gripper = 0.0  
                 

        elif b_type == "STAGNATION_DEVIATION" or b_type == "PREVENTIVE_TRACKING":
            # Trust the policy's gripper state for pure spatial tracking
            pass 

        elif b_type == "STAGNATION_STUCK": 
            # Retreat slightly and apply compliance lift to escape local deadlock
            escape_vector = -direction * (step_size * 0.5) 
            lateral_noise = np.random.normal(0, 0.005, 3) 
            lateral_noise[2] = abs(lateral_noise[2]) # Ensure we don't push downwards
            corrected_pos = curr_pos + escape_vector + lateral_noise
 
            # Hard Z-axis safety lock
            corrected_pos[2] = max(curr_pos[2] + 0.005, corrected_pos[2])
 
        # 3. Silent Logging Module
        if "expert_print_counts" not in globals(): globals()["expert_print_counts"] = {}
        _counts = globals()["expert_print_counts"]
        current_count = _counts.get(b_type, 0)
        
        if current_count < 3:
            print(f"\n[Kinematic Correction] >>> Frame {t:03d} | Type: {b_type} | Dist: {dist:.4f}m")
            print(f"  ├─ Target: {target_edge['object1']} <-> {target_edge['object2']}")
            print(f"  ├─ Vector: {[round(v, 6) for v in (corrected_pos - curr_pos)]}")
            print(f"  └─ Gripper: {policy_action[7]} -> {corrected_gripper}")
            _counts[b_type] = current_count + 1
            if current_count + 1 == 3: print(f"  🔇 [{b_type}] Logs muted...")

        corrected_action = np.concatenate([corrected_pos, corrected_quat, [corrected_gripper]])
        return [round(float(v), 6) for v in corrected_action]

    # ==========================================
    # Data Parsing Utilities
    # ==========================================
    def _get_instruction(self, frame):
        insts = frame["momagraph"].get("task_instruction", "complete the task")
        return random.choice(insts) if isinstance(insts, list) else insts

    def _get_prog(self, graph):
        thresholds = graph.get("task_thresholds", {})
        for k, v in thresholds.items():
            if "progress" in k.lower(): 
                return float(v)
        return 0.0

    def _simplify_graph(self, graph): 
        simple = copy.deepcopy(graph)
        simple.pop("task_instruction", None)
        simple.pop("task_thresholds", None) 
        for edge in simple.get("edges", []):
            edge.pop("is_stable", None) 
            if "geometric_attributes" in edge:
                geo = edge["geometric_attributes"]
                geo.pop("original_delta_3d", None)
                geo.pop("original_dist", None)
                if "delta_3d" in geo: geo["delta_3d"] = [round(v, 6) for v in geo["delta_3d"]]
                if "dist" in geo: geo["dist"] = round(geo["dist"], 6)
        return simple 

    def _add_noise(self, action): 
        noisy = list(action)
        for i in range(3):
            noisy[i] += random.uniform(-0.03, 0.03)
        return [round(v, 6) for v in noisy]

# ==================================================
# Extraction Pipelines
# ==================================================
def process_episodes_to_qwen(task_name, base_dataset_path, num_episodes=1,output_dir=None):

    config = TASK_REGISTRY[task_name]
    task_class_name = "".join([word.capitalize() for word in task_name.split("_")])

    if "chess" in task_name:
        TaskClass = getattr(colosseum.rlbench.tasks, task_class_name)
    else:
        TaskClass = getattr(rlbench.tasks, task_class_name)

    # 1. Extract static BBox parameters
    print(f"Initializing task {task_name} and extracting physical dimensions...")
    env = Environment(action_mode=MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning2(), gripper_action_mode=Discrete()), headless=True)
    env.launch()
    rlbench_task = env.get_task(TaskClass)
    rlbench_task.sample_variation()

    gripper = rlbench_task._robot.gripper
    gripper_max_width = 0.0
    for joint in gripper.joints:
        if joint.get_joint_type() == JointType.PRISMATIC:
            _, interval = joint.get_joint_interval()
            gripper_max_width += interval[1] 

    gripper_max_width = 0.08 if gripper_max_width <= 1e-4 else gripper_max_width

    extents_dict = {"gripper_radius": gripper_max_width / 2.0, "success_radius": 0.05}
    for key, sim_name in config.get("objects", {}).items():
        try:
            bbox = Shape(sim_name).get_bounding_box()
            extents_dict[f"{key}_size"] = np.array([bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]])
        except:
            pass
    env.shutdown()

    # 2. Iterate and construct data
    estimator = ProgressEstimator(manipulate_obj=config["semantic_mo_name"], target_obj=config["semantic_to_name"])
    all_episodes_data = []

    for ep_idx in range(num_episodes):
        ep_path = os.path.join(base_dataset_path, f"episode{ep_idx}") 

        with open(os.path.join(ep_path, "variation_descriptions.pkl"), "rb") as f:
            instructions = pickle.load(f)

        with open(os.path.join(ep_path, "low_dim_obs.pkl"), "rb") as f:
            data_demo = pickle.load(f)
 
        import extract_momagraph
        if hasattr(extract_momagraph, f"_extract_{task_name}"):
            func = getattr(extract_momagraph, f"_extract_{task_name}")
            if hasattr(func, "global_clean_progress"):
                func.global_clean_progress = 0.0

        mock_obs_0 = data_demo[0].__dict__
        prev_obs_dict = None
        episode_data = []

        for t in range(len(data_demo)):
            obs = data_demo[t]
            curr_graph = extract_momagraph_frame(
                task_name=task_name, obs_dict=obs.__dict__, init_obs_dict=mock_obs_0,
                indices=config["indices"], extents=extents_dict,
                instructions=instructions, prev_obs_dict=prev_obs_dict
            )
            progress = estimator.estimate(curr_graph)
 
            if "task_thresholds" not in curr_graph:
                curr_graph["task_thresholds"] = {}
            curr_graph["task_thresholds"][f"{task_name}_progress"] = float(progress)
 
            image_paths = {
                "front": os.path.join(ep_path, "front_rgb", f"{t}.png"),
                "left_shoulder": os.path.join(ep_path, "left_shoulder_rgb", f"{t}.png"),
                "right_shoulder": os.path.join(ep_path, "right_shoulder_rgb", f"{t}.png"),
                # "overhead": os.path.join(ep_path, "overhead_rgb", f"{t}.png"),
                "wrist": os.path.join(ep_path, "wrist_rgb", f"{t}.png")
            }
 
            if t < len(data_demo) - 1:
                next_obs = data_demo[t+1]
                gt_action = list(next_obs.gripper_pose) + [next_obs.gripper_open]
            else:
                gt_action = list(obs.gripper_pose) + [obs.gripper_open]

            gt_action = [round(float(v), 4) for v in gt_action]

            episode_data.append({
                "momagraph": curr_graph,
                "action": gt_action,
                "images": image_paths,
                "gripper_pose": [round(float(v), 4) for v in obs.gripper_pose] 
            }) 
            prev_obs_dict = obs.__dict__

        all_episodes_data.append(episode_data)
  
    # 3. Format Output
    formatter = UnifiedMultiTurnCoTFormatter()
  
    target_dir =output_dir
    os.makedirs(target_dir, exist_ok=True)  
 
    output_path = os.path.join(target_dir, f"{task_name}.json")
    formatter.generate_dataset(all_episodes_data, extents_dict, output_path)
    print(f"Dataset for task {task_name} successfully saved to: {output_path}") 


def process_failed_rollouts_to_qwen(task_name, base_dataset_path, output_dir=None):
    """Processes real failed rollouts from the policy."""
    config = TASK_REGISTRY[task_name]
    task_class_name = "".join([word.capitalize() for word in task_name.split("_")])

    if "chess" in task_name:
        TaskClass = getattr(colosseum.rlbench.tasks, task_class_name)
    else:
        TaskClass = getattr(rlbench.tasks, task_class_name)

    print(f"Initializing failed rollouts for task {task_name} and extracting physical dimensions...")
    env = Environment(action_mode=MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning2(), gripper_action_mode=Discrete()), headless=True)
    env.launch()
    rlbench_task = env.get_task(TaskClass)
    rlbench_task.sample_variation()

    gripper = rlbench_task._robot.gripper
    gripper_max_width = 0.0
    for joint in gripper.joints:
        if joint.get_joint_type() == JointType.PRISMATIC:
            _, interval = joint.get_joint_interval()
            gripper_max_width += interval[1]

    gripper_max_width = 0.08 if gripper_max_width <= 1e-4 else gripper_max_width
    extents_dict = {"gripper_radius": gripper_max_width / 2.0, "success_radius": 0.05}
    for key, sim_name in config.get("objects", {}).items():
        try:
            bbox = Shape(sim_name).get_bounding_box()
            extents_dict[f"{key}_size"] = np.array([bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]])
        except:
            pass
    env.shutdown()
 
    estimator = ProgressEstimator(manipulate_obj=config["semantic_mo_name"], target_obj=config["semantic_to_name"])
    all_episodes_data = []
    episode_dirs = glob.glob(os.path.join(base_dataset_path, "**", "episode*"), recursive=True)
    
    if not episode_dirs:
        print(f"Warning: No failed trajectories found in {base_dataset_path}!")
        return

    print(f"Scanned {len(episode_dirs)} Failed Rollout episodes.")

 
    for ep_path in episode_dirs:
        if os.path.basename(ep_path) == "episodes":
            continue 
        obs_file_path = os.path.join(ep_path, "low_dim_obs.pkl")
        if not os.path.exists(obs_file_path):
            continue
        with open(os.path.join(ep_path, "variation_descriptions.pkl"), "rb") as f:
            instructions = pickle.load(f)

        with open(obs_file_path, "rb") as f:
            data_demo = pickle.load(f)
 
        import extract_momagraph
        if hasattr(extract_momagraph, f"_extract_{task_name}"):
            func = getattr(extract_momagraph, f"_extract_{task_name}")
            if hasattr(func, "global_clean_progress"):
                func.global_clean_progress = 0.0

        mock_obs_0 = data_demo[0].__dict__
        prev_obs_dict = None
        episode_data = []

        for t in range(len(data_demo)):
            obs = data_demo[t]
            obs_dict = obs.__dict__
            
            curr_graph = extract_momagraph_frame(
                task_name=task_name, obs_dict=obs_dict, init_obs_dict=mock_obs_0,
                indices=config["indices"], extents=extents_dict,
                instructions=instructions, prev_obs_dict=prev_obs_dict
            )
            progress = estimator.estimate(curr_graph)

            if "task_thresholds" not in curr_graph:
                curr_graph["task_thresholds"] = {}
            curr_graph["task_thresholds"][f"{task_name}_progress"] = float(progress)

            image_paths = {
                "front": os.path.join(ep_path, "front_rgb", f"{t}.png"),
                "left_shoulder": os.path.join(ep_path, "left_shoulder_rgb", f"{t}.png"),
                "right_shoulder": os.path.join(ep_path, "right_shoulder_rgb", f"{t}.png"),
                "wrist": os.path.join(ep_path, "wrist_rgb", f"{t}.png")
            }
  
            policy_action = obs_dict.get('action', None)
            if policy_action is not None:
                policy_action = [round(float(v), 4) for v in policy_action][:8] # 取前8维 

            episode_data.append({
                "momagraph": curr_graph, 
                "action": policy_action,       
                "policy_action": policy_action,   
                "gripper_pose": [round(float(v), 4) for v in obs_dict['gripper_pose']], 
                "images": image_paths
            }) 
            prev_obs_dict = obs_dict

        all_episodes_data.append(episode_data)
 
    formatter = UnifiedMultiTurnCoTFormatter()
    
    target_dir =output_dir
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, f"{task_name}.json")
    
    formatter.generate_dataset(all_episodes_data, extents_dict, output_path)
    print(f"Failed rollout dataset for task {task_name} successfully saved to: {output_path}") 


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Colosseum SFT Dataset Extractor")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to process (default: all)")
    parser.add_argument("--output_dir", type=str, required=True) 
    parser.add_argument("--mode", type=str, choices=["success", "fail"], required=True)
    
    args = parser.parse_args()

    BASE_PATH = f"/data/colosseum_dataset/train/{args.task}/all_variations/episodes"

     
    if not os.path.exists(BASE_PATH):
        print(f"Cannot find {BASE_PATH}!")
        exit(1)

    episode_dirs = glob.glob(os.path.join(BASE_PATH, "episode*"))
 
    total_episodes = args.episodes if args.episodes is not None else len(episode_dirs)

    if total_episodes == 0:
        print(f"No episode found in {BASE_PATH}!")
        exit(1)

    print(f"[Start Task]: {args.task}")
    print(f"Total {len(episode_dirs)} episodes.")
 
    if args.mode == "success":
        process_episodes_to_qwen(
            task_name=args.task,
            base_dataset_path=BASE_PATH,
            num_episodes=total_episodes,
            output_dir=args.output_dir
        ) 
    else: 
        # Fail mode automatically scans all available directories; --episodes is ignored.
        process_failed_rollouts_to_qwen(
            task_name=args.task,
            base_dataset_path=f"/data/colosseum_dataset/failed_eval_demos/{args.task}",
            output_dir=args.output_dir
        )
import numpy as np
from utils import compute_directional_and_distance, check_universal_contact, check_approach_corridor_world, check_inside_containment, check_holding

# HOLDING Verification (Assume relative displacement < 2mm between frames indicates a rigid kinematic connection)
KINEMATIC_LOCK_TOLERANCE = 0.002

# ==========================================
# MomaGraph Data Parsing Main Engine
# ==========================================
def extract_momagraph_frame(task_name, obs_dict, init_obs_dict, indices, extents, instructions, prev_obs_dict=None):
    """
    Numpy-driven multi-task scene graph extraction engine. 
    Fully decoupled from physical simulators.
    """
    # 1. Extract common states required across all tasks
    task_state = obs_dict['task_low_dim_state']
    gripper_pose = obs_dict['gripper_pose']
    gripper_open = obs_dict['gripper_open']
    
    init_task_state = init_obs_dict['task_low_dim_state']
    init_gripper_pos = init_obs_dict['gripper_pose'][:3]
    
    gripper_radius = extents["gripper_radius"]
    
    # 2. Task Router
    # Dynamically construct the target function name based on task_name
    func_name = f"_extract_{task_name}"
    # Retrieve the function from the global namespace
    extractor = globals().get(func_name)
    if not callable(extractor):
        raise ValueError(f"Unsupported or unimplemented scene graph extractor: {func_name}()")
 
    # Execute the specific extractor to retrieve the scene graph
    return extractor(
        task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
        indices, extents, instructions, prev_obs_dict, obs_dict
    ) 


# ==========================================
# Task-Specific Parsing Logic (Utilizing Shared Utils)
# ==========================================
def _extract_basketball_in_hoop(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                        indices, extents, instructions, prev_obs_dict, obs_dict):
    # Parse dimensions
    ball_radius = extents["mo_size"][0] / 2.0
    hoop_extents = extents["to_size"]
    success_radius = extents["success_radius"]
    
    # Parse poses
    ball_pose = task_state[indices["ball"]]
    success_pose = task_state[indices["success"]]
    hoop_resp_pose = task_state[indices["hoop_respondable"]]
    
    init_ball_pos = init_task_state[indices["ball"]][:3]
    
    # --- Edge 1: gripper -> ball ---
    g_b_relations, g_b_delta, g_b_dist = compute_directional_and_distance(
        gripper_pose[:3], ball_pose[:3], extents["gripper_radius"], ball_radius)
    _, init_g_b_delta, init_g_b_dist = compute_directional_and_distance(
        init_gripper_pos, init_ball_pos, extents["gripper_radius"], ball_radius)

    is_touching = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], ball_pose, ball_radius)
    if is_touching: g_b_relations.append("TOUCHING")

    # Instantaneous displacement calculation (robustness check)
    delta_dist = 0.0
    if prev_obs_dict is not None:
        prev_ball_pos = prev_obs_dict['task_low_dim_state'][indices["ball"]][:3]
        prev_g_b_dist = np.linalg.norm(prev_ball_pos - prev_obs_dict['gripper_pose'][:3])
        delta_dist = abs(g_b_dist - prev_g_b_dist)

    is_holding = False
    if is_touching and gripper_open < 0.1:
        if prev_obs_dict is None or delta_dist < KINEMATIC_LOCK_TOLERANCE:
            g_b_relations.append("HOLDING")
            is_holding = True
            
    edge_gb = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "ball",
        "spatial_relations": list(set(g_b_relations)),
        "geometric_attributes": {
            "delta_3d": g_b_delta.tolist(), "dist": float(g_b_dist),
            "original_delta_3d": init_g_b_delta.tolist(), "original_dist": float(init_g_b_dist)
        },
        "is_touching": "TOUCHING" in g_b_relations,
        "is_stable": is_holding
    }

    # --- Edge 2: ball -> hoop ---
    b_h_relations, b_h_delta, b_h_dist = compute_directional_and_distance(
        ball_pose[:3], success_pose[:3], ball_radius, success_radius)
    _, init_b_h_delta, init_b_h_dist = compute_directional_and_distance(
        init_ball_pos, init_task_state[indices["success"]][:3], ball_radius, success_radius)

    if check_universal_contact(ball_pose[:3], ball_radius, hoop_resp_pose, hoop_extents):
        b_h_relations.append("TOUCHING")

    if check_approach_corridor_world(ball_pose[:3], success_pose[:3], approach_axis=2, approach_dir=1.0,\
        mo_size=ball_radius, tgt_size=hoop_extents):
        b_h_relations.append("AROUND")

    is_inside = False
    if check_inside_containment(ball_pose[:3], ball_radius, success_pose, hoop_extents, top_open=True):
        b_h_relations.append("INSIDE")
        is_inside = True
    else:
        b_h_relations.append("OUTSIDE")
        
    edge_bh = {
        "functional_relationship": "pair_with",
        "object1": "ball", "object2": "hoop",
        "spatial_relations": list(set(b_h_relations)),
        "geometric_attributes": {
            "delta_3d": b_h_delta.tolist(), "dist": float(b_h_dist),
            "original_delta_3d": init_b_h_delta.tolist(), "original_dist": float(init_b_h_dist)
        },
        "is_touching": "TOUCHING" in b_h_relations,
        "is_stable": is_inside
    }

    # Intent Classification
    if is_inside and not is_holding: action_type = "done"
    elif is_holding or (not is_holding and "AROUND" in b_h_relations): action_type = "place"
    else: action_type = "pick"

    return {
        "task_instruction": instructions, 
        "nodes": ["gripper", "ball", "hoop"],
        "edges": [edge_gb, edge_bh],
        "action_type": action_type
    }


def _extract_close_box(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                       indices, extents, instructions, prev_obs_dict, obs_dict):
    # TARGET_ANGLE = 2.6
    CLOSED_DIST = 0.090           # Distance < 9cm indicates fully closed (final frame is ~8.1cm)
    VERTICAL_DIST = 0.160         # Critical point where lid is vertical to desk or nearing closure

    # Parse dimensions
    lid_extents = extents["mo_size"]
    base_extents = extents["to_size"]
    
    lid_pose = task_state[indices["box_lid"]]
    base_pose = task_state[indices["box_base"]]
    joint_angle = task_state[indices["box_joint_angle"]]
    
    init_lid_pos = init_task_state[indices["box_lid"]][:3]
    init_base_pos = init_task_state[indices["box_base"]][:3]

    # --- Edge 1: gripper -> lid ---
    g_l_relations, g_l_delta, g_l_dist = compute_directional_and_distance(
        gripper_pose[:3], lid_pose[:3], extents["gripper_radius"], lid_extents)
    _, init_g_l_delta, init_g_l_dist = compute_directional_and_distance(
        init_gripper_pos, init_lid_pos, extents["gripper_radius"], lid_extents)

    is_touching = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], lid_pose, lid_extents)
    if is_touching: g_l_relations.append("TOUCHING")

    delta_dist = 0.0
    if prev_obs_dict is not None:
        prev_lid_pos = prev_obs_dict['task_low_dim_state'][indices["box_lid"]][:3]
        prev_g_l_dist = np.linalg.norm(prev_lid_pos - prev_obs_dict['gripper_pose'][:3])
        delta_dist = abs(g_l_dist - prev_g_l_dist)

    is_holding = False
    if is_touching and gripper_open < 0.1:
        if prev_obs_dict is None or delta_dist < KINEMATIC_LOCK_TOLERANCE:
            g_l_relations.append("HOLDING")
            is_holding = True
   
    edge_gl = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "box_lid",
        "spatial_relations": list(set(g_l_relations)),
        "geometric_attributes": {
            "delta_3d": g_l_delta.tolist(), "dist": float(g_l_dist),
            "original_delta_3d": init_g_l_delta.tolist(), "original_dist": float(init_g_l_dist)
        },
        "is_touching": "TOUCHING" in g_l_relations,
        "is_stable": is_holding
    }

    # --- Edge 2: lid -> base ---
    l_b_relations, l_b_delta, l_b_dist = compute_directional_and_distance(
        lid_pose[:3], base_pose[:3], lid_extents, base_extents)
    _, init_l_b_delta, init_l_b_dist = compute_directional_and_distance(
        init_lid_pos, init_base_pos, lid_extents, base_extents)

    # simulated_angle = np.interp(l_b_dist, [CLOSED_DIST, init_l_b_dist], [TARGET_ANGLE, 0.0])
    if l_b_dist <= CLOSED_DIST:
        l_b_relations.append("CLOSED")
        action_type = "done"
    elif l_b_dist <= VERTICAL_DIST:
        # Assign AROUND when lid is vertical or near closure
        l_b_relations.append("AROUND")
        action_type = "close"
    elif l_b_dist < init_l_b_dist - 0.01: 
        # Labeled as 'close' if touching the lid or if lid is depressed by 1cm
        action_type = "close"
    else:
        action_type = "pick"
    
       
    edge_lb = {
        "functional_relationship": "open_or_close",
        "object1": "box_lid", "object2": "box_base",
        "spatial_relations": list(set(l_b_relations)),
        "geometric_attributes": {
            "delta_3d": l_b_delta.tolist(), "dist": float(l_b_dist),
            "original_delta_3d": init_l_b_delta.tolist(), "original_dist": float(init_l_b_dist),
            # "angle": float(simulated_angle),             
            # "target_angle": float(TARGET_ANGLE)      
        },
        "is_touching": "TOUCHING" in l_b_relations,
        "is_stable": action_type == "done"
    }

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "box_lid", "box_base"],
        "edges": [edge_gl, edge_lb],
        "action_type": action_type
    }

def _extract_close_laptop_lid(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                              indices, extents, instructions, prev_obs_dict, obs_dict):
    CLOSED_DIST = 0.0095 
    VERTICAL_DIST = 0.02    
 
    lid_extents = extents["mo_size"]
    base_extents = extents["to_size"]
     
    lid_pose = task_state[indices["lid"]]
    base_pose = task_state[indices["base"]]
    
    init_lid_pos = init_task_state[indices["lid"]][:3]
    init_base_pos = init_task_state[indices["base"]][:3]

    # --- Edge 1: gripper -> laptop_lid ---
    g_l_relations, g_l_delta, g_l_dist = compute_directional_and_distance(
        gripper_pose[:3], lid_pose[:3], extents["gripper_radius"], lid_extents)
    _, init_g_l_delta, init_g_l_dist = compute_directional_and_distance(
        init_gripper_pos, init_lid_pos, extents["gripper_radius"], lid_extents)

    is_touching = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], lid_pose, lid_extents)
    if is_touching: g_l_relations.append("TOUCHING")
 
    delta_dist = 0.0
    if prev_obs_dict is not None:
        prev_lid_pos = prev_obs_dict['task_low_dim_state'][indices["lid"]][:3]
        prev_g_l_dist = np.linalg.norm(prev_lid_pos - prev_obs_dict['gripper_pose'][:3])
        delta_dist = abs(g_l_dist - prev_g_l_dist)

    is_holding = False
    if is_touching and gripper_open < 0.1:
        if prev_obs_dict is None or delta_dist < KINEMATIC_LOCK_TOLERANCE:
            g_l_relations.append("HOLDING")
            is_holding = True
    
    edge_gl = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "laptop_lid",
        "spatial_relations": list(set(g_l_relations)),
        "geometric_attributes": {
            "delta_3d": g_l_delta.tolist(), "dist": float(g_l_dist),
            "original_delta_3d": init_g_l_delta.tolist(), "original_dist": float(init_g_l_dist)
        },
        "is_touching": "TOUCHING" in g_l_relations,
        "is_stable": is_holding
    }

    # --- Edge 2: laptop_lid -> laptop_base ---
    l_b_relations, l_b_delta, l_b_dist = compute_directional_and_distance(
        lid_pose[:3], base_pose[:3], lid_extents, base_extents)
    _, init_l_b_delta, init_l_b_dist = compute_directional_and_distance(
        init_lid_pos, init_base_pos, lid_extents, base_extents)

    if l_b_dist <= CLOSED_DIST:
        l_b_relations.append("CLOSED")
        action_type = "done"
    elif l_b_dist <= VERTICAL_DIST:
        # Assign AROUND when lid is vertical; progress likely at ~85%
        l_b_relations.append("AROUND")
        action_type = "close"
    elif l_b_dist < init_l_b_dist - 0.01: 
        # Active closing detected
        action_type = "close"
    else:
        action_type = "pick"
        
    edge_lb = {
        "functional_relationship": "open_or_close",
        "object1": "laptop_lid", "object2": "laptop_base",
        "spatial_relations": list(set(l_b_relations)),
        "geometric_attributes": {
            "delta_3d": l_b_delta.tolist(), "dist": float(l_b_dist),
            "original_delta_3d": init_l_b_delta.tolist(), "original_dist": float(init_l_b_dist),
        },
        "is_touching": "TOUCHING" in l_b_relations,
        "is_stable": action_type == "done"
    }

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "laptop_lid", "laptop_base"],
        "edges": [edge_gl, edge_lb],
        "action_type": action_type
    }  

def _extract_empty_dishwasher(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                              indices, extents, instructions, prev_obs_dict, obs_dict):

    # 1. Parse core entity dimensions
    door_extents = extents.get("door_size", [0.1, 0.1, 0.1])
    tray_extents = extents.get("tray_size", [0.1, 0.1, 0.1])
    plate_extents = extents.get("plate_size", [0.1, 0.1, 0.1])
    dishwasher_extents = extents.get("dishwasher_size", [0.1, 0.1, 0.1])

    # Define physical thresholds and AROUND critical values
    # Use door height/length to calculate dynamic thresholds
    door_length = np.max(door_extents)
    DOOR_OPEN_DELTA = door_length * 0.46  # Threshold: 46% of door length indicates being fully opened/flat
    DOOR_AROUND_DELTA = door_length * 0.32
    plate_radius = np.min(plate_extents) / 2.0
    tray_depth = np.max(tray_extents)
    TRAY_PULL_DELTA = tray_depth * 0.66
    TRAY_AROUND_DELTA = tray_depth * 0.2
    PLATE_DONE_DELTA = 0.08
    PLATE_AROUND_DELTA = 0.001

    # 2. Parse entity poses
    door_pose = task_state[indices["door"]]
    tray_pose = task_state[indices["tray"]]
    plate_pose = task_state[indices["plate"]]
    dishwasher_pose = task_state[indices["dishwasher"]]

    init_door_pos = init_task_state[indices["door"]][:3]
    init_tray_pos = init_task_state[indices["tray"]][:3]
    init_plate_pos = init_task_state[indices["plate"]][:3]
    init_dishwasher_pos = init_task_state[indices["dishwasher"]][:3]

    # ==========================================
    # 3. Spatial Relationship Calculations
    # ==========================================
    # [Door vs Base]
    d_b_relations, d_b_delta, d_b_dist = compute_directional_and_distance(
        door_pose[:3], dishwasher_pose[:3], door_extents, dishwasher_extents)
    _, init_d_b_delta, init_d_b_dist = compute_directional_and_distance(
        init_door_pos, init_dishwasher_pos, door_extents, dishwasher_extents)

    # [Tray vs Base]
    t_b_relations, t_b_delta, t_b_dist = compute_directional_and_distance(
        tray_pose[:3], dishwasher_pose[:3], tray_extents, dishwasher_extents)
    _, init_t_b_delta, init_t_b_dist = compute_directional_and_distance(
        init_tray_pos, init_dishwasher_pos, tray_extents, dishwasher_extents)

    # [Plate vs Tray]  
    p_t_relations, p_t_delta, p_t_dist = compute_directional_and_distance(
        plate_pose[:3], tray_pose[:3], plate_extents, tray_extents)
    _, init_p_t_delta, init_p_t_dist = compute_directional_and_distance(
        init_plate_pos, init_tray_pos, plate_extents, tray_extents)
    if check_universal_contact(plate_pose[:3], plate_radius, tray_pose, tray_extents):
        p_t_relations.append("TOUCHING")

    # [Gripper vs Plate]  
    g_p_relations, g_p_delta, g_p_dist = compute_directional_and_distance(
        gripper_pose[:3], plate_pose[:3], extents["gripper_radius"], plate_extents)
    _, init_g_p_delta, init_g_p_dist = compute_directional_and_distance(
        init_gripper_pos, init_plate_pos, extents["gripper_radius"], plate_extents)
    is_touching_plate = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], plate_pose, plate_extents)
    if is_touching_plate: g_p_relations.append("TOUCHING")

    # [Gripper vs Door] Phase 1 Guiding
    g_d_relations, g_d_delta, g_d_dist = compute_directional_and_distance(
        gripper_pose[:3], door_pose[:3], extents["gripper_radius"], door_extents)
    _, init_g_d_delta, init_g_d_dist = compute_directional_and_distance(
        init_gripper_pos, init_door_pos, extents["gripper_radius"], door_extents)
    is_touching_door = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], door_pose, door_extents)
    if is_touching_door: g_d_relations.append("TOUCHING")

    # [Gripper vs Tray] Phase 2 Guiding
    g_t_relations, g_t_delta, g_t_dist = compute_directional_and_distance(
        gripper_pose[:3], tray_pose[:3], extents["gripper_radius"], tray_extents)
    _, init_g_t_delta, init_g_t_dist = compute_directional_and_distance(
        init_gripper_pos, init_tray_pos, extents["gripper_radius"], tray_extents)
    is_touching_tray = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], tray_pose, tray_extents)
    if is_touching_tray: g_t_relations.append("TOUCHING")

    # is_holding_door = False
    # is_holding_tray = False
    # is_holding_plate = False

    # if not is_door_open:
    #     is_holding_door = check_holding(is_touching_door, g_d_dist, "door", g_d_relations, gripper_open, indices, prev_obs_dict)
        
    # elif is_door_open and not is_tray_pulled:
    #     is_holding_tray = check_holding(is_touching_tray, g_t_dist, "tray", g_t_relations, gripper_open, indices, prev_obs_dict)
        
    # elif is_door_open and is_tray_pulled:
    #     is_holding_plate = check_holding(is_touching_plate, g_p_dist, "plate", g_p_relations, gripper_open, indices, prev_obs_dict)

    # if is_holding_plate: g_p_relations.append("HOLDING")
    # if is_holding_door: g_d_relations.append("HOLDING")
    # if is_holding_tray: g_t_relations.append("HOLDING")

    # ==========================================
    # 4. Intent State Machine (Geometric Judgment)
    # ==========================================
    # a. Door State (Door vs Base)
    delta_d_b = d_b_dist - init_d_b_dist
    is_door_open = prev_obs_dict.get("door_opened", False) if prev_obs_dict else False
    if delta_d_b > DOOR_OPEN_DELTA:
        is_door_open = True
    # Failure perception: If door is pushed back significantly (> 5cm), reset 'opened' state
    elif is_door_open and delta_d_b < DOOR_OPEN_DELTA - 0.05: 
        is_door_open = False
    obs_dict["door_opened"] = is_door_open

    if is_door_open:
        d_b_relations.append("OPENED") 
    elif delta_d_b > DOOR_AROUND_DELTA:
        d_b_relations.append("AROUND")

    # b. Tray State: Pure displacement increment dependency
    delta_t_b = t_b_dist - init_t_b_dist
    is_tray_pulled = prev_obs_dict.get("tray_pulled", False) if prev_obs_dict else False
    if delta_t_b > TRAY_PULL_DELTA:
        is_tray_pulled = True
    # Failure perception: Tray pushed back (> 5cm)
    elif is_tray_pulled and delta_t_b < TRAY_PULL_DELTA - 0.05:
        is_tray_pulled = False
    obs_dict["tray_pulled"] = is_tray_pulled

    if is_tray_pulled:
        t_b_relations.append("OUTSIDE")
    elif delta_t_b > TRAY_AROUND_DELTA:
        t_b_relations.append("AROUND")
    
    tray_radius_eff = np.max(tray_extents) * 0.1  # Effective radius
    is_tray_inside_geo = check_inside_containment(
        pos_obj=tray_pose[:3], radius_obj=tray_radius_eff, 
        pose_container_7d=dishwasher_pose, extents_container=dishwasher_extents, top_open=False
    )
    # INSIDE if geometric center remains within the base
    if is_tray_inside_geo and (not is_tray_pulled):
        t_b_relations.append("INSIDE")

    # c. Plate State: Elevation and displacement dependency
    is_plate_removed = prev_obs_dict.get("plate_removed", False) if prev_obs_dict else False
    delta_p_t = p_t_dist - init_p_t_dist
    # Plate must be far from start AND 'lifted' (increase in Z height relative to tray)
    plate_z = plate_pose[2]
    tray_z = tray_pose[2]
    z_diff = plate_z - tray_z
    # if delta_p_t > PLATE_DONE_DELTA and z_diff > 0.03:
    #     is_plate_removed = True
    # Causal logic lock (Requires door open AND tray pulled out)
    if is_door_open and is_tray_pulled and delta_p_t > PLATE_DONE_DELTA and z_diff > 0.03:
        is_plate_removed = True
 
    # Failure perception: If plate dropped or distance shrunk significantly
    elif is_plate_removed and (z_diff < 0.02 or delta_p_t < PLATE_DONE_DELTA - 0.03): 
        is_plate_removed = False
    obs_dict["plate_removed"] = is_plate_removed

    if is_plate_removed:
        p_t_relations.append("OUTSIDE")
    elif delta_p_t > PLATE_AROUND_DELTA:
        p_t_relations.append("AROUND")

    # ==========================================
    # 5. Holding Verification (Causal Lock)
    # ==========================================
    is_holding_door = False
    is_holding_tray = False
    is_holding_plate = False

    # Permission 1: Door handle can be manipulated at any time
    is_holding_door = check_holding(is_touching_door, g_d_dist, "door", g_d_relations, gripper_open, indices, prev_obs_dict)
        
    # Permission 2: Tray interaction allowed only if door is open
    if is_door_open:
        is_holding_tray = check_holding(is_touching_tray, g_t_dist, "tray", g_t_relations, gripper_open, indices, prev_obs_dict)
        
    # Permission 3: Plate interaction allowed only if door is open AND tray is pulled
    if is_door_open and is_tray_pulled:
        is_holding_plate = check_holding(is_touching_plate, g_p_dist, "plate", g_p_relations, gripper_open, indices, prev_obs_dict)

    # Only HOLDING states verified by causal locks are written to the graph
    if is_holding_plate: g_p_relations.append("HOLDING")
    if is_holding_door: g_d_relations.append("HOLDING")
    if is_holding_tray: g_t_relations.append("HOLDING")

    # Intent back-inference from final state
    if is_plate_removed:
        action_type = "done"
    elif is_door_open and is_tray_pulled:
        action_type = "place" if is_holding_plate else "pick"
    elif is_door_open and not is_tray_pulled:
        action_type = "pull"
    else:
        action_type = "open"
 
    # ==========================================
    # 6. Assembly: MomaGraph Edges
    # ==========================================
    edge_gp = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "plate",
        "spatial_relations": list(set(g_p_relations)),
        "geometric_attributes": {
            "delta_3d": g_p_delta.tolist(), "dist": float(g_p_dist),
            "original_delta_3d": init_g_p_delta.tolist(), "original_dist": float(init_g_p_dist)
        },
        "is_touching": "TOUCHING" in g_p_relations,
        "is_stable": is_holding_plate
    }

    edge_pt = {
        "functional_relationship": "containment",
        "object1": "plate", "object2": "dishwasher_tray",
        "spatial_relations": list(set(p_t_relations)),
        "geometric_attributes": {
            "delta_3d": p_t_delta.tolist(), "dist": float(p_t_dist),
            "original_delta_3d": init_p_t_delta.tolist(), "original_dist": float(init_p_t_dist)
        },
        "is_touching": "TOUCHING" in p_t_relations, 
        "is_stable": action_type == "done" # TODO: MODIFY THIS
    }

    edge_db = {
        "functional_relationship": "open_or_close",
        "object1": "dishwasher_door", "object2": "dishwasher_base",
        "spatial_relations": list(set(d_b_relations)),
        "geometric_attributes": {
            "delta_3d": d_b_delta.tolist(), "dist": float(d_b_dist),
            "original_delta_3d": init_d_b_delta.tolist(), "original_dist": float(init_d_b_dist)
        },
        "is_touching": False, "is_stable": is_door_open # TODO: MODIFY THIS
    }

    edge_tb = {
        "functional_relationship": "containment",
        "object1": "dishwasher_tray", "object2": "dishwasher_base",
        "spatial_relations": list(set(t_b_relations)),
        "geometric_attributes": {
            "delta_3d": t_b_delta.tolist(), "dist": float(t_b_dist),
            "original_delta_3d": init_t_b_delta.tolist(), "original_dist": float(init_t_b_dist)
        },
        "is_touching": False, "is_stable": is_tray_pulled # TODO: MODIFY THIS
    }

    edge_gd = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "dishwasher_door",
        "spatial_relations": list(set(g_d_relations)),
        "geometric_attributes": {
            "delta_3d": g_d_delta.tolist(), "dist": float(g_d_dist),
            "original_delta_3d": init_g_d_delta.tolist(), "original_dist": float(init_g_d_dist)
        },
        "is_touching": "TOUCHING" in g_d_relations,
        "is_stable": is_holding_door
    }

    edge_gt = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "dishwasher_tray",
        "spatial_relations": list(set(g_t_relations)),
        "geometric_attributes": {
            "delta_3d": g_t_delta.tolist(), "dist": float(g_t_dist),
            "original_delta_3d": init_g_t_delta.tolist(), "original_dist": float(init_g_t_dist)
        },
        "is_touching": "TOUCHING" in g_t_relations,
        "is_stable": is_holding_tray
    }

    # ==========================================
    # 7. Dynamic Attention Routing
    # ==========================================
    if action_type == "open":
        focus_edges = [("gripper", "dishwasher_door"), ("dishwasher_door", "dishwasher_base")]
    elif action_type == "pull":
        focus_edges = [("gripper", "dishwasher_tray"), ("dishwasher_tray", "dishwasher_base")]
    else: # pick, place, done
        focus_edges = [("gripper", "plate"), ("plate", "dishwasher_tray")]

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "dishwasher_door", "dishwasher_tray", "plate", "dishwasher_base"],
        "edges": [edge_gd, edge_gt, edge_gp, edge_db, edge_tb, edge_pt], 
        "action_type": action_type,
        "task_thresholds": {
            "door_open": float(DOOR_OPEN_DELTA),       
            "door_around": float(DOOR_AROUND_DELTA),
            "tray_pull": float(TRAY_PULL_DELTA),
            "tray_around": float(TRAY_AROUND_DELTA),
            "plate_done": float(PLATE_DONE_DELTA),
            "plate_around": float(PLATE_AROUND_DELTA)
        },
        "focus_edges": focus_edges 
    }


def _extract_get_ice_from_fridge(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                                 indices, extents, instructions, prev_obs_dict, obs_dict):
 
    cup_extents = extents.get("cup_size", [0.1, 0.1, 0.1])
    fridge_extents = extents.get("fridge_size", [0.5, 0.5, 0.5])
    tongue_extents = extents.get("tongue_size", [0.05, 0.1, 0.05])

    # Extract Z-axis half-height
    cup_half_z = cup_extents[2] / 2.0
    tongue_half_z = tongue_extents[2] / 2.0
    # Approximate XY radius of the cup
    cup_radius_xy = max(cup_extents[0], cup_extents[1]) / 2.0

    # Task thresholds
    PERFECT_Z_DIST = cup_half_z + tongue_half_z
    TONGUE_PUSH_DELTA = 0.015   # Minimal tongue displacement counts as 'pushed'
    
    Z_MARGIN = 0.005   # 0.5cm vertical tolerance
    XY_MARGIN = 0.005  # 0.5cm horizontal center deviation allowed
    
    # TARGET ZONE -> UNDER
    Z_TARGET_TOLERANCE = PERFECT_Z_DIST + Z_MARGIN
    XY_TARGET_RADIUS = cup_radius_xy + XY_MARGIN
    
    # AROUND ZONE -> AROUND
    AROUND_EXTENSION = 0.05
    Z_AROUND_TOLERANCE = Z_TARGET_TOLERANCE + AROUND_EXTENSION
    XY_AROUND_RADIUS = XY_TARGET_RADIUS + AROUND_EXTENSION
 
    cup_pose = task_state[indices["cup"]]
    tongue_pose = task_state[indices["tongue"]]
    joint_pose = task_state[indices["joint"]] 

    init_cup_pos = init_task_state[indices["cup"]][:3]
    init_tongue_pos = init_task_state[indices["tongue"]][:3]
    init_joint_pos = init_task_state[indices["joint"]][:3]
    
    # ==========================================
    # Spatial relationships
    # [Gripper vs Cup] 
    g_c_relations, g_c_delta, g_c_dist = compute_directional_and_distance(
        gripper_pose[:3], cup_pose[:3], extents["gripper_radius"], cup_extents)
    _, init_g_c_delta, init_g_c_dist = compute_directional_and_distance(
        init_gripper_pos, init_cup_pos, extents["gripper_radius"], cup_extents)
    
    is_touching_cup = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], cup_pose, cup_extents)
    if is_touching_cup: g_c_relations.append("TOUCHING")

    # [Cup vs Tongue]
    c_t_relations, c_t_delta, c_t_dist = compute_directional_and_distance(
        cup_pose[:3], tongue_pose[:3], cup_extents, tongue_extents)
    _, init_c_t_delta, init_c_t_dist = compute_directional_and_distance(
        init_cup_pos, init_tongue_pos, cup_extents, tongue_extents)
    
    is_touching_tongue = check_universal_contact(cup_pose[:3], np.min(cup_extents)/2.0, tongue_pose, tongue_extents)
    if is_touching_tongue: c_t_relations.append("TOUCHING")

    # ==========================================
    # Holding and physical state judgment
    # ==========================================
    is_holding_cup = check_holding(is_touching_cup, g_c_dist, "cup", g_c_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_cup: g_c_relations.append("HOLDING")

    # Detect if tongue or joint is pushed
    tongue_movement = np.linalg.norm(tongue_pose[:3] - init_tongue_pos) 
    tongue_push_ratio = min(1.0, tongue_movement / TONGUE_PUSH_DELTA)
    is_tongue_pushed = prev_obs_dict.get("tongue_pushed", False) if prev_obs_dict else False
    if tongue_movement > TONGUE_PUSH_DELTA:
        is_tongue_pushed = True
        
    obs_dict["tongue_pushed"] = is_tongue_pushed  
    # joint_movement = np.linalg.norm(joint_pose[:3] - init_joint_pos)
    # is_tongue_pushed = bool(tongue_movement > TONGUE_PUSH_DELTA or joint_movement > TONGUE_PUSH_DELTA)
    
    if is_tongue_pushed:
        c_t_relations.append("PUSHED")

    # Geometric target area check
    xy_dist = np.linalg.norm(cup_pose[:2] - tongue_pose[:2])
    z_diff = abs(cup_pose[2] - tongue_pose[2])
    # Prevents state flickering caused by wall friction
    has_been_under = prev_obs_dict.get("cup_under", False) if prev_obs_dict else False
    is_cup_at_target = bool(xy_dist <= XY_TARGET_RADIUS and z_diff <= Z_TARGET_TOLERANCE)
    if is_cup_at_target:
        has_been_under = True  
    elif xy_dist > XY_TARGET_RADIUS + 0.03 or z_diff > Z_TARGET_TOLERANCE + 0.03:
        has_been_under = False
        
    obs_dict["cup_under"] = has_been_under

    if has_been_under:
        c_t_relations.append("UNDER") 
        c_t_relations.append("AROUND")
    elif bool(xy_dist <= XY_AROUND_RADIUS and z_diff <= Z_AROUND_TOLERANCE):
        c_t_relations.append("AROUND")

    # Intent classification
    if has_been_under and is_tongue_pushed:
        action_type = "done"
    elif is_holding_cup:
        action_type = "place"
    else: 
        action_type = "pick"
    
    # Edge assembly
    edge_gc = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "cup",
        "spatial_relations": list(set(g_c_relations)),
        "geometric_attributes": {
            "delta_3d": g_c_delta.tolist(), "dist": float(g_c_dist),
            "original_delta_3d": init_g_c_delta.tolist(), "original_dist": float(init_g_c_dist)
        },
        "is_touching": "TOUCHING" in g_c_relations,
        "is_stable": is_holding_cup
    }

    edge_ct = {
        "functional_relationship": "activate",
        "object1": "cup", "object2": "tongue",
        "spatial_relations": list(set(c_t_relations)),
        "geometric_attributes": {
            "delta_3d": c_t_delta.tolist(), "dist": float(c_t_dist),
            "original_delta_3d": init_c_t_delta.tolist(), "original_dist": float(init_c_t_dist)
        },
        "is_touching": "TOUCHING" in c_t_relations,
        "is_stable": is_tongue_pushed
    }

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "cup", "tongue"],  
        "edges": [edge_gc, edge_ct],            
        "action_type": action_type,
        "task_thresholds": {
            "target_z_tolerance": float(Z_TARGET_TOLERANCE),
            "target_xy_radius": float(XY_TARGET_RADIUS),
            "around_z_tolerance": float(Z_AROUND_TOLERANCE),
            "around_xy_radius": float(XY_AROUND_RADIUS),
            "tongue_push_delta": float(TONGUE_PUSH_DELTA),
            "tongue_push_ratio": float(tongue_push_ratio)
        }
    }


def _extract_hockey(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                    indices, extents, instructions, prev_obs_dict, obs_dict):
    # ==========================================
    # 1. Parse entity dimensions & dynamic parameters
    # ==========================================
    stick_extents = extents.get("stick_size", [0.03, 0.03, 0.4])  
    ball_extents = extents.get("ball_size", [0.04, 0.04, 0.04])
    goal_extents = extents.get("goal_size", [0.3, 0.2, 0.2])      

    ball_radius = max(ball_extents) / 2.0
    
    # Dynamic thresholds
    goal_depth = min(goal_extents[0], goal_extents[1]) 
    GOAL_INSIDE_RADIUS_2D = (goal_depth / 2.0) + 0.01  # 1cm tolerance
    GOAL_AROUND_RADIUS_2D = GOAL_INSIDE_RADIUS_2D + 0.15 # 15cm approach zone

    # ==========================================
    # 2. Parse poses
    # ==========================================
    stick_pose = task_state[indices["stick"]]
    ball_pose = task_state[indices["ball"]]
    goal_pose = task_state[indices["goal"]] 

    init_stick_pos = init_task_state[indices["stick"]][:3]
    init_ball_pos = init_task_state[indices["ball"]][:3]
    init_goal_pos = init_task_state[indices["goal"]][:3]

    # 3. Cascaded spatial relationships
    # [Edge 1: Gripper vs Stick] 
    g_s_relations, g_s_delta, g_s_dist = compute_directional_and_distance(
        gripper_pose[:3], stick_pose[:3], extents["gripper_radius"], stick_extents)
    _, init_g_s_delta, init_g_s_dist = compute_directional_and_distance(
        init_gripper_pos, init_stick_pos, extents["gripper_radius"], stick_extents)
    
    is_touching_stick = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], stick_pose, stick_extents)
    if is_touching_stick: g_s_relations.append("TOUCHING")

    # [Edge 2: Stick vs Ball]
    s_b_relations, s_b_delta, s_b_dist = compute_directional_and_distance(
        stick_pose[:3], ball_pose[:3], stick_extents, ball_extents)
    _, init_s_b_delta, init_s_b_dist = compute_directional_and_distance(
        init_stick_pos, init_ball_pos, stick_extents, ball_extents)
    
    # Wider tolerance for contact due to stick length
    is_touching_ball = check_universal_contact(stick_pose[:3], np.max(stick_extents)/2.0 - 0.05, ball_pose, ball_extents)
    if is_touching_ball: s_b_relations.append("TOUCHING")

    # [Edge 3: Ball vs Goal]
    b_g_relations, b_g_delta, b_g_dist = compute_directional_and_distance(
        ball_pose[:3], goal_pose[:3], ball_extents, goal_extents)
    _, init_b_g_delta, init_b_g_dist = compute_directional_and_distance(
        init_ball_pos, init_goal_pos, ball_extents, goal_extents)
    if check_universal_contact(ball_pose[:3], ball_radius, goal_pose, goal_extents): b_g_relations.append("TOUCHING")

    # [Edge 4: Stick vs Goal] (Virtual edge for progress estimation)
    dist_ball_to_goal = np.linalg.norm(b_g_delta[:2])
    init_ball_to_goal = np.linalg.norm(init_b_g_delta[:2])
    # Persistent memory lock: detect if the ball was struck validly
    has_been_hit = prev_obs_dict.get("ball_has_been_hit", False) if prev_obs_dict else False
    if not has_been_hit and np.linalg.norm(ball_pose[:3] - init_ball_pos) > 0.03:  
        has_been_hit = True
    obs_dict["ball_has_been_hit"] = has_been_hit
    # Once hit, stick-to-ball distance remains 0 to prevent progress regression
    dist_stick_to_ball = np.linalg.norm(s_b_delta[:2]) if not has_been_hit else 0.0 
    init_stick_to_ball = np.linalg.norm(init_s_b_delta[:2])
    current_magic_dist = dist_ball_to_goal + dist_stick_to_ball
    init_magic_dist = init_ball_to_goal + init_stick_to_ball

    # Plane (XY) distance for goal detection
    ball_xy = ball_pose[:2]
    goal_xy = goal_pose[:2] 
    dist_xy = np.linalg.norm(ball_xy - goal_xy)
    is_ball_in_goal = bool(dist_xy <= GOAL_INSIDE_RADIUS_2D)
    
    has_been_around_goal = prev_obs_dict.get("ball_around_goal", False) if prev_obs_dict else False
    if dist_xy <= GOAL_AROUND_RADIUS_2D:
        has_been_around_goal = True
    elif dist_xy > GOAL_AROUND_RADIUS_2D + 0.05:  
        has_been_around_goal = False

    obs_dict["ball_around_goal"] = has_been_around_goal

    if is_ball_in_goal:
        b_g_relations.append("INSIDE")
        b_g_relations.append("AROUND")  
    elif has_been_around_goal:
        b_g_relations.append("AROUND")
    else:
        b_g_relations.append("OUTSIDE")

    # Holding judgment
    is_holding_stick = check_holding(is_touching_stick, 0.0, "stick", g_s_relations, gripper_open, indices, prev_obs_dict)
    
    if is_holding_stick: g_s_relations.append("HOLDING")
 
    if is_ball_in_goal:
        action_type = "done"
    elif is_holding_stick:
        action_type = "hit" 
    else:
        action_type = "pick"
 
    edge_gs = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "stick",
        "spatial_relations": list(set(g_s_relations)),
        "geometric_attributes": {
            "delta_3d": g_s_delta.tolist(), "dist": float(g_s_dist),
            "original_delta_3d": init_g_s_delta.tolist(), "original_dist": float(init_g_s_dist)
        },
        "is_touching": "TOUCHING" in g_s_relations,
        "is_stable": is_holding_stick
    }

    edge_sb = {
        "functional_relationship": "activate", # strike/hit
        "object1": "stick", "object2": "ball",
        "spatial_relations": list(set(s_b_relations)),
        "geometric_attributes": {
            "delta_3d": s_b_delta.tolist(), "dist": float(s_b_dist),
            "original_delta_3d": init_s_b_delta.tolist(), "original_dist": float(init_s_b_dist)
        },
        "is_touching": "TOUCHING" in s_b_relations,
        "is_stable": action_type in ["done"]
    }

    edge_bg = {
        "functional_relationship": "pair_with",
        "object1": "ball", "object2": "goal",
        "spatial_relations": list(set(b_g_relations)),
        "geometric_attributes": {
            "delta_3d": b_g_delta.tolist(), "dist": float(b_g_dist),
            "original_delta_3d": init_b_g_delta.tolist(), "original_dist": float(init_b_g_dist)
        },
        "is_touching": "TOUCHING" in b_g_relations, 
        "is_stable": "INSIDE" in b_g_relations
    }

    edge_sg = {
        "functional_relationship": "approach",
        "object1": "stick", "object2": "goal",
        "spatial_relations": list(set(b_g_relations)), 
        "geometric_attributes": { 
            "delta_3d": [float(current_magic_dist)] * 3,       
            "dist": float(current_magic_dist),               
            "original_delta_3d": [float(init_magic_dist)] * 3,
            "original_dist": float(init_magic_dist) 
        },
        "is_touching": False,
        "is_stable": "INSIDE" in b_g_relations
    }
 
    if np.linalg.norm(ball_pose[:3] - init_ball_pos) <= 0.02:
        focus_edges = [("gripper", "stick"), ("stick", "ball")]
    else: # hit, done
        focus_edges = [("stick", "ball"), ("ball", "goal")]

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "stick", "ball", "goal"], 
        "edges": [edge_gs, edge_sb, edge_bg, edge_sg],           
        "action_type": action_type,
        "task_thresholds": {
            "goal_inside_radius_2d": float(GOAL_INSIDE_RADIUS_2D),
            "goal_around_radius_2d": float(GOAL_AROUND_RADIUS_2D),
        },
        "focus_edges": focus_edges 
    }


def _extract_insert_onto_square_peg(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                                    indices, extents, instructions, prev_obs_dict, obs_dict):
    
    # 1. Dynamically identify target pillar for the current episode
    init_sc_pos = init_task_state[indices["success_centre"]][:3]
    
    pillar_keys = ["pillar0", "pillar1", "pillar2"]
    target_pillar_key = None
    min_dist = float('inf')
    
    for key in pillar_keys:
        p_pos = init_task_state[indices[key]][:3]
        # Distance on XY plane
        dist = np.linalg.norm(p_pos[:2] - init_sc_pos[:2])
        if dist < min_dist:
            min_dist = dist
            target_pillar_key = key

    # 2. Parse entity dimensions
    ring_extents = extents.get("ring_size", [0.06, 0.06, 0.02])
    pillar_extents = extents.get(f"{target_pillar_key}_size", [0.04, 0.04, 0.15])
    
    # Square ring on square peg: requires XY overlap
    TARGET_XY_RADIUS = max(pillar_extents[0], pillar_extents[1]) / 2.0 + 0.01
    AROUND_XY_RADIUS = TARGET_XY_RADIUS + 0.001 # Approaching hover zone

    # 3. Parse poses
    ring_pose = task_state[indices["ring"]]
    pillar_pose = task_state[indices[target_pillar_key]]
    
    init_ring_pos = init_task_state[indices["ring"]][:3]
    init_pillar_pos = init_task_state[indices[target_pillar_key]][:3]

    # 4. Spatial relationships (Edge construction)
    # [Edge 1: Gripper vs Ring]  
    g_r_relations, g_r_delta, g_r_dist = compute_directional_and_distance(
        gripper_pose[:3], ring_pose[:3], extents["gripper_radius"], ring_extents)
    _, init_g_r_delta, init_g_r_dist = compute_directional_and_distance(
        init_gripper_pos, init_ring_pos, extents["gripper_radius"], ring_extents)
        
    is_touching_ring = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], ring_pose, ring_extents)
    if is_touching_ring: g_r_relations.append("TOUCHING")
    
    # [Edge 2: Ring vs Target Pillar]
    r_p_relations, r_p_delta, r_p_dist = compute_directional_and_distance(
        ring_pose[:3], pillar_pose[:3], ring_extents, pillar_extents)
    _, init_r_p_delta, init_r_p_dist = compute_directional_and_distance(
        init_ring_pos, init_pillar_pos, ring_extents, pillar_extents)
    is_touching_pillar = check_universal_contact(ring_pose[:3], np.min(ring_extents)/2.0, pillar_pose, pillar_extents)
    if is_touching_pillar: r_p_relations.append("TOUCHING")

    # Holding verification
    is_holding_ring = check_holding(is_touching_ring, g_r_dist, "ring", g_r_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_ring: g_r_relations.append("HOLDING")

    # 5. State machine: XY alignment + Z-axis insertion
    dist_xy = np.linalg.norm(r_p_delta[:2])
    is_aligned_xy = bool(dist_xy <= TARGET_XY_RADIUS)
    # Insertion detected if aligned and ring height matches the pillar base zone
    is_inserted = is_aligned_xy and bool(ring_pose[2] < pillar_pose[2] + 0.04) 
 
    ring_radius = max(ring_extents[0], ring_extents[1]) / 2.0

    if is_inserted:
        r_p_relations.append("INSIDE")
    elif check_approach_corridor_world(
        pos_obj=ring_pose[:3], 
        pos_anchor=pillar_pose[:3], 
        approach_axis=2, 
        approach_dir=1.0,
        mo_size=ring_radius, 
        tgt_size=pillar_extents,
        min_radius=AROUND_XY_RADIUS,  
        min_height=0.02  
    ):
        r_p_relations.append("AROUND")
 
    if is_inserted:
        action_type = "done"
    elif is_holding_ring:
        action_type = "place"
    else:
        action_type = "pick"
 
    edge_gr = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "square_ring",
        "spatial_relations": list(set(g_r_relations)),
        "geometric_attributes": {
            "delta_3d": g_r_delta.tolist(), "dist": float(g_r_dist),
            "original_delta_3d": init_g_r_delta.tolist(), "original_dist": float(init_g_r_dist)
        },
        "is_touching": "TOUCHING" in g_r_relations,
        "is_stable": is_holding_ring
    }
    
    edge_rp = {
        "functional_relationship": "pair_with",
        "object1": "square_ring", "object2": "target_pillar",
        "spatial_relations": list(set(r_p_relations)),
        "geometric_attributes": { 
            "delta_3d": r_p_delta.tolist(), "dist": float(r_p_dist),
            "original_delta_3d": init_r_p_delta.tolist(), "original_dist": float(init_r_p_dist)
        },
        "is_touching": "TOUCHING" in r_p_relations,  
        "is_stable": is_inserted
    }

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "square_ring", "target_pillar"], 
        "edges": [edge_gr, edge_rp],             
        "action_type": action_type,
        "task_thresholds": {
            "target_pillar_id": target_pillar_key,  
            "target_xy_radius": float(TARGET_XY_RADIUS),
            "around_xy_radius": float(AROUND_XY_RADIUS)
        }
    }


def _extract_meat_on_grill(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                           indices, extents, instructions, prev_obs_dict, obs_dict):
    # 1. Dynamic target positioning via NLP
    instruction_str = instructions[0].lower()
    target_meat_key = "chicken" if "chicken" in instruction_str else "steak"

    # 2. Dimensions
    meat_extents = extents.get(f"{target_meat_key}_size", [0.05, 0.05, 0.02])
    grill_extents = extents.get("grill_size", [0.25, 0.25, 0.05])
     
    TARGET_XY_RADIUS = 0.03 
    AROUND_XY_RADIUS = TARGET_XY_RADIUS + 0.03  

    # 3. Poses
    meat_pose = task_state[indices[target_meat_key]]
    grill_pose = task_state[indices["grill"]]
    success_pose = task_state[indices["success"]]
    
    init_meat_pos = init_task_state[indices[target_meat_key]][:3]
    init_grill_pos = init_task_state[indices["grill"]][:3]
    init_success_pos = init_task_state[indices["success"]][:3]

    # 4. Spatial relationships
    # [Edge 1: Gripper vs Meat] 
    g_m_relations, g_m_delta, g_m_dist = compute_directional_and_distance(
        gripper_pose[:3], meat_pose[:3], extents["gripper_radius"], meat_extents)
    _, init_g_m_delta, init_g_m_dist = compute_directional_and_distance(
        init_gripper_pos, init_meat_pos, extents["gripper_radius"], meat_extents)
        
    is_touching_meat = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], meat_pose, meat_extents)
    if is_touching_meat: g_m_relations.append("TOUCHING")
     
    is_holding_meat = check_holding(is_touching_meat, g_m_dist, target_meat_key, g_m_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_meat: g_m_relations.append("HOLDING")

    # [Edge 2: Meat vs Grill]
    m_g_relations, m_g_delta, m_g_dist = compute_directional_and_distance(
        meat_pose[:3], success_pose[:3], meat_extents, [0.01, 0.01, 0.01])  
    _, init_m_g_delta, init_m_g_dist = compute_directional_and_distance(
        init_meat_pos, init_success_pos, meat_extents, [0.01, 0.01, 0.01])
        
    # 5. Intent state machine (XY hover + Z-axis placement)
    dist_xy = np.linalg.norm(m_g_delta[:2])
    dist_z = abs(m_g_delta[2])
    
    # AROUND state hysteresis
    has_been_around = prev_obs_dict.get("meat_around", False) if prev_obs_dict else False
    if dist_xy <= AROUND_XY_RADIUS:
        has_been_around = True
    elif dist_xy > AROUND_XY_RADIUS + 0.04:  
        has_been_around = False

    # ON (Placement) state hysteresis
    has_been_on = prev_obs_dict.get("meat_on", False) if prev_obs_dict else False
    
    # Strict alignment and proximity determines ON
    if dist_xy <= TARGET_XY_RADIUS and dist_z <= 0.05:
        has_been_on = True
    # Failure perception: If meat slides off or bounces high
    elif dist_xy > TARGET_XY_RADIUS + 0.04 or dist_z > 0.08:
        has_been_on = False
 
    if has_been_around: 
        m_g_relations.append("AROUND")
    
    is_on_grill = has_been_on
    if is_on_grill:
        m_g_relations.append("ON")
        m_g_relations.append("TOUCHING")
  
    if is_on_grill and not is_holding_meat:  
        action_type = "done"
    elif is_holding_meat:
        action_type = "place"
    else:
        action_type = "pick"

    # memory stored
    obs_dict["meat_around"] = has_been_around
    obs_dict["meat_on"] = has_been_on
     
    edge_gm = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "target_meat",
        "spatial_relations": list(set(g_m_relations)),
        "geometric_attributes": {
            "delta_3d": g_m_delta.tolist(), "dist": float(g_m_dist),
            "original_delta_3d": init_g_m_delta.tolist(), "original_dist": float(init_g_m_dist)
        },
        "is_touching": bool("TOUCHING" in g_m_relations),
        "is_stable": bool(is_holding_meat)
    }
    
    edge_mg = {
        "functional_relationship": "pair_with",
        "object1": "target_meat", "object2": "grill",
        "spatial_relations": list(set(m_g_relations)),
        "geometric_attributes": {
            "delta_3d": m_g_delta.tolist(), "dist": float(m_g_dist),
            "original_delta_3d": init_m_g_delta.tolist(), "original_dist": float(init_m_g_dist)
        },
        "is_touching": bool("TOUCHING" in m_g_relations),
        "is_stable": bool(is_on_grill)
    }

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "target_meat", "grill"], 
        "edges": [edge_gm, edge_mg],             
        "action_type": action_type,
        "task_thresholds": {
            "target_meat_id": target_meat_key, 
            "target_xy_radius": float(TARGET_XY_RADIUS),
            "around_xy_radius": float(AROUND_XY_RADIUS)
        }
    }


def _extract_move_hanger(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                         indices, extents, instructions, prev_obs_dict, obs_dict):
    # 1. Dimensions
    hanger_extents = extents.get("hanger_size", [0.4, 0.02, 0.2]) 
    hanger_pose = task_state[indices["hanger"]]
    init_hanger_pos = init_task_state[indices["hanger"]][:3]

    # 2. Dynamically identify target rack
    init_rack1_pos = init_task_state[indices["rack1"]][:3]
    init_rack2_pos = init_task_state[indices["rack2"]][:3]
    
    dist_to_rack1 = np.linalg.norm(init_hanger_pos[:2] - init_rack1_pos[:2])
    dist_to_rack2 = np.linalg.norm(init_hanger_pos[:2] - init_rack2_pos[:2])
    
    target_rack_key = "rack1" if dist_to_rack1 > dist_to_rack2 else "rack2"
    target_rack_extents = extents.get(f"{target_rack_key}_size", [0.45, 1.32, 1.45])
    target_rack_pose = task_state[indices[target_rack_key]]
    init_target_rack_pos = init_task_state[indices[target_rack_key]][:3]
  
    TARGET_XY_RADIUS = 0.15
    AROUND_XY_RADIUS = 0.25 

    # 3. Gripper vs Hanger
    g_h_relations, g_h_delta, g_h_dist = compute_directional_and_distance(
        gripper_pose[:3], hanger_pose[:3], extents["gripper_radius"], hanger_extents)
    _, init_g_h_delta, init_g_h_dist = compute_directional_and_distance(
        init_gripper_pos, init_hanger_pos, extents["gripper_radius"], hanger_extents)
        
    is_touching_hanger = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], hanger_pose, hanger_extents)
    if is_touching_hanger: g_h_relations.append("TOUCHING")
    
    is_holding_hanger = check_holding(is_touching_hanger, g_h_dist, "hanger", g_h_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_hanger: g_h_relations.append("HOLDING")

    # 4. Hanger vs Target Rack
    h_r_relations, h_r_delta, h_r_dist = compute_directional_and_distance(
        hanger_pose[:3], target_rack_pose[:3], hanger_extents, target_rack_extents) 
    _, init_h_r_delta, init_h_r_dist = compute_directional_and_distance(
        init_hanger_pos, init_target_rack_pos, hanger_extents, target_rack_extents)

    # 5. State machine
    dx, dy = abs(h_r_delta[0]), abs(h_r_delta[1])
    
    # Elevation check
    z_lift = hanger_pose[2] - init_hanger_pos[2]
    is_dropped_back = bool(z_lift <= 0.02)
    is_lifted = not is_dropped_back 

    is_aligned_xy = bool(dx <= TARGET_XY_RADIUS) and bool(dy <= TARGET_XY_RADIUS)
    is_around_xy = bool(dx <= AROUND_XY_RADIUS) and bool(dy <= AROUND_XY_RADIUS) 
    
    # AROUND
    if is_around_xy and is_lifted:
        h_r_relations.append("AROUND")
 
    is_on_rack = is_aligned_xy and is_dropped_back
    
    if is_on_rack:
        h_r_relations.append("ON")
        h_r_relations.append("TOUCHING")
        if "AROUND" not in h_r_relations:
            h_r_relations.append("AROUND") 
 
    if is_on_rack and not is_holding_hanger: 
        action_type = "done"
    elif is_holding_hanger:
        action_type = "place"
    else:
        action_type = "pick"
 
    edge_gh = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "hanger",
        "spatial_relations": list(set(g_h_relations)), 
        "geometric_attributes": {
            "delta_3d": g_h_delta.tolist(), "dist": float(g_h_dist),
            "original_delta_3d": init_g_h_delta.tolist(), "original_dist": float(init_g_h_dist)
        },
        "is_touching": bool("TOUCHING" in g_h_relations),
        "is_stable": bool(is_holding_hanger)
    }
    
    edge_hr = {
        "functional_relationship": "pair_with",
        "object1": "hanger", "object2": "target_rack", 
        "spatial_relations": list(set(h_r_relations)),
        "geometric_attributes": {
            "delta_3d": h_r_delta.tolist(), "dist": float(h_r_dist),
            "original_delta_3d": init_h_r_delta.tolist(), "original_dist": float(init_h_r_dist)
        },
        "is_touching": bool("TOUCHING" in h_r_relations),
        "is_stable": bool(is_on_rack)
    }

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "hanger", "target_rack"], 
        "edges": [edge_gh, edge_hr],             
        "action_type": action_type,
        "task_thresholds": {
            "target_rack_id": target_rack_key,
            "target_radius": float(TARGET_XY_RADIUS),
            "around_radius": float(AROUND_XY_RADIUS)
        }
    }


def _extract_open_drawer(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                         indices, extents, instructions, prev_obs_dict, obs_dict):
    # 1. NLP target parsing
    instruction_str = instructions[0].lower()
    if "bottom" in instruction_str:
        target_drawer_key = "bottom"
    elif "middle" in instruction_str:
        target_drawer_key = "middle"
    else:
        target_drawer_key = "top"

    # 2. Dimensions
    drawer_extents = extents.get(f"{target_drawer_key}_size", [0.4, 0.2, 0.4])
    drawer_pose = task_state[indices[target_drawer_key]]
    init_drawer_pos = init_task_state[indices[target_drawer_key]][:3]
 
    PULL_SUCCESS_DIST = 0.22
    AROUND_PULL_DIST = 0.15  

    # Spatial relationships
    g_d_relations, g_d_delta, g_d_dist = compute_directional_and_distance(
        gripper_pose[:3], drawer_pose[:3], extents["gripper_radius"], drawer_extents)
    _, init_g_d_delta, init_g_d_dist = compute_directional_and_distance(
        init_gripper_pos, init_drawer_pos, extents["gripper_radius"], drawer_extents)
        
    is_touching = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], drawer_pose, drawer_extents)
    if is_touching: g_d_relations.append("TOUCHING")
     
    is_holding = check_holding(is_touching, g_d_dist, target_drawer_key, g_d_relations, gripper_open, indices, prev_obs_dict)
    if is_holding: g_d_relations.append("HOLDING")

    # Edge 2: drawer -> opened_drawer (Progress indicator)
    pulled_dist = np.linalg.norm(drawer_pose[:3] - init_drawer_pos)

    d_o_relations = [] 
    # is_around = bool(pulled_dist >= AROUND_PULL_DIST)
    is_opened = bool(pulled_dist >= PULL_SUCCESS_DIST)
    is_closed = bool(pulled_dist == 0)

    has_been_around = prev_obs_dict.get("drawer_around", False) if prev_obs_dict else False
    if pulled_dist >= AROUND_PULL_DIST:
        has_been_around = True
    elif pulled_dist < AROUND_PULL_DIST - 0.05:  
        has_been_around = False

    if has_been_around:
        d_o_relations.append("AROUND")
    if is_opened:
        d_o_relations.append("OPENED")
    remaining_dist = float(max(0.0, PULL_SUCCESS_DIST - pulled_dist))
    obs_dict["drawer_around"] = has_been_around
      
    if is_opened:
        action_type = "done"
    elif is_holding or pulled_dist > 0.05:
        action_type = "open"  
    else:
        action_type = "pick"
  
    edge_gd = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "drawer",
        "spatial_relations": list(set(g_d_relations)),
        "geometric_attributes": {
            "delta_3d": g_d_delta.tolist(), "dist": float(g_d_dist),
            "original_delta_3d": init_g_d_delta.tolist(), "original_dist": float(init_g_d_dist)
        },
        "is_touching": bool("TOUCHING" in g_d_relations),
        "is_stable": bool(is_holding)
    }
     
    edge_do = {
        "functional_relationship": "open_or_close",
        "object1": "drawer", 
        "object2": "opened_drawer",
        "spatial_relations": list(set(d_o_relations)),
        "geometric_attributes": { 
            "delta_3d": [remaining_dist] * 3, 
            "dist": float(remaining_dist),         
            "original_delta_3d": [PULL_SUCCESS_DIST] * 3, 
            "original_dist": float(PULL_SUCCESS_DIST)
        },
        "is_touching": True,  
        "is_stable": bool(is_opened) or bool(is_closed)
    }

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "drawer", "opened_drawer"], 
        "edges": [edge_gd, edge_do],  
        "action_type": action_type,
        "task_thresholds": {
            "target_drawer_id": target_drawer_key,
            "pull_success_dist": float(PULL_SUCCESS_DIST),
            "pull_around_dist": float(AROUND_PULL_DIST)
        }
    }


def _extract_place_wine_at_rack_location(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                                         indices, extents, instructions, prev_obs_dict, obs_dict):
    # 1. Target selection
    instruction_str = instructions[0].lower()
    if "left" in instruction_str:
        target_key = "target_left"
    elif "right" in instruction_str:
        target_key = "target_right"
    else:
        target_key = "target_middle"

    # 2. Poses
    wine_pose = task_state[indices["wine"]] 
    target_pos = task_state[indices[target_key]][:3] 
    
    init_wine_pos = init_task_state[indices["wine"]][:3]
    init_target_pos = init_task_state[indices[target_key]][:3]
 
    wine_extents = extents.get("wine_size", [0.08, 0.08, 0.25])
    rack_extents = extents.get("rack_size", [0.2, 0.6, 0.1]) 

    # ==========================================
    # 3. Edge 1: gripper -> wine
    # ==========================================
    g_w_relations, g_w_delta, g_w_dist = compute_directional_and_distance(
        gripper_pose[:3], wine_pose[:3], extents["gripper_radius"], wine_extents)
    _, init_g_w_delta, init_g_w_dist = compute_directional_and_distance(
        init_gripper_pos, init_wine_pos, extents["gripper_radius"], wine_extents)
        
    # is_touching_wine = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], wine_pose, wine_extents) 
    dist_xy = np.linalg.norm(g_w_delta[:2])
    dist_z = abs(g_w_delta[2]) 
    is_touching_wine = bool(dist_xy <= 0.08 and dist_z <= 0.22)
      
    if is_touching_wine: g_w_relations.append("TOUCHING")
    
    # is_holding_wine = check_holding(is_touching_wine, g_w_dist, "wine", g_w_relations, gripper_open, indices, prev_obs_dict)
    is_holding_wine = check_holding(is_touching_wine, 0.0, "wine", g_w_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_wine: g_w_relations.append("HOLDING")

    # ==========================================
    # 4. Edge 2: wine -> target  
    # ========================================== 
    curr_delta = wine_pose[:3] - target_pos
    dist_xy = np.linalg.norm(curr_delta[:2])
    dist_z = abs(curr_delta[2])
     
    init_delta = init_wine_pos - init_target_pos
    init_dist_xy = np.linalg.norm(init_delta[:2])
 
    TARGET_XY_RADIUS = 0.15  
    AROUND_XY_RADIUS = 0.25  
     
    is_lifted = bool(wine_pose[2] > init_wine_pos[2] + 0.05)
    
    w_t_relations = []
     
    if dist_xy <= AROUND_XY_RADIUS and is_lifted:
        w_t_relations.append("AROUND")
 
    # Placement criteria: horizontal alignment and proximity to sensor plane
    is_on_target = bool(dist_xy <= TARGET_XY_RADIUS and dist_z <= 0.10)

    if is_on_target:
        w_t_relations.append("ON")
        w_t_relations.append("TOUCHING")
 
    if is_on_target and not is_holding_wine: 
        action_type = "done"
    elif is_holding_wine:
        action_type = "place"
    else:
        action_type = "pick"
 
    edge_gw = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "wine_bottle",
        "spatial_relations": list(set(g_w_relations)),
        "geometric_attributes": {
            "delta_3d": g_w_delta.tolist(), "dist": float(g_w_dist),
            "original_delta_3d": init_g_w_delta.tolist(), "original_dist": float(init_g_w_dist)
        },
        "is_touching": bool("TOUCHING" in g_w_relations),
        "is_stable": bool(is_holding_wine)
    }
     
    # ProgressEstimator uses delta_3d norm. Forced monotonic decrease value for 'move' phase.
    magic_dist_xy = float(max(0.0, dist_xy))

    edge_wt = {
        "functional_relationship": "pair_with",
        "object1": "wine_bottle", "object2": "rack_section",
        "spatial_relations": list(set(w_t_relations)),
        "geometric_attributes": {
            "delta_3d": [magic_dist_xy] * 3, #curr_delta.tolist(), 
            "dist": float(np.linalg.norm(curr_delta)),
            "original_delta_3d": [float(init_dist_xy)] * 3, #init_delta.tolist(), 
            "original_dist": float(np.linalg.norm(init_delta))
        },
        "is_touching": bool("TOUCHING" in w_t_relations),
        "is_stable": bool(is_on_target)
    }
 
    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "wine_bottle", "rack_section"], 
        "edges": [edge_gw, edge_wt],             
        "action_type": action_type,
        "task_thresholds": {
            "target_id": target_key,  
            "target_radius": float(TARGET_XY_RADIUS),
            "around_radius": float(AROUND_XY_RADIUS)
        }
    }


def _extract_put_money_in_safe(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                               indices, extents, instructions, prev_obs_dict, obs_dict):
    # 1. Target level parsing
    instruction_str = instructions[0].lower()
    if "bottom" in instruction_str:
        target_key = "target_bottom"
    elif "middle" in instruction_str:
        target_key = "target_middle"
    else:
        target_key = "target_top"

    # 2. Poses & Dimensions
    money_pose = task_state[indices["money"]]
    target_pos = task_state[indices[target_key]][:3] 
    safe_pose = task_state[indices["safe"]]
    init_money_pos = init_task_state[indices["money"]][:3]
    init_target_pos = init_task_state[indices[target_key]][:3]

    money_extents = extents.get("money_size", [0.15, 0.08, 0.02])
    safe_extents = extents.get("safe_size", [0.4, 0.4, 0.5]) 
    money_radius = max(money_extents[0], money_extents[1]) / 2.0

    # ==========================================
    # 3. Edge 1: gripper -> money
    # ==========================================
    g_m_relations, g_m_delta, g_m_dist = compute_directional_and_distance(
        gripper_pose[:3], money_pose[:3], extents["gripper_radius"], money_extents)
    _, init_g_m_delta, init_g_m_dist = compute_directional_and_distance(
        init_gripper_pos, init_money_pos, extents["gripper_radius"], money_extents)
        
    is_touching_money = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], money_pose, money_extents)
    if is_touching_money: g_m_relations.append("TOUCHING")
 
    is_holding_raw = check_holding(is_touching_money, 0.0, "money", g_m_relations, gripper_open, indices, prev_obs_dict) 
    is_lifted = bool(money_pose[2] > init_money_pos[2] + 0.02)  
    
    # Final holding criteria: basic lock + actual elevation
    is_holding_money = is_holding_raw and is_lifted 
    # is_holding_money = check_holding(is_touching_money, g_m_dist, "money", g_m_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_money: g_m_relations.append("HOLDING")

    # ==========================================
    # 4. Edge 2: money -> target  
    # ==========================================
    m_t_relations = [] 
    curr_delta = money_pose[:3] - target_pos
    dist_to_target = np.linalg.norm(curr_delta)

    # ---------- FOR EXPERT ----------
    if prev_obs_dict and "min_dist_ever" in prev_obs_dict:
        dist_to_target = min(dist_to_target, prev_obs_dict["min_dist_ever"])
    else:
        dist_to_target = dist_to_target 
    obs_dict["min_dist_ever"] = dist_to_target
    
    SUCCESS_RADIUS = 0.10
    AROUND_RADIUS = 0.20
    
    is_at_target = bool(dist_to_target <= SUCCESS_RADIUS)
     
    has_been_around = prev_obs_dict.get("money_around", False) if prev_obs_dict else False
    if dist_to_target <= AROUND_RADIUS:
        has_been_around = True
    elif dist_to_target > AROUND_RADIUS + 0.05:  
        has_been_around = False

    has_been_inside = prev_obs_dict.get("money_inside", False) if prev_obs_dict else False
    
    is_inside = False
    if is_at_target and not is_holding_money:
        is_inside = True
        has_been_inside = True
    
    if has_been_inside and dist_to_target > 0.10:
        has_been_inside = False
        is_inside = False
        
    if has_been_inside:
        is_inside = True
        
    if is_inside:
        m_t_relations.append("INSIDE")
        m_t_relations.append("ON")
        m_t_relations.append("AROUND")
    elif has_been_around:  
        m_t_relations.append("AROUND")
    else:
        m_t_relations.append("OUTSIDE")

    obs_dict["money_around"] = has_been_around
    obs_dict["money_inside"] = has_been_inside
  
    if is_inside: 
        action_type = "done"
    elif is_holding_money:
        action_type = "place"
    else:
        action_type = "pick"
 
    edge_gm = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "money",
        "spatial_relations": list(set(g_m_relations)),
        "geometric_attributes": {
            "delta_3d": g_m_delta.tolist(), "dist": float(g_m_dist),
            "original_delta_3d": init_g_m_delta.tolist(), "original_dist": float(init_g_m_dist)
        },
        "is_touching": bool("TOUCHING" in g_m_relations),
        "is_stable": bool(is_holding_money)
    }
  
    init_delta = init_money_pos - init_target_pos
    init_dist_3d = float(np.linalg.norm(init_delta))
    
    edge_mt = {
        "functional_relationship": "pair_with",
        "object1": "money", "object2": "safe_shelf",
        "spatial_relations": list(set(m_t_relations)),
        "geometric_attributes": { 
            "delta_3d": [float(dist_to_target)] * 3, 
            "dist": float(dist_to_target),
            "original_delta_3d": [init_dist_3d] * 3, 
            "original_dist": init_dist_3d
        },
        "is_touching": bool(is_inside),
        "is_stable": bool(is_inside)
    }
    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "money", "safe_shelf"], 
        "edges": [edge_gm, edge_mt],             
        "action_type": action_type,
        "task_thresholds": {
            "target_id": target_key
        }
    }


def _extract_reach_and_drag(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                            indices, extents, instructions, prev_obs_dict, obs_dict):
    # 1. Poses & Dimensions
    stick_pose = task_state[indices["stick"]]
    cube_pose = task_state[indices["cube"]]
    success_pose = task_state[indices["success"]]
    
    init_stick_pos = init_task_state[indices["stick"]][:3]
    init_cube_pos = init_task_state[indices["cube"]][:3]
    init_success_pos = init_task_state[indices["success"]][:3]

    stick_extents = extents.get("stick_size", [0.02, 0.02, 0.4])
    cube_extents = extents.get("cube_size", [0.05, 0.05, 0.05])
    target_extents = extents.get("target_size", np.array([0.13, 0.13, 0.01]))

    # Dynamic thresholds
    goal_depth = min(target_extents[0], target_extents[1]) 
    GOAL_INSIDE_RADIUS_2D = (goal_depth / 2.0) + 0.03   
    GOAL_AROUND_RADIUS_2D = GOAL_INSIDE_RADIUS_2D + 0.08  

    # 2. Spatial relationships
    # Edge 1: gripper -> stick
    g_s_relations, g_s_delta, g_s_dist = compute_directional_and_distance(
        gripper_pose[:3], stick_pose[:3], extents["gripper_radius"], stick_extents)
    _, init_g_s_delta, init_g_s_dist = compute_directional_and_distance(
        init_gripper_pos, init_stick_pos, extents["gripper_radius"], stick_extents)
        
    is_touching_stick = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], stick_pose, stick_extents)
    if is_touching_stick: g_s_relations.append("TOUCHING")
    
    is_holding_stick = check_holding(is_touching_stick, g_s_dist, "stick", g_s_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_stick: g_s_relations.append("HOLDING")

    #  Edge 2: stick -> cube
    s_c_relations, s_c_delta, s_c_dist = compute_directional_and_distance(
        stick_pose[:3], cube_pose[:3], stick_extents, cube_extents)
    _, init_s_c_delta, init_s_c_dist = compute_directional_and_distance(
        init_stick_pos, init_cube_pos, stick_extents, cube_extents)
     
    is_touching_cube = check_universal_contact(stick_pose[:3], np.max(stick_extents)/2.0 - 0.05, cube_pose, cube_extents)
    if is_touching_cube: s_c_relations.append("TOUCHING")

    #  Edge 3: cube -> target_zone   
    c_t_relations, c_t_delta, c_t_dist = compute_directional_and_distance(
        cube_pose[:3], success_pose[:3], cube_extents, [0.01, 0.01, 0.01])
    _, init_c_t_delta, init_c_t_dist = compute_directional_and_distance(
        init_cube_pos, init_success_pos, cube_extents, [0.01, 0.01, 0.01])
    if check_universal_contact(cube_pose[:3], max(cube_extents[0], cube_extents[1]) / 2.0, success_pose, [0.01, 0.01, 0.01]): 
        c_t_relations.append("TOUCHING")

    # Edge 4: Cube vs Target (Virtual edge for progress tracking) 
    dist_cube_to_goal = np.linalg.norm(c_t_delta[:2])
    init_cube_to_goal = np.linalg.norm(init_c_t_delta[:2])
    dist_stick_to_cube = np.linalg.norm(s_c_delta[:2]) if not bool(np.linalg.norm(cube_pose[:3] - init_cube_pos) > 0.02) else 0.0
    init_stick_to_cube = np.linalg.norm(init_s_c_delta[:2])
    current_magic_dist = dist_cube_to_goal + dist_stick_to_cube
    init_magic_dist = init_cube_to_goal + init_stick_to_cube

    dist_xy = np.linalg.norm(cube_pose[:2] - success_pose[:2])
    is_inside = bool(dist_xy <= GOAL_INSIDE_RADIUS_2D)
    if is_inside: 
        c_t_relations.append("INSIDE")
        c_t_relations.append("ON")
    elif dist_xy <= GOAL_AROUND_RADIUS_2D:
        c_t_relations.append("AROUND") 
    else:
        c_t_relations.append("OUTSIDE")

    if is_inside:
        action_type = "done"
    elif is_holding_stick:
        action_type = "drag" 
    else:
        action_type = "pick"

    edge_gs = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "stick",
        "spatial_relations": list(set(g_s_relations)),
        "geometric_attributes": {
            "delta_3d": g_s_delta.tolist(), "dist": float(g_s_dist),
            "original_delta_3d": init_g_s_delta.tolist(), "original_dist": float(init_g_s_dist)
        },
        "is_touching": bool(is_touching_stick),
        "is_stable": bool(is_holding_stick)
    }

    edge_sc = {
        "functional_relationship": "activate", # strike/hit
        "object1": "stick", "object2": "cube",
        "spatial_relations": list(set(s_c_relations)),
        "geometric_attributes": {
            "delta_3d": s_c_delta.tolist(), "dist": float(s_c_dist),
            "original_delta_3d": init_s_c_delta.tolist(), "original_dist": float(init_s_c_dist)
        },
        "is_touching": "TOUCHING" in s_c_relations,
        "is_stable": action_type in ["done"]
    }

    edge_ct = {
        "functional_relationship": "pair_with",
        "object1": "cube", "object2": "target_zone",
        "spatial_relations": list(set(c_t_relations)),
        "geometric_attributes": {
            "delta_3d": c_t_delta.tolist(), "dist": float(c_t_dist),
            "original_delta_3d": init_c_t_delta.tolist(), "original_dist": float(init_c_t_dist)
        },
        "is_touching": bool(is_inside),
        "is_stable": bool(is_inside)
    }

    edge_sg = {
        "functional_relationship": "approach",
        "object1": "stick", "object2": "target_zone",
        "spatial_relations": list(set(c_t_relations)), 
        "geometric_attributes": {
            "delta_3d": [float(current_magic_dist), 0.0, 0.0],       
            "dist": float(current_magic_dist),               
            "original_delta_3d": [float(init_magic_dist), 0.0, 0.0],
            "original_dist": float(init_magic_dist) 
        },
        "is_touching": False,
        "is_stable": "INSIDE" in c_t_relations
    }

    if np.linalg.norm(cube_pose[:3] - init_cube_pos) <= 0.02:
        focus_edges = [("gripper", "stick"), ("stick", "cube")]
    else: # hit, done
        focus_edges = [("stick", "cube"), ("cube", "target_zone")]

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "stick", "cube", "target_zone"], 
        "edges": [edge_gs, edge_sc, edge_ct, edge_sg],           
        "action_type": action_type,
        "task_thresholds": {
            "goal_inside_radius_2d": float(GOAL_INSIDE_RADIUS_2D),
            "goal_around_radius_2d": float(GOAL_AROUND_RADIUS_2D),
        },
        "focus_edges": focus_edges 
    }


def _extract_scoop_with_spatula(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                                indices, extents, instructions, prev_obs_dict, obs_dict):

    SUCCESS_XY_DIST = 0.073
    AROUND_XY_DIST = 0.075
    LIFT_HEIGHT_THRESHOLD = 0.025 
 
    spatula_pose = task_state[indices["spatula"]]
    cube_spawn_pos = init_task_state[indices["cuboid"]][:3]
    success_pose = task_state[indices["success"]][:3]
    
    init_spatula_pos = init_task_state[indices["spatula"]][:3]

    spatula_extents = extents.get("spatula_size", np.array([0.03, 0.12, 0.28])) 
    cube_extents = extents.get("cuboid_size", np.array([0.02, 0.02, 0.02]))
    
    # ==========================================
    # Edge 1: gripper -> spatula  
    # ==========================================
    g_s_relations, g_s_delta, g_s_dist = compute_directional_and_distance(
        gripper_pose[:3], spatula_pose[:3], extents["gripper_radius"], spatula_extents)
    _, init_g_s_delta, init_g_s_dist = compute_directional_and_distance(
        init_gripper_pos, init_spatula_pos, extents["gripper_radius"], spatula_extents)
    
    is_touching_spatula = bool(g_s_dist < (np.max(spatula_extents)/2.0 + extents["gripper_radius"] + 0.1))
    if is_touching_spatula: g_s_relations.append("TOUCHING")
    is_holding_spatula = is_touching_spatula and bool(gripper_open < 0.1)
    if is_holding_spatula: g_s_relations.append("HOLDING")

    # ==========================================
    # Edge 2: spatula -> cube (Tool support edge)
    # ==========================================
    s_c_relations, s_c_delta, s_c_dist = compute_directional_and_distance(
        spatula_pose[:3], cube_spawn_pos, spatula_extents, cube_extents)
    _, init_s_c_delta, init_s_c_dist = compute_directional_and_distance(
        init_spatula_pos, cube_spawn_pos, spatula_extents, cube_extents)
    has_lifted = prev_obs_dict.get("has_lifted_cube", False) if prev_obs_dict else False
    is_lifted = bool(spatula_pose[2] > cube_spawn_pos[2] + LIFT_HEIGHT_THRESHOLD)
    # is_cube_on_spatula = is_lifted and is_holding_spatula and bool(s_c_dist < 0.20)
    if is_lifted and is_holding_spatula and bool(s_c_dist < 0.20):
        is_cube_on_spatula = True
        has_lifted = True
    elif has_lifted and is_holding_spatula:
        is_cube_on_spatula = True
    else:
        is_cube_on_spatula = False

    if s_c_dist < 0.20: s_c_relations.append("TOUCHING")
    if is_cube_on_spatula: 
        s_c_relations.append("ON") 

    # Progress calculation
    dist_to_target_xy = np.linalg.norm(spatula_pose[:2] - success_pose[:2])
    
    has_lifted = prev_obs_dict.get("has_lifted_cube", False) if prev_obs_dict else False
    
    if is_cube_on_spatula: 
        has_lifted = True
    p1_dist = g_s_dist if not is_holding_spatula else 0.0
    p2_dist = s_c_dist if (is_holding_spatula and not has_lifted) else 0.0
    p3_dist = dist_to_target_xy
    current_magic_dist = p1_dist + p2_dist + p3_dist

    init_p3 = np.linalg.norm(cube_spawn_pos[:2] - success_pose[:2]) 
    init_magic_dist = init_g_s_dist + init_s_c_dist + init_p3

    obs_dict["has_lifted_cube"] = has_lifted

    is_near_target = bool(dist_to_target_xy <= AROUND_XY_DIST)
    is_at_target = bool(dist_to_target_xy <= SUCCESS_XY_DIST)
    
    s_t_relations = []
    if is_holding_spatula:
        if is_near_target: s_t_relations.append("AROUND")
        if is_at_target and is_cube_on_spatula: s_t_relations.append("INSIDE")

    if "INSIDE" in s_t_relations:
        action_type = "done"
    elif is_holding_spatula: 
        action_type = "scoop" 
    else:
        action_type = "pick"
 
    edge_gs = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "spatula",
        "spatial_relations": list(set(g_s_relations)),
        "geometric_attributes": {
            "delta_3d": g_s_delta.tolist(), "dist": float(g_s_dist), 
            "original_delta_3d": init_g_s_delta.tolist(), "original_dist": float(init_g_s_dist)
        },
        "is_touching": is_touching_spatula, "is_stable": is_holding_spatula
    }

    edge_sc = {
        "functional_relationship": "activate",
        "object1": "spatula", "object2": "cube",
        "spatial_relations": list(set(s_c_relations)),  
        "geometric_attributes": {
            "delta_3d": s_c_delta.tolist(), "dist": float(s_c_dist),
            "original_delta_3d": init_s_c_delta.tolist(), "original_dist": float(init_s_c_dist)
        },
        "is_touching": bool("TOUCHING" in s_c_relations), "is_stable": is_cube_on_spatula
    }
    # Edge 3: spatula -> target_zone
    edge_st = {
        "functional_relationship": "pair_with",
        "object1": "spatula", "object2": "target_zone",
        "spatial_relations": s_t_relations,
        "geometric_attributes": {
            # Injects magic_dist into X and Z axes for Estimator compatibility
            "delta_3d": [float(current_magic_dist), 0.0, float(current_magic_dist)],
            "dist": float(current_magic_dist),
            "original_delta_3d": [float(init_magic_dist), 0.0, float(init_magic_dist)],
            "original_dist": float(init_magic_dist)
        },
        "is_touching": False, "is_stable": (action_type == "done")
    }

    if not is_cube_on_spatula:
        focus_edges = [("gripper", "spatula"), ("spatula", "cube")]
    else:
        focus_edges = [("spatula", "cube"), ("spatula", "target_zone")]

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "spatula", "cube", "target_zone"], 
        "edges": [edge_gs, edge_sc, edge_st],             
        "action_type": action_type,
        "task_thresholds": {
            "target_radius": float(SUCCESS_XY_DIST),
            "around_radius": float(AROUND_XY_DIST)
        },
        "focus_edges": focus_edges
    }


def _extract_setup_chess(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                         indices, extents, instructions, prev_obs_dict, obs_dict):
    
    SUCCESS_XY_DIST = 0.015
    AROUND_XY_DIST = 0.015

    # 1. Reshape pose matrices
    pieces_state = task_state[indices["pieces_slice"]].reshape(32, 7)
    detectors_state = task_state[indices["detectors_slice"]].reshape(32, 7)
    
    init_pieces_state = init_task_state[indices["pieces_slice"]].reshape(32, 7)
    init_detectors_state = init_task_state[indices["detectors_slice"]].reshape(32, 7)

    piece_size = extents.get("chess_piece_size", [0.03, 0.03, 0.06])
    sensor_radius = extents.get("success_radius", 0.02)
    target_size = np.array([sensor_radius, sensor_radius, 0.01])

    # 2. Dynamic target mapping & global stage tracking
    if prev_obs_dict and "target_mapping" in prev_obs_dict.get("task_thresholds", {}):
        target_mapping = prev_obs_dict["task_thresholds"]["target_mapping"]
        total_stages = prev_obs_dict["task_thresholds"]["total_stages"]
    else:
        target_mapping = {}
        empty_detectors = []
        # Find empty slots (detectors far from any piece)
        for d_idx in range(32):
            d_pos = init_detectors_state[d_idx, :2]
            min_dist_to_pieces = np.min(np.linalg.norm(init_pieces_state[:, :2] - d_pos, axis=1))
            if min_dist_to_pieces > SUCCESS_XY_DIST:
                empty_detectors.append(d_idx)
        displaced_pieces = []
        for p_idx in range(32):
            p_pos = init_pieces_state[p_idx, :2]
            if np.min(np.linalg.norm(init_detectors_state[:, :2] - p_pos, axis=1)) > SUCCESS_XY_DIST:
                displaced_pieces.append(p_idx)
        # for i in range(len(displaced_pieces)):
        #     target_mapping[str(displaced_pieces[i])] = empty_detectors[i]

        # total_stages = max(1, len(displaced_pieces)) 
        pairs = min(len(displaced_pieces), len(empty_detectors))
        for i in range(pairs):
            target_mapping[str(displaced_pieces[i])] = empty_detectors[i]
        total_stages = max(1, pairs)

    # prev_active_idx = prev_obs_dict.get("task_thresholds", {}).get("active_piece_index") if prev_obs_dict else None
    # prev_action = prev_obs_dict.get("action_type") if prev_obs_dict else None

    # displaced_indices = []
    # for p_idx_str, target_d_idx in target_mapping.items():
    #     p_idx = int(p_idx_str)
    #     p_pos = pieces_state[p_idx, :2]
    #     target_pos = detectors_state[target_d_idx, :2]
    #     dist_to_target = np.linalg.norm(target_pos - p_pos)
        
    #     is_touch = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], pieces_state[p_idx], piece_size)
    #     is_held = is_touch and gripper_open < 0.1
        
    #     is_just_placed = (p_idx == prev_active_idx) and (prev_action == "place") and (not is_held) and (dist_to_target <= SUCCESS_XY_DIST)
    #     if dist_to_target > SUCCESS_XY_DIST or is_held or is_just_placed:
    #         displaced_indices.append(p_idx)

    # current_stage = total_stages - len(displaced_indices)
    # current_stage = max(0, min(total_stages - 1, current_stage))

    # Progress calculation
    completed_indices = prev_obs_dict.get("completed_indices", []).copy() if prev_obs_dict else []
    for p_idx_str, target_d_idx in target_mapping.items():
        p_idx = int(p_idx_str)
        real_dist = np.linalg.norm(pieces_state[p_idx, :2] - detectors_state[target_d_idx, :2])
        gripper_to_piece_dist = np.linalg.norm(gripper_pose[:3] - pieces_state[p_idx, :3])
        is_manipulating = bool(gripper_to_piece_dist < 0.05)
        if real_dist <= SUCCESS_XY_DIST and not is_manipulating:
            if p_idx not in completed_indices:
                completed_indices.append(p_idx)      
        # elif real_dist > 0.08 and p_idx in completed_indices and not is_manipulating:
        #     completed_indices.remove(p_idx)  
    if prev_obs_dict:
        prev_completed = prev_obs_dict.get("completed_indices", [])
        for p in prev_completed:
            if p not in completed_indices:
                completed_indices.append(p)

    # Lock active piece (manipulated or intended)
    displaced_indices = [int(p) for p in target_mapping.keys() if int(p) not in completed_indices]
    if displaced_indices:
        active_idx = displaced_indices[0]
        min_g_dist = float('inf') 

        for i in displaced_indices:
            p_pos = pieces_state[i, :3]
            g_dist = np.linalg.norm(gripper_pose[:3] - p_pos)
            # is_touch = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], pieces_state[i], piece_size)
            
            # if is_touch and gripper_open < 0.1:
            #     active_idx = i
            #     break
            if g_dist < min_g_dist:
                min_g_dist = g_dist
                active_idx = i
    else:
        active_idx = int(list(target_mapping.keys())[0])
        # if prev_active_idx is not None:
        #     active_idx = int(prev_active_idx)
        # else:
        #     active_idx = int(list(target_mapping.keys())[0])
    
    # ==========================================
    # 6. Target retrieval
    # ==========================================
    active_piece_pose = pieces_state[active_idx]
    best_target_idx = target_mapping[str(active_idx)]
    
    active_target_pose = detectors_state[best_target_idx]
    init_active_piece_pos = init_pieces_state[active_idx][:3]
    init_active_target_pos = init_detectors_state[best_target_idx][:3]

    # ==========================================
    # 3. Edge 1: gripper -> chess_piece
    # ==========================================
    g_p_relations, g_p_delta, g_p_dist = compute_directional_and_distance(
        gripper_pose[:3], active_piece_pose[:3], extents["gripper_radius"], piece_size)
    _, init_g_p_delta, init_g_p_dist = compute_directional_and_distance(
        init_gripper_pos, init_active_piece_pos, extents["gripper_radius"], piece_size)

    is_touching_piece = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], active_piece_pose, piece_size)
    if is_touching_piece: g_p_relations.append("TOUCHING")

    # delta_dist = 0.0
    # if prev_obs_dict is not None:
    #     prev_pieces_state = prev_obs_dict['task_low_dim_state'][indices["pieces_slice"]].reshape(32, 7)
    #     prev_piece_pos = prev_pieces_state[active_idx, :3]
    #     prev_g_p_dist = np.linalg.norm(prev_obs_dict['gripper_pose'][:3] - prev_piece_pos)
    #     delta_dist = abs(g_p_dist - prev_g_p_dist)
    # is_holding_piece = False
    # if is_touching_piece and gripper_open < 0.1:
    #     if prev_obs_dict is None or delta_dist < KINEMATIC_LOCK_TOLERANCE:
    #         is_holding_piece = True
    #         g_p_relations.append("HOLDING")
    
    is_holding_piece = (gripper_to_piece_dist < 0.04) and gripper_open < 0.8
    if is_holding_piece: g_p_relations.append("HOLDING")

    # ==========================================
    # 4. Edge 2: chess_piece -> target_zone
    # ==========================================
    p_t_relations, p_t_delta, p_t_dist = compute_directional_and_distance(
        active_piece_pose[:3], active_target_pose[:3], piece_size, target_size)
    _, init_p_t_delta, init_p_t_dist = compute_directional_and_distance(
        init_active_piece_pos, init_active_target_pos, piece_size, target_size)

    dist_xy = np.linalg.norm(p_t_delta[:2])
    dist_z = abs(p_t_delta[2])
    
    is_inside = bool(dist_xy <= SUCCESS_XY_DIST and dist_z <= 0.10)
    # is_around = bool(dist_xy <= AROUND_XY_DIST)

    # if is_inside or is_just_placed:
    #     p_t_relations.append("INSIDE")
    #     p_t_relations.append("ON")
    #     p_t_relations.append("AROUND")
    # elif is_around:
    #     p_t_relations.append("AROUND") 
    if is_inside:
        p_t_relations.extend(["INSIDE", "ON", "AROUND"])
    elif dist_xy <= AROUND_XY_DIST:
        p_t_relations.append("AROUND")
    
    # if not is_around and p_t_dist > 0.1:
    #     p_t_relations.append("FAR")
    # else:
    #     p_t_relations.append("CLOSE")
    
    # if not is_inside:
    #     p_t_relations.append("OUTSIDE") 

    base_progress = len(completed_indices) / total_stages

    if init_p_t_dist < 0.01:
        local_increment = 0.0
    else:
        local_increment = (1.0 - (p_t_dist / init_p_t_dist)) * 0.9
        local_increment = max(0.0, min(0.9, local_increment))

    final_prog = base_progress + (local_increment / total_stages)

    # Global monotonic lock per episode
    if prev_obs_dict:
        prev_completed = prev_obs_dict.get("completed_indices", [])
        prev_final_prog = prev_obs_dict.get("max_progress_ever", 0.0)
        # if len(completed_indices) < len(prev_completed):
        #     pass
        # else:
        #     final_prog = max(final_prog, prev_final_prog)
        final_prog = max(final_prog, prev_final_prog)

    obs_dict["max_progress_ever"] = final_prog
    obs_dict["completed_indices"] = completed_indices.copy()
 
    if is_inside and not is_holding_piece:#
        action_type = "done" 
    elif is_holding_piece:
        action_type = "place" 
    else:
        action_type = "pick"

    edge_gp = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "chess_piece",
        "spatial_relations": list(set(g_p_relations)),
        "geometric_attributes": {
            "delta_3d": g_p_delta.tolist(), "dist": float(g_p_dist),
            "original_delta_3d": init_g_p_delta.tolist(), "original_dist": float(init_g_p_dist)
        }, 
        "is_touching": bool(is_touching_piece), 
        "is_stable": bool(is_holding_piece)
    }

    # edge_pt = {
    #     "functional_relationship": "pair_with",
    #     "object1": "chess_piece", "object2": "target_zone",
    #     "spatial_relations": list(set(p_t_relations)),
    #     "geometric_attributes": {
    #         "delta_3d": p_t_delta.tolist(), "dist": float(p_t_dist),
    #         "original_delta_3d": init_p_t_delta.tolist(), "original_dist": float(init_p_t_dist)
    #     }, 
    #     "is_touching": bool(is_inside), 
    #     "is_stable": bool(is_inside)
    # }
    edge_pt = {
        "functional_relationship": "pair_with",
        "object1": "chess_piece", "object2": "target_zone",
        "spatial_relations": list(set(p_t_relations)),
        "geometric_attributes": {
            "delta_3d": p_t_delta.tolist(), 
            "dist": float(p_t_dist), 
            "original_dist": float(p_t_dist + 0.001) if is_inside else float(init_p_t_dist)
        }, 
        "is_touching": bool(is_inside), 
        "is_stable": bool(is_inside)
    }
 
    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "chess_piece", "target_zone"], 
        "edges": [edge_gp, edge_pt],             
        "action_type": action_type,
        "task_thresholds": {
            "setup_chess_progress": float(final_prog), 
            "target_radius": float(SUCCESS_XY_DIST),
            "around_radius": float(AROUND_XY_DIST),
            "active_piece_index": int(active_idx),
            "total_stages": int(total_stages),
            "current_stage": len(completed_indices),
            "target_mapping": target_mapping,
            "completed_indices": completed_indices.copy()
        }
    }
 

def _extract_slide_block_to_target(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                                 indices, extents, instructions, prev_obs_dict, obs_dict):
    # 1. Dimensions & Poses
    block_extents = extents.get("block_size", [0.05, 0.05, 0.05])
    success_radius = extents.get("success_radius", 0.04)  

    block_pose = task_state[indices["block"]]
    success_pose = task_state[indices["success"]]
    
    init_block_pos = init_task_state[indices["block"]][:3]
    init_success_pos = init_task_state[indices["success"]][:3]

    # --- Edge 1: gripper -> block ---
    g_b_relations, g_b_delta, g_b_dist = compute_directional_and_distance(
        gripper_pose[:3], block_pose[:3], extents["gripper_radius"], block_extents)
    _, init_g_b_delta, init_g_b_dist = compute_directional_and_distance(
        init_gripper_pos, init_block_pos, extents["gripper_radius"], block_extents)

    is_touching_block = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], block_pose, block_extents)
    if is_touching_block: g_b_relations.append("TOUCHING")

    is_holding = check_holding(is_touching_block, g_b_dist, "block", g_b_relations, gripper_open, indices, prev_obs_dict)
    if is_holding: g_b_relations.append("HOLDING")

    # --- Edge 2: block -> target ---
    b_t_relations, b_t_delta, b_t_dist = compute_directional_and_distance(
        block_pose[:3], success_pose[:3], block_extents, success_radius)
    _, init_b_t_delta, init_b_t_dist = compute_directional_and_distance(
        init_block_pos, init_success_pos, block_extents, success_radius)

    dist_xy = np.linalg.norm(block_pose[:2] - success_pose[:2])

    TARGET_RADIUS = success_radius  
    AROUND_RADIUS = TARGET_RADIUS + 0.08 

    is_inside = bool(dist_xy <= TARGET_RADIUS)
    is_around = bool(dist_xy <= AROUND_RADIUS)

    if is_inside:
        b_t_relations.append("INSIDE")
        b_t_relations.append("ON")
    elif is_around:
        b_t_relations.append("AROUND")
    if not is_inside:
        b_t_relations.append("OUTSIDE")


    if is_inside:
        action_type = "done"
    elif is_holding or is_touching_block:
        action_type = "place" # Represents pushing towards target in slide tasks
    else:
        action_type = "pick"

    edge_gb = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "block",
        "spatial_relations": list(set(g_b_relations)),
        "geometric_attributes": {
            "delta_3d": g_b_delta.tolist(), "dist": float(g_b_dist),
            "original_delta_3d": init_g_b_delta.tolist(), "original_dist": float(init_g_b_dist)
        },
        "is_touching": bool(is_touching_block),
        "is_stable": bool(is_holding)
    }

    edge_bt = {
        "functional_relationship": "pair_with",
        "object1": "block", "object2": "target_zone",
        "spatial_relations": list(set(b_t_relations)),
        "geometric_attributes": {
            "delta_3d": b_t_delta.tolist(), "dist": float(b_t_dist),
            "original_delta_3d": init_b_t_delta.tolist(), "original_dist": float(init_b_t_dist)
        },
        "is_touching": bool(is_inside),
        "is_stable": bool(is_inside)
    }

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "block", "target_zone"],
        "edges": [edge_gb, edge_bt],
        "action_type": action_type,
        "task_thresholds": {
            "target_radius": float(TARGET_RADIUS),
            "around_radius": float(AROUND_RADIUS)
        }
    }  


def _extract_stack_cups(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                        indices, extents, instructions, prev_obs_dict, obs_dict):
    cup1_pose = task_state[indices["cup1"]]
    init_cup1_pos = init_task_state[indices["cup1"]][:3]

    # is_frame_zero = (prev_obs_dict is None) or (
    #     np.allclose(gripper_pose[:3], init_gripper_pos, atol=1e-4) and 
    #     np.allclose(cup1_pose[:3], init_cup1_pos, atol=1e-4)
    # )

    # if is_frame_zero:
    #     _extract_stack_cups.c1_stacked = False
    #     _extract_stack_cups.c3_stacked = False
    #     _extract_stack_cups.stage2_done = False
    #     if hasattr(_extract_stack_cups, "stage1_gripper_pos"):
    #         del _extract_stack_cups.stage1_gripper_pos

    # is_c1_stacked = getattr(_extract_stack_cups, "c1_stacked", False)
    # is_c3_stacked = getattr(_extract_stack_cups, "c3_stacked", False)

    # --- 1. State retrieval from memory ---
    is_c1_stacked = prev_obs_dict.get("c1_stacked", False) if prev_obs_dict else False
    is_c3_stacked = prev_obs_dict.get("c3_stacked", False) if prev_obs_dict else False
    stage1_gripper_pos = prev_obs_dict.get("stage1_gripper_pos", None) if prev_obs_dict else None

    # Dimensions & Poses
    cup2_pose = task_state[indices["cup2"]]
    cup3_pose = task_state[indices["cup3"]] 

    init_cup3_pos = init_task_state[indices["cup3"]][:3]
    init_cup2_pos = init_task_state[indices["cup2"]][:3]
    cup_size = extents.get("cup1_size", [0.05, 0.05, 0.08]) 
    success_radius = extents.get("success_radius", 0.04)  
 
    TARGET_XY_RADIUS = 0.12 
    TARGET_Z_RADIUS = 0.18 
    AROUND_RADIUS = 0.13

    # Independent stacking detection
    c1_dist_xy = np.linalg.norm(cup1_pose[:2] - cup2_pose[:2]) 
    c1_dist_z = abs(cup1_pose[2] - cup2_pose[2])
    
    c3_dist_x = abs(cup3_pose[0] - cup2_pose[0]) 
    c3_dist_y = abs(cup3_pose[1] - cup2_pose[1])
    c3_dist_z = abs(cup3_pose[2] - cup2_pose[2])
    
    g_c1_dist = np.linalg.norm(gripper_pose[:3] - cup1_pose[:3])
    g_c3_dist = np.linalg.norm(gripper_pose[:3] - cup3_pose[:3])

    # Capture release event
    prev_gripper_open = prev_obs_dict['gripper_open'] if prev_obs_dict is not None else 1.0
    just_released = (prev_gripper_open < 0.1) and (gripper_open >= 0.1)
    just_stacked_cup = None
 
    # if bool(c1_dist_xy <= TARGET_XY_RADIUS/2 and c1_dist_z <= TARGET_Z_RADIUS) and just_released:
    #     if not is_c1_stacked:
    #         is_c1_stacked = True 
    #         _extract_stack_cups.c1_stacked = True
    #         just_stacked_cup = "cup1"
    #         _extract_stack_cups.stage1_gripper_pos = gripper_pose[:3].copy()
        
    # if (abs(c3_dist_x) <= TARGET_XY_RADIUS) and (abs(c3_dist_y) <= TARGET_XY_RADIUS/2) and (abs(c3_dist_z) <= TARGET_Z_RADIUS) and just_released:
    #     if not is_c3_stacked:
    #         is_c3_stacked = True 
    #         _extract_stack_cups.c3_stacked = True
    #         just_stacked_cup = "cup3"
    #         _extract_stack_cups.stage1_gripper_pos = gripper_pose[:3].copy()

    if bool(c1_dist_xy <= TARGET_XY_RADIUS/2 and c1_dist_z <= TARGET_Z_RADIUS) and just_released:
        if not is_c1_stacked:
            is_c1_stacked = True
            stage1_gripper_pos = gripper_pose[:3].tolist() 
            
    if (abs(c3_dist_x) <= TARGET_XY_RADIUS) and (abs(c3_dist_y) <= TARGET_XY_RADIUS/2) and (abs(c3_dist_z) <= TARGET_Z_RADIUS) and just_released:
        if not is_c3_stacked:
            is_c3_stacked = True
            stage1_gripper_pos = gripper_pose[:3].tolist() 

    # Delay stage transition by one frame for release detection
    real_stage = int(is_c1_stacked) + int(is_c3_stacked)
    current_stage = real_stage - 1 if just_stacked_cup else real_stage
    # dynamic_init_gripper_pos = getattr(_extract_stack_cups, "stage1_gripper_pos", init_gripper_pos)
    dynamic_init_gripper_pos = np.array(stage1_gripper_pos) if stage1_gripper_pos is not None else init_gripper_pos
    
    init_g_c1_dist = np.linalg.norm(dynamic_init_gripper_pos - init_cup1_pos)
    init_g_c3_dist = np.linalg.norm(dynamic_init_gripper_pos - init_cup3_pos)
    
    approach_c1 = init_g_c1_dist - g_c1_dist
    approach_c3 = init_g_c3_dist - g_c3_dist

    # Intent inference logic
    if just_stacked_cup:
        active_cup_key = just_stacked_cup
        active_pose = cup1_pose if active_cup_key == "cup1" else cup3_pose
        init_active_pos = init_cup1_pos if active_cup_key == "cup1" else init_cup3_pos 
    else:
        if current_stage == 0:
            active_cup_key = "cup1" if approach_c1 >= approach_c3 else "cup3" 
            if approach_c1 >= approach_c3 and g_c1_dist <= g_c3_dist: active_cup_key = "cup1" 
            elif approach_c3 >= approach_c1 and g_c3_dist <= g_c1_dist: active_cup_key = "cup3"
        elif current_stage == 1:
            active_cup_key = "cup3" if getattr(_extract_stack_cups, "c1_stacked", False) else "cup1"
        else:
            active_cup_key = "cup1"

        active_pose = cup1_pose if active_cup_key == "cup1" else cup3_pose
        init_active_pos = init_cup1_pos if active_cup_key == "cup1" else init_cup3_pos
 
    # --- Edge 1: gripper -> active_cup ---
    g_c_relations, g_c_delta, g_c_dist = compute_directional_and_distance(
        gripper_pose[:3], active_pose[:3], extents["gripper_radius"], cup_size)
    _, init_g_c_delta, init_g_c_dist = compute_directional_and_distance(
        dynamic_init_gripper_pos, init_active_pos, extents["gripper_radius"], cup_size)

    is_touching_cup = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], active_pose, cup_size)
    if is_touching_cup: g_c_relations.append("TOUCHING")

    is_holding_cup = check_holding(is_touching_cup, g_c_dist, active_cup_key, g_c_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_cup: g_c_relations.append("HOLDING")

    # --- Edge 2: active_cup -> target_zone ---
    c_t_relations, c_t_delta, c_t_dist = compute_directional_and_distance(
        active_pose[:3], cup2_pose[:3], np.max(cup_size)/2, np.max(cup_size)/2)
    _, init_c_t_delta, init_c_t_dist = compute_directional_and_distance(
        init_active_pos, init_cup2_pos, np.max(cup_size)/2, np.max(cup_size)/2)
 
    dist_xy = np.linalg.norm(c_t_delta[:2])
    dist_z = abs(c_t_delta[2])
     
    if not hasattr(_extract_stack_cups, "stage2_done") or not prev_obs_dict:
        _extract_stack_cups.stage2_done = False

    # Tolerance lock for Stage 2 success
    if current_stage == 1 and just_released and dist_xy <= TARGET_XY_RADIUS + 0.08 and dist_z <= TARGET_Z_RADIUS + 0.15:
        _extract_stack_cups.stage2_done = True

    # Elevation lock: must be lifted 3cm from table to count as stacked
    is_lifted_from_table = bool(active_pose[2] > init_active_pos[2] + 0.03)

    if real_stage == 2 or getattr(_extract_stack_cups, "stage2_done", False):
        is_inside_logic = True
    elif current_stage == 0 and real_stage == 1 and just_released: 
        is_inside_logic = True
    else:
        is_inside_logic = bool(dist_xy <= TARGET_XY_RADIUS/2 and dist_z <= TARGET_Z_RADIUS and not is_holding_cup and is_lifted_from_table)

    active_cup_radius = max(cup_size[0], cup_size[1]) / 2.0

    is_around_logic = check_approach_corridor_world(
        pos_obj=active_pose[:3], 
        pos_anchor=cup2_pose[:3],     
        approach_axis=2,              
        approach_dir=1.0,             
        mo_size=active_cup_radius, 
        tgt_size=cup_size, 
        min_radius=AROUND_RADIUS,     
        min_height=0.90               
    )

    if is_inside_logic:
        c_t_relations.append("INSIDE")
        c_t_relations.append("ON")
        c_t_relations.append("AROUND")
    elif is_around_logic and is_holding_cup:
        c_t_relations.append("AROUND")
 
    if current_stage == 2 or real_stage == 2 or getattr(_extract_stack_cups, "stage2_done", False):
        action_type = "done"
    elif is_holding_cup:
        action_type = "place"
    else:
        action_type = "pick"  

    edge_gc = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "active_cup",
        "spatial_relations": list(set(g_c_relations)),
        "geometric_attributes": {
            "delta_3d": g_c_delta.tolist(), "dist": float(g_c_dist),
            "original_delta_3d": init_g_c_delta.tolist(), "original_dist": float(init_g_c_dist)
        },
        "is_touching": bool(is_touching_cup),
        "is_stable": bool(is_holding_cup)
    }

    edge_ct = {
        "functional_relationship": "pair_with",
        "object1": "active_cup", "object2": "target_zone",
        "spatial_relations": list(set(c_t_relations)),
        "geometric_attributes": {
            "delta_3d": c_t_delta.tolist(), "dist": float(c_t_dist),
            "original_delta_3d": init_c_t_delta.tolist(), "original_dist": float(init_c_t_dist)
        },
        "is_touching": bool(is_inside_logic),
        "is_stable": bool(is_inside_logic)
    }


    obs_dict["c1_stacked"] = is_c1_stacked
    obs_dict["c3_stacked"] = is_c3_stacked
    obs_dict["stage1_gripper_pos"] = stage1_gripper_pos

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "active_cup", "target_zone"],
        "edges": [edge_gc, edge_ct],
        "action_type": action_type,
        "task_thresholds": {
            "current_stage": int(current_stage),
            "total_stages": 2, 
            "active_cup_name": active_cup_key, 
            "target_radius": float(TARGET_XY_RADIUS),
            "around_radius": float(AROUND_RADIUS)
        }
    }


def _extract_straighten_rope(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                             indices, extents, instructions, prev_obs_dict, obs_dict):
    # 1. Initialization
    end_keys = ["head", "tail"]
    target_keys = {"head": "success_head", "tail": "success_tail"}
    
    poses = {k: task_state[indices[k]] for k in end_keys + list(target_keys.values())}
    init_poses = {k: init_task_state[indices[k]][:3] for k in end_keys + list(target_keys.values())}

    end_size = extents.get("head_size", [0.03, 0.03, 0.03])
    success_radius = extents.get("success_radius", 0.05)
    
    # 2. End-point status scanning
    end_status = {}
    completed_count = 0 
    completed_ends = prev_obs_dict.get("completed_ends", []) if prev_obs_dict else []
    
    for k in end_keys:
        curr_pose_7d = poses[k]
        tgt_p = poses[target_keys[k]][:3]
        
        dist_xy = np.linalg.norm(curr_pose_7d[:2] - tgt_p[:2])
        is_in_zone = dist_xy <= success_radius
        
        is_touch = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], curr_pose_7d, end_size)
        is_held = is_touch and gripper_open < 0.1
        
        # Maintains completion state unless moved far (>15cm) from target
        if is_in_zone and k not in completed_ends:
            completed_ends.append(k)
        elif dist_xy > 0.15 and k in completed_ends:
            completed_ends.remove(k)
            
        is_finished = (k in completed_ends)
        # if is_finished:
        #     completed_count += 1
            
        end_status[k] = {"is_finished": is_finished, "is_held": is_held, "dist_xy": dist_xy}
 
    # 3. Active Target Determination
    held_ends = [k for k in end_keys if end_status[k]["is_held"]]
    
    if held_ends:
        active_key = held_ends[0]
    else:
        unfinished = [k for k in end_keys if not end_status[k]["is_finished"]]
        if not unfinished:
            active_key = end_keys[-1]
        else:
            active_key = min(unfinished, key=lambda k: np.linalg.norm(gripper_pose[:3] - poses[k][:3]))

    # 4. Progress anchor points
    completed_count = len(completed_ends)
    total_stages = 2
    current_stage = min(completed_count, total_stages - 1)

    if current_stage == 0:
        v_start_gripper = init_gripper_pos
    else:
        other_key = "tail" if active_key == "head" else "head"
        v_start_gripper = init_poses[target_keys[other_key]]

    v_start_end = init_poses[active_key]
    v_start_target = init_poses[target_keys[active_key]]

    # --- Edge 1: gripper -> rope_end ---
    g_e_rels, g_e_delta, g_e_dist = compute_directional_and_distance(
        gripper_pose[:3], poses[active_key][:3], extents["gripper_radius"], end_size)

    if current_stage == 0:
        actual_v_start_gripper = init_gripper_pos
    else:
        actual_v_start_gripper = prev_obs_dict['gripper_pose'][:3] if prev_obs_dict else init_gripper_pos
        
    _, init_g_e_delta, init_g_e_dist = compute_directional_and_distance(
        np.array(actual_v_start_gripper), np.array(v_start_end), extents["gripper_radius"], end_size) 
    
    is_touching = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], poses[active_key], end_size)
    if is_touching: g_e_rels.append("TOUCHING")
    
    is_holding = end_status[active_key]["is_held"]
    if is_holding: g_e_rels.append("HOLDING")

    # --- Edge 2: rope_end -> target_patch ---
    e_t_rels, e_t_delta, e_t_dist = compute_directional_and_distance(
        poses[active_key][:3], poses[target_keys[active_key]][:3], end_size, success_radius)
    
    _, init_e_t_delta, init_e_t_dist = compute_directional_and_distance(
        np.array(init_poses[active_key]), np.array(init_poses[target_keys[active_key]]), end_size, success_radius)

    dist_xy_active = end_status[active_key]["dist_xy"]
    if dist_xy_active <= success_radius:
        e_t_rels.extend(["INSIDE", "ON", "TOUCHING", "AROUND"])
    elif dist_xy_active <= success_radius + 0.03:
        e_t_rels.append("AROUND")
    else:
        e_t_rels.append("OUTSIDE")


    if completed_count == total_stages:
        action_type = "done"
    elif is_holding:
        action_type = "pull"
    else:
        action_type = "pick"

    # global_stage_progress = current_stage / total_stages
    # local_progress = (1.0 - e_t_dist / init_e_t_dist) / total_stages
    # final_progress = global_stage_progress + max(0.0, local_progress)

    # global_stage_progress = current_stage / total_stages
    # local_progress = (1.0 - e_t_dist / init_e_t_dist) / total_stages
    # final_progress = global_stage_progress + max(0.0, local_progress)
 
    # if prev_obs_dict:
    #     historical_p = prev_obs_dict.get("max_p", 0.0)
    #     if (historical_p - final_progress) < 0.15:
    #         final_progress = max(final_progress, historical_p) 
    
    # obs_dict["max_p"] = final_progress
    # obs_dict["last_stage"] = current_stage
    obs_dict["completed_ends"] = completed_ends

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "rope_end", "target_patch"],
        "edges": [
            {
                "functional_relationship": "manipulate",
                "object1": "gripper", "object2": "rope_end",
                "spatial_relations": list(set(g_e_rels)),
                "geometric_attributes": {
                    "delta_3d": g_e_delta.tolist(), "dist": float(g_e_dist),
                    "original_delta_3d": init_g_e_delta.tolist(), "original_dist": float(init_g_e_dist)
                },
                "is_touching": bool(is_touching),
                "is_stable": bool(is_holding)
            },
            {
                "functional_relationship": "pair_with",
                "object1": "rope_end", "object2": "target_patch",
                "spatial_relations": list(set(e_t_rels)),
                "geometric_attributes": {
                    "delta_3d": e_t_delta.tolist(), "dist": float(e_t_dist),
                    "original_delta_3d": init_e_t_delta.tolist(), "original_dist": float(init_e_t_dist)
                },
                "is_touching": bool("TOUCHING" in e_t_rels),
                "is_stable": bool(end_status[active_key]["is_finished"])
            }
        ],
        "action_type": action_type,
        "task_thresholds": {
            "active_end": active_key,
            "current_stage": int(current_stage),
            "total_stages": int(total_stages)
        }
    }


def _extract_turn_oven_on(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                          indices, extents, instructions, prev_obs_dict, obs_dict):
    # 1. Initialization
    knob_extents = extents.get("knob_size", [0.05, 0.05, 0.05])
    oven_extents = extents.get("oven_size", [0.6, 0.5, 0.6])
    
    # RLBench Standard: JointCondition(knob_joint, 1.5)
    TARGET_ROTATION = 1.5
    
    knob_pose = task_state[indices["knob"]]
    oven_pose = task_state[indices["oven"]]
    current_joint_pos = task_state[indices["knob_joint"]]
    
    init_knob_pos = init_task_state[indices["knob"]][:3]
    init_oven_pos = init_task_state[indices["oven"]][:3]
    init_joint_pos = init_task_state[indices["knob_joint"]]


    # --- Edge 1: gripper -> knob ---
    g_k_rels, g_k_delta, g_k_dist = compute_directional_and_distance(
        gripper_pose[:3], knob_pose[:3], extents["gripper_radius"], knob_extents)
    _, init_g_k_delta, init_g_k_dist = compute_directional_and_distance(
        init_gripper_pos, init_knob_pos, extents["gripper_radius"], knob_extents)

    is_touching_knob = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], knob_pose, knob_extents)
    if is_touching_knob: g_k_rels.append("TOUCHING")

    # is_holding = check_holding(is_touching_knob, g_k_dist, "knob", g_k_rels, gripper_open, indices, prev_obs_dict)
    is_holding = is_touching_knob and bool(gripper_open < 0.1)
    if is_holding: g_k_rels.append("HOLDING")

    # --- Edge 2: knob -> oven ---
    k_o_rels, k_o_delta, k_o_dist = compute_directional_and_distance(
        knob_pose[:3], oven_pose[:3], knob_extents, oven_extents)
    _, init_k_o_delta, init_k_o_dist = compute_directional_and_distance(
        init_knob_pos, init_oven_pos, knob_extents, oven_extents)
    # rotation_delta = abs(current_joint_pos - init_joint_pos)

    # ---------- FOR FAILURE ---------- 
    # rotation_delta = max(0.0, current_joint_pos - init_joint_pos) 
    # ---------- FOR EXPERT ---------- 
    # Rotation progress
    raw_rotation_delta = max(0.0, current_joint_pos - init_joint_pos)  
    curr_prog = min(1.0, raw_rotation_delta / TARGET_ROTATION) 
    if prev_obs_dict:
        curr_prog = max(curr_prog, prev_obs_dict.get("max_progress_ever", 0.0)) 
    obs_dict["max_progress_ever"] = curr_prog 
    rotation_delta = curr_prog * TARGET_ROTATION

    is_turned = bool(rotation_delta >= (TARGET_ROTATION - 0.05))
  
    if is_turned: 
        k_o_rels.append("ON")
    else:
        k_o_rels.append("OFF")
    
    k_o_rels.append("TOUCHING")

    if is_turned:
        action_type = "done"
    elif is_holding:
        action_type = "rotate"
    else:
        action_type = "pick"

    # Virtual "remaining rotation distance" for ProgressEstimator
    remaining_rot = max(0.0, TARGET_ROTATION - rotation_delta) 

    edge_gk = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "oven_knob",
        "spatial_relations": list(set(g_k_rels)),
        "geometric_attributes": {
            "delta_3d": g_k_delta.tolist(), "dist": float(g_k_dist),
            "original_delta_3d": init_g_k_delta.tolist(), "original_dist": float(init_g_k_dist)
        },
        "is_touching": bool(is_touching_knob),
        "is_stable": bool(is_holding)
    }

    edge_ko = {
        "functional_relationship": "activate",
        "object1": "oven_knob", "object2": "oven",
        "spatial_relations": list(set(k_o_rels)),
        "geometric_attributes": {
            "delta_3d": [remaining_rot, 0.0, 0.0], 
            "dist": float(remaining_rot),
            "original_delta_3d": [TARGET_ROTATION, 0.0, 0.0],
            "original_dist": float(TARGET_ROTATION)
        },
        "is_touching": True,
        "is_stable": bool(is_turned)
    }

    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "oven_knob", "oven"],
        "edges": [edge_gk, edge_ko],
        "action_type": action_type,
        "task_thresholds": {
            "turn_oven_on_progress": float(curr_prog), 
            "target_rotation": float(TARGET_ROTATION),
            "current_rotation": float(rotation_delta)
        }
    }
 

def _extract_wipe_desk(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                      indices, extents, instructions, prev_obs_dict, obs_dict):
    # 1. Parse entities
    sponge_extents = extents.get("sponge_size", [0.05, 0.08, 0.03])
    desk_extents = extents.get("dirt_boundary_size", [0.3, 0.3, 0.01])
    
    sponge_pose = task_state[indices["sponge"]]
    boundary_pose = task_state[indices["dirt_boundary"]]
    
    init_sponge_pos = init_task_state[indices["sponge"]][:3]
    init_boundary_pos = init_task_state[indices["dirt_boundary"]][:3]

    # 2. Physical relations (Strict Z-axis contact)
    g_s_rels, g_s_delta, g_s_dist = compute_directional_and_distance(
        gripper_pose[:3], sponge_pose[:3], extents["gripper_radius"], sponge_extents)
    _, init_g_s_delta, init_g_s_dist = compute_directional_and_distance(
        init_gripper_pos, init_sponge_pos, extents["gripper_radius"], sponge_extents)

    is_touching_sponge = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], sponge_pose, sponge_extents)
    if is_touching_sponge: g_s_rels.append("TOUCHING")

    is_holding = check_holding(is_touching_sponge, g_s_dist, "sponge", g_s_rels, gripper_open, indices, prev_obs_dict)
    if is_holding: g_s_rels.append("HOLDING")

    s_d_rels, s_d_delta, s_d_dist = compute_directional_and_distance(
        sponge_pose[:3], boundary_pose[:3], sponge_extents, desk_extents)
    _, init_s_d_delta, init_s_d_dist = compute_directional_and_distance(
        init_sponge_pos, init_boundary_pos, sponge_extents, desk_extents)

    dist_xy = np.linalg.norm(sponge_pose[:2] - boundary_pose[:2])
    dist_z = abs(sponge_pose[2] - boundary_pose[2])
    
    # Sponge Z-size is ~3.2cm (radius ~1.6cm). Distance < 2.5cm implies tight contact with desk surface.
    is_touching_desk = bool(dist_z < 0.025) 
    is_inside_area = bool(dist_xy < np.max(desk_extents))

    dist_to_init = np.linalg.norm(gripper_pose[:3] - init_gripper_pos)
    is_first_frame = (prev_obs_dict is None) or (dist_to_init < 0.001)

    clean_prog = prev_obs_dict.get("clean_progress", 0.0) if prev_obs_dict else 0.0
    if is_first_frame: clean_prog = 0.0

    # if not hasattr(_extract_wipe_desk, "global_clean_progress") or is_first_frame:
    #     # _extract_wipe_desk.global_clean_progress = 0.0
    #     clean_prog = 0.0

    # Incremental progress based on active friction displacement
    if is_holding and is_touching_desk and is_inside_area:
        action_type = "wipe"
        if prev_obs_dict and not is_first_frame:
            prev_sponge_pos = prev_obs_dict['task_low_dim_state'][indices["sponge"]][:3]
            move_delta = np.linalg.norm(sponge_pose[:3] - prev_sponge_pos)
            inc = 0.0002 + move_delta * 0.6
            # _extract_wipe_desk.global_clean_progress = min(0.98, _extract_wipe_desk.global_clean_progress + inc) 
            clean_prog = min(0.98, clean_prog + inc)
    elif is_holding:
        action_type = "place" # Transporting but hovering
    else:
        action_type = "pick"

    if clean_prog > 0.95 and not is_holding:
        action_type = "done"
        clean_prog = 1.0

    if action_type == "done":
        s_d_rels.extend(["ON", "TOUCHING", "AROUND"])
    else:
        if is_touching_desk and is_inside_area:
            s_d_rels.extend(["TOUCHING", "ON"])
        elif is_inside_area:
            s_d_rels.append("AROUND")
        else:
            s_d_rels.append("OUTSIDE")

    magic_dist = 1.0 - clean_prog

    edge_gs = {
        "functional_relationship": "manipulate",
        "object1": "gripper", "object2": "sponge",
        "spatial_relations": list(set(g_s_rels)),
        "geometric_attributes": {
            "delta_3d": g_s_delta.tolist(), "dist": float(g_s_dist),
            "original_delta_3d": init_g_s_delta.tolist(), "original_dist": float(init_g_s_dist)
        },
        "is_touching": bool(is_touching_sponge),
        "is_stable": bool(is_holding)
    }

    edge_sd = {
        "functional_relationship": "clean",
        "object1": "sponge", "object2": "desk_surface",
        "spatial_relations": list(set(s_d_rels)),
        "geometric_attributes": {
            "delta_3d": s_d_delta.tolist(), 
            "dist": float(magic_dist), 
            "original_delta_3d": [1.0, 0.0, 0.0], 
            "original_dist": 1.0
        },
        "is_touching": bool(is_touching_desk),
        "is_stable": action_type == "done"
    }
    obs_dict["clean_progress"] = clean_prog
    return {
        "task_instruction": instructions,
        "nodes": ["gripper", "sponge", "desk_surface"],
        "edges": [edge_gs, edge_sd],
        "action_type": action_type,
        "task_thresholds": {
            "clean_progress": float(clean_prog) 
        }
    }
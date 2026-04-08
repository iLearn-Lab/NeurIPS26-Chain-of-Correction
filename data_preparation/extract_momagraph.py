import numpy as np
from utils import compute_directional_and_distance, check_universal_contact, check_approach_corridor_world, check_inside_containment, check_holding

# HOLDING 校验 (假设帧间相对位移小于 2mm 视为刚性连接)
KINEMATIC_LOCK_TOLERANCE = 0.002

# ==========================================
# MomaGraph 数据解析主引擎
# ==========================================
def extract_momagraph_frame(task_name, obs_dict, init_obs_dict, indices, extents, instructions, prev_obs_dict=None):
    """
    由 Numpy 驱动的多任务图谱提取引擎。与任何物理仿真器完全解耦。
    """
    # 1. 提取所有任务都需要的公共状态
    task_state = obs_dict['task_low_dim_state']
    gripper_pose = obs_dict['gripper_pose']
    gripper_open = obs_dict['gripper_open']
    
    init_task_state = init_obs_dict['task_low_dim_state']
    init_gripper_pos = init_obs_dict['gripper_pose'][:3]
    
    gripper_radius = extents["gripper_radius"]
    
    # 2. 任务分发器 (Task Router)
    # 根据 task_name 自动拼接出对应的目标函数名
    func_name = f"_extract_{task_name}"
    # 从当前全局命名空间中寻找该函数
    extractor = globals().get(func_name)
    if not callable(extractor):
        raise ValueError(f"未受支持或未实现的场景图谱提取器: {func_name}()")
 
    # 统一调用提取器获取图谱
    return extractor(
        task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
        indices, extents, instructions, prev_obs_dict, obs_dict
    ) 


# ==========================================
# 具体任务的解析逻辑分支 (复用 Utils)
# ==========================================
def _extract_basketball_in_hoop(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                        indices, extents, instructions, prev_obs_dict, obs_dict):
    # 解析尺寸
    ball_radius = extents["mo_size"][0] / 2.0
    hoop_extents = extents["to_size"]
    success_radius = extents["success_radius"]
    
    # 解析位姿
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

    # 瞬时位移计算 (防作弊)
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

    # 宏观意图
    if is_inside and not is_holding: action_type = "done"
    elif is_holding or (not is_holding and "AROUND" in b_h_relations): action_type = "place"
    else: action_type = "pick"

    return {
        "task_instruction": instructions, # 完整 list 或者 模拟采样 np.random.choice(INSTRUCTIONS)
        "nodes": ["gripper", "ball", "hoop"],
        "edges": [edge_gb, edge_bh],
        "action_type": action_type
    }


def _extract_close_box(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                       indices, extents, instructions, prev_obs_dict, obs_dict):
    # TARGET_ANGLE = 2.6
    CLOSED_DIST = 0.090           # 设定小于 9cm 视为完全闭合 (末帧是 8.1cm)
    VERTICAL_DIST = 0.160         # 箱盖垂直于桌面或接近闭合的临界点 (约16cm)

    # 解析尺寸
    lid_extents = extents["mo_size"]
    base_extents = extents["to_size"]
    
    # 解析位姿
    # print("[DEBUG] mo_size = :", extents["mo_size"])
    # print("[DEBUG] to_size = :", extents["to_size"])
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

    # 瞬时位移计算 (防作弊)
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
        # 盖子达到垂直或接近闭合时，赋予 AROUND
        l_b_relations.append("AROUND")
        action_type = "close"
    elif l_b_dist < init_l_b_dist - 0.01: 
        # 只要碰到盖子或压下去了1cm，就视为正在关闭
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
            # "angle": float(simulated_angle),            # 关箱子任务的核心几何特征 反推出来的角度
            # "target_angle": float(TARGET_ANGLE)     # 传出目标角度供估算器使用
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
    # 解析尺寸
    lid_extents = extents["mo_size"]
    base_extents = extents["to_size"]
    
    # 解析位姿
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

    # 瞬时位移计算 (防作弊)
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
        # 盖子达到垂直或接近闭合时，赋予 AROUND，进度条准备跃升 85%
        l_b_relations.append("AROUND")
        action_type = "close"
    elif l_b_dist < init_l_b_dist - 0.01: 
        # 只要碰到盖子或压下去了1cm，就视为正在关闭
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

    # 1. 解析所有核心实体的尺寸
    door_extents = extents.get("door_size", [0.1, 0.1, 0.1])
    tray_extents = extents.get("tray_size", [0.1, 0.1, 0.1])
    plate_extents = extents.get("plate_size", [0.1, 0.1, 0.1])
    dishwasher_extents = extents.get("dishwasher_size", [0.1, 0.1, 0.1])

    # 定义物理阈值与 AROUND 临界值
    # 用门自身的高度/长度来计算动态阈值
    door_length = np.max(door_extents)
    DOOR_OPEN_DELTA = door_length * 0.46  # 动态阈值：门长度的 46% 视为彻底压平打开
    DOOR_AROUND_DELTA = door_length * 0.32
    plate_radius = np.min(plate_extents) / 2.0
    tray_depth = np.max(tray_extents)
    TRAY_PULL_DELTA = tray_depth * 0.66
    TRAY_AROUND_DELTA = tray_depth * 0.2
    PLATE_DONE_DELTA = 0.08
    PLATE_AROUND_DELTA = 0.001

    # 2. 解析所有实体的位姿
    door_pose = task_state[indices["door"]]
    tray_pose = task_state[indices["tray"]]
    plate_pose = task_state[indices["plate"]]
    dishwasher_pose = task_state[indices["dishwasher"]]

    init_door_pos = init_task_state[indices["door"]][:3]
    init_tray_pos = init_task_state[indices["tray"]][:3]
    init_plate_pos = init_task_state[indices["plate"]][:3]
    init_dishwasher_pos = init_task_state[indices["dishwasher"]][:3]

    # ==========================================
    # 3. 计算各个核心物理部件的空间距离关系
    # ==========================================
    # [门 vs 底座]
    d_b_relations, d_b_delta, d_b_dist = compute_directional_and_distance(
        door_pose[:3], dishwasher_pose[:3], door_extents, dishwasher_extents)
    _, init_d_b_delta, init_d_b_dist = compute_directional_and_distance(
        init_door_pos, init_dishwasher_pos, door_extents, dishwasher_extents)

    # [托盘 vs 底座]
    t_b_relations, t_b_delta, t_b_dist = compute_directional_and_distance(
        tray_pose[:3], dishwasher_pose[:3], tray_extents, dishwasher_extents)
    _, init_t_b_delta, init_t_b_dist = compute_directional_and_distance(
        init_tray_pos, init_dishwasher_pos, tray_extents, dishwasher_extents)

    # [盘子 vs 托盘]  
    p_t_relations, p_t_delta, p_t_dist = compute_directional_and_distance(
        plate_pose[:3], tray_pose[:3], plate_extents, tray_extents)
    _, init_p_t_delta, init_p_t_dist = compute_directional_and_distance(
        init_plate_pos, init_tray_pos, plate_extents, tray_extents)
    if check_universal_contact(plate_pose[:3], plate_radius, tray_pose, tray_extents):
        p_t_relations.append("TOUCHING")

    # [夹爪 vs 盘子] 
    g_p_relations, g_p_delta, g_p_dist = compute_directional_and_distance(
        gripper_pose[:3], plate_pose[:3], extents["gripper_radius"], plate_extents)
    _, init_g_p_delta, init_g_p_dist = compute_directional_and_distance(
        init_gripper_pos, init_plate_pos, extents["gripper_radius"], plate_extents)
    is_touching_plate = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], plate_pose, plate_extents)
    if is_touching_plate: g_p_relations.append("TOUCHING")

    # [夹爪 vs 门] 引导第一阶段
    g_d_relations, g_d_delta, g_d_dist = compute_directional_and_distance(
        gripper_pose[:3], door_pose[:3], extents["gripper_radius"], door_extents)
    _, init_g_d_delta, init_g_d_dist = compute_directional_and_distance(
        init_gripper_pos, init_door_pos, extents["gripper_radius"], door_extents)
    is_touching_door = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], door_pose, door_extents)
    if is_touching_door: g_d_relations.append("TOUCHING")

    # [夹爪 vs 托盘] 引导第二阶段
    g_t_relations, g_t_delta, g_t_dist = compute_directional_and_distance(
        gripper_pose[:3], tray_pose[:3], extents["gripper_radius"], tray_extents)
    _, init_g_t_delta, init_g_t_dist = compute_directional_and_distance(
        init_gripper_pos, init_tray_pos, extents["gripper_radius"], tray_extents)
    is_touching_tray = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], tray_pose, tray_extents)
    if is_touching_tray: g_t_relations.append("TOUCHING")

    # # ==========================================
    # # 4. 抓取判定 (针对任务全过程中需要夹爪操作的物体)
    # # ========================================== 
    # is_holding_door = False
    # is_holding_tray = False
    # is_holding_plate = False

    # # 阶段 1：门还没开，只能抓门把手！
    # if not is_door_open:
    #     is_holding_door = check_holding(is_touching_door, g_d_dist, "door", g_d_relations, gripper_open, indices, prev_obs_dict)
        
    # # 阶段 2：门开了，但托盘没拉出来，只能抓托盘！
    # elif is_door_open and not is_tray_pulled:
    #     is_holding_tray = check_holding(is_touching_tray, g_t_dist, "tray", g_t_relations, gripper_open, indices, prev_obs_dict)
        
    # # 阶段 3：门开了且托盘拉出来了，才能抓盘子！
    # elif is_door_open and is_tray_pulled:
    #     is_holding_plate = check_holding(is_touching_plate, g_p_dist, "plate", g_p_relations, gripper_open, indices, prev_obs_dict)

    # # 只有经过因果锁校验的 HOLDING，才被允许写入图谱
    # if is_holding_plate: g_p_relations.append("HOLDING")
    # if is_holding_door: g_d_relations.append("HOLDING")
    # if is_holding_tray: g_t_relations.append("HOLDING")

    # ==========================================
    # 5. 长序列宏观意图状态机 (State Machine)   几何判断
    # ==========================================
    # a. 门的状态 (Door vs Base) 用门自身的高度/长度来计算动态阈值
    delta_d_b = d_b_dist - init_d_b_dist
    is_door_open = prev_obs_dict.get("door_opened", False) if prev_obs_dict else False
    if delta_d_b > DOOR_OPEN_DELTA:
        is_door_open = True
    # 失败感知：如果门被严重推回 (退回超过 5cm)，解除完成状态
    elif is_door_open and delta_d_b < DOOR_OPEN_DELTA - 0.05: 
        is_door_open = False
    obs_dict["door_opened"] = is_door_open

    if is_door_open:
        d_b_relations.append("OPENED") 
    elif delta_d_b > DOOR_AROUND_DELTA:
        d_b_relations.append("AROUND")

    # b. 托盘状态：纯粹依赖位移增量
    delta_t_b = t_b_dist - init_t_b_dist
    is_tray_pulled = prev_obs_dict.get("tray_pulled", False) if prev_obs_dict else False
    if delta_t_b > TRAY_PULL_DELTA:
        is_tray_pulled = True
    # 失败感知：托盘被严重推回 (超过 5cm)
    elif is_tray_pulled and delta_t_b < TRAY_PULL_DELTA - 0.05:
        is_tray_pulled = False
    obs_dict["tray_pulled"] = is_tray_pulled

    if is_tray_pulled:
        t_b_relations.append("OUTSIDE")
    elif delta_t_b > TRAY_AROUND_DELTA:
        t_b_relations.append("AROUND")
    
    tray_radius_eff = np.max(tray_extents) * 0.1  # 有效半径
    is_tray_inside_geo = check_inside_containment(
        pos_obj=tray_pose[:3], radius_obj=tray_radius_eff, 
        pose_container_7d=dishwasher_pose, extents_container=dishwasher_extents, top_open=False
    )
    # 只要几何中心还在盒子里，就认为是 INSIDE
    if is_tray_inside_geo and (not is_tray_pulled):
        t_b_relations.append("INSIDE")

    # c. 盘子状态：纯粹依赖位移增量
    is_plate_removed = prev_obs_dict.get("plate_removed", False) if prev_obs_dict else False
    delta_p_t = p_t_dist - init_p_t_dist
    # 盘子不仅要离初始位置远，更重要的是它得“被提起来” (相对托盘的 Z 轴高度升高)
    # 取 Z 轴差值 (注意仿真器坐标系，通常 index 2 是 Z 轴)
    plate_z = plate_pose[2]
    tray_z = tray_pose[2]
    z_diff = plate_z - tray_z
    # if delta_p_t > PLATE_DONE_DELTA and z_diff > 0.03:
    #     is_plate_removed = True
    # 失败轨迹标注：加入因果逻辑锁 (必须满足：门已开 且 托盘已拉出)
    if is_door_open and is_tray_pulled and delta_p_t > PLATE_DONE_DELTA and z_diff > 0.03:
        is_plate_removed = True
 
    # 增加失败感知：如果盘子掉落，距离缩水 (比如掉回托盘上)，必须解除拿出状态！
    elif is_plate_removed and (z_diff < 0.02 or delta_p_t < PLATE_DONE_DELTA - 0.03): 
        is_plate_removed = False
    obs_dict["plate_removed"] = is_plate_removed

    if is_plate_removed:
        p_t_relations.append("OUTSIDE")
    elif delta_p_t > PLATE_AROUND_DELTA:
        p_t_relations.append("AROUND")

    # ==========================================
    # 4. 抓取判定 (针对任务全过程中需要夹爪操作的物体)
    # ==========================================  
    is_holding_door = False
    is_holding_tray = False
    is_holding_plate = False

    # 权限 1：任何时候都可以抓门把手 (比如开门、或者没拉好继续拉)
    is_holding_door = check_holding(is_touching_door, g_d_dist, "door", g_d_relations, gripper_open, indices, prev_obs_dict)
        
    # 权限 2：只有门开了，托盘暴露出来，才【允许】发生抓取托盘的物理接触
    if is_door_open:
        is_holding_tray = check_holding(is_touching_tray, g_t_dist, "tray", g_t_relations, gripper_open, indices, prev_obs_dict)
        
    # 权限 3：只有门开了 且 托盘拉出来了，盘子暴露出来，才【允许】发生抓取盘子的物理接触
    if is_door_open and is_tray_pulled:
        is_holding_plate = check_holding(is_touching_plate, g_p_dist, "plate", g_p_relations, gripper_open, indices, prev_obs_dict)

    # 只有经过因果锁校验的 HOLDING，才被允许写入图谱
    if is_holding_plate: g_p_relations.append("HOLDING")
    if is_holding_door: g_d_relations.append("HOLDING")
    if is_holding_tray: g_t_relations.append("HOLDING")

    # 从最终阶段往前反推动作意图
    if is_plate_removed:
        action_type = "done"
    elif is_door_open and is_tray_pulled:
        action_type = "place" if is_holding_plate else "pick"
    elif is_door_open and not is_tray_pulled:
        action_type = "pull"
    else:
        action_type = "open"
 
    # ==========================================
    # 6. 组装 MomaGraph 边结构
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
    # 7. 动态注意力路由 (Dynamic Attention Routing)
    #  根据当前任务阶段，只聚焦最核心的 2 条边
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
    # ==========================================
    # 1. 解析所有核心实体的尺寸
    # ==========================================
    cup_extents = extents.get("cup_size", [0.1, 0.1, 0.1])
    fridge_extents = extents.get("fridge_size", [0.5, 0.5, 0.5])
    tongue_extents = extents.get("tongue_size", [0.05, 0.1, 0.05])

    # 提取 Z 轴半高 (中心点到顶/底的距离)
    cup_half_z = cup_extents[2] / 2.0
    tongue_half_z = tongue_extents[2] / 2.0
    # 提取杯子在 XY 平面上的近似半径 (取宽深的较大值的一半)
    cup_radius_xy = max(cup_extents[0], cup_extents[1]) / 2.0

    # 定义任务相关的物理临界阈值
    # 理论上的完美 Z 轴中心距离 (杯子顶部紧贴 tongue 底部)
    PERFECT_Z_DIST = cup_half_z + tongue_half_z
    TONGUE_PUSH_DELTA = 0.015   # 接冰口开关(tongue)发生微小位移即视为被 push
    # 设定允许的误差宽容度 (Margin)
    Z_MARGIN = 0.005   # 高度允许 0.5cm 的上下浮动空间
    XY_MARGIN = 0.005  # 水平允许杯子中心偏离 0.5cm
    # 目标区域 (TARGET ZONE -> UNDER)
    Z_TARGET_TOLERANCE = PERFECT_Z_DIST + Z_MARGIN
    XY_TARGET_RADIUS = cup_radius_xy + XY_MARGIN
    # 靠近区域 (AROUND ZONE -> AROUND)
    # 比目标区域再往外延展 5cm
    AROUND_EXTENSION = 0.05
    Z_AROUND_TOLERANCE = Z_TARGET_TOLERANCE + AROUND_EXTENSION
    XY_AROUND_RADIUS = XY_TARGET_RADIUS + AROUND_EXTENSION

    # ==========================================
    # 2. 解析所有实体的位姿
    # ==========================================
    cup_pose = task_state[indices["cup"]]
    tongue_pose = task_state[indices["tongue"]]
    joint_pose = task_state[indices["joint"]] 

    init_cup_pos = init_task_state[indices["cup"]][:3]
    init_tongue_pos = init_task_state[indices["tongue"]][:3]
    init_joint_pos = init_task_state[indices["joint"]][:3]
    
    # ==========================================
    # 3. 计算各个核心物理部件的空间距离关系
    # ==========================================
    # [夹爪 vs 杯子] 
    g_c_relations, g_c_delta, g_c_dist = compute_directional_and_distance(
        gripper_pose[:3], cup_pose[:3], extents["gripper_radius"], cup_extents)
    _, init_g_c_delta, init_g_c_dist = compute_directional_and_distance(
        init_gripper_pos, init_cup_pos, extents["gripper_radius"], cup_extents)
    
    is_touching_cup = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], cup_pose, cup_extents)
    if is_touching_cup: g_c_relations.append("TOUCHING")

    # [杯子 vs 接冰口开关 (Tongue)]
    c_t_relations, c_t_delta, c_t_dist = compute_directional_and_distance(
        cup_pose[:3], tongue_pose[:3], cup_extents, tongue_extents)
    _, init_c_t_delta, init_c_t_dist = compute_directional_and_distance(
        init_cup_pos, init_tongue_pos, cup_extents, tongue_extents)
    
    is_touching_tongue = check_universal_contact(cup_pose[:3], np.min(cup_extents)/2.0, tongue_pose, tongue_extents)
    if is_touching_tongue: c_t_relations.append("TOUCHING")

    # ==========================================
    # 4. 抓取与物理状态判定
    # ==========================================
    is_holding_cup = check_holding(is_touching_cup, g_c_dist, "cup", g_c_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_cup: g_c_relations.append("HOLDING")

    # 判断接冰片(tongue)或关节(joint)是否被推开
    tongue_movement = np.linalg.norm(tongue_pose[:3] - init_tongue_pos)
    # 计算一个 0~1 的推压比例 
    tongue_push_ratio = min(1.0, tongue_movement / TONGUE_PUSH_DELTA)
    is_tongue_pushed = prev_obs_dict.get("tongue_pushed", False) if prev_obs_dict else False
    if tongue_movement > TONGUE_PUSH_DELTA:
        is_tongue_pushed = True
        
    obs_dict["tongue_pushed"] = is_tongue_pushed # 存入记忆
    # joint_movement = np.linalg.norm(joint_pose[:3] - init_joint_pos)
    # is_tongue_pushed = bool(tongue_movement > TONGUE_PUSH_DELTA or joint_movement > TONGUE_PUSH_DELTA)
    
    if is_tongue_pushed:
        c_t_relations.append("PUSHED")

    # 纯几何目标区域判定 
    # 分离计算水平面(XY)的对齐距离 和 垂直面(Z)的高度差
    xy_dist = np.linalg.norm(cup_pose[:2] - tongue_pose[:2])
    z_diff = abs(cup_pose[2] - tongue_pose[2])
    # 防止杯壁摩擦导致的 UNDER/AROUND 闪烁
    has_been_under = prev_obs_dict.get("cup_under", False) if prev_obs_dict else False
    is_cup_at_target = bool(xy_dist <= XY_TARGET_RADIUS and z_diff <= Z_TARGET_TOLERANCE)
    if is_cup_at_target:
        has_been_under = True #  表达进入接冰区域
    elif xy_dist > XY_TARGET_RADIUS + 0.03 or z_diff > Z_TARGET_TOLERANCE + 0.03:
        has_been_under = False
        
    obs_dict["cup_under"] = has_been_under

    if has_been_under:
        c_t_relations.append("UNDER") 
        c_t_relations.append("AROUND")
    elif bool(xy_dist <= XY_AROUND_RADIUS and z_diff <= Z_AROUND_TOLERANCE):
        c_t_relations.append("AROUND")

    # ==========================================
    # 5. 长序列宏观意图状态机 (State Machine)
    # ==========================================
    # 任务成功标准：杯子稳稳处于接冰口下方，且接冰片被按压过
    if has_been_under and is_tongue_pushed:
        action_type = "done"
    elif is_holding_cup:
        action_type = "place"
    else:
        # 还没抓稳杯子
        action_type = "pick"
    
    # ==========================================
    # 6. 组装 MomaGraph 边结构
    # ==========================================
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
    # 1. 解析所有核心实体的尺寸 & 动态符号化参数
    # ==========================================
    stick_extents = extents.get("stick_size", [0.03, 0.03, 0.4]) # 冰球棍通常较长
    ball_extents = extents.get("ball_size", [0.04, 0.04, 0.04])
    goal_extents = extents.get("goal_size", [0.3, 0.2, 0.2])     # 球门

    ball_radius = max(ball_extents) / 2.0
    
    # 动态阈值推导
    goal_depth = min(goal_extents[0], goal_extents[1]) 
    GOAL_INSIDE_RADIUS_2D = (goal_depth / 2.0) + 0.01  # 给 1cm 的容差
    GOAL_AROUND_RADIUS_2D = GOAL_INSIDE_RADIUS_2D + 0.15 # AROUND 区大一点，15cm

    # ==========================================
    # 2. 解析所有实体的位姿
    # ==========================================
    stick_pose = task_state[indices["stick"]]
    ball_pose = task_state[indices["ball"]]
    goal_pose = task_state[indices["goal"]] 

    init_stick_pos = init_task_state[indices["stick"]][:3]
    init_ball_pos = init_task_state[indices["ball"]][:3]
    init_goal_pos = init_task_state[indices["goal"]][:3]

    # ==========================================
    # 3. 计算级联物理部件的空间距离关系
    # ==========================================
    # [Edge 1: 夹爪 vs 冰球棍] 
    g_s_relations, g_s_delta, g_s_dist = compute_directional_and_distance(
        gripper_pose[:3], stick_pose[:3], extents["gripper_radius"], stick_extents)
    _, init_g_s_delta, init_g_s_dist = compute_directional_and_distance(
        init_gripper_pos, init_stick_pos, extents["gripper_radius"], stick_extents)
    
    is_touching_stick = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], stick_pose, stick_extents)
    if is_touching_stick: g_s_relations.append("TOUCHING")

    # [Edge 2: 冰球棍 vs 冰球]
    s_b_relations, s_b_delta, s_b_dist = compute_directional_and_distance(
        stick_pose[:3], ball_pose[:3], stick_extents, ball_extents)
    _, init_s_b_delta, init_s_b_dist = compute_directional_and_distance(
        init_stick_pos, init_ball_pos, stick_extents, ball_extents)
    
    # 因为球棍较长，接触判定可以稍微放宽容差
    is_touching_ball = check_universal_contact(stick_pose[:3], np.max(stick_extents)/2.0 - 0.05, ball_pose, ball_extents)
    if is_touching_ball: s_b_relations.append("TOUCHING")

    # [Edge 3: 冰球 vs 球门]
    b_g_relations, b_g_delta, b_g_dist = compute_directional_and_distance(
        ball_pose[:3], goal_pose[:3], ball_extents, goal_extents)
    _, init_b_g_delta, init_b_g_dist = compute_directional_and_distance(
        init_ball_pos, init_goal_pos, ball_extents, goal_extents)
    if check_universal_contact(ball_pose[:3], ball_radius, goal_pose, goal_extents): b_g_relations.append("TOUCHING")

    # [Edge 4: 冰球棍 vs 球门] (专供进度条计算器使用的虚边)
    dist_ball_to_goal = np.linalg.norm(b_g_delta[:2])
    init_ball_to_goal = np.linalg.norm(init_b_g_delta[:2])
    # 持久化记忆锁，判断球是否【曾经】被有效击打过
    has_been_hit = prev_obs_dict.get("ball_has_been_hit", False) if prev_obs_dict else False
    if not has_been_hit and np.linalg.norm(ball_pose[:3] - init_ball_pos) > 0.03: # 阈值稍微放宽到 3cm，防底噪
        has_been_hit = True
    obs_dict["ball_has_been_hit"] = has_been_hit
    # 一旦球被击打过，has_been_hit 永久为 True，棍找球的距离永久为 0
    # 这样就算球在中途停下，进度也不会倒退
    dist_stick_to_ball = np.linalg.norm(s_b_delta[:2]) if not has_been_hit else 0.0 
    init_stick_to_ball = np.linalg.norm(init_s_b_delta[:2])
    current_magic_dist = dist_ball_to_goal + dist_stick_to_ball
    init_magic_dist = init_ball_to_goal + init_stick_to_ball

    # 使用 2D 平面(XY)距离来判定进球
    ball_xy = ball_pose[:2]
    goal_xy = goal_pose[:2]
    # 计算平面欧几里得距离
    dist_xy = np.linalg.norm(ball_xy - goal_xy)
    is_ball_in_goal = bool(dist_xy <= GOAL_INSIDE_RADIUS_2D)
    
    has_been_around_goal = prev_obs_dict.get("ball_around_goal", False) if prev_obs_dict else False
    if dist_xy <= GOAL_AROUND_RADIUS_2D:
        has_been_around_goal = True
    elif dist_xy > GOAL_AROUND_RADIUS_2D + 0.05: # 退回 5cm 外才解除
        has_been_around_goal = False

    obs_dict["ball_around_goal"] = has_been_around_goal

    if is_ball_in_goal:
        b_g_relations.append("INSIDE")
        b_g_relations.append("AROUND") # 补齐语义
    elif has_been_around_goal:
        b_g_relations.append("AROUND")
    else:
        b_g_relations.append("OUTSIDE")

    # ==========================================
    # 4. 抓取与动态物理状态判定
    # ==========================================
    # 校验是否稳固抓取球棍
    # is_holding_stick = check_holding(is_touching_stick, g_s_dist, "stick", g_s_relations, gripper_open, indices, prev_obs_dict) 
    is_holding_stick = check_holding(is_touching_stick, 0.0, "stick", g_s_relations, gripper_open, indices, prev_obs_dict)
    
    if is_holding_stick: g_s_relations.append("HOLDING")

    # ==========================================
    # 5. 长序列宏观意图状态机 (State Machine)
    # ==========================================
    if is_ball_in_goal:
        action_type = "done"
    elif is_holding_stick:
        action_type = "hit" 
    else:
        action_type = "pick"

    # ==========================================
    # 6. 组装 MomaGraph 边结构
    # ==========================================
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
        "functional_relationship": "activate", # 或者用 strike/hit
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

    # ==========================================
    # 7. 动态注意力路由 (Dynamic Attention Routing)
    # ==========================================
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
    
    # ==========================================
    # 1. 动态寻找本回合的真实目标柱子 (Target Pillar)
    # ==========================================
    init_sc_pos = init_task_state[indices["success_centre"]][:3]
    
    pillar_keys = ["pillar0", "pillar1", "pillar2"]
    target_pillar_key = None
    min_dist = float('inf')
    
    for key in pillar_keys:
        p_pos = init_task_state[indices[key]][:3]
        # 计算 3 个柱子与 success_centre 的 XY 平面距离
        dist = np.linalg.norm(p_pos[:2] - init_sc_pos[:2])
        if dist < min_dist:
            min_dist = dist
            target_pillar_key = key

    # ==========================================
    # 2. 解析核心实体的尺寸 & 动态符号化参数
    # ==========================================
    ring_extents = extents.get("ring_size", [0.06, 0.06, 0.02])
    pillar_extents = extents.get(f"{target_pillar_key}_size", [0.04, 0.04, 0.15])
    
    # 方环套方柱：只要环的内径与柱子外径在 XY 平面上重合即可
    # 用柱子的宽度一半加上 1cm 的宽松对齐容差
    TARGET_XY_RADIUS = max(pillar_extents[0], pillar_extents[1]) / 2.0 + 0.01
    AROUND_XY_RADIUS = TARGET_XY_RADIUS + 0.001 # 准备插入的悬停区域

    # ==========================================
    # 3. 解析所有实体的位姿
    # ==========================================
    ring_pose = task_state[indices["ring"]]
    pillar_pose = task_state[indices[target_pillar_key]]
    
    init_ring_pos = init_task_state[indices["ring"]][:3]
    init_pillar_pos = init_task_state[indices[target_pillar_key]][:3]

    # ==========================================
    # 4. 计算空间距离关系 (Edge 构建)
    # ==========================================
    # [Edge 1: 夹爪 vs 方环] 
    g_r_relations, g_r_delta, g_r_dist = compute_directional_and_distance(
        gripper_pose[:3], ring_pose[:3], extents["gripper_radius"], ring_extents)
    _, init_g_r_delta, init_g_r_dist = compute_directional_and_distance(
        init_gripper_pos, init_ring_pos, extents["gripper_radius"], ring_extents)
        
    is_touching_ring = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], ring_pose, ring_extents)
    if is_touching_ring: g_r_relations.append("TOUCHING")
    
    # [Edge 2: 方环 vs 目标柱子]
    r_p_relations, r_p_delta, r_p_dist = compute_directional_and_distance(
        ring_pose[:3], pillar_pose[:3], ring_extents, pillar_extents)
    _, init_r_p_delta, init_r_p_dist = compute_directional_and_distance(
        init_ring_pos, init_pillar_pos, ring_extents, pillar_extents)
    is_touching_pillar = check_universal_contact(ring_pose[:3], np.min(ring_extents)/2.0, pillar_pose, pillar_extents)
    if is_touching_pillar: r_p_relations.append("TOUCHING")

    # 抓取状态校验
    is_holding_ring = check_holding(is_touching_ring, g_r_dist, "ring", g_r_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_ring: g_r_relations.append("HOLDING")

    # ==========================================
    # 5. 几何状态机判定 (XY对齐 + Z轴插入)
    # ==========================================
    dist_xy = np.linalg.norm(r_p_delta[:2])
    is_aligned_xy = bool(dist_xy <= TARGET_XY_RADIUS)
    # 插入判定：XY对齐，且方环的高度落入柱子的底座附近 (给了 4cm 的悬浮容错)
    is_inserted = is_aligned_xy and bool(ring_pose[2] < pillar_pose[2] + 0.04) 

    # 计算环的等效半径（用于传给函数，尽管这里会被兜底机制覆盖）
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
        min_radius=AROUND_XY_RADIUS, # <--- 核心：注入 2D 容差
        min_height=0.02 # Z轴进场跑道长度设为 2cm
    ):
        r_p_relations.append("AROUND")

    # 宏观意图
    if is_inserted:
        action_type = "done"
    elif is_holding_ring:
        action_type = "place"
    else:
        action_type = "pick"

    # ==========================================
    # 6. 组装 MomaGraph 边结构
    # ==========================================
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
            # ProgressEstimator 会自动提取 XY 用于 move，提取 Z 用于 place
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
            "target_pillar_id": target_pillar_key, # 记录这回合到底套的是哪根柱子
            "target_xy_radius": float(TARGET_XY_RADIUS),
            "around_xy_radius": float(AROUND_XY_RADIUS)
        }
    }


def _extract_meat_on_grill(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                           indices, extents, instructions, prev_obs_dict, obs_dict):
    # ==========================================
    # 1. 动态目标定位 (NLP Instruction Parsing)
    # ==========================================
    # 从第一条语言指令中判断当前 episode 要求烤什么肉
    instruction_str = instructions[0].lower()
    target_meat_key = "chicken" if "chicken" in instruction_str else "steak"

    # ==========================================
    # 2. 解析核心实体的尺寸 & 动态符号化参数
    # ==========================================
    meat_extents = extents.get(f"{target_meat_key}_size", [0.05, 0.05, 0.02])
    grill_extents = extents.get("grill_size", [0.25, 0.25, 0.05])
    
    # 烤架表面很大，目标是放在上面。定义 XY 对齐容差
    # 取烤架较短的边的一半，减去肉的半径，保证肉落在网格内
    TARGET_XY_RADIUS = 0.03 
    AROUND_XY_RADIUS = TARGET_XY_RADIUS + 0.03 # 放宽 3cm 作为 AROUND 进场跑道

    # ==========================================
    # 3. 解析所有实体的位姿
    # ==========================================
    meat_pose = task_state[indices[target_meat_key]]
    grill_pose = task_state[indices["grill"]]
    success_pose = task_state[indices["success"]]
    
    init_meat_pos = init_task_state[indices[target_meat_key]][:3]
    init_grill_pos = init_task_state[indices["grill"]][:3]
    init_success_pos = init_task_state[indices["success"]][:3]

    # ==========================================
    # 4. 计算空间距离关系 (Edge 构建)
    # ==========================================
    # [Edge 1: 夹爪 vs 肉] 
    g_m_relations, g_m_delta, g_m_dist = compute_directional_and_distance(
        gripper_pose[:3], meat_pose[:3], extents["gripper_radius"], meat_extents)
    _, init_g_m_delta, init_g_m_dist = compute_directional_and_distance(
        init_gripper_pos, init_meat_pos, extents["gripper_radius"], meat_extents)
        
    is_touching_meat = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], meat_pose, meat_extents)
    if is_touching_meat: g_m_relations.append("TOUCHING")
    
    # 抓取状态校验
    is_holding_meat = check_holding(is_touching_meat, g_m_dist, target_meat_key, g_m_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_meat: g_m_relations.append("HOLDING")

    # [Edge 2: 肉 vs 烤架]
    m_g_relations, m_g_delta, m_g_dist = compute_directional_and_distance(
        meat_pose[:3], success_pose[:3], meat_extents, [0.01, 0.01, 0.01]) # 传感器极小，给个 1cm 假尺寸
    _, init_m_g_delta, init_m_g_dist = compute_directional_and_distance(
        init_meat_pos, init_success_pos, meat_extents, [0.01, 0.01, 0.01])
        
    # ==========================================
    # 5. 几何状态机判定 (XY悬停 + Z轴单边贴合放置)
    # ==========================================
    dist_xy = np.linalg.norm(m_g_delta[:2])
    dist_z = abs(m_g_delta[2])
    
    # --- AROUND 状态防抖 ---
    has_been_around = prev_obs_dict.get("meat_around", False) if prev_obs_dict else False
    if dist_xy <= AROUND_XY_RADIUS:
        has_been_around = True
    elif dist_xy > AROUND_XY_RADIUS + 0.04: # 退回安全区外 4cm 再解除
        has_been_around = False

    # --- ON (放置) 状态防抖 ---
    has_been_on = prev_obs_dict.get("meat_on", False) if prev_obs_dict else False
    
    # 严格对齐且贴合时，判定为 ON
    if dist_xy <= TARGET_XY_RADIUS and dist_z <= 0.05:
        has_been_on = True
    # 💥 失败感知：如果肉滑出烤架太远 (超出目标容差 4cm) 或被高高弹起，解除 ON 状态
    elif dist_xy > TARGET_XY_RADIUS + 0.04 or dist_z > 0.08:
        has_been_on = False

    # 关系装配
    if has_been_around: 
        m_g_relations.append("AROUND")
    
    is_on_grill = has_been_on
    if is_on_grill:
        m_g_relations.append("ON")
        m_g_relations.append("TOUCHING")
 
    # 宏观意图判定
    if is_on_grill and not is_holding_meat: # 必须松开夹爪 (NothingGrasped) 才算完成
        action_type = "done"
    elif is_holding_meat:
        action_type = "place"
    else:
        action_type = "pick"

    # 将记忆存入 obs_dict
    obs_dict["meat_around"] = has_been_around
    obs_dict["meat_on"] = has_been_on
    
    # ==========================================
    # 6. 组装 MomaGraph 边结构  
    # ==========================================
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
    # ==========================================
    # 1. 提取物理尺寸
    # ==========================================
    hanger_extents = extents.get("hanger_size", [0.4, 0.02, 0.2]) 
    hanger_pose = task_state[indices["hanger"]]
    init_hanger_pos = init_task_state[indices["hanger"]][:3]

    # ==========================================
    # 2. 动态寻找目标架子 (Target Rack)
    # ==========================================
    # 比较初始时刻衣架与两个架子的 XY 距离，离得远的就是要挂过去的目标架子
    init_rack1_pos = init_task_state[indices["rack1"]][:3]
    init_rack2_pos = init_task_state[indices["rack2"]][:3]
    
    dist_to_rack1 = np.linalg.norm(init_hanger_pos[:2] - init_rack1_pos[:2])
    dist_to_rack2 = np.linalg.norm(init_hanger_pos[:2] - init_rack2_pos[:2])
    
    target_rack_key = "rack1" if dist_to_rack1 > dist_to_rack2 else "rack2"
    target_rack_extents = extents.get(f"{target_rack_key}_size", [0.45, 1.32, 1.45])
    target_rack_pose = task_state[indices[target_rack_key]]
    init_target_rack_pos = init_task_state[indices[target_rack_key]][:3]

    # 动态设定对齐容差  
    TARGET_XY_RADIUS = 0.15
    AROUND_XY_RADIUS = 0.25 

    # ==========================================
    # 3. 计算夹爪与衣架的物理关系
    # ==========================================
    g_h_relations, g_h_delta, g_h_dist = compute_directional_and_distance(
        gripper_pose[:3], hanger_pose[:3], extents["gripper_radius"], hanger_extents)
    _, init_g_h_delta, init_g_h_dist = compute_directional_and_distance(
        init_gripper_pos, init_hanger_pos, extents["gripper_radius"], hanger_extents)
        
    is_touching_hanger = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], hanger_pose, hanger_extents)
    if is_touching_hanger: g_h_relations.append("TOUCHING")
    
    is_holding_hanger = check_holding(is_touching_hanger, g_h_dist, "hanger", g_h_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_hanger: g_h_relations.append("HOLDING")

    # ==========================================
    # 4. 计算衣架与目标架子的物理关系
    # ==========================================
    h_r_relations, h_r_delta, h_r_dist = compute_directional_and_distance(
        hanger_pose[:3], target_rack_pose[:3], hanger_extents, target_rack_extents) 
    _, init_h_r_delta, init_h_r_dist = compute_directional_and_distance(
        init_hanger_pos, init_target_rack_pos, hanger_extents, target_rack_extents)

    # ==========================================
    # 5. 几何状态机判定 
    # ==========================================
    # 提取衣架距离目标架子中心的 X 和 Y 绝对偏差
    dx, dy = abs(h_r_delta[0]), abs(h_r_delta[1])
    
    # 高度判定：基于两个架子等高，只要回到接近初始高度就算落下
    # 同时判断是否被抬起脱离了原架子
    z_lift = hanger_pose[2] - init_hanger_pos[2]
    is_dropped_back = bool(z_lift <= 0.02)
    is_lifted = not is_dropped_back 

    is_aligned_xy = bool(dx <= TARGET_XY_RADIUS) and bool(dy <= TARGET_XY_RADIUS)
    is_around_xy = bool(dx <= AROUND_XY_RADIUS) and bool(dy <= AROUND_XY_RADIUS) 
    
    # 激活 AROUND
    if is_around_xy and is_lifted:
        h_r_relations.append("AROUND")

    # 放置判定：XY 对齐了目标架子中心，并且放下来了
    is_on_rack = is_aligned_xy and is_dropped_back
    
    if is_on_rack:
        h_r_relations.append("ON")
        h_r_relations.append("TOUCHING")
        if "AROUND" not in h_r_relations:
            h_r_relations.append("AROUND") 

    # ==========================================
    # 6. 宏观意图判定
    # ==========================================
    if is_on_rack and not is_holding_hanger: 
        action_type = "done"
    elif is_holding_hanger:
        action_type = "place"
    else:
        action_type = "pick"

    # ==========================================
    # 7. 组装 MomaGraph
    # ==========================================
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
            # 这里的 delta_3d 完美反映了衣架向目标架子靠近的物理真实过程
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
    # ==========================================
    # 1. 动态目标定位 (NLP Instruction Parsing)
    # ==========================================
    instruction_str = instructions[0].lower()
    if "bottom" in instruction_str:
        target_drawer_key = "bottom"
    elif "middle" in instruction_str:
        target_drawer_key = "middle"
    else:
        target_drawer_key = "top"

    # ==========================================
    # 2. 解析核心实体的尺寸
    # ==========================================
    drawer_extents = extents.get(f"{target_drawer_key}_size", [0.4, 0.2, 0.4])
    drawer_pose = task_state[indices[target_drawer_key]]
    init_drawer_pos = init_task_state[indices[target_drawer_key]][:3]

    # 官方规定的成功位移是拉出 15cm
    PULL_SUCCESS_DIST = 0.22
    AROUND_PULL_DIST = 0.15 # 拉出 15cm 时触发 AROUND

    # ==========================================
    # 4. 计算空间距离关系 (Edge 构建)
    # ========================================== 
    g_d_relations, g_d_delta, g_d_dist = compute_directional_and_distance(
        gripper_pose[:3], drawer_pose[:3], extents["gripper_radius"], drawer_extents)
    _, init_g_d_delta, init_g_d_dist = compute_directional_and_distance(
        init_gripper_pos, init_drawer_pos, extents["gripper_radius"], drawer_extents)
        
    is_touching = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], drawer_pose, drawer_extents)
    if is_touching: g_d_relations.append("TOUCHING")
    
    # 校验是否稳固抓取了把手
    is_holding = check_holding(is_touching, g_d_dist, target_drawer_key, g_d_relations, gripper_open, indices, prev_obs_dict)
    if is_holding: g_d_relations.append("HOLDING")

    # ==========================================
    # 计算 Edge 2: drawer -> opened_drawer (进度)
    # ==========================================
    # 计算抽屉相对于初始位置拉开了多远 (L2 范数是最稳健的进度指标)
    pulled_dist = np.linalg.norm(drawer_pose[:3] - init_drawer_pos)

    d_o_relations = [] 
    # is_around = bool(pulled_dist >= AROUND_PULL_DIST)
    is_opened = bool(pulled_dist >= PULL_SUCCESS_DIST)
    is_closed = bool(pulled_dist == 0)

    has_been_around = prev_obs_dict.get("drawer_around", False) if prev_obs_dict else False
    if pulled_dist >= AROUND_PULL_DIST:
        has_been_around = True
    elif pulled_dist < AROUND_PULL_DIST - 0.05: # 只有抽屉被推回去 5cm 以上才解除
        has_been_around = False

    if has_been_around:
        d_o_relations.append("AROUND")
    if is_opened:
        d_o_relations.append("OPENED")
    remaining_dist = float(max(0.0, PULL_SUCCESS_DIST - pulled_dist))
    obs_dict["drawer_around"] = has_been_around
     
    # 宏观意图判定
    if is_opened:
        action_type = "done"
    elif is_holding or pulled_dist > 0.05:
        action_type = "open" # 对应拉抽屉动作
    else:
        action_type = "pick"
 
    # ==========================================
    # 6. 组装 MomaGraph
    # ==========================================
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
    
    # --- Edge 2: 状态边 (任务进度) ---
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
        "is_touching": True, # 抽屉永远在框架的滑轨上
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
    # ==========================================
    # 1. 动态目标选择
    # ==========================================
    instruction_str = instructions[0].lower()
    if "left" in instruction_str:
        target_key = "target_left"
    elif "right" in instruction_str:
        target_key = "target_right"
    else:
        target_key = "target_middle"

    # ==========================================
    # 2. 提取位姿
    # ==========================================
    wine_pose = task_state[indices["wine"]]
    # 即便传感器不是 Shape，但 indices 拿到的前 3 位依然是它的 XYZ 中心坐标
    target_pos = task_state[indices[target_key]][:3] 
    
    init_wine_pos = init_task_state[indices["wine"]][:3]
    init_target_pos = init_task_state[indices[target_key]][:3]

    # 解析尺寸
    wine_extents = extents.get("wine_size", [0.08, 0.08, 0.25])
    rack_extents = extents.get("rack_size", [0.2, 0.6, 0.1]) # 假设尺寸

    # ==========================================
    # 3. 计算边 1: gripper -> wine
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
    # 4. 计算边 2: wine -> target (通过 XYZ 坐标计算)
    # ==========================================
    # 计算当前酒瓶中心到目标传感器的水平和垂直距离
    curr_delta = wine_pose[:3] - target_pos
    dist_xy = np.linalg.norm(curr_delta[:2])
    dist_z = abs(curr_delta[2])
    
    # 初始的水平距离（作为进度的分母）
    init_delta = init_wine_pos - init_target_pos
    init_dist_xy = np.linalg.norm(init_delta[:2])

    # 判定阈值
    TARGET_XY_RADIUS = 0.15 # 水平对齐容差
    AROUND_XY_RADIUS = 0.25 # 接近半径
    
    # 动作标志位
    is_lifted = bool(wine_pose[2] > init_wine_pos[2] + 0.05)
    
    w_t_relations = []
    
    # 水平接近判定
    if dist_xy <= AROUND_XY_RADIUS and is_lifted:
        w_t_relations.append("AROUND")

    # 放平判定。由于瓶子旋转了，其 Z 坐标中心会显著下降。
    # 只要水平对齐，且 Z 轴高度接近传感器平面（允许 10cm 误差），即视为放置成功
    is_on_target = bool(dist_xy <= TARGET_XY_RADIUS and dist_z <= 0.10)

    if is_on_target:
        w_t_relations.append("ON")
        w_t_relations.append("TOUCHING")

    # ==========================================
    # 5. 意图与组装
    # ==========================================
    if is_on_target and not is_holding_wine: 
        action_type = "done"
    elif is_holding_wine:
        action_type = "place"
    else:
        action_type = "pick"

    # 构建 MomaGraph
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
    
    # ProgressEstimator 会提取 delta_3d 的范数
    # 为了防止旋转引起的抖动，在 move 阶段（0-85%）强制喂给它一个基于水平位移的单调递减值
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
            "target_id": target_key, # 具体的 left/right 信息存在 thresholds 里备查
            "target_radius": float(TARGET_XY_RADIUS),
            "around_radius": float(AROUND_XY_RADIUS)
        }
    }


def _extract_put_money_in_safe(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                               indices, extents, instructions, prev_obs_dict, obs_dict):
    # ==========================================
    # 1. 动态目标层级解析
    # ==========================================
    instruction_str = instructions[0].lower()
    if "bottom" in instruction_str:
        target_key = "target_bottom"
    elif "middle" in instruction_str:
        target_key = "target_middle"
    else:
        target_key = "target_top"

    # ==========================================
    # 2. 提取位姿与尺寸
    # ==========================================
    money_pose = task_state[indices["money"]]
    target_pos = task_state[indices[target_key]][:3] 
    safe_pose = task_state[indices["safe"]]
    init_money_pos = init_task_state[indices["money"]][:3]
    init_target_pos = init_task_state[indices[target_key]][:3]

    money_extents = extents.get("money_size", [0.15, 0.08, 0.02])
    safe_extents = extents.get("safe_size", [0.4, 0.4, 0.5]) 
    money_radius = max(money_extents[0], money_extents[1]) / 2.0

    # ==========================================
    # 3. 计算边 1: gripper -> money
    # ==========================================
    g_m_relations, g_m_delta, g_m_dist = compute_directional_and_distance(
        gripper_pose[:3], money_pose[:3], extents["gripper_radius"], money_extents)
    _, init_g_m_delta, init_g_m_dist = compute_directional_and_distance(
        init_gripper_pos, init_money_pos, extents["gripper_radius"], money_extents)
        
    is_touching_money = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], money_pose, money_extents)
    if is_touching_money: g_m_relations.append("TOUCHING")
 
    is_holding_raw = check_holding(is_touching_money, 0.0, "money", g_m_relations, gripper_open, indices, prev_obs_dict) 
    is_lifted = bool(money_pose[2] > init_money_pos[2] + 0.02) # 只要离开桌面 2 厘米即可
    
    # 最终判定：必须满足底层 holding 条件，且钞票确实离开了桌面
    is_holding_money = is_holding_raw and is_lifted 
    # is_holding_money = check_holding(is_touching_money, g_m_dist, "money", g_m_relations, gripper_open, indices, prev_obs_dict)
    if is_holding_money: g_m_relations.append("HOLDING")

    # ==========================================
    # 4. 计算边 2: money -> target  
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
    
    #  增加 AROUND 状态的迟滞锁 (防边界横跳)
    has_been_around = prev_obs_dict.get("money_around", False) if prev_obs_dict else False
    if dist_to_target <= AROUND_RADIUS:
        has_been_around = True
    elif dist_to_target > AROUND_RADIUS + 0.05: # 退回 20cm 外才算丢失 AROUND
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
    elif has_been_around: # 使用防抖后的变量
        m_t_relations.append("AROUND")
    else:
        m_t_relations.append("OUTSIDE")

    obs_dict["money_around"] = has_been_around
    obs_dict["money_inside"] = has_been_inside
 
    # ==========================================
    # 5. 意图与组装
    # ==========================================
    if is_inside: 
        action_type = "done"
    elif is_holding_money:
        action_type = "place"
    else:
        action_type = "pick"

    # 构建 MomaGraph
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
            #  维度欺骗   将 3D 欧氏距离复制到 3 个轴。
            # 这样无论 ProgressEstimator 取 X、Y 还是默认的 Z 轴，它拿到的都是绝对递减的三维距离 
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
    # ==========================================
    # 1. 提取实体位姿与尺寸
    # ==========================================
    stick_pose = task_state[indices["stick"]]
    cube_pose = task_state[indices["cube"]]
    success_pose = task_state[indices["success"]]
    
    init_stick_pos = init_task_state[indices["stick"]][:3]
    init_cube_pos = init_task_state[indices["cube"]][:3]
    init_success_pos = init_task_state[indices["success"]][:3]

    stick_extents = extents.get("stick_size", [0.02, 0.02, 0.4])
    cube_extents = extents.get("cube_size", [0.05, 0.05, 0.05])
    target_extents = extents.get("target_size", np.array([0.13, 0.13, 0.01]))
    # --- 动态阈值定义 ---
    goal_depth = min(target_extents[0], target_extents[1]) 
    GOAL_INSIDE_RADIUS_2D = (goal_depth / 2.0) + 0.03  # 给 1cm 的容差
    GOAL_AROUND_RADIUS_2D = GOAL_INSIDE_RADIUS_2D + 0.08 # AROUND 区大一点，15cm

    # ==========================================
    # 2. 计算级联物理部件的空间距离关系 
    # ==========================================
    # Edge 1: gripper -> stick (抓取棍子)
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
    
    # 因为球棍较长，接触判定可以稍微放宽容差
    is_touching_cube = check_universal_contact(stick_pose[:3], np.max(stick_extents)/2.0 - 0.05, cube_pose, cube_extents)
    if is_touching_cube: s_c_relations.append("TOUCHING")

    #  Edge 3: cube -> target_zone (方块进度) 
    c_t_relations, c_t_delta, c_t_dist = compute_directional_and_distance(
        cube_pose[:3], success_pose[:3], cube_extents, [0.01, 0.01, 0.01])
    _, init_c_t_delta, init_c_t_dist = compute_directional_and_distance(
        init_cube_pos, init_success_pos, cube_extents, [0.01, 0.01, 0.01])
    if check_universal_contact(cube_pose[:3], max(cube_extents[0], cube_extents[1]) / 2.0, success_pose, [0.01, 0.01, 0.01]): 
        c_t_relations.append("TOUCHING")

    # [Edge 4: 方块 vs 目标] (专供进度条计算器使用的虚边) 
    dist_cube_to_goal = np.linalg.norm(c_t_delta[:2])
    init_cube_to_goal = np.linalg.norm(init_c_t_delta[:2])
    dist_stick_to_cube = np.linalg.norm(s_c_delta[:2]) if not bool(np.linalg.norm(cube_pose[:3] - init_cube_pos) > 0.02) else 0.0 #  移动超过 2cm
    init_stick_to_cube = np.linalg.norm(init_s_c_delta[:2])
    # 拼装成一维的魔法总距离 
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

    # ==========================================
    # 3. 宏观意图与状态机
    # ==========================================
    if is_inside:
        action_type = "done"
    elif is_holding_stick:
        # 如果拿着棍子，即便没碰到方块，当前的意图也是 drag
        action_type = "drag" 
    else:
        action_type = "pick"

    # ==========================================
    # 4. 组装 Edge 结构
    # ==========================================
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
        "functional_relationship": "activate", # 或者用 strike/hit
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
            # 计算器默认提取 delta_3d 的前两轴算距离，所以我们直接把魔法标量放在 X 轴
            "delta_3d": [float(current_magic_dist), 0.0, 0.0],       
            "dist": float(current_magic_dist),               
            "original_delta_3d": [float(init_magic_dist), 0.0, 0.0],
            "original_dist": float(init_magic_dist) 
        },
        "is_touching": False,
        "is_stable": "INSIDE" in c_t_relations
    }

    # ==========================================
    #  动态注意力路由 (Dynamic Attention Routing)
    # ==========================================
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
    # 物理阈值
    SUCCESS_XY_DIST = 0.073
    AROUND_XY_DIST = 0.075
    LIFT_HEIGHT_THRESHOLD = 0.025 

    # ==========================================
    # 1. 提取位姿 (注意：只信任 spatula 和 success)
    # ==========================================
    spatula_pose = task_state[indices["spatula"]]
    # 强制使用初始位置作为物体的“出生点”锚点
    cube_spawn_pos = init_task_state[indices["cuboid"]][:3]
    # 目标锚点
    success_pose = task_state[indices["success"]][:3]
    
    init_spatula_pos = init_task_state[indices["spatula"]][:3]

    spatula_extents = extents.get("spatula_size", np.array([0.03, 0.12, 0.28])) 
    cube_extents = extents.get("cuboid_size", np.array([0.02, 0.02, 0.02]))
    
    # ==========================================
    # 2. Edge 1: gripper -> spatula  
    # ==========================================
    g_s_relations, g_s_delta, g_s_dist = compute_directional_and_distance(
        gripper_pose[:3], spatula_pose[:3], extents["gripper_radius"], spatula_extents)
    _, init_g_s_delta, init_g_s_dist = compute_directional_and_distance(
        init_gripper_pos, init_spatula_pos, extents["gripper_radius"], spatula_extents)
    
    # 触碰与抓取判定
    is_touching_spatula = bool(g_s_dist < (np.max(spatula_extents)/2.0 + extents["gripper_radius"] + 0.1))
    if is_touching_spatula: g_s_relations.append("TOUCHING")
    is_holding_spatula = is_touching_spatula and bool(gripper_open < 0.1)
    if is_holding_spatula: g_s_relations.append("HOLDING")

    # ==========================================
    # 3. Edge 2: spatula -> cube (工具支撑边) 不参与进度计算
    # ==========================================
    s_c_relations, s_c_delta, s_c_dist = compute_directional_and_distance(
        spatula_pose[:3], cube_spawn_pos, spatula_extents, cube_extents)
    _, init_s_c_delta, init_s_c_dist = compute_directional_and_distance(
        init_spatula_pos, cube_spawn_pos, spatula_extents, cube_extents)
    # 获取历史状态
    has_lifted = prev_obs_dict.get("has_lifted_cube", False) if prev_obs_dict else False
    # 核心推理：如果铲子Z轴显著高于方块出生点，视为完成铲起
    is_lifted = bool(spatula_pose[2] > cube_spawn_pos[2] + LIFT_HEIGHT_THRESHOLD)
    # is_cube_on_spatula = is_lifted and is_holding_spatula and bool(s_c_dist < 0.20)
    if is_lifted and is_holding_spatula and bool(s_c_dist < 0.20):
        # 第一次在出生点附近成功铲起
        is_cube_on_spatula = True
        has_lifted = True
    elif has_lifted and is_holding_spatula:
        #  既然方块坐标是死的，只要之前铲起来了且铲子没掉， 就认为方块在铲子上 
        is_cube_on_spatula = True
    else:
        is_cube_on_spatula = False

    if s_c_dist < 0.20: s_c_relations.append("TOUCHING") # 靠近出生点
    if is_cube_on_spatula: 
        s_c_relations.append("ON") 

    # ==========================================
    # 4. 计算进度 (完全基于铲子的路径) 单调递减的总距离场
    # ==========================================
    # 进度阶段 3：当前铲子(代表方块)距离目标的 XY 平面距离
    dist_to_target_xy = np.linalg.norm(spatula_pose[:2] - success_pose[:2])
    
    #  获取历史状态： 锁死“方块被铲起”的记忆
    has_lifted = prev_obs_dict.get("has_lifted_cube", False) if prev_obs_dict else False
    
    # 更新历史状态 (单向锁：一旦铲起，绝不退回)
    if is_cube_on_spatula: 
        has_lifted = True
    # p1: 手找铲子
    p1_dist = g_s_dist if not is_holding_spatula else 0.0
    # p2: 铲子找物体 (一旦 lift，这一段距离视为归零)
    p2_dist = s_c_dist if (is_holding_spatula and not has_lifted) else 0.0
    # p3: 铲子去终点
    p3_dist = dist_to_target_xy
    current_magic_dist = p1_dist + p2_dist + p3_dist

    init_p3 = np.linalg.norm(cube_spawn_pos[:2] - success_pose[:2]) 
    init_magic_dist = init_g_s_dist + init_s_c_dist + init_p3

    # 将状态存入 obs_dict (只需要存 has_lifted)
    obs_dict["has_lifted_cube"] = has_lifted

    # ==========================================
    # 5. 状态空间关系与意图拦截器
    # ==========================================
    is_near_target = bool(dist_to_target_xy <= AROUND_XY_DIST)
    is_at_target = bool(dist_to_target_xy <= SUCCESS_XY_DIST)
    
    s_t_relations = []
    if is_holding_spatula:
        if is_near_target: s_t_relations.append("AROUND")
        if is_at_target and is_cube_on_spatula: s_t_relations.append("INSIDE")

    # 意图切换
    if "INSIDE" in s_t_relations:
        action_type = "done"
    elif is_holding_spatula: 
        action_type = "scoop" 
    else:
        action_type = "pick"
 
    # ==========================================
    # 6. 组装 Edge (移除死寂的 cube 节点)
    # ==========================================
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
    # 虽不参与进度计算，但保留了任务的真实物理语义 (铲子与方块的互动)
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
    # Edge 3: spatula -> target_zone (对齐 Estimator 的 target_obj)
    edge_st = {
        "functional_relationship": "pair_with",
        "object1": "spatula", "object2": "target_zone",
        "spatial_relations": s_t_relations,
        "geometric_attributes": {
            # 【细节】：将 magic_dist 同时注入 X 轴和 Z 轴
            # 防止 Estimator 因为 drop_axis = 2 而提取到 0 导致进度卡死
            "delta_3d": [float(current_magic_dist), 0.0, float(current_magic_dist)],
            "dist": float(current_magic_dist),
            "original_delta_3d": [float(init_magic_dist), 0.0, float(init_magic_dist)],
            "original_dist": float(init_magic_dist)
        },
        "is_touching": False, "is_stable": (action_type == "done")
    }

    # ==========================================
    # 动态注意力路由
    # ==========================================
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
    # 物理阈值 (棋子较小，容差收紧)
    SUCCESS_XY_DIST = 0.015
    AROUND_XY_DIST = 0.015

    # ==========================================
    # 1. 批量重塑位姿矩阵 (32 个棋子，每个 7 维)
    # ==========================================
    pieces_state = task_state[indices["pieces_slice"]].reshape(32, 7)
    detectors_state = task_state[indices["detectors_slice"]].reshape(32, 7)
    
    init_pieces_state = init_task_state[indices["pieces_slice"]].reshape(32, 7)
    init_detectors_state = init_task_state[indices["detectors_slice"]].reshape(32, 7)

    piece_size = extents.get("chess_piece_size", [0.03, 0.03, 0.06])  # 典型棋子包围盒近似
    sensor_radius = extents.get("success_radius", 0.02)
    target_size = np.array([sensor_radius, sensor_radius, 0.01]) # 棋盘格近似, success 传感器

    # ==========================================
    # 2. 动态锁定活跃棋子 (Active Piece) & 计算全局阶段与原配追踪
    # ==========================================
    # 1. 读取或计算全局总阶段数 (仅在第一帧计算初始有多少个空位)
    # 逻辑：如果一个探测器距离所有棋子都很远，那它就是一个没人占的空位
    if prev_obs_dict and "target_mapping" in prev_obs_dict.get("task_thresholds", {}):
        target_mapping = prev_obs_dict["task_thresholds"]["target_mapping"]
        total_stages = prev_obs_dict["task_thresholds"]["total_stages"]
    else:
        #  只在 0 帧执行一次 
        target_mapping = {}
        empty_detectors = []
        # 找空位
        for d_idx in range(32):
            d_pos = init_detectors_state[d_idx, :2]
            # 算这个探测器离最近的棋子有多远
            min_dist_to_pieces = np.min(np.linalg.norm(init_pieces_state[:, :2] - d_pos, axis=1))
            if min_dist_to_pieces > SUCCESS_XY_DIST:
                empty_detectors.append(d_idx)
        displaced_pieces = []
        for p_idx in range(32):
            p_pos = init_pieces_state[p_idx, :2]
            if np.min(np.linalg.norm(init_detectors_state[:, :2] - p_pos, axis=1)) > SUCCESS_XY_DIST:
                displaced_pieces.append(p_idx)
        # #  RLBench 初始化时，把棋子拿出来的顺序通常对应空位的顺序
        # for i in range(len(displaced_pieces)):
        #     target_mapping[str(displaced_pieces[i])] = empty_detectors[i]

        # total_stages = max(1, len(displaced_pieces)) 
        pairs = min(len(displaced_pieces), len(empty_detectors))
        for i in range(pairs):
            target_mapping[str(displaced_pieces[i])] = empty_detectors[i]
        total_stages = max(1, pairs)

    # # 3. 找出当前未归位的棋子 (条件：要么不在格子里，要么正在被抓着)  
    # prev_active_idx = prev_obs_dict.get("task_thresholds", {}).get("active_piece_index") if prev_obs_dict else None
    # prev_action = prev_obs_dict.get("action_type") if prev_obs_dict else None

    # displaced_indices = []
    # for p_idx_str, target_d_idx in target_mapping.items():
    #     p_idx = int(p_idx_str)
    #     p_pos = pieces_state[p_idx, :2]
    #     target_pos = detectors_state[target_d_idx, :2]
        
    #     # 核心：只计算棋子和它【专属目标】的距离
    #     dist_to_target = np.linalg.norm(target_pos - p_pos)
        
    #     is_touch = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], pieces_state[p_idx], piece_size)
    #     is_held = is_touch and gripper_open < 0.1
        
    #     is_just_placed = (p_idx == prev_active_idx) and (prev_action == "place") and (not is_held) and (dist_to_target <= SUCCESS_XY_DIST)
        
    #     # 如果离老家远，或者正被拿着，或者刚放下（快门延迟），就算没完成
    #     if dist_to_target > SUCCESS_XY_DIST or is_held or is_just_placed:
    #         displaced_indices.append(p_idx)

    # 4. 进度指针计算 
    # current_stage = total_stages - len(displaced_indices)
    # current_stage = max(0, min(total_stages - 1, current_stage))
    # 获取上一帧已完成的棋子集合 
    completed_indices = prev_obs_dict.get("completed_indices", []).copy() if prev_obs_dict else []
    # 更新已完成列表
    for p_idx_str, target_d_idx in target_mapping.items():
        p_idx = int(p_idx_str)
        # 计算实时距离
        real_dist = np.linalg.norm(pieces_state[p_idx, :2] - detectors_state[target_d_idx, :2])
        # 用物理距离判定是否被抓取，抛弃不可靠的 gripper_open
        gripper_to_piece_dist = np.linalg.norm(gripper_pose[:3] - pieces_state[p_idx, :3])
        is_manipulating = bool(gripper_to_piece_dist < 0.05) # 手在5厘米内，就是在玩它
        if real_dist <= SUCCESS_XY_DIST and not is_manipulating:
            if p_idx not in completed_indices:
                completed_indices.append(p_idx)      
        # # 失败感知逻辑：如果一个已完成的棋子被猛烈撞飞（例如 > 5cm）
        # elif real_dist > 0.08 and p_idx in completed_indices and not is_manipulating:
        #     completed_indices.remove(p_idx) # 此时进度会如实下跌，触发纠偏 
    if prev_obs_dict:
        prev_completed = prev_obs_dict.get("completed_indices", [])
        for p in prev_completed:
            if p not in completed_indices:
                completed_indices.append(p)

    # 5. 锁定当前机械臂正在操作或意图操作的 active_piece  
    # 找出所有还没完成的棋子
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
        #     # 极小概率兜底：开局就全放好了
        #     active_idx = int(list(target_mapping.keys())[0])
    
    # ==========================================
    # 6. 从映射表中取出它的专属老家作为 Target
    # ==========================================
    active_piece_pose = pieces_state[active_idx]
    best_target_idx = target_mapping[str(active_idx)]
    
    active_target_pose = detectors_state[best_target_idx]
    init_active_piece_pos = init_pieces_state[active_idx][:3]
    init_active_target_pos = init_detectors_state[best_target_idx][:3]

    # ==========================================
    # 3. 计算 Edge 1: gripper -> chess_piece
    # ==========================================
    g_p_relations, g_p_delta, g_p_dist = compute_directional_and_distance(
        gripper_pose[:3], active_piece_pose[:3], extents["gripper_radius"], piece_size)
    _, init_g_p_delta, init_g_p_dist = compute_directional_and_distance(
        init_gripper_pos, init_active_piece_pos, extents["gripper_radius"], piece_size)

    is_touching_piece = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], active_piece_pose, piece_size)
    if is_touching_piece: g_p_relations.append("TOUCHING")

    # # 专为 32 颗棋子打造的动态运动学防误判锁 (Kinematic Lock)
    # delta_dist = 0.0
    # if prev_obs_dict is not None:
    #     # 从上一帧的数据切片中，精准提取同一个 active_idx 棋子的旧位置
    #     prev_pieces_state = prev_obs_dict['task_low_dim_state'][indices["pieces_slice"]].reshape(32, 7)
    #     prev_piece_pos = prev_pieces_state[active_idx, :3]
    #     prev_g_p_dist = np.linalg.norm(prev_obs_dict['gripper_pose'][:3] - prev_piece_pos)
    #     # 计算两帧之间，手与该棋子的相对位移差
    #     delta_dist = abs(g_p_dist - prev_g_p_dist)
    # # 严谨的 HOLDING 判定
    # is_holding_piece = False
    # if is_touching_piece and gripper_open < 0.1:
    #     # 相对位移极小（小于 KINEMATIC_LOCK_TOLERANCE），说明棋子死死地黏在手上
    #     if prev_obs_dict is None or delta_dist < KINEMATIC_LOCK_TOLERANCE:
    #         is_holding_piece = True
    #         g_p_relations.append("HOLDING")
    is_holding_piece = (gripper_to_piece_dist < 0.04) and gripper_open < 0.8
    if is_holding_piece: g_p_relations.append("HOLDING")

    # ==========================================
    # 4. 计算 Edge 2: chess_piece -> target_zone
    # ==========================================
    p_t_relations, p_t_delta, p_t_dist = compute_directional_and_distance(
        active_piece_pose[:3], active_target_pose[:3], piece_size, target_size)
    _, init_p_t_delta, init_p_t_dist = compute_directional_and_distance(
        init_active_piece_pos, init_active_target_pos, piece_size, target_size)

    dist_xy = np.linalg.norm(p_t_delta[:2])
    # 注意：这里取绝对值，且容忍探测器埋在地下或悬空的位移
    dist_z = abs(p_t_delta[2])
    
    # INSIDE 判定：水平极其接近(3cm)，垂直允许 10cm 以内的偏移(探测器可能埋深)
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
    
    # # 远近语义补充：如果不是 AROUND 且距离较远
    # if not is_around and p_t_dist > 0.1:
    #     p_t_relations.append("FAR")
    # else:
    #     p_t_relations.append("CLOSE")
    
    # if not is_inside:
    #     p_t_relations.append("OUTSIDE") 

    # 1. 基础进度：完全由摆放成功的棋子数决定
    base_progress = len(completed_indices) / total_stages

    if init_p_t_dist < 0.01:
        local_increment = 0.0
    else:
        local_increment = (1.0 - (p_t_dist / init_p_t_dist)) * 0.9
        local_increment = max(0.0, min(0.9, local_increment))

    # 3. 最终组合进度
    final_prog = base_progress + (local_increment / total_stages)

    # 4. 保护：Episode 内全局单调锁 (必须加！)
    # 只有这样才能消灭那 179 次报警
    if prev_obs_dict:
        # 从 Answer 字符串或 obs_dict 中提取上一帧的最终进度
        prev_completed = prev_obs_dict.get("completed_indices", [])
        prev_final_prog = prev_obs_dict.get("max_progress_ever", 0.0)
        # if len(completed_indices) < len(prev_completed):
        #     # 允许进度真实下跌！打破单调锁，让外层的纠偏逻辑捕捉到这个异常！
        #     pass
        # else:
        #     final_prog = max(final_prog, prev_final_prog)
        final_prog = max(final_prog, prev_final_prog)

    # 存入 obs_dict 供下一帧使用
    obs_dict["max_progress_ever"] = final_prog
    obs_dict["completed_indices"] = completed_indices.copy()
 
    # ==========================================
    # 5. 宏观意图与组装
    # ==========================================
    # 注意：这里的 action_type 是针对当前正在操作的这颗棋子而言的 
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
 
    # ==========================================
    # 在函数的最后，将阶段数据注入 task_thresholds
    # ==========================================
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
            # 将原配映射表传给下一帧
            "target_mapping": target_mapping,
            "completed_indices": completed_indices.copy()
        }
    }
 

def _extract_slide_block_to_target(task_state, init_task_state, gripper_pose, gripper_open, init_gripper_pos, 
                                 indices, extents, instructions, prev_obs_dict, obs_dict):
    # ==========================================
    # 1. 解析实体尺寸与位姿
    # ==========================================
    block_extents = extents.get("block_size", [0.05, 0.05, 0.05])
    # 目标区域通常是一个平面的传感器范围
    success_radius = extents.get("success_radius", 0.04)  

    block_pose = task_state[indices["block"]]
    success_pose = task_state[indices["success"]]
    
    init_block_pos = init_task_state[indices["block"]][:3]
    init_success_pos = init_task_state[indices["success"]][:3]

    # ==========================================
    # 2. 计算物理关系 (Edges)
    # ==========================================
    # --- Edge 1: gripper -> block (操控边) ---
    g_b_relations, g_b_delta, g_b_dist = compute_directional_and_distance(
        gripper_pose[:3], block_pose[:3], extents["gripper_radius"], block_extents)
    _, init_g_b_delta, init_g_b_dist = compute_directional_and_distance(
        init_gripper_pos, init_block_pos, extents["gripper_radius"], block_extents)

    # 接触判定 (由于是“推”的动作，TOUCHING 容差可以稍大)
    is_touching_block = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], block_pose, block_extents)
    if is_touching_block: g_b_relations.append("TOUCHING")

    # 抓取/稳固接触判定 (Kinematic Lock)
    is_holding = check_holding(is_touching_block, g_b_dist, "block", g_b_relations, gripper_open, indices, prev_obs_dict)
    if is_holding: g_b_relations.append("HOLDING")

    # --- Edge 2: block -> target (目标边) ---
    b_t_relations, b_t_delta, b_t_dist = compute_directional_and_distance(
        block_pose[:3], success_pose[:3], block_extents, success_radius)
    _, init_b_t_delta, init_b_t_dist = compute_directional_and_distance(
        init_block_pos, init_success_pos, block_extents, success_radius)

    # 计算水平距离（滑动任务的核心指标）
    dist_xy = np.linalg.norm(block_pose[:2] - success_pose[:2])
    
    # 阈值定义
    TARGET_RADIUS = success_radius  # 成功半径 (通常 4cm)
    AROUND_RADIUS = TARGET_RADIUS + 0.08 # 接近半径 (12cm)

    is_inside = bool(dist_xy <= TARGET_RADIUS)
    is_around = bool(dist_xy <= AROUND_RADIUS)

    if is_inside:
        b_t_relations.append("INSIDE")
        b_t_relations.append("ON")
    elif is_around:
        b_t_relations.append("AROUND")
    if not is_inside:
        b_t_relations.append("OUTSIDE")

    # ==========================================
    # 3. 宏观意图状态机
    # ==========================================
    # 成功条件：方块覆盖传感器 (is_inside) 且 机器人停止施力/松手
    if is_inside:
        action_type = "done"
    elif is_holding or is_touching_block:
        action_type = "place" # 在滑动任务中，place 表达“向目标位置推进”
    else:
        action_type = "pick" # 表达“移动向方块准备推”

    # ==========================================
    # 4. 组装返回数据
    # ==========================================
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
        "is_touching": bool(is_inside), # 滑动任务中，在目标区即视为 touch
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

    #  重置条件：如果没传 prev_obs_dict，或者夹爪和杯子都处于绝对的初始位置，强行认定为第 0 帧！
    # is_frame_zero = (prev_obs_dict is None) or (
    #     np.allclose(gripper_pose[:3], init_gripper_pos, atol=1e-4) and 
    #     np.allclose(cup1_pose[:3], init_cup1_pos, atol=1e-4)
    # )

    # ==========================================
    # 0. 跨帧状态记忆锁 (防渲染阶段状态污染)
    # ==========================================
    # if is_frame_zero:
    #     _extract_stack_cups.c1_stacked = False
    #     _extract_stack_cups.c3_stacked = False
    #     _extract_stack_cups.stage2_done = False
    #     if hasattr(_extract_stack_cups, "stage1_gripper_pos"):
    #         del _extract_stack_cups.stage1_gripper_pos

    # is_c1_stacked = getattr(_extract_stack_cups, "c1_stacked", False)
    # is_c3_stacked = getattr(_extract_stack_cups, "c3_stacked", False)
    # --- 1. 从 prev_obs_dict 恢复状态 ---
    is_c1_stacked = prev_obs_dict.get("c1_stacked", False) if prev_obs_dict else False
    is_c3_stacked = prev_obs_dict.get("c3_stacked", False) if prev_obs_dict else False
    stage1_gripper_pos = prev_obs_dict.get("stage1_gripper_pos", None) if prev_obs_dict else None

    # ==========================================
    # 1. 提取实体位姿与尺寸
    # ==========================================
    cup2_pose = task_state[indices["cup2"]]
    cup3_pose = task_state[indices["cup3"]] 

    init_cup3_pos = init_task_state[indices["cup3"]][:3]
    init_cup2_pos = init_task_state[indices["cup2"]][:3]
    cup_size = extents.get("cup1_size", [0.05, 0.05, 0.08]) 
    success_radius = extents.get("success_radius", 0.04)  
 
    TARGET_XY_RADIUS = 0.12  # 水平对齐容差 
    TARGET_Z_RADIUS = 0.18   # 垂直高度容差 
    AROUND_RADIUS = 0.13     # 进场走廊 13cm

    # ==========================================
    # 2. 独立判定两个杯子的堆叠状态
    # ==========================================
    c1_dist_xy = np.linalg.norm(cup1_pose[:2] - cup2_pose[:2]) 
    c1_dist_z = abs(cup1_pose[2] - cup2_pose[2])
    
    c3_dist_x = abs(cup3_pose[0] - cup2_pose[0]) 
    c3_dist_y = abs(cup3_pose[1] - cup2_pose[1])
    c3_dist_z = abs(cup3_pose[2] - cup2_pose[2])
    
    g_c1_dist = np.linalg.norm(gripper_pose[:3] - cup1_pose[:3])
    g_c3_dist = np.linalg.norm(gripper_pose[:3] - cup3_pose[:3])

    # 捕捉“松手”动作 (上一帧抓紧，当前帧松开)
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
            stage1_gripper_pos = gripper_pose[:3].tolist() # 存入 local 变量
            
    if (abs(c3_dist_x) <= TARGET_XY_RADIUS) and (abs(c3_dist_y) <= TARGET_XY_RADIUS/2) and (abs(c3_dist_z) <= TARGET_Z_RADIUS) and just_released:
        if not is_c3_stacked:
            is_c3_stacked = True
            stage1_gripper_pos = gripper_pose[:3].tolist() # 存入 local 变量

    # 【核心修正】：将阶段切换延迟一帧。如果是刚松手的那一帧，强行维持上一阶段的 stage
    real_stage = int(is_c1_stacked) + int(is_c3_stacked)
    current_stage = real_stage - 1 if just_stacked_cup else real_stage
    # dynamic_init_gripper_pos = getattr(_extract_stack_cups, "stage1_gripper_pos", init_gripper_pos)
    dynamic_init_gripper_pos = np.array(stage1_gripper_pos) if stage1_gripper_pos is not None else init_gripper_pos
    
    # ==========================================
    # 3. 动态注意力路由：智能推断当前意图
    # ========================================== 
    init_g_c1_dist = np.linalg.norm(dynamic_init_gripper_pos - init_cup1_pos)
    init_g_c3_dist = np.linalg.norm(dynamic_init_gripper_pos - init_cup3_pos)
    
    approach_c1 = init_g_c1_dist - g_c1_dist
    approach_c3 = init_g_c3_dist - g_c3_dist

    # 【核心修正】：如果是刚松手的那一帧，焦点强行锁在刚叠好的杯子上以输出 done
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
 
    # ==========================================
    # 4. 计算 MomaGraph 边关系 (围绕 active_cup)
    # ==========================================
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
     
    # 初始化第二阶段胜利锁
    if not hasattr(_extract_stack_cups, "stage2_done") or not prev_obs_dict:
        _extract_stack_cups.stage2_done = False

    # 【第二阶段宽容胜利锁】：仅在松手的那一瞬间生效！
    if current_stage == 1 and just_released and dist_xy <= TARGET_XY_RADIUS + 0.08 and dist_z <= TARGET_Z_RADIUS + 0.15:
        _extract_stack_cups.stage2_done = True

    # 【新增高空锁】：杯子必须离开初始桌面至少 3cm，才允许被日常判定为放好
    is_lifted_from_table = bool(active_pose[2] > init_active_pos[2] + 0.03)

    # 逻辑判断：有锁保底，无锁时必须满足严格容差 + 离地要求
    if real_stage == 2 or getattr(_extract_stack_cups, "stage2_done", False):
        is_inside_logic = True
    elif current_stage == 0 and real_stage == 1 and just_released: 
        # 维持第一阶段松手延迟帧的 True 状态
        is_inside_logic = True
    else:
        # 日常帧判定：必须离地，防止把桌子上待机的杯子误判为已叠好
        is_inside_logic = bool(dist_xy <= TARGET_XY_RADIUS/2 and dist_z <= TARGET_Z_RADIUS and not is_holding_cup and is_lifted_from_table)

    # 计算杯子的等效半径 (用于进场函数)
    active_cup_radius = max(cup_size[0], cup_size[1]) / 2.0

    # 使用通用进场走廊函数判定 AROUND
    is_around_logic = check_approach_corridor_world(
        pos_obj=active_pose[:3], 
        pos_anchor=cup2_pose[:3],     
        approach_axis=2,              
        approach_dir=1.0,             
        mo_size=active_cup_radius, 
        tgt_size=cup_size, 
        min_radius=AROUND_RADIUS,     
        min_height=0.90               # 稍微调高天花板，防止高塔出界
    )

    if is_inside_logic:
        c_t_relations.append("INSIDE")
        c_t_relations.append("ON")
        c_t_relations.append("AROUND")
    elif is_around_logic and is_holding_cup:
        c_t_relations.append("AROUND")
 
    # ==========================================
    # 5. 宏观意图与状态机
    # ==========================================
    if current_stage == 2 or real_stage == 2 or getattr(_extract_stack_cups, "stage2_done", False):
        action_type = "done"
    elif is_holding_cup:
        action_type = "place"
    else:
        action_type = "pick"  

    # ==========================================
    # 6. 组装返回数据
    # ==========================================
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

    # --- 在返回前，把状态传给下一帧 ---
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
    # ==========================================
    # 1. 实体解析与参数初始化
    # ==========================================
    end_keys = ["head", "tail"]
    target_keys = {"head": "success_head", "tail": "success_tail"}
    
    # 提取位姿
    poses = {k: task_state[indices[k]] for k in end_keys + list(target_keys.values())}
    init_poses = {k: init_task_state[indices[k]][:3] for k in end_keys + list(target_keys.values())}
    
    # 尺寸参数
    end_size = extents.get("head_size", [0.03, 0.03, 0.03])
    success_radius = extents.get("success_radius", 0.05)
    
    # ==========================================
    # 2. 局部状态扫描 (端点完成度判定)
    # ==========================================
    end_status = {}
    completed_count = 0 
    completed_ends = prev_obs_dict.get("completed_ends", []) if prev_obs_dict else []
    
    for k in end_keys:
        curr_pose_7d = poses[k]
        tgt_p = poses[target_keys[k]][:3]
        
        # 物理判定：端点与对应传感器的 XY 距离
        dist_xy = np.linalg.norm(curr_pose_7d[:2] - tgt_p[:2])
        is_in_zone = dist_xy <= success_radius
        
        # 接触判定
        is_touch = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], curr_pose_7d, end_size)
        is_held = is_touch and gripper_open < 0.1
        
        # 完成标志：进入区域且手已放开
        # is_finished = is_in_zone and not is_held
        # 只要进过区域且目前没被撞飞（<15cm），就保持完成状态
        if is_in_zone and k not in completed_ends:
            completed_ends.append(k)
        elif dist_xy > 0.15 and k in completed_ends:
            completed_ends.remove(k)
            
        is_finished = (k in completed_ends)
        # if is_finished:
        #     completed_count += 1
            
        end_status[k] = {"is_finished": is_finished, "is_held": is_held, "dist_xy": dist_xy}
 
    # ==========================================
    # 3. 动态确定当前操作目标 (Active End)
    # ==========================================
    # 规则：如果有一端正在被抓，则该端为 active；否则，选择离手最近的未完成端
    held_ends = [k for k in end_keys if end_status[k]["is_held"]]
    
    if held_ends:
        active_key = held_ends[0]
    else:
        unfinished = [k for k in end_keys if not end_status[k]["is_finished"]]
        if not unfinished:
            active_key = end_keys[-1] # 全部完成后指向末尾
        else:
            active_key = min(unfinished, key=lambda k: np.linalg.norm(gripper_pose[:3] - poses[k][:3]))

    # ==========================================
    # 4. 进度阶段与虚拟锚点 (进度条核心)
    # ==========================================
    completed_count = len(completed_ends)
    total_stages = 2
    current_stage = min(completed_count, total_stages - 1)

    # 这里的锚点设计是为了让 ProgressEstimator 的线性插值能工作：
    # 第一段起点为初始位置；第二段起点为第一段刚完成时的位置
    if current_stage == 0:
        v_start_gripper = init_gripper_pos
    else:
        # 第二段时，我们认为机械臂是从上一个目标点飞向下一个端点的
        other_key = "tail" if active_key == "head" else "head"
        v_start_gripper = init_poses[target_keys[other_key]]

    v_start_end = init_poses[active_key]
    v_start_target = init_poses[target_keys[active_key]]

    # ==========================================
    # 5. 构建图谱边 (Edges)
    # ==========================================
    # --- Edge 1: gripper -> rope_end ---
    g_e_rels, g_e_delta, g_e_dist = compute_directional_and_distance(
        gripper_pose[:3], poses[active_key][:3], extents["gripper_radius"], end_size)
    
    #  不要用 target 作为起点，对于 gripper，起点永远是上一阶段结束时的位置
    # 如果是第一阶段，使用 init_gripper_pos
    if current_stage == 0:
        actual_v_start_gripper = init_gripper_pos
    else:
        # 如果是第二阶段，起点是上一帧机械臂所在的位置 (即放开第一端的位置)
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
    
    #  绳子端的 original_dist 必须基于初始位姿，否则分母太小
    _, init_e_t_delta, init_e_t_dist = compute_directional_and_distance(
        np.array(init_poses[active_key]), np.array(init_poses[target_keys[active_key]]), end_size, success_radius)
 
    # 只有当 XY 距离小于 3cm (非常接近目标) 且正在抓持时，才给 AROUND
    dist_xy_active = end_status[active_key]["dist_xy"]
    if dist_xy_active <= success_radius:
        e_t_rels.extend(["INSIDE", "ON", "TOUCHING", "AROUND"])
    elif dist_xy_active <= success_radius + 0.03: # 从 0.10 缩减到 0.03
        e_t_rels.append("AROUND")
    else:
        e_t_rels.append("OUTSIDE")

    # ==========================================
    # 6. 动作意图与组装
    # ==========================================
    if completed_count == total_stages:
        action_type = "done"
    elif is_holding:
        action_type = "pull" # 绳子任务使用 pull 动作
    else:
        action_type = "pick"

    # # 阶段保护补丁
    # global_stage_progress = current_stage / total_stages
    # local_progress = (1.0 - e_t_dist / init_e_t_dist) / total_stages
    # final_progress = global_stage_progress + max(0.0, local_progress)

    # # 确保进度在同一阶段内绝不回退 
    # global_stage_progress = current_stage / total_stages
    # # 这里的 local_progress 基于当前阶段的剩余距离
    # local_progress = (1.0 - e_t_dist / init_e_t_dist) / total_stages
    # final_progress = global_stage_progress + max(0.0, local_progress)

    # # 安全地获取历史信息 
    # if prev_obs_dict:
    #     historical_p = prev_obs_dict.get("max_p", 0.0)
    #     # 只有当跌幅超过 15% 时（通常是阶段跳变或严重脱手），才允许进度掉落
    #     # 这样既能触发 Branch 2 训练重规划，又能过滤掉 0.01 级别的摩擦噪声
    #     if (historical_p - final_progress) < 0.15:
    #         final_progress = max(final_progress, historical_p) 
    
    # # 将当前状态存入 obs_dict 供下一帧使用
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
    # ==========================================
    # 1. 实体解析与参数初始化
    # ==========================================
    knob_extents = extents.get("knob_size", [0.05, 0.05, 0.05])
    oven_extents = extents.get("oven_size", [0.6, 0.5, 0.6])
    
    # 目标：RLBench 定义 JointCondition(knob_joint, 1.5)
    TARGET_ROTATION = 1.5
    
    # 提取实时状态
    knob_pose = task_state[indices["knob"]]
    oven_pose = task_state[indices["oven"]]
    # 💡 关节位置通常在 7 维位姿块之后或通过特定索引获取
    current_joint_pos = task_state[indices["knob_joint"]]
    
    init_knob_pos = init_task_state[indices["knob"]][:3]
    init_oven_pos = init_task_state[indices["oven"]][:3]
    init_joint_pos = init_task_state[indices["knob_joint"]]

    # ==========================================
    # 2. 计算物理关系 (Edges)
    # ==========================================
    # --- Edge 1: gripper -> knob (操控边) ---
    g_k_rels, g_k_delta, g_k_dist = compute_directional_and_distance(
        gripper_pose[:3], knob_pose[:3], extents["gripper_radius"], knob_extents)
    _, init_g_k_delta, init_g_k_dist = compute_directional_and_distance(
        init_gripper_pos, init_knob_pos, extents["gripper_radius"], knob_extents)

    is_touching_knob = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], knob_pose, knob_extents)
    if is_touching_knob: g_k_rels.append("TOUCHING")

    # is_holding = check_holding(is_touching_knob, g_k_dist, "knob", g_k_rels, gripper_open, indices, prev_obs_dict)
    is_holding = is_touching_knob and bool(gripper_open < 0.1)
    if is_holding: g_k_rels.append("HOLDING")

    # --- Edge 2: knob -> oven (状态边) ---
    # 旋转任务中，相对位移几乎为0，进度由关节角决定
    k_o_rels, k_o_delta, k_o_dist = compute_directional_and_distance(
        knob_pose[:3], oven_pose[:3], knob_extents, oven_extents)
    _, init_k_o_delta, init_k_o_dist = compute_directional_and_distance(
        init_knob_pos, init_oven_pos, knob_extents, oven_extents)

    # 计算旋转进度
    # rotation_delta = abs(current_joint_pos - init_joint_pos)

    # ---------- FOR FAILURE ---------- 
    # rotation_delta = max(0.0, current_joint_pos - init_joint_pos) 
    # ---------- FOR EXPERT ---------- 
    raw_rotation_delta = max(0.0, current_joint_pos - init_joint_pos)  
    curr_prog = min(1.0, raw_rotation_delta / TARGET_ROTATION) 
    if prev_obs_dict:
        curr_prog = max(curr_prog, prev_obs_dict.get("max_progress_ever", 0.0)) 
    obs_dict["max_progress_ever"] = curr_prog 
    rotation_delta = curr_prog * TARGET_ROTATION

    is_turned = bool(rotation_delta >= (TARGET_ROTATION - 0.05)) # 提前一点判定成功，防止尾部抖动
  
    # 空间关系补充
    if is_turned: 
        k_o_rels.append("ON")
    else:
        k_o_rels.append("OFF")
    
    # 旋钮永远在烤箱上
    k_o_rels.append("TOUCHING")

    # ==========================================
    # 3. 意图状态机与进度魔法 (Magic Delta)
    # ==========================================
    if is_turned:
        action_type = "done"
    elif is_holding:
        action_type = "rotate"  # 旋转意图
    else:
        action_type = "pick"

    # 💡 核心技巧：ProgressEstimator 默认根据 dist 缩减算进度。
    # 我们构造一个虚拟的“剩余角度距离”喂给 edge_ko，让进度条根据旋转角度上涨。
    remaining_rot = max(0.0, TARGET_ROTATION - rotation_delta) 
    # ==========================================
    # 4. 组装返回
    # ==========================================
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
            # 这里的魔法距离确保了进度条与旋转同步
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
    # ==========================================
    # 1. 实体解析
    # ==========================================
    sponge_extents = extents.get("sponge_size", [0.05, 0.08, 0.03])
    desk_extents = extents.get("dirt_boundary_size", [0.3, 0.3, 0.01])
    
    sponge_pose = task_state[indices["sponge"]]
    boundary_pose = task_state[indices["dirt_boundary"]]
    
    init_sponge_pos = init_task_state[indices["sponge"]][:3]
    init_boundary_pos = init_task_state[indices["dirt_boundary"]][:3]

    # ==========================================
    # 2. 物理关系提取 (严格计算 Z 轴接触)
    # ==========================================
    g_s_rels, g_s_delta, g_s_dist = compute_directional_and_distance(
        gripper_pose[:3], sponge_pose[:3], extents["gripper_radius"], sponge_extents)
    _, init_g_s_delta, init_g_s_dist = compute_directional_and_distance(
        init_gripper_pos, init_sponge_pos, extents["gripper_radius"], sponge_extents)

    is_touching_sponge = check_universal_contact(gripper_pose[:3], extents["gripper_radius"], sponge_pose, sponge_extents)
    if is_touching_sponge: g_s_rels.append("TOUCHING")

    is_holding = check_holding(is_touching_sponge, g_s_dist, "sponge", g_s_rels, gripper_open, indices, prev_obs_dict)
    if is_holding: g_s_rels.append("HOLDING")

    # 提取海绵与桌面的 XYZ 偏差
    s_d_rels, s_d_delta, s_d_dist = compute_directional_and_distance(
        sponge_pose[:3], boundary_pose[:3], sponge_extents, desk_extents)
    _, init_s_d_delta, init_s_d_dist = compute_directional_and_distance(
        init_sponge_pos, init_boundary_pos, sponge_extents, desk_extents)

    dist_xy = np.linalg.norm(sponge_pose[:2] - boundary_pose[:2])
    dist_z = abs(sponge_pose[2] - boundary_pose[2])
    
    # 💡 精准判定：海绵 Z 尺寸约 3.2cm，半径为 1.6cm。距离小于 2.5cm 视为紧贴物理碰撞面！
    is_touching_desk = bool(dist_z < 0.025) 
    is_inside_area = bool(dist_xy < np.max(desk_extents))

    # ==========================================
    # 3. 意图与物理接地进度 (Physical-Grounded Progress)
    # ==========================================
    # 首帧物理探测器 (清空记忆)
    dist_to_init = np.linalg.norm(gripper_pose[:3] - init_gripper_pos)
    is_first_frame = (prev_obs_dict is None) or (dist_to_init < 0.001)

    clean_prog = prev_obs_dict.get("clean_progress", 0.0) if prev_obs_dict else 0.0
    if is_first_frame: clean_prog = 0.0

    # if not hasattr(_extract_wipe_desk, "global_clean_progress") or is_first_frame:
    #     # _extract_wipe_desk.global_clean_progress = 0.0
    #     clean_prog = 0.0

    # 💡 核心意图分配：必须“抓着”且“紧贴桌面”，才算在 wipe！
    if is_holding and is_touching_desk and is_inside_area:
        action_type = "wipe"
        # 只有在真正在桌面上摩擦时，才根据位移“消耗”进度
        if prev_obs_dict and not is_first_frame:
            prev_sponge_pos = prev_obs_dict['task_low_dim_state'][indices["sponge"]][:3]
            move_delta = np.linalg.norm(sponge_pose[:3] - prev_sponge_pos)
            inc = 0.0002 + move_delta * 0.6 # 静止不动只涨 0.1%，动起来涨幅大幅增加
            # _extract_wipe_desk.global_clean_progress = min(0.98, _extract_wipe_desk.global_clean_progress + inc) 
            clean_prog = min(0.98, clean_prog + inc)
    elif is_holding:
        action_type = "place" # 拿着但悬空，视为正在搬运
    else:
        action_type = "pick"

    # 任务完成兜底
    if clean_prog > 0.95 and not is_holding:
        action_type = "done"
        clean_prog = 1.0

    # ==========================================
    # 4. 显式图谱关系分配 (明确 TOUCHING)
    # ==========================================
    if action_type == "done":
        s_d_rels.extend(["ON", "TOUCHING", "AROUND"])
    else:
        # 💡 回应你的质疑：清晰、显式地反馈 TOUCHING 状态
        if is_touching_desk and is_inside_area:
            s_d_rels.extend(["TOUCHING", "ON"])
        elif is_inside_area:
            s_d_rels.append("AROUND") # 悬空在污渍上方
        else:
            s_d_rels.append("OUTSIDE")

    magic_dist = 1.0 - clean_prog

    # ==========================================
    # 5. 组装返回
    # ==========================================
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
        "is_touching": bool(is_touching_desk), # 💡 这里也显式映射物理状态
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
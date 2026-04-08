import numpy as np
from scipy.spatial.transform import Rotation as R
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, Scene

# =================================================
#  Visualization and video rendering
# =================================================
def draw_wrapped_text(img, text, pos_x, pos_y, font, font_scale, color, thickness, max_width):
    """
    OpenCV 自动换行文本绘制工具
    """
    import cv2
    # 针对图谱列表，以逗号 ", " 作为分割边界
    words = text.split(", ") 
    if not words: return pos_y
    
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        test_line = current_line + ", " + word
        text_width = cv2.getTextSize(test_line, font, font_scale, thickness)[0][0]
        
        if text_width > max_width:
            # 如果超宽，就把之前的行存起来，换新行
            lines.append(current_line + ",") 
            current_line = word
        else:
            # 没超宽，继续拼接
            current_line = test_line
            
    lines.append(current_line) # 加上最后一行

    # 计算字体实际高度，并加上1.5倍行距
    line_height = int(cv2.getTextSize("Tg", font, font_scale, thickness)[0][1] * 1.6)
    
    current_y = pos_y
    for line in lines:
        cv2.putText(img, line, (pos_x, current_y), font, font_scale, color, thickness)
        current_y += line_height # Y坐标下移
        
    return current_y # 返回最终的 Y 坐标，防止与后面的文字重叠



# ==========================================
# 核心物理与几何计算函数
# ==========================================

def compute_directional_and_distance(pos_src, pos_tgt, size_src_radius, size_tgt_radius, \
    close_margin=0.10, directional_eps=1e-3):
    """
    计算方向和距离关系 (pos_src 相对于 pos_tgt 的关系, 引入 CLOSE 阈值)
    """
    relations = []
    delta_3d = pos_tgt - pos_src
    dist = np.linalg.norm(delta_3d)
    
    def _to_scalar_radius(size_input):
        size_arr = np.atleast_1d(size_input)
        if len(size_arr) == 1:
            return float(size_arr[0])
        elif len(size_arr) == 3:
            # 取长方体的外接球半径: 1/2 * 对角线长度
            return float(np.linalg.norm(size_arr) / 2.0)
        return 0.0

    r_src = _to_scalar_radius(size_src_radius)
    r_tgt = _to_scalar_radius(size_tgt_radius)
    
    # 动态 CLOSE 阈值：两者外接球半径 + 语义容差 (此时保证了三个变量都是纯标量)
    dynamic_threshold = r_src + r_tgt + close_margin
    # dynamic_threshold = size_src_radius + size_tgt_radius + close_margin
    if dist < dynamic_threshold: 
        relations.append("CLOSE")
    else: 
        relations.append("FAR")
    
    if delta_3d[0] > directional_eps: relations.append("LEFT OF")
    elif delta_3d[0] < -directional_eps: relations.append("RIGHT OF")
    
    if delta_3d[1] > directional_eps: relations.append("BEHIND")
    elif delta_3d[1] < -directional_eps: relations.append("IN FRONT OF")
    
    if delta_3d[2] > directional_eps: relations.append("LOWER THAN")
    elif delta_3d[2] < -directional_eps: relations.append("HIGHER THAN")
    
    return relations, delta_3d, dist


def check_universal_contact(pos_src, radius_src, pose_tgt_7d, tgt_size, touching_eps=0.001):
    """
    通用表面接触检测器
    :param tgt_size: 目标物体的尺寸。
                     - 传入浮点数/单元素列表 (例如: 0.05 或 [0.05]): 视目标为球体，值为其半径。
                     - 传入 3D 向量 (例如: [0.1, 0.2, 0.1]): 视目标为长方体，值为长宽高。
    :param eps: 虚拟皮肤厚度 (容差)，默认 1mm。
    """
    tgt_pos = pose_tgt_7d[:3]
    size_array = np.atleast_1d(tgt_size)
    
    # 场景1：目标是球体 (输入为标量或长度为1的数组)
    if len(size_array) == 1:
        tgt_radius = float(size_array[0])
        dist = np.linalg.norm(tgt_pos - pos_src)
        # 两球心距离减去两球半径，剩余缝隙小于 touching_eps 视为接触
        return dist - radius_src - tgt_radius <= touching_eps
        
    # 场景2：目标是长方体 (输入为长度为3的数组，采用局部坐标系 OBB 干涉法)
    elif len(size_array) == 3:
        tgt_quat = pose_tgt_7d[3:]
        inv_rotation = R.from_quat(tgt_quat).as_matrix().T
        local_pos = inv_rotation.dot(pos_src - tgt_pos)
    
        hx, hy, hz = size_array / 2.0
    
        dx = abs(local_pos[0]) - hx - radius_src
        dy = abs(local_pos[1]) - hy - radius_src
        dz = abs(local_pos[2]) - hz - radius_src
    
        return max(dx, dy, dz) <= touching_eps
    else:
        raise ValueError(f"tgt_size 维度错误！必须是标量(球体)或3D向量(长方体)，当前收到: {tgt_size}")


def check_holding(is_touching, current_dist, obj_idx_name, relations_list, gripper_open, indices, prev_obs_dict):
    # HOLDING 校验 (假设帧间相对位移小于 2mm 视为刚性连接)
    KINEMATIC_LOCK_TOLERANCE = 0.002
    delta_dist = 0.0
    if prev_obs_dict is not None:
        prev_pos = prev_obs_dict['task_low_dim_state'][indices[obj_idx_name]][:3]
        prev_g_mo_dist = np.linalg.norm(prev_pos - prev_obs_dict['gripper_pose'][:3])
        delta_dist = abs(current_dist - prev_g_mo_dist)

    is_holding = False
    if is_touching and gripper_open < 0.1:
        if prev_obs_dict is None or delta_dist < KINEMATIC_LOCK_TOLERANCE:
            relations_list.append("HOLDING")
            return True
        
    return False


def check_inside_containment(pos_obj, radius_obj, pose_container_7d, extents_container, top_open=True):
    """
    通用 INSIDE 容器包容判定器 (适配抽屉、纸箱、篮筐等任何长方体容器)
    - pos_obj: 被装物体的中心坐标
    - extents_container: 容器的长宽高 [size_x, size_y, size_z]
    - top_open: 容器是否是顶部敞开的 (通常纸箱、篮筐顶部是敞开的)
    """
    cont_pos = pose_container_7d[:3]
    cont_quat = pose_container_7d[3:]
    
    # 转移到容器局部坐标系
    inv_rotation = R.from_quat(cont_quat).as_matrix().T
    local_pos = inv_rotation.dot(pos_obj - cont_pos)
    
    hx, hy, hz = extents_container / 2.0
    
    # 核心判断：物体的边界不能超出容器的内壁
    # 这里加上 radius_obj 是为了保证不仅中心进去了，边缘也进去了
    inside_x = abs(local_pos[0]) + radius_obj <= hx
    inside_y = abs(local_pos[1]) + radius_obj <= hy
    
    # 对于 Z 轴（高度）的特殊处理：
    if top_open:
        # 如果是纸箱/篮筐，允许物体从上方探出一点，但底部必须在箱子内部
        inside_z = local_pos[2] - radius_obj >= -hz
    else:
        # 如果是微波炉/抽屉，必须完全在内部
        inside_z = abs(local_pos[2]) + radius_obj <= hz
        
    return inside_x and inside_y and inside_z


def check_approach_corridor_world(pos_obj, pos_anchor, approach_axis=2, approach_dir=1.0, \
    mo_size=None, tgt_size=None, min_radius=0.01, min_height=0.60, relax_factor=1.2, depth_multiplier=2.0):
    """
    基于世界坐标系的绝对进场走廊 (专门针对固定不倾斜的容器入口，支持 XY, XZ, YZ 任意平面对齐)
    E.g.
        投篮 / 把东西放进箱子 (Z轴进场，XY对齐)
            check_approach_corridor_universal(ball, hoop, approach_axis=2, approach_dir=1.0)
        把抽屉推回去 (Y轴进场，XZ对齐)
            check_approach_corridor_universal(drawer, desk, approach_axis=1, approach_dir=-1.0)
        把插头水平插进墙壁插座 (X轴进场，YZ对齐)
            check_approach_corridor_universal(plug, socket, approach_axis=0, approach_dir=1.0)
    - pos_obj: 篮球中心
    - pos_anchor: 篮筐中心(success传感器)
    :param approach_axis: 进场的主导轴。0代表X轴, 1代表Y轴, 2代表Z轴(默认)。
    :param approach_dir: 进场的方向。1.0表示从正方向靠近(如从上往下), -1.0表示从负方向靠近。

    - min_radius: 最小对齐容差兜底 (默认 1cm，精密装配请调小)。
    - min_height: 最小进场跑道长度兜底 (默认 10cm)。
    - relax_factor: 走廊半径宽松系数 (默认 1.2，控制平面对齐的宽容度)。
    - depth_multiplier: 走廊深度倍数 (默认 2.0，控制主导轴进场跑道的长度)。
    """
    # 1. 计算进场主导轴上的深度差   如篮球比篮筐中心高多少
    delta_depth = (pos_obj[approach_axis] - pos_anchor[approach_axis]) * approach_dir
    
    # 2. 动态提取非主导轴（即对齐平面）
    # 如果 approach_axis=2(Z)，则 plane_indices=[0, 1] (XY平面)
    # 如果 approach_axis=0(X)，则 plane_indices=[1, 2] (YZ平面)
    plane_indices = [i for i in range(3) if i != approach_axis]
    
    # 3. 计算对齐平面上的投影偏差
    plane_obj = pos_obj[plane_indices]
    plane_anchor = pos_anchor[plane_indices]
    plane_dist = np.linalg.norm(plane_obj - plane_anchor)
    
    # ==========================================
    # 4. 动态自适应走廊尺寸计算
    # ==========================================
    # 从对齐平面提取目标容器的开口尺寸
    target_opening_radius = min(tgt_size[plane_indices[0]], tgt_size[plane_indices[1]]) / 2.0
    
    # 走廊半径 = (容器开口 - 物体半径) * 宽松系数，并加兜底
    corridor_radius = max(min_radius, (target_opening_radius - mo_size) * relax_factor)
    
    # 修复：从进场主导轴提取容器深度，而不是死抠 tgt_size[2]
    target_depth = tgt_size[approach_axis]
    corridor_height = max(min_height, target_depth * depth_multiplier)
    
    # 5. 最终判定
    return (0.0 < delta_depth < corridor_height) and (plane_dist < corridor_radius)


# 用来计算前后两帧图谱差异的函数
def get_graph_diff(prev_g, curr_g, config):
    diffs = []
    # 1. 宏观意图变化
    if prev_g["action_type"] != curr_g["action_type"]:
        diffs.append(f"🎯 意图: {prev_g['action_type']} -> {curr_g['action_type']}")
  
    # 辅助函数：通过对象名字极其安全地找边，彻底抛弃死板的 List 索引
    def _find_edge(edges_list, obj1, obj2):
        for e in edges_list:
            if e.get("object1") == obj1 and e.get("object2") == obj2:
                return e
        return None

    # 动态获取当前阶段需要关注的焦点边 (若无焦点，则退化为单任务的 config 配置)
    focus_edges = curr_g.get("focus_edges")
    if not focus_edges:
        mo_name = config.get("semantic_mo_name", "object")
        to_name = config.get("semantic_to_name", "target")
        focus_edges = [("gripper", mo_name), (mo_name, to_name)]

    # 2. 边关系与几何属性变化
    for obj1, obj2 in focus_edges:
        edge_name = f"{obj1}->{obj2}"
        
        # 精准定位上一帧和当前帧的对应边
        prev_edge = _find_edge(prev_g.get("edges", []), obj1, obj2)
        curr_edge = _find_edge(curr_g.get("edges", []), obj1, obj2)
        
        # 兜底防御：如果图谱里缺失了这条边，直接跳过防报错
        if not prev_edge or not curr_edge:
            continue
            
        # 空间关系增删
        p_rels = set(prev_edge.get("spatial_relations", []))
        c_rels = set(curr_edge.get("spatial_relations", []))
        added, removed = c_rels - p_rels, p_rels - c_rels
        if added: diffs.append(f"➕ {edge_name} 新增: {list(added)}")
        if removed: diffs.append(f"➖ {edge_name} 移除: {list(removed)}")
        
        # 关键布尔状态翻转
        for attr in ["is_touching", "is_stable"]:
            p_val, c_val = prev_edge.get(attr), curr_edge.get(attr)
            if p_val != c_val:
                diffs.append(f"🔄 {edge_name} [{attr}]: {p_val} -> {c_val}")

    prev_thr = prev_g.get("task_thresholds", {})
    curr_thr = curr_g.get("task_thresholds", {})

    # 监测焦点杯子的切换
    if prev_thr.get("active_cup_name") != curr_thr.get("active_cup_name"): #  or curr_thr.get("active_cup_name")
        diffs.append(f"🔄 task_thresholds [active_cup_name]: {prev_thr.get('active_cup_name')} -> {curr_thr.get('active_cup_name')}") #   

    # 监测任务阶段的演进
    if prev_thr.get("current_stage") != curr_thr.get("current_stage"): # or curr_thr.get("current_stage") 
        diffs.append(f"🔄 task_thresholds [current_stage]: {prev_thr.get('current_stage')} -> {curr_thr.get('current_stage')}") # 
                
    return diffs



class EndEffectorPoseViaPlanning2(EndEffectorPoseViaPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        action[:3] = np.clip(
            action[:3],
            np.array(
                [scene._workspace_minx, scene._workspace_miny, scene._workspace_minz]
            )
            + 1e-7,
            np.array(
                [scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]
            )
            - 1e-7,
        )

        super().action(scene, action, ignore_collisions)
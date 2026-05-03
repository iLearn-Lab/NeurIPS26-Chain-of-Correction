import numpy as np
from scipy.spatial.transform import Rotation as R
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, Scene

# =================================================
#  Visualization and video rendering
# =================================================
def draw_wrapped_text(img, text, pos_x, pos_y, font, font_scale, color, thickness, max_width):
    """
    OpenCV tool for automatic text wrapping based on a specified pixel width.
    Splits text by commas to maintain semantic integrity of graph relations.
    
    Returns:
        int: The final Y-coordinate after drawing all lines, used to prevent overlap.
    """
    import cv2
    # Split graph relations using comma boundaries
    words = text.split(", ") 
    if not words: return pos_y
    
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        test_line = current_line + ", " + word
        text_width = cv2.getTextSize(test_line, font, font_scale, thickness)[0][0]
        
        if text_width > max_width:
            # If width is exceeded, store the current line and start a new one
            lines.append(current_line + ",") 
            current_line = word
        else:
            current_line = test_line
            
    lines.append(current_line)

    # Calculate font height and apply 1.6x line spacing
    line_height = int(cv2.getTextSize("Tg", font, font_scale, thickness)[0][1] * 1.6)
    
    current_y = pos_y
    for line in lines:
        cv2.putText(img, line, (pos_x, current_y), font, font_scale, color, thickness)
        current_y += line_height
        
    return current_y



# ==========================================
# Core Physics and Geometric Computations
# ==========================================

def compute_directional_and_distance(pos_src, pos_tgt, size_src_radius, size_tgt_radius, \
    close_margin=0.10, directional_eps=1e-3):
    """
    Calculates directional and distance relationships between a source and target position.
    Specifically computes source relative to target (e.g., 'LEFT OF' the target).
    
    Args:
        size_src_radius/size_tgt_radius: Can be scalar (radius) or 3D vector (extents).
                                        Cuboids are approximated via bounding spheres.
    """
    relations = []
    delta_3d = pos_tgt - pos_src
    dist = np.linalg.norm(delta_3d)
    
    def _to_scalar_radius(size_input):
        size_arr = np.atleast_1d(size_input)
        if len(size_arr) == 1:
            return float(size_arr[0])
        elif len(size_arr) == 3:
            # Approximate radius using half the diagonal length of the cuboid
            return float(np.linalg.norm(size_arr) / 2.0)
        return 0.0

    r_src = _to_scalar_radius(size_src_radius)
    r_tgt = _to_scalar_radius(size_tgt_radius)
    
    # Dynamic proximity threshold: Sum of radii + semantic margin
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
    Generic surface contact detector supporting Sphere-Sphere and Sphere-OBB interference.
    
    Args:
        tgt_size: Float/List[1] for sphere radius, or Array[3] for cuboid dimensions.
        touching_eps: Virtual skin thickness (tolerance), default 1mm.
    """
    tgt_pos = pose_tgt_7d[:3]
    size_array = np.atleast_1d(tgt_size)
    
    # Scenario 1: Target is a Sphere
    if len(size_array) == 1:
        tgt_radius = float(size_array[0])
        dist = np.linalg.norm(tgt_pos - pos_src) 
        return dist - radius_src - tgt_radius <= touching_eps
        
    # Scenario 2: Target is a Cuboid (Oriented Bounding Box interference)
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
        raise ValueError(f"Invalid tgt_size dimensions. Expected scalar or 3D vector, got: {tgt_size}")


def check_holding(is_touching, current_dist, obj_idx_name, relations_list, gripper_open, indices, prev_obs_dict):
    """
    Verifies if an object is being 'held' based on gripper state and kinematic locking.
    Kinematic lock is assumed if the relative displacement between gripper and object is < 2mm.
    """
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
    Generic occupancy detector for cuboid containers (drawers, boxes, baskets).
    
    Args:
        pos_obj: Center coordinates of the object.
        radius_obj: Radius of the object (to ensure entire volume is inside).
        top_open: If True, allows the object to protrude from the top (e.g., a basket).
    """
    cont_pos = pose_container_7d[:3]
    cont_quat = pose_container_7d[3:]
    
    # Transform object position to container local frame
    inv_rotation = R.from_quat(cont_quat).as_matrix().T
    local_pos = inv_rotation.dot(pos_obj - cont_pos)
    
    hx, hy, hz = extents_container / 2.0
    
    # XY boundary check
    inside_x = abs(local_pos[0]) + radius_obj <= hx
    inside_y = abs(local_pos[1]) + radius_obj <= hy
    
    # Z-axis (height) handling
    if top_open:
        # Must be above the container floor, but can extend above the top rim
        inside_z = local_pos[2] - radius_obj >= -hz
    else:
        # Must be fully enclosed within Z bounds
        inside_z = abs(local_pos[2]) + radius_obj <= hz
        
    return inside_x and inside_y and inside_z


def check_approach_corridor_world(pos_obj, pos_anchor, approach_axis=2, approach_dir=1.0, \
    mo_size=None, tgt_size=None, min_radius=0.01, min_height=0.60, relax_factor=1.2, depth_multiplier=2.0):
    """
    Defines an absolute approach corridor in world coordinates for container entry.
    Supports alignment on XY, XZ, or YZ planes.
    
    Examples:
        Basketball (Z-axis approach, XY alignment): axis=2, dir=1.0
        Drawer (Y-axis approach, XZ alignment): axis=1, dir=-1.0
    """
    # 1. Calculate depth difference on the primary approach axis
    delta_depth = (pos_obj[approach_axis] - pos_anchor[approach_axis]) * approach_dir
    
    # 2. Extract non-dominant axes (the alignment plane)
    plane_indices = [i for i in range(3) if i != approach_axis]
    
    # 3. Compute projection deviation on the alignment plane
    plane_obj = pos_obj[plane_indices]
    plane_anchor = pos_anchor[plane_indices]
    plane_dist = np.linalg.norm(plane_obj - plane_anchor)
    
    # ==========================================
    # 4. Adaptive corridor radius calculation
    target_opening_radius = min(tgt_size[plane_indices[0]], tgt_size[plane_indices[1]]) / 2.0
    
    corridor_radius = max(min_radius, (target_opening_radius - mo_size) * relax_factor)
    # 5. Corridor height calculation based on container depth
    target_depth = tgt_size[approach_axis]
    corridor_height = max(min_height, target_depth * depth_multiplier)
    
    
    return (0.0 < delta_depth < corridor_height) and (plane_dist < corridor_radius)



def get_graph_diff(prev_g, curr_g, config):
    """
    Computes semantic differences between two scene graph frames.
    Monitors intention shifts, relation additions/removals, and boolean state flips.
    """
    diffs = []
    
    # 1. Global intention change
    if prev_g["action_type"] != curr_g["action_type"]:
        diffs.append(f"Intent Shift: {prev_g['action_type']} -> {curr_g['action_type']}")
  
    def _find_edge(edges_list, obj1, obj2):
        for e in edges_list:
            if e.get("object1") == obj1 and e.get("object2") == obj2:
                return e
        return None

    # Determine focus edges for the current task stage
    focus_edges = curr_g.get("focus_edges")
    if not focus_edges:
        mo_name = config.get("semantic_mo_name", "object")
        to_name = config.get("semantic_to_name", "target")
        focus_edges = [("gripper", mo_name), (mo_name, to_name)]

    # Monitor edge and geometric attribute changes
    for obj1, obj2 in focus_edges:
        edge_name = f"{obj1}->{obj2}"
        
        prev_edge = _find_edge(prev_g.get("edges", []), obj1, obj2)
        curr_edge = _find_edge(curr_g.get("edges", []), obj1, obj2)
        
        if not prev_edge or not curr_edge:
            continue
            
        # Set operations for added/removed relations
        p_rels = set(prev_edge.get("spatial_relations", []))
        c_rels = set(curr_edge.get("spatial_relations", []))
        added, removed = c_rels - p_rels, p_rels - c_rels
        if added: diffs.append(f"Added {edge_name}: {list(added)}")
        if removed: diffs.append(f"Removed {edge_name}: {list(removed)}")
        
        # Boolean attribute monitoring
        for attr in ["is_touching", "is_stable"]:
            p_val, c_val = prev_edge.get(attr), curr_edge.get(attr)
            if p_val != c_val:
                diffs.append(f"State Flip {edge_name} [{attr}]: {p_val} -> {c_val}")

    prev_thr = prev_g.get("task_thresholds", {})
    curr_thr = curr_g.get("task_thresholds", {})
 
    for key in ["active_cup_name", "current_stage"]:
        if prev_thr.get(key) != curr_thr.get(key):
            diffs.append(f"Threshold Update [{key}]: {prev_thr.get(key)} -> {curr_thr.get(key)}")
                
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
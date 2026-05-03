import numpy as np

class ProgressEstimator:
    """
    Heuristic progress estimation engine driven by MomaGraph scene graphs.
    It maps physical spatial relationships and topological states into a 
    normalized progress value [0.0, 1.0].
    """
    def __init__(self, manipulate_obj: str, target_obj: str, drop_axis: int = 2, custom_milestones: dict = None):
        """
        :param manipulate_obj: Name of the object being manipulated (e.g., "ball").
        :param target_obj: Name of the target container or anchor (e.g., "hoop").
        :param drop_axis: The dominant axis for the final release/insertion phase (0:X, 1:Y, 2:Z). Default is 2.
        :param custom_milestones: Optional dictionary to override default progress stage thresholds.
        """
        self.manipulate_obj = manipulate_obj
        self.target_obj = target_obj
        self.drop_axis = drop_axis
        
        # ====================================================================
        # Default milestones for standard short-sequence manipulation tasks.
        # These represent normalized progress values at specific semantic states.
        # ====================================================================
        self.milestones = {
            "p1_approach_max": 0.30,   # Max progress during the approach phase
            "p1_around_val": 0.32,     # Reached gripper alignment (AROUND)
            "p1_touch_val": 0.35,      # Initial contact (TOUCHING)
            "p2_hold_start": 0.50,     # Stable grasp established (HOLDING)
            "p3_align_start": 0.85,    # Aligned with target (AROUND) - critical threshold
            "p3_drop_max": 0.95,       # Maximum progress during the final drop/insertion
            "done_val": 1.0            # Task completion
        }
        
        if custom_milestones:
            self.milestones.update(custom_milestones)
            
        # State memory variables to ensure monotonic distance normalization
        self.align_start_dist = None  
        self.transport_start_dist = None


    def _get_edge(self, edges: list, obj1: str, obj2: str):
        """Retrieves an edge between two specific objects from the graph."""
        for edge in edges:
            if edge["object1"] == obj1 and edge["object2"] == obj2:
                return edge
        return None

    def _compute_linear_progress(self, current_val, max_val, base_prog, max_prog):
        """Standard decay-based linear interpolation: maps current_val [max_val -> 0] to [base_prog -> max_prog]"""
        if max_val <= 1e-5: return max_prog
        ratio = 1.0 - (current_val / max_val)
        ratio = max(0.0, min(1.0, ratio)) 
        return base_prog + ratio * (max_prog - base_prog) 


    def _interpolate(self, current_val, val_start, val_end, prog_start, prog_end):
        """Maps a physical value from range [val_start, val_end] to progress range [prog_start, prog_end]"""
        denominator = val_end - val_start
        if abs(denominator) <= 1e-5: 
            return prog_end
        ratio = (current_val - val_start) / denominator
        ratio = max(0.0, min(1.0, ratio))
        return prog_start + ratio * (prog_end - prog_start)

    def estimate(self, momagraph_frame) -> float:
        """
        Estimates task progress based on the current scene graph frame.
        Handles multi-stage tasks via fractal mapping and specialized tasks via intent-based logic.
        """
        edges = momagraph_frame.get("edges", [])
        action_type = momagraph_frame.get("action_type", "")
        thresholds = momagraph_frame.get("task_thresholds", {})
        ms = self.milestones

        # Dynamic Attention Routing: Focus on relevant edges based on current state 
        if "focus_edges" not in momagraph_frame:
            edge_gb = self._get_edge(edges, "gripper", self.manipulate_obj)
            edge_bh = self._get_edge(edges, self.manipulate_obj, self.target_obj)
            
            gb_rels = edge_gb.get("spatial_relations", []) if edge_gb else []
            bh_rels = edge_bh.get("spatial_relations", []) if edge_bh else [] 
            if "HOLDING" in gb_rels or "AROUND" in bh_rels:
                # Phases 2 & 3: Object is secured or near target. Focus on transport and stability.
                momagraph_frame["focus_edges"] = [("gripper", self.manipulate_obj), (self.manipulate_obj, self.target_obj)]
            else:
                # Phase 1: Hand approaching the object (Pick). Target is currently irrelevant.
                momagraph_frame["focus_edges"] = [("gripper", self.manipulate_obj)]
        
        # Task Termination
        if action_type == "done":
            return ms["done_val"]

        # Specialized Intent: Wipe Task
        # Directly utilizes the externally calculated cleaning progress.
        if action_type == "wipe" and "clean_progress" in thresholds:
            wipe_prog = thresholds["clean_progress"]
            # Maps clean_progress to the transport/done phase [50% -> 100%]
            return self._compute_linear_progress(
                1.0 - wipe_prog, 1.0, 
                ms["p2_hold_start"], ms["done_val"]
            )

        # Topological Graph Perception: Multi-stage Fractal Mapping
        # Triggered if multiple 'manipulate' edges exist (e.g., 'empty_dishwasher').
        manipulate_edges = [e for e in edges if e.get("functional_relationship") == "manipulate"]
        n_stages = max(1, len(manipulate_edges))

        if n_stages > 1:
            w = 1.0 / n_stages # Weight per sub-stage

            # Fractal Milestones: Dynamically generated based on sub-stage window
            s1_start = 0.0
            s1_touch = s1_start + w * ms["p1_touch_val"]
            s1_around = s1_start + w * ms["p3_align_start"]
            s1_end = w * 1.0
            
            s2_start = w * 1.0
            s2_touch = s2_start + w * ms["p1_touch_val"]
            s2_around = s2_start + w * ms["p3_align_start"]
            s2_end = w * 2.0
            
            s3_start = w * 2.0
            s3_touch = s3_start + w * ms["p1_touch_val"]
            s3_around = s3_start + w * ms["p3_align_start"]
            s3_end = s3_start + w * ms["p3_drop_max"]

            # Complex long-sequence logic (specific to multi-object dishwasher task) 
            edge_db = self._get_edge(edges, "dishwasher_door", "dishwasher_base")
            edge_tb = self._get_edge(edges, "dishwasher_tray", "dishwasher_base")
            edge_pt = self._get_edge(edges, "plate", "dishwasher_tray")
            
            # --- Sub-stage 1: Open Door [0% - 33%] ---
            if action_type == "open":
                edge_gd = self._get_edge(edges, "gripper", "dishwasher_door")
                gd_rels = edge_gd.get("spatial_relations", []) if edge_gd else []
                if not edge_db: return s1_start
                dist = edge_db["geometric_attributes"]["dist"]
                orig_dist = edge_db["geometric_attributes"]["original_dist"]
                delta = dist - orig_dist
                rels = edge_db.get("spatial_relations", [])
                
                DOOR_OPEN = thresholds.get("door_open", 0.25)
                DOOR_AROUND = thresholds.get("door_around", 0.15)
                
                if "OPENED" in rels: 
                    return s1_end
                elif "AROUND" in rels:
                    return self._interpolate(delta, DOOR_AROUND, DOOR_OPEN, s1_around, s1_end)
                elif "HOLDING" in gd_rels or delta > 0.005: 
                    return self._interpolate(delta, 0.0, DOOR_AROUND, s1_touch, s1_around)
                else:
                    gd_dist = edge_gd["geometric_attributes"]["dist"] if edge_gd else 1.0
                    gd_orig = edge_gd["geometric_attributes"]["original_dist"] if edge_gd else 1.0
                    return self._interpolate(gd_dist, gd_orig, 0.0, s1_start, s1_touch)

            # --- Sub-stage 2: Pull Tray [33% - 66%] ---
            elif action_type == "pull":
                edge_gt = self._get_edge(edges, "gripper", "dishwasher_tray")
                gt_rels = edge_gt.get("spatial_relations", []) if edge_gt else []

                if not edge_tb: return s2_start
                dist = edge_tb["geometric_attributes"]["dist"]
                orig_dist = edge_tb["geometric_attributes"]["original_dist"]
                delta = dist - orig_dist
                rels = edge_tb.get("spatial_relations", [])
                
                TRAY_PULL = thresholds.get("tray_pull", 0.15)
                TRAY_AROUND = thresholds.get("tray_around", 0.09)
                
                if "OUTSIDE" in rels:
                    return s2_end
                elif "AROUND" in rels:
                    return self._interpolate(delta, TRAY_AROUND, TRAY_PULL, s2_around, s2_end)
                elif "HOLDING" in gt_rels or delta > 0.005:
                    return self._interpolate(delta, 0.0, TRAY_AROUND, s2_touch, s2_around)
                else:
                    gt_dist = edge_gt["geometric_attributes"]["dist"] if edge_gt else 1.0
                    gt_orig = edge_gt["geometric_attributes"]["original_dist"] if edge_gt else 1.0
                    return self._interpolate(gt_dist, gt_orig, 0.0, s2_start, s2_touch)

            # --- Sub-stage 3: Remove Plate [66% - 100%] ---
            elif action_type in ["pick", "place"]:
                edge_gp = self._get_edge(edges, "gripper", "plate")
                gp_rels = edge_gp.get("spatial_relations", []) if edge_gp else []
                
                if not edge_pt: return s3_start
                dist = edge_pt["geometric_attributes"]["dist"]
                orig_dist = edge_pt["geometric_attributes"]["original_dist"]
                delta = dist - orig_dist
                rels = edge_pt.get("spatial_relations", [])
                
                PLATE_DONE = thresholds.get("plate_done", 0.05)
                PLATE_AROUND = thresholds.get("plate_around", 0.02)
                
                if "OUTSIDE" in rels:
                    return s3_end
                elif "AROUND" in rels:
                    return self._interpolate(delta, PLATE_AROUND, PLATE_DONE, s3_around, s3_end)
                elif "HOLDING" in gp_rels or delta > 0.005:
                    return self._interpolate(delta, 0.0, PLATE_AROUND, s3_touch, s3_around)
                else:
                    gp_dist = edge_gp["geometric_attributes"]["dist"] if edge_gp else 1.0
                    gp_orig = edge_gp["geometric_attributes"]["original_dist"] if edge_gp else 1.0
                    return self._interpolate(gp_dist, gp_orig, 0.0, s3_start, s3_touch)

        # ====================================================================
        # Universal Logic for Short-Sequence / Single-Stage Tasks
        # ====================================================================
        edge_gb = self._get_edge(edges, "gripper", self.manipulate_obj)
        edge_bh = self._get_edge(edges, self.manipulate_obj, self.target_obj)
        
        if not edge_gb or not edge_bh: return 0.0
            
        gb_relations = set(edge_gb.get("spatial_relations", []))
        bh_relations = set(edge_bh.get("spatial_relations", []))
        gb_geo = edge_gb.get("geometric_attributes", {})
        bh_geo = edge_bh.get("geometric_attributes", {})

        # Reset distance memory when target switching is detected in multi-stage tasks (e.g., Chess)
        current_stage = thresholds.get("current_stage", 0)
        if hasattr(self, "last_stage_memory") and current_stage != self.last_stage_memory:
            self.transport_start_dist = None
            self.align_start_dist = None
        self.last_stage_memory = current_stage

        prog = 0.0 

        # Specialized Branch: Fridge Task (Tongue Interaction)
        # Progress decouples from spatial distance and depends solely on haptic push depth.
        if "tongue" in self.target_obj and "AROUND" in bh_relations:
            push_ratio = thresholds.get("tongue_push_ratio", 0.0)
            return ms["p3_align_start"] + push_ratio * (ms["p3_drop_max"] - ms["p3_align_start"])
            
        # Specialized Branch: Hinge Closure Tasks (Laptop/Box)
        if action_type == "close":
            current_dist = bh_geo.get("dist", 0.0)
            orig_dist = bh_geo.get("original_dist", 0.276)
            
            if "laptop" in self.target_obj:
                CLOSED_DIST = 0.0095 
                VERTICAL_DIST = 0.02
            else:
                CLOSED_DIST = 0.090
                VERTICAL_DIST = 0.160

            if "AROUND" in bh_relations:
                # Final closing phase [85% -> 95%]
                denominator = VERTICAL_DIST - CLOSED_DIST
                if denominator <= 1e-5: prog = ms["p3_drop_max"]
                else:
                    ratio = (VERTICAL_DIST - current_dist) / denominator
                    ratio = max(0.0, min(1.0, ratio))
                    prog = ms["p3_align_start"] + ratio * (ms["p3_drop_max"] - ms["p3_align_start"])

            else:
                # Initial closing phase [50% -> 85%]
                denominator = orig_dist - VERTICAL_DIST
                if denominator <= 1e-5: prog = ms["p3_align_start"]
                else:
                    ratio = (orig_dist - current_dist) / denominator
                    ratio = max(0.0, min(1.0, ratio))
                    prog = ms["p2_hold_start"] + ratio * (ms["p3_align_start"] - ms["p2_hold_start"])
    
        # Phase 3: Alignment and Place/Drop/Insert
        elif "AROUND" in bh_relations:
            delta_val = abs(bh_geo.get("delta_3d", [0, 0, 0])[self.drop_axis])
            
            if self.align_start_dist is None or delta_val > self.align_start_dist:
                self.align_start_dist = delta_val 
                
            prog = self._compute_linear_progress(
                delta_val, self.align_start_dist, 
                ms["p3_align_start"], ms["p3_drop_max"]
            )

        # Phase 2: Object Transport (Holding)
        elif "HOLDING" in gb_relations:
            delta_3d_bh = bh_geo.get("delta_3d", [1.0, 1.0, 1.0])
            # Use planar distance (excluding drop axis) to track transport progress
            plane_delta = [v for i, v in enumerate(delta_3d_bh) if i != self.drop_axis]
            dist_plane = np.linalg.norm(plane_delta)
            
            if self.transport_start_dist is None or dist_plane > self.transport_start_dist:
                self.transport_start_dist = dist_plane 
                  
            prog = self._compute_linear_progress(
                dist_plane, self.transport_start_dist, 
                ms["p2_hold_start"], ms["p3_align_start"]
            )

        # Phase 1: Approach (Pick)
        else:
            if "TOUCHING" in gb_relations:
                prog = ms["p1_touch_val"]
            elif "AROUND" in gb_relations:
                prog = ms["p1_around_val"]
            else:
                dist_gb = gb_geo.get("dist", 1.0)
                orig_dist_gb = gb_geo.get("original_dist", 1.0)
                prog = self._compute_linear_progress(
                    dist_gb, orig_dist_gb, 
                    0.0, ms["p1_approach_max"]
                )

        # Global Multi-stage Mapping: Scales sub-stage progress into the global [0, 1] range.
        total_stages = thresholds.get("total_stages", 1)
        if total_stages > 1: 
            current_stage = thresholds.get("current_stage", 0)
              
            if current_stage >= total_stages:
                return 1.0 # Lock at completion
                
            w = 1.0 / total_stages 
            return (current_stage * w) + (prog * w)

        return prog

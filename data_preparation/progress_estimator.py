import numpy as np

class ProgressEstimator:
    def __init__(self, manipulate_obj: str, target_obj: str, drop_axis: int = 2, custom_milestones: dict = None):
        """
        :param manipulate_obj: 被机械臂操作的物体名称 (例如 "ball", "red_block")
        :param target_obj: 目标容器/物体的名称 (例如 "hoop", "blue_block")
        :param drop_axis: 最终释放/插入阶段的主导轴 (0:X, 1:Y, 2:Z)。默认 2 (Z轴向下掉落)
        :param custom_milestones: 允许外部传入自定义的进度阶段比例字典，覆盖默认值
        """
        self.manipulate_obj = manipulate_obj
        self.target_obj = target_obj
        self.drop_axis = drop_axis
        
        # ==========================================
        # 基础/通用短序列任务进度阶段的里程碑参数 (Hyperparameters)
        # ==========================================
        self.milestones = {
            "p1_approach_max": 0.30,   # 靠近极限
            "p1_around_val": 0.32,     # 夹爪对齐(AROUND)
            "p1_touch_val": 0.35,      # 夹爪触碰(TOUCHING)
            "p2_hold_start": 0.50,     # 抓稳(HOLDING)
            "p3_align_start": 0.85,    # 目标对齐(AROUND)(接近完成的临界点)
            "p3_drop_max": 0.95,       # 下落/插入/放置极限
            "done_val": 1.0            # 完成
        }
        
        # 允许外部重写这些参数
        if custom_milestones:
            self.milestones.update(custom_milestones)
            
        # 状态记忆变量
        self.align_start_dist = None  
        self.transport_start_dist = None


    def _get_edge(self, edges: list, obj1: str, obj2: str):
        for edge in edges:
            if edge["object1"] == obj1 and edge["object2"] == obj2:
                return edge
        return None

    def _compute_linear_progress(self, current_val, max_val, base_prog, max_prog):
        if max_val <= 1e-5: return max_prog
        ratio = 1.0 - (current_val / max_val)
        ratio = max(0.0, min(1.0, ratio)) 
        return base_prog + ratio * (max_prog - base_prog)
        # 兼容旧逻辑的函数 (本质上是从 max_val 衰减到 0)
        # return self._interpolate(current_val, max_val, 0.0, base_prog, max_prog)

    # 通用线性插值器
    def _interpolate(self, current_val, val_start, val_end, prog_start, prog_end):
        """将物理值 (current_val) 从 [val_start, val_end] 完美映射到进度域 [prog_start, prog_end]"""
        denominator = val_end - val_start
        if abs(denominator) <= 1e-5: 
            return prog_end
        ratio = (current_val - val_start) / denominator
        ratio = max(0.0, min(1.0, ratio)) # 严格截断防越界
        return prog_start + ratio * (prog_end - prog_start)

    def estimate(self, momagraph_frame) -> float:
        edges = momagraph_frame.get("edges", [])
        action_type = momagraph_frame.get("action_type", "")
        thresholds = momagraph_frame.get("task_thresholds", {})
        ms = self.milestones  # 简写，方便调用

        # 动态注意力路由 (Dynamic Attention Routing) 
        if "focus_edges" not in momagraph_frame:
            edge_gb = self._get_edge(edges, "gripper", self.manipulate_obj)
            edge_bh = self._get_edge(edges, self.manipulate_obj, self.target_obj)
            
            gb_rels = edge_gb.get("spatial_relations", []) if edge_gb else []
            bh_rels = edge_bh.get("spatial_relations", []) if edge_bh else [] 
            if "HOLDING" in gb_rels or "AROUND" in bh_rels:
                # 阶段 2 & 3: 已经抓稳了物体，或者已经逼近目标区域。
                # 此时的焦点是把物体送过去，同时保持抓取稳定。
                momagraph_frame["focus_edges"] = [("gripper", self.manipulate_obj), (self.manipulate_obj, self.target_obj)]
            else:
                # 阶段 1: 手还在空中找物体的阶段 (Pick)。
                # 此时目标容器(target_obj)完全不重要，唯一焦点就是手和物体的距离 
                momagraph_frame["focus_edges"] = [("gripper", self.manipulate_obj)]
        
        # ==========================================
        # 任务完成 (100%)
        # ==========================================
        if action_type == "done":
            return ms["done_val"]

        # ====================================================================
        #  特殊意图  (Wipe 任务专用)
        # 当意图是 wipe 时，直接读取底层计算好的 clean_progress，无视任何物理插值 
        # ====================================================================
        if action_type == "wipe" and "clean_progress" in thresholds:
            # progress 范围被提取器限制在 0.0 ~ 0.98 之间
            # 将其映射到估算器的运送阶段 (50% ~ 100%)
            wipe_prog = thresholds["clean_progress"]
            # 让它从 p2_hold_start (50%) 开始，平滑涨向 done_val (100%)
            return self._compute_linear_progress(
                1.0 - wipe_prog, 1.0, 
                ms["p2_hold_start"], ms["done_val"]
            )

        # ====================================================================
        # 拓扑图智能感知：动态计算阶段数量
        # 核心逻辑：如果图谱中存在超过 1 条的 manipulate (夹爪操作) 边，说明这是一个具有多个操作目标的复杂多阶段任务 (如: empty_dishwasher)
        # ====================================================================
        manipulate_edges = [e for e in edges if e.get("functional_relationship") == "manipulate"]
        n_stages = max(1, len(manipulate_edges))

        if n_stages > 1:
            w = 1.0 / n_stages

            # 分形映射：基于基准节奏动态生成子阶段里程碑
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

            # 动态寻找长序列中的核心边 
            edge_db = self._get_edge(edges, "dishwasher_door", "dishwasher_base")
            edge_tb = self._get_edge(edges, "dishwasher_tray", "dishwasher_base")
            edge_pt = self._get_edge(edges, "plate", "dishwasher_tray")
            
            # --- 阶段 1: 开门 (0% ~ 33%) ---
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
                    # 抓紧了门或者门已经开始动了
                    return self._interpolate(delta, 0.0, DOOR_AROUND, s1_touch, s1_around)
                else:
                    # 夹爪正在接近门把手的漫长过程
                    gd_dist = edge_gd["geometric_attributes"]["dist"] if edge_gd else 1.0
                    gd_orig = edge_gd["geometric_attributes"]["original_dist"] if edge_gd else 1.0
                    return self._interpolate(gd_dist, gd_orig, 0.0, s1_start, s1_touch)

            # --- 阶段 2: 拉托盘 (33% ~ 66%) ---
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
                    # 开完门后，机械臂向下移动去抓托盘的过程
                    gt_dist = edge_gt["geometric_attributes"]["dist"] if edge_gt else 1.0
                    gt_orig = edge_gt["geometric_attributes"]["original_dist"] if edge_gt else 1.0
                    return self._interpolate(gt_dist, gt_orig, 0.0, s2_start, s2_touch)

            # --- 阶段 3: 拿取并移除盘子 (66% ~ 100%) ---
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
        # 以下均为短序列 / 单阶段 / 多阶段动态游标任务 通用通道
        # ====================================================================

        # 动态获取边
        edge_gb = self._get_edge(edges, "gripper", self.manipulate_obj)
        edge_bh = self._get_edge(edges, self.manipulate_obj, self.target_obj)
        
        if not edge_gb or not edge_bh: return 0.0
            
        gb_relations = set(edge_gb.get("spatial_relations", []))
        bh_relations = set(edge_bh.get("spatial_relations", []))
        gb_geo = edge_gb.get("geometric_attributes", {})
        bh_geo = edge_bh.get("geometric_attributes", {})

        # 💡 核心保护：检测到多阶段任务(如棋子)切换目标时，立即重置内部的距离记忆变量
        current_stage = thresholds.get("current_stage", 0)
        if hasattr(self, "last_stage_memory") and current_stage != self.last_stage_memory:
            self.transport_start_dist = None
            self.align_start_dist = None
        self.last_stage_memory = current_stage

        # 存放计算出的当前子阶段进度 (0.0 ~ 1.0)
        prog = 0.0 

        # ==========================================
        # 专属分支：接冰块任务 (tongue 被推压)
        # 物理原理：一旦杯子 AROUND 接冰口，进度脱离空间距离，
        # 完全由机械推压的深度 (tongue_push_ratio) 决定 85% ~ 95% 的涨幅。
        # ==========================================
        if "tongue" in self.target_obj and "AROUND" in bh_relations:
            push_ratio = thresholds.get("tongue_push_ratio", 0.0)
            return ms["p3_align_start"] + push_ratio * (ms["p3_drop_max"] - ms["p3_align_start"])
            
        # ==========================================
        # 专属分支：铰链闭合任务 (action_type == "close")
        # ==========================================
        if action_type == "close":
            current_dist = bh_geo.get("dist", 0.0)
            orig_dist = bh_geo.get("original_dist", 0.276)
            
            if "laptop" in self.target_obj:  # 笔记本任务
                CLOSED_DIST = 0.0095 
                VERTICAL_DIST = 0.02
            else:
                CLOSED_DIST = 0.090    # 完全闭合
                VERTICAL_DIST = 0.160  # 垂直于桌面/接近闭合 (对应提取器中的 AROUND 阈值)

            # 阶段 3 ：已经达到 AROUND 状态 (接近闭合)
            # 进度直接跃升至 85%，并在 85% ~ 95% 之间做最后的收尾插值
            if "AROUND" in bh_relations:
                denominator = VERTICAL_DIST - CLOSED_DIST
                if denominator <= 1e-5: prog = ms["p3_drop_max"]
                else:
                    ratio = (VERTICAL_DIST - current_dist) / denominator
                    ratio = max(0.0, min(1.0, ratio))
                    prog = ms["p3_align_start"] + ratio * (ms["p3_drop_max"] - ms["p3_align_start"])

            # 阶段 2 ：刚刚开始往下压
            # 进度从 50% 起步，在 50% ~ 85% 之间平滑上涨
            else:
                denominator = orig_dist - VERTICAL_DIST
                if denominator <= 1e-5: prog = ms["p3_align_start"]
                else:
                    ratio = (orig_dist - current_dist) / denominator
                    ratio = max(0.0, min(1.0, ratio))
                    prog = ms["p2_hold_start"] + ratio * (ms["p3_align_start"] - ms["p2_hold_start"])
    
        # ==========================================
        # 阶段 3: 对齐与插入/掉落/放置阶段 
        # ==========================================
        elif "AROUND" in bh_relations:
            delta_val = abs(bh_geo.get("delta_3d", [0, 0, 0])[self.drop_axis])
            
            if self.align_start_dist is None or delta_val > self.align_start_dist:
                self.align_start_dist = delta_val 
                
            prog = self._compute_linear_progress(
                delta_val, self.align_start_dist, 
                ms["p3_align_start"], ms["p3_drop_max"]
            )

        # ==========================================
        # 阶段 2: 运送目标 
        # ==========================================
        elif "HOLDING" in gb_relations:
            delta_3d_bh = bh_geo.get("delta_3d", [1.0, 1.0, 1.0])
            
            plane_delta = [v for i, v in enumerate(delta_3d_bh) if i != self.drop_axis]
            dist_plane = np.linalg.norm(plane_delta)
            
            if self.transport_start_dist is None or dist_plane > self.transport_start_dist:
                self.transport_start_dist = dist_plane 
                  
            prog = self._compute_linear_progress(
                dist_plane, self.transport_start_dist, 
                ms["p2_hold_start"], ms["p3_align_start"]
            )

        # ==========================================
        # 阶段 1: 接近目标
        # ==========================================
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

        # ==========================================
        # 全局分形映射 (自动处理多目标连续任务)
        # ==========================================
        total_stages = thresholds.get("total_stages", 1)
        if total_stages > 1:
            # w = 1.0 / total_stages
            # # 缩放至全局区间: 基础进度 + 局部进度增量
            # return (current_stage * w) + (prog * w) 
            current_stage = thresholds.get("current_stage", 0)
             
            # 如果已经完成所有阶段，直接锁死在 1.0，不要再叠加残余进度
            if current_stage >= total_stages:
                return 1.0
                
            w = 1.0 / total_stages
            # 只有在未完成的阶段，才进行缩放
            return (current_stage * w) + (prog * w)

        return prog

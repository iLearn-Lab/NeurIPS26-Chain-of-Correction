import numpy as np

class ProgressCorrector:
    def __init__(self, config: dict = None):
        """
        基于进度评估器 (ProgressEstimator) 的底层物理约束，进行纠错干预。
        """
        self.cfg = {
            "tau_hold": 0.50,       # 运送阶段底线 (HOLDING)
            "tau_drop_safe": 0.85,  # 安全掉落区底线 (AROUND)
            "tau_dev": 0.02,        # 偏离容差 (任何大于2%的回退都是拓扑状态丢失)
            "eps_stuck": 1e-4,      # 进度停滞极小值
            "n_max": 15,            # 最大允许停滞步数
            "noise_std": 0.02,      # 探索噪声大小
            "lambda_factor": 0.8    # 牵引力度
        }
        if config: self.cfg.update(config)
        self.stuck_counter = 0

    def _get_vector(self, graph: dict, obj1: str, obj2: str) -> np.ndarray:
        for edge in graph.get("edges", []):
            if edge["object1"] == obj1 and edge["object2"] == obj2:
                return np.array(edge["geometric_attributes"]["delta_3d"])
        return np.zeros(3)

    def step(self, p_prev: float, p_curr: float, p_next: float, 
             graph: dict, a_pred: np.ndarray, eef_velocity: float = None) -> tuple[str, np.ndarray, dict]:
        """
        核心纠错推演。
        :param eef_velocity: (可选) 机械臂末端速度标量。用于防止误判安全抬升为卡死。
        """
        c = self.cfg
        v_gb = self._get_vector(graph, "gripper", "ball")
        v_bh = self._get_vector(graph, "ball", "hoop")
        
        # ==========================================
        # 1. 错后纠正 (Post-Error Correction)
        # ==========================================
        # 1.1 灾难性掉落 (半路掉球)
        if p_prev >= c["tau_hold"] and p_curr < c["tau_hold"]:
            graph_mod = graph.copy()
            graph_mod["nodes"].append("dropped_ball") # 注入掉球意图
            return "POST_DROP", a_pred, graph_mod

        # 1.2 物理卡死检测 (结合速度判断，防止误杀抬升和避障)
        if abs(p_curr - p_prev) < c["eps_stuck"]:
            # 如果提供了速度，且速度大于 1cm/s，说明在做有效的三维运动(如拔高)，不算卡死！
            if eef_velocity is not None and eef_velocity > 0.01:
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        if self.stuck_counter >= c["n_max"]:
            a_noisy = a_pred + np.random.normal(0, c["noise_std"], size=a_pred.shape)
            return "POST_STUCK", a_noisy, graph

        # ==========================================
        # 2. 错前预防 (Pre-Error Correction)
        # ==========================================
        # 2.1 预防掉落 (预测在半空松手，且没在安全走廊上方)
        if p_curr >= c["tau_hold"] and p_next < c["tau_hold"]:
            # 强制覆盖动作，把球死死拉向篮筐上方，禁止网络张开夹爪
            a_corrected = c["lambda_factor"] * v_bh
            return "PRE_PREVENT_DROP", a_corrected, graph

        # 2.2 纠正轨迹偏离 / 防止拓扑状态丢失
        # 只要进度发生倒退，说明最高水位线逻辑失效，必定是发生了物理剥离
        if (p_curr - p_next) > c["tau_dev"]:
            if p_curr < c["tau_hold"]:
                # 还没抓稳就偏离了，说明手在乱晃 -> 拉向球
                a_corrected = c["lambda_factor"] * v_gb
            else:
                # 抓稳了但在运送中途偏离了方向 -> 拉向目标
                a_corrected = c["lambda_factor"] * v_bh
            return "PRE_CORRECT_DEVIATION", a_corrected, graph

        # ==========================================
        # 3. 正常执行
        # ==========================================
        return "NORMAL", a_pred, graph
import numpy as np

class ProgressCorrector:
    """
    Implements corrective intervention and failure recovery logic based on 
    underlying physical constraints and milestones from the ProgressEstimator.
    """

    def __init__(self, config: dict = None):
        """
        Initializes the corrector with physics-based hyperparameters.

        Args:
            config (dict): Optional dictionary to override default thresholds.
        """
        self.cfg = {
            "tau_hold": 0.50,         # Delivery phase baseline (HOLDING established)
            "tau_drop_safe": 0.85,    # Safe release zone baseline (AROUND target)
            "tau_dev": 0.02,          # Deviation tolerance (regressions > 2% indicate state loss)
            "eps_stuck": 1e-4,        # Progress stagnation epsilon
            "n_max": 15,              # Maximum allowable stagnation steps before intervention
            "noise_std": 0.02,        # Standard deviation for exploration/recovery noise
            "lambda_factor": 0.8      # Traction gain for corrective actions
        }
        if config:
            self.cfg.update(config)
        self.stuck_counter = 0

    def _get_vector(self, graph: dict, obj1: str, obj2: str) -> np.ndarray:
        """Extracts the 3D delta vector between two entities from the scene graph."""
        for edge in graph.get("edges", []):
            if edge["object1"] == obj1 and edge["object2"] == obj2:
                return np.array(edge["geometric_attributes"]["delta_3d"])
        return np.zeros(3)

    def step(self, p_prev: float, p_curr: float, p_next: float, 
             graph: dict, a_pred: np.ndarray, eef_velocity: float = None) -> tuple[str, np.ndarray, dict]:
        """
        Core logic for error inference and corrective steering.

        Args:
            p_prev (float): Progress in the previous frame.
            p_curr (float): Progress in the current frame.
            p_next (float): Predicted progress for the next frame.
            graph (dict): Current MomaGraph scene graph.
            a_pred (np.ndarray): Predicted action from the policy network.
            eef_velocity (float, optional): Scalar End-Effector velocity to filter false-positive stasis.

        Returns:
            tuple: (Intervention Mode, Output Action, Modified Scene Graph)
        """
        c = self.cfg
        v_gb = self._get_vector(graph, "gripper", "ball")
        v_bh = self._get_vector(graph, "ball", "hoop")
        
        # ==========================================
        # 1. Post-Error Recovery
        # ==========================================

        # 1.1 Catastrophic Drop Detection (Loss of object during transport)
        if p_prev >= c["tau_hold"] and p_curr < c["tau_hold"]:
            graph_mod = graph.copy()
            graph_mod["nodes"].append("dropped_ball") # Inject failure intent for the VLM/Policy
            return "POST_DROP", a_pred, graph_mod

        # 1.2 Physical Stasis/Stuck Detection
        if abs(p_curr - p_prev) < c["eps_stuck"]:
            # If velocity > 1cm/s, assume valid 3D motion (e.g., lifting or avoidance) and reset counter
            if eef_velocity is not None and eef_velocity > 0.01:
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        if self.stuck_counter >= c["n_max"]:
            # Inject noise to escape local minima or physical stagnation points
            a_noisy = a_pred + np.random.normal(0, c["noise_std"], size=a_pred.shape)
            return "POST_STUCK", a_noisy, graph

        # ==========================================
        # 2. Pre-emptive Error Prevention
        # ==========================================

        # 2.1 Drop Prevention (Predicted premature release outside safe target corridor)
        if p_curr >= c["tau_hold"] and p_next < c["tau_hold"]:
            # Force override: pull object toward target and suppress gripper release
            a_corrected = c["lambda_factor"] * v_bh
            return "PRE_PREVENT_DROP", a_corrected, graph

        # 2.2 Trajectory Deviation / State Consistency Correction
        # Any regression in progress implies a failure in monotonic progress logic.
        if (p_curr - p_next) > c["tau_dev"]:
            if p_curr < c["tau_hold"]:
                # Deviation during approach: steer gripper toward the object
                a_corrected = c["lambda_factor"] * v_gb
            else:
                # Deviation during transport: steer object toward the target
                a_corrected = c["lambda_factor"] * v_bh
            return "PRE_CORRECT_DEVIATION", a_corrected, graph

        # ==========================================
        # 3. Nominal Execution
        # ==========================================
        return "NORMAL", a_pred, graph
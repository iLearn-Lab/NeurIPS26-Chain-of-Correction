import os
import pickle
import json
import argparse
import rlbench
import uvicorn 
import colosseum
import numpy as np
import threading 
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from pyrep.objects import Shape
from pyrep.errors import PyRepError 
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from pyrep.const import JointType

# Custom utility and framework imports
from data_preparation.utils import EndEffectorPoseViaPlanning2
from data_preparation.progress_estimator import ProgressEstimator
from data_preparation.main_annotation import TASK_REGISTRY, extract_momagraph_frame

app = FastAPI()

# ==========================================
# Global State and Synchronization
# ==========================================
ENV = None
TASK_INSTANCES = {}
EXTENTS_CACHE = {} 
SIM_LOCK = threading.Lock()
# ==========================================

class EvalRequest(BaseModel):
    """Schema for action evaluation requests."""
    task: str
    action: list
    episode_id: int
    frame_id: int
    baseline_progress: float
    source: str = ""
    ep_path: str = ""  # Absolute path from client

def get_task_extents(rlbench_task, config):
    """
    Extracts physical dimensions of the robot gripper and task-relevant objects.
    
    Returns:
        dict: A dictionary containing radii and 3D bounding box extents.
    """
    gripper = rlbench_task._robot.gripper
    gripper_max_width = 0.0
    for joint in gripper.joints:
        if joint.get_joint_type() == JointType.PRISMATIC:
            _, interval = joint.get_joint_interval()
            gripper_max_width += interval[1]
    
    # Fallback to standard 8cm width if detection fails
    gripper_max_width = 0.08 if gripper_max_width <= 1e-4 else gripper_max_width

    extents_dict = {"gripper_radius": gripper_max_width / 2.0, "success_radius": 0.05}
    
    # Iterate through objects defined in the registry to cache their bounding boxes
    for key, sim_name in config.get("objects", {}).items():
        try:
            bbox = Shape(sim_name).get_bounding_box()
            extents_dict[f"{key}_size"] = np.array([bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]])
        except Exception: 
            pass
    return extents_dict

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Managed lifespan for the FastAPI application to handle simulation startup/shutdown."""
    global ENV 
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True 
    obs_config.joint_velocities = True
    obs_config.gripper_open = True 
    obs_config.task_low_dim_state = True 
    obs_config.gripper_pose = True
    
    # Configure action modes aligned with evaluation benchmarks
    arm_action_mode = EndEffectorPoseViaPlanning2()
    gripper_action_mode = Discrete()
 
    def compatible_action(old_action_fn):
        def wrapper(scene, action, *args, **kwargs):
            return old_action_fn(scene, action)
        return wrapper

    # Ensure compatibility with older RLBench scene API calls
    arm_action_mode.action = compatible_action(arm_action_mode.action)
    gripper_action_mode.action = compatible_action(gripper_action_mode.action)

    action_mode = MoveArmThenGripper(arm_action_mode, gripper_action_mode)
     
    ENV = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True
    )
    ENV.launch()
    print("[Status] RLBench simulator successfully initialized in headless mode.")
    yield 
    ENV.shutdown()

app = FastAPI(lifespan=lifespan)

def restore_state_to_frame(rlbench_task, task_name, episode_id, frame_id, base_data_path):
    """
    Restores the simulator state to a specific frame from a recorded demonstration.
    
    Implements robust path resolution to handle variations in directory naming conventions.
    """
    # Candidate path resolution logic
    path_orig = os.path.join(base_data_path, task_name, "all_variations", "episodes", f"episode{episode_id}")
    path_no_suffix = ""
    if task_name.endswith("_0"):
        path_no_suffix = os.path.join(base_data_path, task_name[:-2], "all_variations", "episodes", f"episode{episode_id}")
    path_with_suffix = os.path.join(base_data_path, f"{task_name}_0", "all_variations", "episodes", f"episode{episode_id}")
    
    valid_path = None
    for p in [path_orig, path_no_suffix, path_with_suffix]:
        if p and os.path.exists(p):
            valid_path = p
            break
            
    if not valid_path:
        raise FileNotFoundError(f"[Error] Could not locate episode directory. Search space: {[path_orig, path_no_suffix, path_with_suffix]}")

    obs_file = os.path.join(valid_path, "low_dim_obs.pkl")
    if not os.path.exists(obs_file):
        raise FileNotFoundError(f"[Error] Physical state file missing: {obs_file}")

    with open(obs_file, 'rb') as f:
        obs_list = pickle.load(f)
        
    safe_frame_id = min(frame_id, len(obs_list) - 1)
    target_obs = obs_list[safe_frame_id]
    
    # Restore robot state
    rlbench_task._robot.arm.set_joint_positions(target_obs.joint_positions)
    rlbench_task._robot.gripper.set_joint_positions([target_obs.gripper_joint_positions[0]] * 2) 
    
    # Restore object poses from recorded misc metadata
    for obj_name, pose in target_obs.misc.items():
        try:
            obj_handle = Shape(obj_name)
            obj_handle.set_pose(pose)
        except Exception:
            continue 

    return target_obs

@app.post("/evaluate")
def evaluate_action(data: EvalRequest):
    """
    Evaluates a proposed action by restoring simulator state, executing the step,
    and calculating the resulting progress delta.
    """
    with SIM_LOCK:
        try:
            original_task_name = data.task
            core_task_name = original_task_name[:-2] if original_task_name.endswith("_0") else original_task_name
            
            if core_task_name not in TASK_REGISTRY:
                print(f"[Warning] Unknown task received: {core_task_name}")
                return {"delta_progress": -0.1}
 
            # Route to either training or failed demonstration subsets
            base_path = "/data/colosseum_dataset/failed_eval_demos" if "failed" in data.source else "/data/colosseum_dataset/train"
            
            config = TASK_REGISTRY[core_task_name]
            
            # Lazy loading and caching of task instances
            if core_task_name not in TASK_INSTANCES:
                task_class_name = "".join([word.capitalize() for word in core_task_name.split("_")])
                TaskClass = getattr(colosseum.rlbench.tasks if "chess" in core_task_name else rlbench.tasks, task_class_name)
                TASK_INSTANCES[core_task_name] = ENV.get_task(TaskClass) 
                EXTENTS_CACHE[core_task_name] = get_task_extents(TASK_INSTANCES[core_task_name], config)

            task = TASK_INSTANCES[core_task_name]
            extents = EXTENTS_CACHE[core_task_name]
 
            try:
                task.sample_variation()
                task.reset()
            except Exception:
                pass
            
            # Phase 1: State Restoration
            init_obs_dict = restore_state_to_frame(task, original_task_name, data.episode_id, data.frame_id, base_path)
            task._has_reset = True
            task._reset_called = True
            if hasattr(task, '_scene'):
                task._scene.has_init_episode = True
 
            # Phase 2: Action Pre-processing and Clipping
            action = np.array(data.action, dtype=np.float32)
            action[0] = np.clip(action[0], 0.1, 0.8) # Workspace X
            action[1] = np.clip(action[1], -0.5, 0.5) # Workspace Y
            action[2] = np.clip(action[2], 0.7, 1.5)  # Workspace Z
            
            # Quaternion normalization to ensure valid rotation
            quat = action[3:7]
            norm = np.linalg.norm(quat)
            action[3:7] = quat / norm if norm > 1e-6 else [0.0, 0.0, 0.0, 1.0]
 
            # Phase 3: Simulator Execution
            try:
                obs, reward, terminate = task.step(action.tolist())
            except RuntimeError:
                # Handle simulator solver crashes due to invalid action poses
                return {"delta_progress": -0.05}
            
            if getattr(obs, 'task_low_dim_state', None) is None:
                print(f"[Warning] Null physical state returned for task: {core_task_name}")
                return {"delta_progress": -0.05}

            # Phase 4: Perception and Progress Delta Calculation
            try:
                curr_graph = extract_momagraph_frame(
                    task_name=core_task_name, 
                    obs_dict=obs.__dict__, 
                    init_obs_dict=init_obs_dict.__dict__,
                    indices=config["indices"], 
                    extents=extents,
                    instructions=[""], 
                    prev_obs_dict=init_obs_dict.__dict__
                )
                
                estimator = ProgressEstimator(
                    manipulate_obj=config["semantic_mo_name"], 
                    target_obj=config["semantic_to_name"]
                )
                new_progress = estimator.estimate(curr_graph) 
                delta_p = new_progress - data.baseline_progress
                 
            except (TypeError, IndexError, KeyError) as e:
                print(f"[Warning] Scene graph extraction failed for {core_task_name}: {e}")
                delta_p = -0.05
            
            return {"delta_progress": float(delta_p)}
            
        except Exception as e:
            print(f"[Critical Error] Unexpected exception: {e}")
            traceback.print_exc()
            return {"delta_progress": -0.1}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--task", type=str, default=None, help="Optionally limit this instance to a specific task.")
    args = parser.parse_args()
    
    # Entry point for Uvicorn ASGI server
    uvicorn.run(app, host="0.0.0.0", port=args.port)
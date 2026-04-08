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

# 使用与测试对齐的 Planning 模式
from data_preparation.utils import EndEffectorPoseViaPlanning2
from data_preparation.progress_estimator import ProgressEstimator
from data_preparation.main_annotation import TASK_REGISTRY, extract_momagraph_frame

app = FastAPI()

# ================= 全局变量与锁 =================
ENV = None
TASK_INSTANCES = {}
EXTENTS_CACHE = {} 
SIM_LOCK = threading.Lock()
# ================================================

class EvalRequest(BaseModel):
    task: str
    action: list
    episode_id: int
    frame_id: int
    baseline_progress: float
    source: str = ""
    ep_path: str = ""  # 接收客户端发来的绝对路径

def get_task_extents(rlbench_task, config):
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
        except Exception: 
            pass
    return extents_dict

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENV 
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True 
    obs_config.joint_velocities = True
    obs_config.gripper_open = True 
    obs_config.task_low_dim_state = True 
    obs_config.gripper_pose = True
    
    arm_action_mode = EndEffectorPoseViaPlanning2()
    gripper_action_mode = Discrete()
 
    def compatible_action(old_action_fn):
        def wrapper(scene, action, *args, **kwargs):
            return old_action_fn(scene, action)
        return wrapper

    arm_action_mode.action = compatible_action(arm_action_mode.action)
    gripper_action_mode.action = compatible_action(gripper_action_mode.action)

    action_mode = MoveArmThenGripper(arm_action_mode, gripper_action_mode)
     
    ENV = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True
    )
    ENV.launch()
    print("✅ RLBench 仿真器已成功在后台启动并待命！")
    yield 

app = FastAPI(lifespan=lifespan)

def restore_state_to_frame(rlbench_task, task_name, episode_id, frame_id, base_data_path):
    """
    终极探路版：不依赖客户端路径，服务端自动遍历所有可能的命名后缀
    """
    # 候选 1：客户端传来的原始名字 (比如 hockey_0 或 hockey)
    path1 = os.path.join(base_data_path, task_name, "all_variations", "episodes", f"episode{episode_id}")
    
    # 候选 2：尝试去掉 _0
    path2 = ""
    if task_name.endswith("_0"):
        path2 = os.path.join(base_data_path, task_name[:-2], "all_variations", "episodes", f"episode{episode_id}")
        
    # 候选 3：尝试强行加上 _0
    path3 = os.path.join(base_data_path, f"{task_name}_0", "all_variations", "episodes", f"episode{episode_id}")
    
    valid_path = None
    for p in [path1, path2, path3]:
        if p and os.path.exists(p):
            valid_path = p
            break
            
    if not valid_path:
        raise FileNotFoundError(f"❌ 无法找到匹配的 episode 目录。尝试了:\n1. {path1}\n2. {path2}\n3. {path3}")

    obs_file = os.path.join(valid_path, "low_dim_obs.pkl")
    if not os.path.exists(obs_file):
        raise FileNotFoundError(f"❌ 关键物理文件缺失: {obs_file}")

    with open(obs_file, 'rb') as f:
        obs_list = pickle.load(f)
        
    safe_frame_id = min(frame_id, len(obs_list) - 1)
    target_obs = obs_list[safe_frame_id]
    
    rlbench_task._robot.arm.set_joint_positions(target_obs.joint_positions)
    rlbench_task._robot.gripper.set_joint_positions([target_obs.gripper_joint_positions[0]] * 2) 
    
    for obj_name, pose in target_obs.misc.items():
        try:
            obj_handle = Shape(obj_name)
            obj_handle.set_pose(pose)
        except Exception:
            continue 

    return target_obs

@app.post("/evaluate")
def evaluate_action(data: EvalRequest):
    with SIM_LOCK:
        try:
            original_task_name = data.task
            core_task_name = original_task_name[:-2] if original_task_name.endswith("_0") else original_task_name
            
            if core_task_name not in TASK_REGISTRY:
                print(f"⚠️ 拦截到未知任务名: {core_task_name}")
                return {"delta_progress": -0.1}
 
            base_path = "/data/colosseum_dataset/failed_eval_demos" if "failed" in data.source else "/data/colosseum_dataset/train"
            
            config = TASK_REGISTRY[core_task_name]
                       
            if core_task_name not in TASK_INSTANCES:
                task_class_name = "".join([word.capitalize() for word in core_task_name.split("_")])
                TaskClass = getattr(colosseum.rlbench.tasks if "chess" in core_task_name else rlbench.tasks, task_class_name)
                TASK_INSTANCES[core_task_name] = ENV.get_task(TaskClass) 
                EXTENTS_CACHE[core_task_name] = get_task_extents(TASK_INSTANCES[core_task_name], config)

            task = TASK_INSTANCES[core_task_name]
            extents = EXTENTS_CACHE[core_task_name]
 
            try:
                task.sample_variation()
            except Exception:
                pass

            # 正规 reset (抑制多余日志)
            try:
                task.reset()
            except Exception:
                pass
            
            # 使用客户端发来的绝对路径加载
            init_obs_dict = restore_state_to_frame(task, original_task_name, data.episode_id, data.frame_id, base_path)

            task._has_reset = True
            task._reset_called = True
            if hasattr(task, '_scene'):
                task._scene.has_init_episode = True
 
            action = np.array(data.action, dtype=np.float32)
            action[0] = np.clip(action[0], 0.1, 0.8)
            action[1] = np.clip(action[1], -0.5, 0.5)
            action[2] = np.clip(action[2], 0.7, 1.5)
            
            quat = action[3:7]
            norm = np.linalg.norm(quat)
            action[3:7] = quat / norm if norm > 1e-6 else [0.0, 0.0, 0.0, 1.0]
 
            try:
                obs, reward, terminate = task.step(action.tolist())
            except RuntimeError as e:
                # 如果动作离谱导致 V-REP 规划器崩溃，静默返回惩罚分，不打印吓人的红字
                return {"delta_progress": -0.05}
            
            # 如果 obs 依然有问题（比如 task_low_dim_state 是 None），予以保护
            if getattr(obs, 'task_low_dim_state', None) is None:
                print(f"⚠️ 无法获取物理坐标状态 (任务:{core_task_name})")
                return {"delta_progress": -0.05}

            try:
                curr_graph = extract_momagraph_frame(
                    task_name=core_task_name, 
                    obs_dict=obs.__dict__, 
                    init_obs_dict=init_obs_dict.__dict__,
                    indices=config["indices"], 
                    extents=extents,
                    instructions=[""], # 🌟 核心修复 3：传列表防止 IndexError
                    prev_obs_dict=init_obs_dict.__dict__
                )
                
                estimator = ProgressEstimator(manipulate_obj=config["semantic_mo_name"], target_obj=config["semantic_to_name"])
                new_progress = estimator.estimate(curr_graph) 
                delta_p = new_progress - data.baseline_progress
                 
            except (TypeError, IndexError, KeyError) as e:
                print(f"⚠️ 场景图提取失败 (任务:{core_task_name}), 疑似环境初始化不全: {e}")
                traceback.print_exc()  # 🌟 加上这行，让所有隐藏的 Bug 无所遁形
                delta_p = -0.05
            
            return {"delta_progress": float(delta_p)}
            
        except Exception as e:
            print(f"🔴 [致命错误] {e}")
            traceback.print_exc()
            return {"delta_progress": -0.1}

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8080)
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--task", type=str, default=None, help="指定该实例只负责的任务名")
    args = parser.parse_args()
    
    # 如果指定了任务，可以在启动时预加载，进一步提速
    # if args.task:
    #     print(f"🚀 正在预加载任务: {args.task}")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
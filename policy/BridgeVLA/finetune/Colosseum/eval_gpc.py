import requests
import json
import base64
from io import BytesIO
from PIL import Image
from termcolor import cprint
# -------------------------

import os
import yaml
import csv
import torch
import cv2
import shutil
import numpy as np
from omegaconf import OmegaConf
from multiprocessing import Value
import copy
from copy import deepcopy
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
 
from PIL import Image
from termcolor import cprint
# -------------------------
 
from collect_failure_cases import save_failed_episode

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.utils.log_writer import LogWriter
from yarr.agents.agent import VideoSummary, ActResult

import bridgevla.mvt.config as default_mvt_cfg
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.config as default_exp_cfg

from bridgevla.mvt.mvt import MVT
from bridgevla.libs.peract.helpers import utils
from utils.custom_rlbench_env import CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv
from utils.peract_utils_colosseum import CAMERAS, SCENE_BOUNDS, IMAGE_SIZE, get_official_peract
from utils.rlbench_planning import EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning
from bridgevla.utils.rvt_utils import TensorboardManager, get_eval_parser, COLOSSEUM_TASKS
from bridgevla.utils.rvt_utils import load_agent as load_agent_state

from colosseum import ASSETS_CONFIGS_FOLDER, ASSETS_JSON_FOLDER, TASKS_PY_FOLDER, TASKS_TTM_FOLDER
from colosseum.rlbench.utils import ObservationConfigExt, check_and_make, name_to_class, save_demo


TASK_PROMPTS = {
    "meat_on_grill": "The robot needs to place the piece of meat onto the grill.",
    "move_hanger": "The robot needs to move the clothes hanger to the specified location.",
    "open_drawer": "The robot needs to successfully pull the drawer open.",
    "place_wine_at_rack_location": "The robot needs to grasp the wine bottle and place it securely on the rack.",
    "put_money_in_safe": "The robot needs to place the stack of money inside the safe.",
    "scoop_with_spatula": "The robot needs to use the spatula to scoop up the target object.",
    "setup_chess": "The robot needs to stand the fallen chess piece upright on the board.",
    "turn_oven_on": "The robot needs to grasp and turn the oven dial to switch it on."
}

class GPCRanker:
    def __init__(self, task_name, host="localhost", port=6065):
        self.url = f"http://{host}:{port}"
        self.task_name = task_name
        self.num_trial = 8  
        cprint(f"Connected to GPC-RANK server: {self.url}", "cyan")

    def rank_actions(self, raw_obs, candidates, lang_goal, step_idx):
        images_b64 = [] 
        for cam in ['front', 'left_shoulder', 'right_shoulder', 'wrist']:
            img_key = f"{cam}_rgb"
            if hasattr(raw_obs, img_key):
                img_array = getattr(raw_obs, img_key)
                pil_img = Image.fromarray(img_array)
                buffered = BytesIO()
                pil_img.save(buffered, format="JPEG")
                images_b64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
 
        candidates_list = [c.tolist() if isinstance(c, np.ndarray) else c for c in candidates]

        payload = {
            "instruction": lang_goal,
            "candidates": candidates_list,
            "images_b64": images_b64,
            "time_step": step_idx,
            "task_name": self.task_name 
        }

        try:
            response = requests.post(f"{self.url}/rank", json=payload, timeout=60)
            best_idx = response.json().get("best_action_idx", 0)
            return best_idx
        except Exception as e:
            cprint(f" GPC server processing failure: {e}, back to candidate 0", "red")
            return 0 

    def reset(self):
        try:
            requests.post(f"{self.url}/reset", timeout=5)
        except:
            pass

class GPCAgentWrapper: 
    def __init__(self, agent, ranker, env):
        self.agent = agent
        self.ranker = ranker
        self.env = env
        self.step_counter = 0
        
        self.agent.use_sampling = True
        self.agent.num_samples = self.ranker.num_trial
        self.agent.sampling_temp = 0.5 

    def act(self, step, obs, **kwargs):
        raw_act_result = self.agent.act(step, obs, **kwargs)
        
        candidates = []
        if isinstance(raw_act_result, list):
            for res in raw_act_result:
                candidates.append(res if isinstance(res, np.ndarray) else res.action)
        else:
            candidates.append(raw_act_result if isinstance(raw_act_result, np.ndarray) else raw_act_result.action)
        
        raw_obs = self.env.latest_raw_obs
        lang_goal = self.env._lang_goal
        
        best_idx = self.ranker.rank_actions(raw_obs, candidates, lang_goal, self.step_counter)
        self.step_counter += 1
        
        return ActResult(
            action=candidates[best_idx],
            observation_elements={},
            replay_elements={},
            info={}
        )

    def __getattr__(self, name):
        return getattr(self.agent, name)
 

def load_agent(model_path=None, exp_cfg_path=None, mvt_cfg_path=None, eval_log_dir="", device=0, use_input_place_with_mean=False):
    device = f"cuda:{device}"
    assert model_path is not None
    model_folder = os.path.join(os.path.dirname(model_path))
    exp_cfg = default_exp_cfg.get_cfg_defaults()
    if exp_cfg_path != None: exp_cfg.merge_from_file(exp_cfg_path)
    else: exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))
    if not use_input_place_with_mean:
        old_place_with_mean = exp_cfg.rvt.place_with_mean
        exp_cfg.rvt.place_with_mean = True
    exp_cfg.freeze()
    mvt_cfg = default_mvt_cfg.get_cfg_defaults()
    if mvt_cfg_path != None: mvt_cfg.merge_from_file(mvt_cfg_path)
    else: mvt_cfg.merge_from_file(os.path.join(model_folder, "mvt_cfg.yaml"))
    mvt_cfg.freeze()
    if mvt_cfg.stage_two:
        exp_cfg.defrost()
        exp_cfg.rvt.place_with_mean = old_place_with_mean
        exp_cfg.freeze()
    rvt = MVT(renderer_device=device, **mvt_cfg)
    agent = bridgevla_agent.RVTAgent(
        network=rvt.to(device), image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two, rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS, cameras=CAMERAS, log_dir=f"{eval_log_dir}/eval_run",
        **exp_cfg.peract, **exp_cfg.rvt,
    )
    agent.build(training=False, device=device)
    load_agent_state(model_path, agent)
    agent.eval()
    return agent


@torch.no_grad()
def eval(agent, tasks, eval_datafolder, start_episode=0, eval_episodes=25, episode_length=25, replay_ground_truth=False, device=0, headless=True, logging=False, log_dir=None, verbose=True, save_video=False, save_failed=True, gpc_wm_dir=""):
 
    agent.eval()
    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")
    if save_failed:
        obs_config.joint_velocities = True
        obs_config.joint_positions = True
        obs_config.joint_forces = True
        obs_config.gripper_open = True
        obs_config.gripper_pose = True
        obs_config.gripper_matrix = True
        obs_config.gripper_joint_positions = True
        obs_config.gripper_touch_forces = True
        obs_config.task_low_dim_state = True
    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)
    task_files = [t.replace(".py", "") for t in os.listdir(rlbench_task.TASKS_PATH) if t != "__init__.py" and t.endswith(".py")]
    task_classes = []
    if tasks[0] == "all": tasks = COLOSSEUM_TASKS
    task_class_variation_idx = []
    task_class_base = []
    for task in tasks:
        task_class_base.append('_'.join(task.split('_')[:-1]))
        if task_class_base[-1] not in task_files: raise ValueError('Task %s not recognised!.' % task)
        task_class = name_to_class(task_class_base[-1], TASKS_PY_FOLDER)
        task_class_variation_idx.append(int(task.split('_')[-1]))
        task_classes.append(task_class)
    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes, observation_config=obs_config, action_mode=action_mode,
        dataset_root=eval_datafolder, episode_length=episode_length, headless=headless,
        swap_task_every=eval_episodes, include_lang_goal_in_obs=True, time_in_state=True,
        record_every_n=1 if save_video else -1, base_cfg_name=task_class_base,
        task_class_variation_idx=task_class_variation_idx,
    )
    eval_env.eval = True
    device = f"cuda:{device}"
    if logging:
        assert log_dir is not None
        csv_file = "eval_results.csv"
        if not os.path.exists(os.path.join(log_dir, csv_file)):
            with open(os.path.join(log_dir, csv_file), "w") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()
    rollout_generator = RolloutGenerator(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)
  
    target_task_name = task_class_base[0]
    ranker = GPCRanker(task_name=target_task_name, host="localhost", port=6065)
    wrapped_agent = GPCAgentWrapper(agent, ranker, eval_env)

    eval_env.launch()
    num_tasks = len(tasks)
    step_signal = Value("i", -1)
    scores = []

    for task_id in range(num_tasks):
        task_rewards = []
        language_goals=[]
        total_count=0
        for ep in range(start_episode, start_episode + eval_episodes):
            ranker.reset()
            wrapped_agent.step_counter = 0
            episode_rollout = []
            generator = rollout_generator.generator(
                step_signal=step_signal, env=eval_env, agent=wrapped_agent, 
                episode_length=episode_length, timesteps=1, eval=True,
                eval_demo_seed=ep, record_enabled=False, replay_ground_truth=replay_ground_truth,
                vlm_corrector=None, 
            )
            try:
                step_idx = 0
                for replay_transition in generator:
                    if verbose: print(f"   step {step_idx}: Selected action = {replay_transition.action[:8]}")
                    step_idx += 1
                    if save_failed:
                        pure_obs = copy.deepcopy(eval_env.latest_raw_obs)
                        pure_obs.action = replay_transition.action
                        if not isinstance(replay_transition.info, dict): replay_transition.info = {}
                        replay_transition.info['pure_rlbench_obs'] = pure_obs
                    episode_rollout.append(replay_transition)
                total_count+=1
            except StopIteration as e:
                assert False
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            language_goals.append(lang_goal)
            if verbose: print(f"Evaluating {task_name} | Episode {ep} | Total Episode {total_count} |Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}")
            
            if reward <= 99 and save_failed:
                try:
                    current_task_base = task_class_base[task_id]
                    current_variation = task_class_variation_idx[task_id]
                    pure_obs_history = [t.info['pure_rlbench_obs'] for t in episode_rollout]
                    save_failed_episode(
                        transitions=pure_obs_history, task_name=current_task_base, variation=current_variation,
                        episode_id=ep, descriptions=[lang_goal], base_dir="/data/colosseum_dataset/failed_eval_demos"
                    )
                except Exception as e:
                    print(f"Fail to save failure rollouts: {e}")

        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]
        if logging:
            with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_results = {"task": task_name}
                for s in summaries:
                    if s.name == "eval_envs/return": csv_results["success rate"] = s.value
                    elif s.name == "eval_envs/length": csv_results["length"] = s.value
                    elif s.name == "eval_envs/total_transitions": csv_results["total_transitions"] = s.value
                    if "eval" in s.name: s.name = "%s/%s" % (s.name, task_name)
                csv_writer.writerow(csv_results)
        else:
            for s in summaries:
                if "eval" in s.name: s.name = "%s/%s" % (s.name, task_name)

        if len(summaries) > 0: task_score = [s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name][0]
        else: task_score = "unknown"

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")
        scores.append(task_score)

    eval_env.shutdown()
    if logging: csv_fp.close()
    agent.train()
    return scores

def get_model_index(filename):
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try: return int(filename[:-4].split("_")[-1])
        except: return None
    return None

def _eval(args):
    assert args.model_name is not None
    model_path=os.path.join(args.model_folder, args.model_name)
    tb = TensorboardManager(args.eval_log_dir)
    tasks_to_eval = deepcopy(args.tasks)

    model_idx = get_model_index(model_path)
    if model_idx is None: model_idx = 0

    agent = load_agent(
        model_path=model_path, exp_cfg_path=args.exp_cfg_path, mvt_cfg_path=args.mvt_cfg_path,
        eval_log_dir=args.eval_log_dir, device=args.device, use_input_place_with_mean=args.use_input_place_with_mean
    )

    agent_eval_log_dir = os.path.join(args.eval_log_dir, os.path.basename(model_path).split(".")[0])
    os.makedirs(agent_eval_log_dir, exist_ok=True)
    
    scores = eval(
        agent=agent, tasks=tasks_to_eval, eval_datafolder=args.eval_datafolder,
        start_episode=args.start_episode, eval_episodes=args.eval_episodes,
        episode_length=args.episode_length, replay_ground_truth=args.ground_truth,
        device=args.device, headless=args.headless, logging=True, log_dir=agent_eval_log_dir,
        verbose=True, save_video=args.save_video, save_failed=args.save_failed,
        gpc_wm_dir=args.gpc_wm_dir  
    )
    print(f"model {model_path}, scores {scores}")
    task_scores = {tasks_to_eval[i]: scores[i] for i in range(len(tasks_to_eval))}
    tb.update("eval", model_idx, task_scores)
    tb.writer.flush()
    tb.close()

if __name__ == "__main__":
    parser = get_eval_parser()
    parser.add_argument("--save_failed", action="store_true", default=False, help="Save failed episodes")
    parser.add_argument("--gpc_wm_dir", type=str, default="/data/colosseum_dataset/gpc_data/world_model_ckpt", help="Root directory of GPC world models")
    args = parser.parse_args()

    if args.log_name is None: args.log_name = "none"
    args.eval_log_dir = os.path.join(args.model_folder, args.log_name)
    os.makedirs(args.eval_log_dir, exist_ok=True)

    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)

    _eval(args)

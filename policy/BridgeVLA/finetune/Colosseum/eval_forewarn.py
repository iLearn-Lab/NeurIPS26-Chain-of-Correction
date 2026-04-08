import os
import torch
import numpy as np
import copy
from collections import deque
from termcolor import cprint
 
from eval import load_agent 
from utils.custom_rlbench_env import CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv
from utils.peract_utils_colosseum import CAMERAS, IMAGE_SIZE
from bridgevla.libs.peract.helpers import utils
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from utils.rlbench_planning import EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning
from bridgevla.utils.rvt_utils import get_eval_parser, name_to_class, COLOSSEUM_TASKS, TASKS_PY_FOLDER
 
from llama_recipes.models.mllama_model import MllamaForConditionalGenerationWM, MllamaWMProcessor

class ForewarnRanker:
    """FOREWARN 潜空间动作筛选器"""
    def __init__(self, task_name, vlm_path, wm_path, device, num_candidates=8):
        self.device = device
        self.num_candidates = num_candidates
        cprint(f"--- Loading FOREWARN Ranker for {task_name} ---", "magenta")
        
        # 1. 加载微调后的 Llama-WM (包含已训练好的 Projector)
        self.model = MllamaForConditionalGenerationWM.from_pretrained(
            vlm_path, torch_dtype=torch.bfloat16, device_map=device
        )
        # 2. 内部初始化世界模型 (WM)
        # 注意：initialize_vision_model 会读取 config 里的 wm_path
        self.model.initialize_vision_model() 
        self.model.eval()
        
        self.processor = MllamaWMProcessor.from_pretrained(vlm_path)

    @torch.no_grad()
    def rank_actions(self, history_obs, candidates, lang_goal):
        """对候选动作进行潜空间打分"""
        # 这里封装发送给 Llama 进行 Stage 2 选择的逻辑
        # 简单起见，这里逻辑对应 llama_wm_infer.py 的生成过程
        # 返回得分最高的 action 索引
        return 0 # 占位符：实际逻辑需对接 model.generate 并解析 text 输出的 ID

def eval_forewarn(args):
    # 1. 加载基础策略 (Base Policy, e.g., BridgeVLA)
    base_agent = load_agent(
        model_path=os.path.join(args.model_folder, args.model_name),
        device=args.device
    )

    # 2. 环境初始化 (同原 eval.py)
    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")
    action_mode = MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete())
    
    # 3. 任务循环
    tasks = args.tasks if args.tasks[0] != "all" else COLOSSEUM_TASKS
    for task_name in tasks:
        # 🌟 为每个任务初始化专用的 FOREWARN 打分器
        # 路径需要根据你训练好的 VLM ckpt 动态调整
        ranker = ForewarnRanker(
            task_name=task_name,
            vlm_path=f"/data/finetuned_models/Forewarn_VLMs/{task_name}_ckpt",
            wm_path=f"logs/dreamer_cont/{task_name}/...", # 指向你正在练的模型
            device=args.device
        )

        env = CustomMultiTaskRLBenchEnv(
            task_classes=[name_to_class('_'.join(task_name.split('_')[:-1]), TASKS_PY_FOLDER)],
            observation_config=obs_config,
            action_mode=action_mode,
            dataset_root=args.eval_datafolder,
            headless=args.headless,
            device=args.device
        )
        env.launch()

        for ep in range(args.eval_episodes):
            obs = env.reset()
            history_buffer = deque(maxlen=5) # 维护 5 帧历史
            done = False
            
            while not done:
                # A. 基础策略给出建议动作
                base_action = base_agent.predict(obs) 
                
                # B. 生成扰动候选 (Action Proposals)
                candidates = []
                candidates.append(base_action)
                for _ in range(ranker.num_candidates - 1):
                    noise = np.random.normal(0, 0.05, size=base_action.shape)
                    noise[-1] = 0 # 保持夹爪闭合状态离散性
                    candidates.append(base_action + noise)
                
                # C. 潜空间打分选优 (FOREWARN Core)
                # 使用 history_buffer 和 candidates 调用 ranker
                best_action = ranker.rank_actions(history_buffer, candidates, env._lang_goal)
                
                # D. 环境步进
                obs, reward, done, info = env.step(candidates[best_action])
                history_buffer.append(obs)
                
            print(f"Task {task_name} Ep {ep} Success: {reward > 99}")
        env.shutdown()

if __name__ == "__main__":
    parser = get_eval_parser()
    args = parser.parse_args()
    eval_forewarn(args)
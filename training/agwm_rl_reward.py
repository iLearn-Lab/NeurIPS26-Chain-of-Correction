import re
import json
import requests
import numpy as np


# 1. 定义端口映射表 (需与启动脚本一致)
TASK_PORT_MAP = {
    "hockey": 8001,
    "wipe_desk": 8002,
    "setup_chess": 8003,
    "stack_cups": 8004,
    "empty_dishwasher": 8005,
    "straighten_rope": 8006,
    "turn_oven_on": 8007,
    "put_money_in_safe": 8008,
    "move_hanger": 8009,
    "get_ice_from_fridge": 8010,
    "insert_onto_square_peg": 8011,
    "place_wine_at_rack_location": 8012,
    "scoop_with_spatula": 8013,
    "meat_on_grill": 8014,
    "open_drawer": 8015
}




REWARD_SERVER_URL = "http://127.0.0.1:8080/evaluate"


# ====================================================
# 🌟 强化学习超参矩阵 (Hyperparameter Matrix)
# ====================================================
W_FORMAT    = 2.0   # 基础格式约束
W_PHYSICAL  = 15.0  # 物理执行分 (delta_p)
W_SUCCESS   = 10.0  # 任务完成大奖

# --- 创新点: 认知与推演权重 ---
W_COG_EDGE     = 1.0  # 识别核心拓扑边
W_COG_CURR_P   = 1.0  # 感知当前进度
W_COG_FUTURE_P = 0.0  # 预判未来进度 (前向预测，权重更高)
W_COG_CORRECT  = 1.0  # 纠错意图判断正确
# ====================================================

def extract_action_from_response(response_str):
    try:
        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, str(response_str), re.DOTALL)
        if not match:
            json_pattern = r"\{.*?\}"
            match = re.search(json_pattern, str(response_str), re.DOTALL)
            if not match:
                return None
        
        data = json.loads(match.group(1) if "```json" in str(response_str) else match.group(0))
        action = data.get("corrected_action", [])
        correction_needed = data.get("correction_needed", False)
        
        if len(action) == 8:
            return {"action": action, "correction_needed": correction_needed}
        else:
            return None
    except Exception:
        return None



def extract_response_components(response_str):
    """提取思维链 <think> 和最终 JSON 动作"""
    think_content = ""
    think_match = re.search(r"<think>(.*?)</think>", str(response_str), re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        
    action_data = None
    try:
        json_pattern = r"```json\s*(.*?)\s*```"
        match = re.search(json_pattern, str(response_str), re.DOTALL)
        if not match:
            json_pattern = r"\{.*?\}"
            match = re.search(json_pattern, str(response_str), re.DOTALL)
            
        if match:
            raw_json = json.loads(match.group(1) if "```json" in str(response_str) else match.group(0))
            action = raw_json.get("corrected_action", [])
            correction_needed = raw_json.get("correction_needed", False)
            if len(action) == 8:
                action_data = {"action": action, "correction_needed": correction_needed}
    except Exception:
        pass
        
    return think_content, action_data



def compute_score(reward_inputs, **kwargs):
    scores = []
    
    for item in reward_inputs:
        solution_str = str(item["response"])
         
        print("\n" + "="*50)
        print("👀 [模型原始输出 (RAW OUTPUT)] 开始:")
        print(solution_str)
        print("="*50 + "\n")

        extra_info_dict = item.get('extra_info', {})
        info = extra_info_dict.get('extra_info', extra_info_dict) if isinstance(extra_info_dict, dict) else {}

        real_task_name = info.get('task', 'unknown')
        baseline_p = float(info.get('baseline_progress', 0.0))
        gt_correction = info.get('is_correction_sample', False) 
        gt_focus_edges = info.get('focus_edges', [])
        if not gt_focus_edges:
            task_thresholds = info.get('task_thresholds', {})
            gt_focus_edges = task_thresholds.get('focus_edges', []) if isinstance(task_thresholds, dict) else []

        # 解析路径
        ep_path_absolute = ""
        image_paths = item.get('images', []) 
        if isinstance(image_paths, list) and len(image_paths) > 0:
            match = re.search(r"(.*?/episodes/episode\d+)/", image_paths[0])
            if match: ep_path_absolute = match.group(1)

        # 提取模型输出
        think_content = ""
        think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL | re.IGNORECASE)
        if think_match: think_content = think_match.group(1).lower()
        else:
            split_text = solution_str.split("```json")
            if len(split_text) > 0:
                think_content = split_text[0].lower()

        parsed = extract_action_from_response(solution_str) 
        if not parsed: 
            print(f"⚠️ [{real_task_name}] 格式错误, 给予 {-W_FORMAT} 惩罚")
            scores.append({"overall": float(-W_FORMAT)})
            continue

        print(f"✅ [开始评估] {real_task_name} | 纠错: {parsed['correction_needed']}")

        # 2. 动态路由
        port = TASK_PORT_MAP.get(real_task_name, 8080) # 找不到就走默认
        target_url = f"http://127.0.0.1:{port}/evaluate"

        # ==========================================
        # 🧠 创新点 1: 场景图推演 (Focus Edges)
        # ==========================================
        metric_edge = 0.0
        if gt_focus_edges:
            matched_edges = 0
            for edge in gt_focus_edges:
                node1, node2 = str(edge[0]).replace('_', ' ').lower(), str(edge[1]).replace('_', ' ').lower()
                # 检查两个核心节点是否在思维链的同一句话或附近出现
                if node1 in think_content and node2 in think_content:
                    matched_edges += 1
            metric_edge = matched_edges / len(gt_focus_edges) # 归一化到 [0, 1]
            if matched_edges > 0:
                print(f"  ├─ [认知] 拓扑边命中: {matched_edges}/{len(gt_focus_edges)} -> 得分 {metric_edge:.2f}")
        else: 
            print("  ├─ [认知] ⚠️ 拓扑边: 数据集缺少真值，无法给分")
 
        # ==========================================
        # 🧠 创新点 2a: 当前进度感知
        # ==========================================
        metric_curr_p = 0.0 
        curr_p_match = re.search(r"progress[^0-9]*([0-9]+\.[0-9]+)", think_content)

        if curr_p_match:
            pred_curr_p = float(curr_p_match.group(1))
            error_curr = abs(pred_curr_p - baseline_p)
            metric_curr_p = max(0.0, 1.0 - (error_curr / 0.1)) # 10% 内按比例给分
            if metric_curr_p > 0:
                print(f"  ├─ [认知] 进度感知精准: 模型预估 {pred_curr_p}, 真实 {baseline_p:.3f} -> 得分 {metric_curr_p:.2f}")
            else:
                print(f"  ├─ [认知] 进度感知偏差过大: 模型预估 {pred_curr_p}, 真实 {baseline_p:.3f} -> 得分 0.00")
        else:
            print("  ├─ [认知] ⚠️ 进度感知: 未能在文本中提取到 progress 关键字和数字")

        # ==========================================
        # 纠错意图判断
        # ==========================================
        metric_correct = 1.0 if (parsed['correction_needed'] == gt_correction) else -1.0
        print(f"  ├─ [认知] 纠错意图: 模型 {parsed['correction_needed']}, 真实 {gt_correction} -> 得分 {metric_correct:.2f}")

        # ==========================================
        # 🦾 物理执行与 🧠 创新点 2b (未来进度预判)
        # ==========================================
        metric_phys = 0.0
        metric_future_p = 0.0
        metric_success = 0.0
        
        if parsed['correction_needed']:
            try:
                payload = {
                    "task": real_task_name,
                    "episode_id": info.get('episode_id'),
                    "frame_id": info.get('frame_id'),
                    "action": parsed['action'],
                    "baseline_progress": baseline_p,
                    "ep_path": ep_path_absolute
                }
                # resp = requests.post(REWARD_SERVER_URL, json=payload, timeout=15)
                # 3. 发送请求到对应的任务服务器
                resp = requests.post(target_url, json=payload, timeout=15)
                
                if resp.status_code == 200:
                    delta_p = resp.json().get("delta_progress", 0.0)
                    print(f"  ├─ [物理] 引擎返回 delta_p: {delta_p:.4f}")
                    
                    if delta_p > 0.01:
                        metric_phys = delta_p # 归一化的物理进度
                        if baseline_p + delta_p >= 0.95:
                            metric_success = 1.0 # 完成大奖
                          
                        # # 🧠 创新点 2b: 预判未来进度 (只有动作合法且有效时，预判才有意义)
                        # fut_p_match = re.search(r"expected progress.*?(\d+\.\d+)", think_content)
                        # if fut_p_match:
                        #     pred_fut_p = float(fut_p_match.group(1))
                        #     real_fut_p = baseline_p + delta_p
                        #     error_fut = abs(pred_fut_p - real_fut_p)
                        #     # 未来预测更难，容差放宽到 15% (0.15)
                        #     metric_future_p = max(0.0, 1.0 - (error_fut / 0.15))
                        #     print(f"  ├─ [认知] 未来进度预判: {pred_fut_p} (真实:{real_fut_p:.2f}) -> {metric_future_p:.2f}")
                    else:
                        metric_phys = delta_p * 2.0 # 失败或倒退给予惩罚系数放大
            except Exception:
                metric_phys = -0.1 # 严重崩溃
 
        # ==========================================
        # 加权求和与截断
        # ==========================================
        total_reward = (
            W_FORMAT * 1.0 + 
            W_COG_EDGE * metric_edge + 
            W_COG_CURR_P * metric_curr_p + 
            # W_COG_FUTURE_P * metric_future_p + 
            W_COG_CORRECT * metric_correct +
            W_PHYSICAL * metric_phys + 
            W_SUCCESS * metric_success
        )
        
        # 将分数限制在健康区间
        final_score = float(np.clip(total_reward, -5.0, 20.0))
        
        print(f"🏆 [最终 Reward] = {final_score:.2f} \n")
        scores.append({"overall": final_score})
        
    return scores

# def compute_score(reward_inputs, **kwargs):
#     scores = []
    
#     for item in reward_inputs:
#         solution_str = item["response"]
        
#         extra_info_dict = item.get('extra_info', {})
#         info = extra_info_dict.get('extra_info', extra_info_dict) if isinstance(extra_info_dict, dict) else {}

#         real_task_name = info.get('task', 'unknown')
#         image_paths = item.get('images', []) 
        
#         # 🌟 终极绝杀：直接从图片绝对路径中截取 episode 目录！
#         ep_path_absolute = ""
#         if isinstance(image_paths, list) and len(image_paths) > 0:
#             # 匹配到 episodeXXX 为止的完整路径
#             match = re.search(r"(.*?/episodes/episode\d+)/", image_paths[0])
#             if match:
#                 ep_path_absolute = match.group(1)

#         parsed = extract_action_from_response(solution_str)
#         if not parsed: 
#             print("⚠️ [Reward 判定] 模型未输出合法 JSON -> Reward = -0.5")
#             scores.append({"overall": -0.5})
#             continue
            
#         print(f"✅ [解析成功] 任务: {real_task_name} | 动作: {parsed['action']} | 纠错: {parsed['correction_needed']}")

#         delta_p = 0.0
#         try:
#             payload = {
#                 "task": real_task_name,
#                 "episode_id": info.get('episode_id'),
#                 "frame_id": info.get('frame_id'),
#                 "action": parsed['action'],
#                 "baseline_progress": info.get('baseline_progress', 0.0),
#                 "ep_path": ep_path_absolute  # 🌟 直接把绝对路径传给服务器！
#             }
            
#             resp = requests.post(REWARD_SERVER_URL, json=payload, timeout=15)
            
#             if resp.status_code == 200:
#                 delta_p = resp.json().get("delta_progress", 0.0)
#                 print(f"🟢 [仿真器返回] 进度增量 delta_p = {delta_p}")
#             else:
#                 print(f"🔴 [仿真器报错] HTTP 状态码: {resp.status_code}")
                
#         except requests.exceptions.Timeout:
#             print("🔴 [请求超时] 物理服务器未响应！")
#         except Exception as e:
#             print(f"🔴 [通信断开] 无法连接到仿真器。报错: {e}")
    
#         reward = float(delta_p * 20.0)
#         if parsed['correction_needed'] == info.get('is_correction_sample'):
#             reward += 0.1
        
#         print(f"🏆 [当前样本最终 Reward] = {reward}\n")
#         scores.append({"overall": float(reward)})
        
#     return scores 
 
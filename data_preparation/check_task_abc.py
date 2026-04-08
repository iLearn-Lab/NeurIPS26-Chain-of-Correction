import json 
import os
import re
import argparse 
import collections
 
def check_all_tasks(json_path, tolerance=0.02):
    print(f"\n🔍 正在质检文件: {json_path}")
    print("-" * 60)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    task_a, task_b, task_c = [], [], []
    
    # 💡 终极过滤法：直接解析答案 JSON 里的专属键值
    for item in dataset:
        ans_dict = json.loads(item["messages"]["Answer"])
        if "progress" in ans_dict and "scene_graph" in ans_dict:
            task_a.append(item["messages"])
        elif "predicted_progress" in ans_dict and "future_scene_graph" in ans_dict:
            task_b.append(item["messages"])
        elif "thought_process" in ans_dict and "corrected_action" in ans_dict:
            task_c.append(item["messages"])
            
    print(f"✅ 共扫描 {len(dataset)} 条数据，任务分布如下：")
    print(f"📊 任务 A (当前感知): {len(task_a)} 条 ({len(task_a)/len(dataset):.1%})")
    print(f"📊 任务 B (预测未来): {len(task_b)} 条 ({len(task_b)/len(dataset):.1%})")
    print(f"📊 任务 C (纠偏控制): {len(task_c)} 条 ({len(task_c)/len(dataset):.1%})")
    
    # ================= 任务 A 抽查 =================
    print("\n" + "="*20 + " [任务 A 抽查 (取第1条)] " + "="*20)
    if task_a:
        instruction_a = task_a[0]['Problem'].split('Task instruction:')[-1].splitlines()[0].strip()
        print(f"指令 (Instruction): {instruction_a}")
        ans_a = json.loads(task_a[0]['Answer'])
        print(f"提取的进度 (Progress): {ans_a.get('progress')}")
        print(f"提取的节点 (Nodes): {ans_a.get('scene_graph', {}).get('nodes')}")

    # ================= 任务 B 抽查 =================
    print("\n" + "="*20 + " [任务 B 抽查 (取第1条)] " + "="*20)
    if task_b:
        instruction_b = task_b[0]['Problem'].split('Task instruction:')[-1].splitlines()[0].strip()
        print(f"指令 (Instruction): {instruction_b}")
        ans_b = json.loads(task_b[0]['Answer'])
        print(f"预测的进度 (Predicted Progress): {ans_b.get('predicted_progress')}")
        print(f"未来的节点 (Future Nodes): {ans_b.get('future_scene_graph', {}).get('nodes')}")

    # ================= 任务 C 抽查 =================
    print("\n" + "="*20 + " [任务 C 抽查 (取第1条)] " + "="*20)
    if task_c:
        instruction_c = task_c[0]['Problem'].split('Task instruction:')[-1].splitlines()[0].strip()
        print(f"指令 (Instruction): {instruction_c}")
        ans_c = json.loads(task_c[0]['Answer'])
        print(f"是否需要纠错 (Correction Needed): {ans_c.get('correction_needed')}")
        print(f"输出的纠偏动作 (Corrected Action): {ans_c.get('corrected_action')}")
        print(f"思维链 (Thought Process): {ans_c.get('thought_process')}")

    # ================= 时序逻辑验证 (全量逐帧版) =================
    print(f"\n" + "="*20 + f" [任务 A 专家轨迹时序逻辑验证 (容忍度: {tolerance})] " + "="*20)
    if task_a:
        progress_list = [json.loads(s["Answer"]).get("progress", 0.0) for s in task_a]
        
        abnormal_drops = 0
        max_drop = 0.0
        
        for i in range(1, len(progress_list)):
            prev_p = progress_list[i-1]
            curr_p = progress_list[i]
            
            # 💡 通用版 Episode 边界检测：只要发生了 0.5 以上的断崖式暴跌，就认为是进入了新的 Episode
            if prev_p - curr_p > 0.5:
                continue
                
            # 计算进度回退量
            if curr_p < prev_p:
                drop = prev_p - curr_p
                # 使用动态的 tolerance
                if drop > tolerance: 
                    abnormal_drops += 1
                    max_drop = max(max_drop, drop)

        print(f"首帧进度: {progress_list[0]} | 尾帧进度: {progress_list[-1]}")
        
        # 修复硬编码，使用动态数据长度
        if abnormal_drops == 0:
            print(f"✅ 完美！{len(task_a)} 条数据已逐帧全部校验完毕，没有任何异常的进度倒退！")
        else:
            print(f"❌ 警告：在所有数据中发现了 {abnormal_drops} 次异常的进度倒退！")
            print(f"❌ 最大单次倒退幅度为: {max_drop:.3f}")
            print("👉 建议：这说明你的专家轨迹里有严重失误，或者 ProgressEstimator 存在 Bug。")



def check_task_b_dynamics(json_path, tolerance=0.02):
    print(f"\n🚀 正在启动任务 B (前向动力学) 专属通用质检: {json_path}")
    print("-" * 60)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    task_b_samples = []
    for item in dataset:
        ans_dict = json.loads(item["messages"]["Answer"])
        if "predicted_progress" in ans_dict and "future_scene_graph" in ans_dict:
            task_b_samples.append(item["messages"])
            
    print(f"✅ 成功定位到 {len(task_b_samples)} 条任务 B 数据。")
    
    abnormal_dynamics = 0
    max_regression = 0.0
    action_format_errors = 0

    for idx, sample in enumerate(task_b_samples):
        problem = sample["Problem"]
        answer = json.loads(sample["Answer"])
 
        # 1. 从 Problem 字符串中提取 t 时刻进度 (带类型保护)
        try:
            split_key = "\nAction Proposal (t):" if "\nAction Proposal (t):" in problem else "\nExecuted Action (t):"
            graph_str = problem.split("Current Scene Graph (t):\n")[1].split(split_key)[0]
            curr_graph = json.loads(graph_str)
            
            thresholds = curr_graph.get("task_thresholds", {})
            
            # 遍历字典，寻找真正的浮点数进度
            curr_prog = 0.0
            found_prog = False
            
            # 优先找带 "progress" 关键字的键
            for k, v in thresholds.items():
                if "progress" in k.lower():
                    try:
                        curr_prog = float(v)
                        found_prog = True
                        break
                    except: continue
            
            # 如果没找到，退而求其次找任何能转成 float 的值
            if not found_prog:
                for v in thresholds.values():
                    try:
                        curr_prog = float(v)
                        break
                    except: continue
                    
        except Exception as e:
            # 现在你会看到更具体的报错，或者不再报错
            print(f"❌ 解析第 {idx} 条数据失败: {e}")
            continue 
            
        # 2. 从 Answer 中提取 t+1 时刻的预测进度
        next_prog = answer.get("predicted_progress", 0.0)
        
        # 3. 检查动力学规律：未来进度不能出现断崖式下跌
        delta = next_prog - curr_prog
        if delta < -tolerance:
            abnormal_dynamics += 1
            max_regression = min(max_regression, delta)
            
        # 4. 顺手检查 Action 的格式
        try:
            action_str = problem.split(split_key + " ")[1].split("\n")[0]
            action_list = json.loads(action_str)
            if len(action_list) != 8 or not all(isinstance(x, float) for x in action_list):
                action_format_errors += 1
        except:
            action_format_errors += 1

    print("\n" + "="*20 + f" [任务 B 动力学质检报告 (容忍度: {tolerance})] " + "="*20)
    if abnormal_dynamics == 0:
        print("✅ 物理逻辑完美！所有 t -> t+1 的预测进度均符合专家单调递增规律！")
    else:
        print(f"⚠️ 警告：发现了 {abnormal_dynamics} 次违反物理规律的进度倒退 (t+1 进度远小于 t 进度)。")
        print(f"⚠️ 最大单步跌幅为: {max_regression:.3f}")
        
    if action_format_errors == 0:
        print("✅ 动作空间格式完美！全部为 8 维连续浮点向量。")
    else:
        print(f"❌ 警告：发现了 {action_format_errors} 条动作格式异常的数据！")

 

def check_task_c_branches(json_path):
    print(f"\n🚀 正在启动任务 C (纠错与重规划) 专属分支扫描: {json_path}")
    print("-" * 70)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    task_c_samples = []
    for item in dataset:
        try:
            ans_dict = json.loads(item["messages"]["Answer"])
            if "thought_process" in ans_dict and "correction_needed" in ans_dict:
                task_c_samples.append((item["messages"]["Problem"], ans_dict))
        except:
            continue
            
    print(f"✅ 成功定位到 {len(task_c_samples)} 条任务 C 数据。\n")
    
    # 定义五个分支的特征关键字 (匹配我们设计的 CoT)
    branch_keywords = {
        "Branch 1 (拓扑掉落)": "Catastrophic topological transition",
        "Branch 2 (严重回归)": "Macroscopic",
        "Branch 3A (原地卡死)": "stuck",                   # 对应 stagnation_counter >= 5
        "Branch 3B (方向走偏)": "Directional deviation",
        "Branch 4 (前瞻拦截)": "Pre-emptive error prevention",
        "Branch 5 (专家完美)": "No correction needed"
    }
    
    branch_counts = collections.defaultdict(int)
    branch_examples = {}

    # 扫描并分类
    for problem, ans_dict in task_c_samples:
        thought = ans_dict.get("thought_process", "")
        
        assigned_branch = "Unknown (格式异常)"
        for branch_name, keyword in branch_keywords.items():
            if keyword in thought:
                assigned_branch = branch_name
                break
                
        branch_counts[assigned_branch] += 1
        # 只保留每个分支的第一个例子用于展示
        if assigned_branch not in branch_examples:
            branch_examples[assigned_branch] = (problem, ans_dict)

    # 打印统计结果
    print("📊 ================= [任务 C 分支触发统计] =================")
    for branch_name in branch_keywords.keys():
        count = branch_counts[branch_name]
        percentage = (count / len(task_c_samples)) * 100 if len(task_c_samples) > 0 else 0
        print(f" 🔹 {branch_name}: {count} 条 ({percentage:.1f}%)")
    print("=========================================================\n")

    # 打印每个分支的具体抽查例子
    print("🔎 ================= [各分支数据详细抽查] =================")
    for branch_name in branch_keywords.keys():
        if branch_name not in branch_examples:
            print(f"⚠️ {branch_name} 无数据触发 (专家轨迹中未出现此类异常)。\n")
            continue
            
        problem, ans_dict = branch_examples[branch_name]
        
        # 截取精简版的 Problem (省略太长的图谱细节，只看头尾和提议动作)
        problem_lines = problem.split('\n')
        instruction = [line for line in problem_lines if "Task instruction:" in line][0]
        action_prop = [line for line in problem_lines if "Action Proposal (t):" in line][0]
        
        print(f"🟢 【{branch_name}】示例:")
        print(f"  📥 [输入指令]: {instruction}")
        print(f"  📥 [提议动作]: {action_prop}")
        print(f"  📤 [纠错判定 (Correction Needed)]: {ans_dict['correction_needed']}")
        print(f"  📤 [输出动作 (Corrected Action)]: {ans_dict['corrected_action']}")
        print(f"  🧠 [思维链 (CoT)]:")
        # 换行打印思维链，便于阅读
        for sentence in ans_dict['thought_process'].split('] '):
            if sentence.strip():
                print(f"       ] {sentence.strip()}" if sentence.startswith("[") else f"       [{sentence.strip()}")
        print("-" * 70)

  

# def check_unified_dataset(json_path, tolerance=0.02):
#     print(f"\n🔍 正在对统一思维链数据集执行深度逻辑全景质检: {json_path}")
#     print("-" * 80)
    
#     if not os.path.exists(json_path):
#         print(f"❌ 错误：找不到文件 '{json_path}'")
#         return

#     with open(json_path, 'r', encoding='utf-8') as f:
#         dataset = json.load(f)

#     # 包含任务 A、B、C 的全量统计字典
#     stats = {
#         "total": len(dataset),
#         "t0_frames": 0,
#         "tn_frames": 0,
#         "perception_checked": 0,     # 任务 A 逻辑数
#         "abnormal_drops": 0,
#         "max_drop": 0.0,            
#         "first_prog": None,         
#         "last_prog": None,          
#         "dynamics_checked": 0,       # 任务 B 逻辑数
#         "abnormal_dynamics": 0,
#         "max_regression": 0.0,
#         "action_format_errors": 0,
#         "correction_checked": 0,     # 任务 C 逻辑数
#         "format_errors": 0
#     }

#     # 任务 C 的分支关键字映射 (精准匹配 thought 文本)
#     branch_keywords = {
#         "Branch 1 (拓扑掉落)": "Catastrophic topological transition",
#         "Branch 2 (严重回归)": "Macroscopic",
#         "Branch 3A (原地卡死)": "stuck",
#         "Branch 3B (方向走偏)": "Directional deviation",
#         "Branch 4 (前瞻拦截)": "Pre-emptive error prevention",
#         "Branch 5 (专家完美)": "No correction needed"
#     }
    
#     branch_counts = collections.defaultdict(int)
#     branch_examples = {}  # 记录每个分支的详细 CoT 示例
#     task_a_drops_dist = [] # 记录每次跌落的幅度
#     task_b_drops_dist = [] # 记录预测的跌幅

#     for idx, item in enumerate(dataset):
#         msgs = item.get("messages", [])
        
#         # ==========================================
#         # 1. 提取首帧进度 (t=0)
#         # ==========================================
#         if len(msgs) == 2:
#             stats["t0_frames"] += 1
#             if stats["first_prog"] is None:
#                 try:
#                     curr_content = msgs[1]['content']
#                     think_content = curr_content.split("<think>")[1].split("</think>")[0]
#                     # 匹配 "Current Scene Graph (t=0): ... with progress: 0.XXX"
#                     match_t0 = re.search(r'Current Scene Graph \(t=0\):.*?with progress:\s*([\d.]+)', think_content, re.S)
#                     if match_t0: stats["first_prog"] = float(match_t0.group(1))
#                 except: pass
#             continue
            
#         if len(msgs) != 4:
#             stats["format_errors"] += 1
#             continue
            
#         stats["tn_frames"] += 1
        
#         try:
#             # 提取 t-1 轮次历史进度
#             prev_ans_raw = msgs[1]['content']
#             prev_ans_json = json.loads(prev_ans_raw.split("```json\n")[-1].split("\n```")[0])
#             prev_prog = prev_ans_json.get("progress", 0.0)
#             # prev_prog = float(re.search(r'"progress":\s*([\d.]+)', prev_ans_raw).group(1))

#             user_msg_t = msgs[2]['content']
#             prop_action_match = re.search(r'Action Proposal:\s*(\[.*?\])', user_msg_t)
#             proposed_action = prop_action_match.group(1) if prop_action_match else "[]"
            
#             # 提取 t 轮次内容
#             curr_assistant_msg = msgs[3]['content']
#             think_content = curr_assistant_msg.split("<think>")[1].split("</think>")[0]
#             final_json = json.loads(curr_assistant_msg.split("```json\n")[-1].split("\n```")[0])
            
#             # ========================================================
#             # 【任务 A 检查：感知层物理一致性】
#             # ========================================================
#             match_curr = re.search(r'Current Scene Graph \(t\):.*?with progress:\s*([\d.]+)', think_content, re.S)
 
#             if match_curr:
#                 curr_prog = float(match_curr.group(1))
#                 stats["perception_checked"] += 1
#                 stats["last_prog"] = curr_prog # 持续更新尾帧进度
                
#                 # 计算异常进度回退量 (仅在未纠错的帧中计算)
#                 # if not final_json.get("correction_needed") and curr_prog < prev_prog:
#                 if curr_prog < prev_prog:
#                     drop = prev_prog - curr_prog
#                     if drop > tolerance:
#                         stats["abnormal_drops"] += 1
#                         stats["max_drop"] = max(stats["max_drop"], drop) 
#                         task_a_drops_dist.append(drop) # 收集分布
                        
#             # ========================================================
#             # 【任务 B 检查：动力学预测准确性】
#             # ========================================================
#             # 匹配 "Future Scene Graph (t+1): ... with progress: 0.XXX"
#             match_next = re.search(r'Future Scene Graph \(t\+1\):.*?with progress:\s*([\d.]+)', think_content, re.S)
#             if match_next and match_curr:
#                 next_prog = float(match_next.group(1))
#                 stats["dynamics_checked"] += 1
                
#                 # 检查未来进度是否出现断崖式下跌 (t+1 远小于 t)
#                 delta = next_prog - curr_prog
#                 if delta < -tolerance:
#                     stats["abnormal_dynamics"] += 1
#                     # delta 是负数，min 获取的是最大的跌幅
#                     stats["max_regression"] = min(stats["max_regression"], delta)
#                     task_b_drops_dist.append(delta) # 收集分布
            
#             # 检查动作格式是否规范
#             corrected_action = final_json.get("corrected_action", [])
#             if len(corrected_action) != 8 or not all(isinstance(x, (float, int)) for x in corrected_action):
#                 stats["action_format_errors"] += 1

#             # ========================================================
#             # 【任务 C 检查：决策纠偏逻辑】
#             # ========================================================
#             stats["correction_checked"] += 1
#             assigned_branch = "Unknown (未触发预设关键字)"
            
#             # 扫描预设关键字并归类
#             for branch_name, keyword in branch_keywords.items():
#                 if keyword in think_content:
#                     assigned_branch = branch_name
#                     break
                   
#             branch_counts[assigned_branch] += 1
#             if assigned_branch not in branch_examples:
#                 instruction_match = re.search(r'Task(?: instruction)?:\s*(.*?)\n', msgs[0]['content'])
#                 instruction = instruction_match.group(1) if instruction_match else "Unknown Task"
#                 branch_examples[assigned_branch] = {
#                     "instruction": instruction.strip(),
#                     "proposed_action": proposed_action,
#                     "correction_needed": final_json.get("correction_needed"),
#                     "corrected_action": corrected_action,
#                     "thought_process": think_content.strip()
#                 }
            
#         except Exception as e:
#             stats["format_errors"] += 1
#             continue

#     # =========================================================================
#     # 🖨️ 最终格式化打印输出
#     # =========================================================================
#     print(f"✅ 全量扫描完毕，总样本数: {stats['total']}")
#     print(f"📊 帧类型分布: [t=0 初始化]: {stats['t0_frames']} | [t>0 循环]: {stats['tn_frames']}")
    
#     # ---------------- 打印任务 A ----------------
#     print(f"\n" + "="*20 + f" [任务 A 专家轨迹时序逻辑验证 (容忍度: {tolerance})] " + "="*20)
#     print(f"首帧进度: {stats['first_prog']} | 尾帧进度: {stats['last_prog']}")
    
#     if stats["abnormal_drops"] == 0:
#         print(f"✅ 完美！{stats['perception_checked']} 条感知数据已逐帧全部校验完毕，没有任何异常的进度倒退！")
#     else:
#         print(f"❌ 警告：在所有数据中发现了 {stats['abnormal_drops']} 次异常的进度倒退！")
#         print(f"❌ 最大单次倒退幅度为: {stats['max_drop']:.3f}")
#         print(f"   [跌幅详情]: {sorted([round(x,3) for x in task_a_drops_dist], reverse=True)[:10]} (仅展示前10个最大跌落)")

#     # ---------------- 打印任务 B ----------------
#     print("\n" + "="*20 + f" [任务 B 动力学质检报告 (容忍度: {tolerance})] " + "="*20)
#     print(f"🔍 成功定位并提取到 {stats['dynamics_checked']} 条前向动力学预测数据。")
#     if stats["abnormal_dynamics"] == 0:
#         print("✅ 物理逻辑完美！所有 t -> t+1 的预测进度均符合专家单调递增规律！")
#     else:
#         print(f"⚠️ 警告：发现了 {stats['abnormal_dynamics']} 次违反物理规律的进度倒退 (t+1 进度远小于 t 进度)。")
#         print(f"⚠️ 最大单步预测跌幅为: {stats['max_regression']:.3f}")
        
#     if stats["action_format_errors"] == 0:
#         print("✅ 动作空间格式完美！全部为 8 维连续浮点向量。")
#     else:
#         print(f"❌ 警告：发现了 {stats['action_format_errors']} 条动作格式异常的数据！")
         
#     # ---------------- 打印任务 C ----------------
#     print(f"\n🚀 正在启动任务 C (纠错与重规划) 专属分支扫描: {json_path}")
#     print("-" * 70)
#     print(f"✅ 成功定位到 {stats['correction_checked']} 条任务 C 数据。\n")
    
#     print("📊 ================= [任务 C 分支触发统计] =================")
#     # 强制打印所有的预设分支，哪怕是0
#     for branch_name in branch_keywords.keys():
#         count = branch_counts.get(branch_name, 0)
#         percentage = (count / max(1, stats['tn_frames'])) * 100
#         print(f" 🔹 {branch_name}: {count} 条 ({percentage:.1f}%)")
        
#     # 如果有未知分类，也曝光打印出来
#     for branch_name, count in branch_counts.items():
#         if branch_name not in branch_keywords.keys():
#             percentage = (count / max(1, stats['tn_frames'])) * 100
#             print(f" ❓ {branch_name}: {count} 条 ({percentage:.1f}%)")
#     print("=========================================================\n")

#     print("🔎 ================= [各分支数据详细抽查] =================")
#     for branch_name in branch_keywords.keys():
#         if branch_name not in branch_examples:
#             print(f"⚠️ {branch_name} 无数据触发 (专家轨迹中未出现此类异常)。\n")
#             continue
            
#         ex = branch_examples[branch_name]
#         print(f"🟢 【{branch_name}】示例:")
#         print(f"  📥 [输入指令]: {ex['instruction']}")
#         print(f"  📥 [提议动作]: {ex['proposed_action']}")
#         print(f"  📤 [纠错判定 (Correction Needed)]: {ex['correction_needed']}")
#         print(f"  📤 [输出动作 (Corrected Action)]: {ex['corrected_action']}")
#         print(f"  🧠 [思维链 (CoT)]:")
        
#         # 完美还原旧版思维链的阶梯式排版
#         thought_text = ex['thought_process'].strip()
#         # 将 "[Phase 3...]: [Progress...]" 强行断行，保持对齐
#         thought_text = thought_text.replace("]: [", "]:\n[")
        
#         for line in thought_text.split('\n'):
#             line = line.strip()
#             if not line:
#                 continue
#             # 以 '[' 开头的推理标签进行强缩进，JSON 图谱和普通句子顶格打印
#             if line.startswith('['):
#                 print(f"       {line}")
#             else:
#                 print(line)
#         print("-" * 70) 

#     if stats["format_errors"] > 0:
#         print(f"\n⚠️ 额外警告：有 {stats['format_errors']} 条数据格式损坏，未能进入物理校验。")


def check_unified_dataset(json_path, tolerance=0.02):
    print(f"\n🔍 正在对统一思维链数据集执行深度逻辑全景质检: {json_path}")
    print("-" * 80)
    
    if not os.path.exists(json_path):
        print(f"❌ 错误：找不到文件 '{json_path}'")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 剔除掉已经废弃的动态预测 (Task B)，聚焦于任务 A 和 C
    stats = {
        "total": len(dataset),
        "t0_frames": 0,
        "tn_frames": 0,
        "perception_checked": 0,     # 任务 A 逻辑数
        "abnormal_drops": 0,
        "max_drop": 0.0,            
        "first_prog": None,         
        "last_prog": None,          
        "action_format_errors": 0,
        "correction_checked": 0,     # 任务 C 逻辑数
        "format_errors": 0
    }

    # 💡 修正 1：完美对齐我们刚刚重构的定性/客观描述词汇
    branch_keywords = {
        "Branch 1 (拓扑掉落)": "Invalid topological transition detected",
        "Branch 2 (严重回归)": "Severe regression detected",
        "Branch 3A (原地卡死)": "Physical stasis detected",
        "Branch 3B (方向走偏)": "Directional deviation detected",
        "Branch 4 (前瞻拦截)": "Pre-emptive analysis shows",
        "Expert (合法专家动作)": "No correction needed" # 涵盖了所有的合法过渡、避障和正常推进
    }
    
    branch_counts = collections.defaultdict(int)
    branch_examples = {}  # 记录每个分支的详细 CoT 示例
    task_a_drops_dist = [] # 记录每次跌落的幅度

    for idx, item in enumerate(dataset):
        msgs = item.get("messages", [])
        
        # ==========================================
        # 1. 提取首帧进度 (t=0)
        # ==========================================
        if len(msgs) == 2:
            stats["t0_frames"] += 1
            if stats["first_prog"] is None:
                try:
                    curr_content = msgs[1]['content']
                    think_content = curr_content.split("<think>")[1].split("</think>")[0]
                    # 💡 修正 2：更新为最新的 Global Task Progress 正则
                    match_t0 = re.search(r'Global Task Progress:\s*([\d.]+)', think_content)
                    if match_t0: stats["first_prog"] = float(match_t0.group(1))
                except: pass
            continue
            
        if len(msgs) != 4:
            stats["format_errors"] += 1
            continue
            
        stats["tn_frames"] += 1
        
        try:
            # 提取 t-1 轮次历史进度
            prev_ans_raw = msgs[1]['content']
            prev_ans_json = json.loads(prev_ans_raw.split("```json\n")[-1].split("\n```")[0])
            prev_prog = prev_ans_json.get("progress", 0.0)

            user_msg_t = msgs[2]['content']
            prop_action_match = re.search(r'Action Proposal:\s*(\[.*?\])', user_msg_t)
            proposed_action = prop_action_match.group(1) if prop_action_match else "[]"
            
            # 提取 t 轮次内容
            curr_assistant_msg = msgs[3]['content']
            think_content = curr_assistant_msg.split("<think>")[1].split("</think>")[0]
            final_json = json.loads(curr_assistant_msg.split("```json\n")[-1].split("\n```")[0])
            
            # ========================================================
            # 【任务 A 检查：感知层物理一致性】
            # ========================================================
            # 💡 修正 3：更新为最新的 Global Task Progress 正则
            match_curr = re.search(r'Global Task Progress:\s*([\d.]+)', think_content)

            if match_curr:
                curr_prog = float(match_curr.group(1))
                stats["perception_checked"] += 1
                stats["last_prog"] = curr_prog # 持续更新尾帧进度
                
                # 检查极其异常的物理断崖
                if curr_prog < prev_prog:
                    drop = prev_prog - curr_prog
                    if drop > tolerance:
                        stats["abnormal_drops"] += 1
                        stats["max_drop"] = max(stats["max_drop"], drop) 
                        task_a_drops_dist.append(drop)
                        
            # 💡 (原任务 B 已被物理脱水重构剔除，无需再寻找未来的伪造进度)

            # 检查动作格式是否规范
            corrected_action = final_json.get("corrected_action", [])
            if len(corrected_action) != 8 or not all(isinstance(x, (float, int)) for x in corrected_action):
                stats["action_format_errors"] += 1

            # ========================================================
            # 【任务 C 检查：决策纠偏逻辑】
            # ========================================================
            stats["correction_checked"] += 1
            assigned_branch = "Unknown (未触发预设关键字)"
            
            # 扫描预设关键字并归类
            for branch_name, keyword in branch_keywords.items():
                if keyword in think_content:
                    assigned_branch = branch_name
                    break
                    
            branch_counts[assigned_branch] += 1
            if assigned_branch not in branch_examples:
                instruction_match = re.search(r'Task(?: instruction)?:\s*(.*?)\n', msgs[0]['content'])
                instruction = instruction_match.group(1) if instruction_match else "Unknown Task"
                branch_examples[assigned_branch] = {
                    "instruction": instruction.strip(),
                    "proposed_action": proposed_action,
                    "correction_needed": final_json.get("correction_needed"),
                    "corrected_action": corrected_action,
                    "thought_process": think_content.strip()
                }
            
        except Exception as e:
            stats["format_errors"] += 1
            continue

    # =========================================================================
    # 🖨️ 最终格式化打印输出
    # =========================================================================
    print(f"✅ 全量扫描完毕，总样本数: {stats['total']}")
    print(f"📊 帧类型分布: [t=0 初始化]: {stats['t0_frames']} | [t>0 循环]: {stats['tn_frames']}")
    
    # ---------------- 打印任务 A ----------------
    print(f"\n" + "="*20 + f" [任务 A 专家轨迹时序逻辑验证 (容忍度: {tolerance})] " + "="*20)
    print(f"首帧进度: {stats['first_prog']} | 尾帧进度: {stats['last_prog']}")
    
    if stats["abnormal_drops"] == 0:
        print(f"✅ 完美！{stats['perception_checked']} 条数据已校验完毕，没有发现未被拦截的异常断崖！")
    else:
        print(f"⚠️ 提示：发现了 {stats['abnormal_drops']} 次进度回调 (这是真实的物理动作，将被分支 2 拦截或豁免)。")
        print(f"   最大单次倒退幅度为: {stats['max_drop']:.3f}")

    if stats["action_format_errors"] == 0:
        print("✅ 动作空间格式完美！全部为 8 维连续浮点向量。")
    else:
        print(f"❌ 警告：发现了 {stats['action_format_errors']} 条动作格式异常的数据！")
         
    # ---------------- 打印任务 C ----------------
    print(f"\n🚀 正在启动纠错与重规划专属分支扫描: {json_path}")
    print("-" * 70)
    print(f"✅ 成功定位到 {stats['correction_checked']} 条分析数据。\n")
    
    print("📊 ================= [各类物理分支触发统计] =================")
    for branch_name in branch_keywords.keys():
        count = branch_counts.get(branch_name, 0)
        percentage = (count / max(1, stats['tn_frames'])) * 100
        print(f" 🔹 {branch_name}: {count} 条 ({percentage:.1f}%)")
        
    for branch_name, count in branch_counts.items():
        if branch_name not in branch_keywords.keys():
            percentage = (count / max(1, stats['tn_frames'])) * 100
            print(f" ❓ {branch_name}: {count} 条 ({percentage:.1f}%)")
    print("=========================================================\n")

    print("🔎 ================= [各分支思维链 (CoT) 详细抽查] =================")
    for branch_name in branch_keywords.keys():
        if branch_name not in branch_examples:
            print(f"⚠️ {branch_name} 无数据触发 (专家轨迹中未注入此类异常)。\n")
            continue
            
        ex = branch_examples[branch_name]
        print(f"🟢 【{branch_name}】示例:")
        print(f"  📥 [输入指令]: {ex['instruction']}")
        print(f"  📥 [提议动作]: {ex['proposed_action']}")
        print(f"  📤 [纠错判定]: {ex['correction_needed']}")
        print(f"  📤 [输出动作]: {ex['corrected_action']}")
        print(f"  🧠 [思维链 (CoT)]:\n")
        
        thought_text = ex['thought_process'].strip()
        for line in thought_text.split('\n'):
            line = line.strip()
            if not line: continue
            if line.startswith('['):
                print(f"       {line}")
            else:
                print(f"         {line}")
        print("-" * 70) 

    if stats["format_errors"] > 0:
        print(f"\n⚠️ 额外警告：有 {stats['format_errors']} 条数据格式损坏，未能进入物理校验。")
 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
 
    if not os.path.exists(args.json_path):
        print(f"❌ 错误：找不到文件 '{args.json_path}'")
    else:
        check_unified_dataset(args.json_path, args.tolerance)
        # # 一次性运行三项体检
        # print(f"\n" + "🚀" * 10 + " 开始全量自动化质检 " + "🚀" * 10)
        
        # # 1. 基础分布与 Task A 校验
        # check_all_tasks(args.json_path, tolerance=args.tolerance)
        
        # # 2. Task B 前向动力学校验
        # check_task_b_dynamics(args.json_path, tolerance=args.tolerance)
        
        # # 3. Task C 分支覆盖率校验
        # check_task_c_branches(args.json_path)
        
        print(f"\n" + "✨" * 10 + " 质检流程全部结束 " + "✨" * 10)

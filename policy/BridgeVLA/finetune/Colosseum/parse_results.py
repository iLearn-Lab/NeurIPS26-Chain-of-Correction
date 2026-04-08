# import os
# import glob
# import pandas as pd

# def merge_colosseum_results():
#     # 1. 设置你的根目录（这里就是包含各个变体号文件夹的上一级目录）
#     base_dir = "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM/opensource_colosseum/epoch_80_trial_trial_1"
    
#     # 2. 使用 glob 语法深度匹配所有 eval_results.csv
#     # 路径模式：base_dir / 变体号(14) / 任务名(scoop_with_spatula) / 模型层(model_80) / eval_results.csv
#     search_pattern = os.path.join(base_dir, "*", "*", "*", "eval_results.csv")
#     csv_files = glob.glob(search_pattern)
    
#     if not csv_files:
#         print("❌ 没有找到任何 eval_results.csv 文件，请确认路径是否正确或程序是否已产出结果。")
#         return

#     print(f"🔍 找到了 {len(csv_files)} 个评测结果文件，正在合并...")
    
#     all_data = []
#     for file in csv_files:
#         try:
#             # 3. 通过路径切割，巧妙提取“任务名”和“变体号”
#             # 以 .../14/scoop_with_spatula/model_80/eval_results.csv 为例：
#             path_parts = file.split(os.sep)
#             variation = path_parts[-4]   # 倒数第4层是变体号
#             task_name = path_parts[-3]   # 倒数第3层是任务名
            
#             # 4. 读取这个子 CSV 表格
#             df = pd.read_csv(file)
            
#             # 5. 在表格的最前面增加两列，打上这是哪个任务、哪个变体的标签
#             df.insert(0, 'Variation', variation)
#             df.insert(0, 'Task', task_name)
            
#             all_data.append(df)
            
#         except Exception as e:
#             print(f"⚠️ 读取文件出错: {file}\n错误信息: {e}")

#     # 6. 将所有读取到的小表格纵向拼接成一个超级大表
#     if all_data:
#         final_df = pd.concat(all_data, ignore_index=True)
        
#         # 7. 排序一下，让表格看起来更整齐（先按任务排，再按变体排）
#         # 假设 Variation 读取进来是字符串，先转成数字以便正确排序 (0, 1, 2... 14)
#         final_df['Variation'] = pd.to_numeric(final_df['Variation'], errors='ignore')
#         final_df = final_df.sort_values(by=['Task', 'Variation'])
        
#         # 8. 保存输出
#         output_name = "Final_Merged_Colosseum_Results.csv"
#         final_df.to_csv(output_name, index=False)
#         print("="*50)
#         print(f"✅ 大功告成！合并了 {len(csv_files)} 个文件的结果。")
#         print(f"📊 最终总表已生成: {os.path.abspath(output_name)}")
#         print("="*50)

# if __name__ == "__main__":
#     merge_colosseum_results()

import os
import pandas as pd

# ================= 配置区 =================
# 包含不同 trial 的根目录列表
BASE_DIRS = [
    "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM/opensource_colosseum/epoch_80_trial_trial_1",
    "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM/opensource_colosseum/epoch_80_trial_trial_2"
]

# 输出的文件名
OUTPUT_FILE = "colosseum_all_tasks_merged.xlsx"
# ==========================================

def main():
    print("🚀 开始扫描并合并评估结果 (仅提取 task 和 success rate)...")
    
    # 存储所有任务的数据
    # 结构: {"get_ice_from_fridge": [{"label": "get_ice_from_fridge_2", "score": 88, "var_id": 2}, ...]}
    all_data = {}

    # 遍历所有的索引变体文件夹 0 到 14
    for var_id in range(15):
        var_str = str(var_id)
        
        # 遍历所有的 trial 文件夹
        for base_dir in BASE_DIRS:
            var_path = os.path.join(base_dir, var_str)
            if not os.path.exists(var_path):
                continue
            
            # 遍历该索引下的所有基础任务名 (作为字典的 Key 分组)
            for task_name in os.listdir(var_path):
                task_path = os.path.join(var_path, task_name)
                if not os.path.isdir(task_path):
                    continue
                
                csv_path = os.path.join(task_path, "model_80", "eval_results.csv")
                if os.path.exists(csv_path):
                    try:
                        # 🌟 核心：精确切割，只读 'task' 和 'success rate' 这两列
                        df = pd.read_csv(csv_path, usecols=['task', 'success rate'])
                        
                        if task_name not in all_data:
                            all_data[task_name] = []
                            
                        # 遍历这几行数据并存入对应任务的列表中
                        for _, row in df.iterrows():
                            all_data[task_name].append({
                                "label": str(row['task']),
                                "score": row['success rate'],
                                "var_id": var_id  # 存下索引，用于最终的严格排序
                            })
                    except Exception as e:
                        print(f"⚠️ 读取报错跳过 {csv_path}: {e}")

    if not all_data:
        print("❌ 提取失败：没有找到任何包含对应列的 CSV 文件，请检查路径。")
        return

    # ================= 拼装最终的 DataFrame =================
    final_cols = {}
    
    # 为每个任务生成两列，并严格按 var_id 排序
    for task, records in all_data.items():
        # 🌟 核心排序逻辑：保证同一变体索引的数据挨在一起 (2, 2, 2 -> 3, 3 -> 4...)
        records.sort(key=lambda x: x['var_id'])
        
        final_cols[f"{task}_Task"] = [r['label'] for r in records]
        final_cols[f"{task}_SuccessRate"] = [r['score'] for r in records]

    # 因为有的任务评估可能跑崩了导致行数不同，用 None (空单元格) 补齐短的列
    max_len = max([len(v) for v in final_cols.values()])
    for col in final_cols:
        final_cols[col] += [None] * (max_len - len(final_cols[col]))

    # 保存为 Excel
    final_df = pd.DataFrame(final_cols)
    final_df.to_excel(OUTPUT_FILE, index=False)
    
    print(f"\n✅ 搞定！汇总表格已生成: {OUTPUT_FILE}")
    print(f"📊 共梳理了 {len(all_data)} 个任务的数据，每个任务独占两列。")

if __name__ == "__main__":
    main()


import os
import pandas as pd
 
BASE_DIRS = [
    "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM/opensource_colosseum/epoch_80_trial_trial_1",
    "/home/zxn/Chain-of-Correction/policy/BridgeVLA/checkpoints/COLOSSEUM/opensource_colosseum/epoch_80_trial_trial_2"
]
 
OUTPUT_FILE = "colosseum_all_tasks_merged.xlsx"
# ==========================================

def main():
    all_data = {}
 
    for var_id in range(15):
        var_str = str(var_id)
         
        for base_dir in BASE_DIRS:
            var_path = os.path.join(base_dir, var_str)
            if not os.path.exists(var_path):
                continue
             
            for task_name in os.listdir(var_path):
                task_path = os.path.join(var_path, task_name)
                if not os.path.isdir(task_path):
                    continue
                
                csv_path = os.path.join(task_path, "model_80", "eval_results.csv")
                if os.path.exists(csv_path):
                    try: 
                        df = pd.read_csv(csv_path, usecols=['task', 'success rate'])
                        
                        if task_name not in all_data:
                            all_data[task_name] = []
                             
                        for _, row in df.iterrows():
                            all_data[task_name].append({
                                "label": str(row['task']),
                                "score": row['success rate'],
                                "var_id": var_id 
                            })
                    except Exception as e:
                        print(f"{e}")

    if not all_data:
        print("No CSV found.")
        return
 
    final_cols = {}
     
    for task, records in all_data.items(): 
        records.sort(key=lambda x: x['var_id'])
        
        final_cols[f"{task}_Task"] = [r['label'] for r in records]
        final_cols[f"{task}_SuccessRate"] = [r['score'] for r in records]
 
    max_len = max([len(v) for v in final_cols.values()])
    for col in final_cols:
        final_cols[col] += [None] * (max_len - len(final_cols[col]))
 
    final_df = pd.DataFrame(final_cols)
    final_df.to_excel(OUTPUT_FILE, index=False)
    
    print(f"\n Table successfully generated as {OUTPUT_FILE}") 

if __name__ == "__main__":
    main()
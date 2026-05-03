# python data_preparation/build_rldata_multi_task.py --false_multiplier 1.5 (--do_val_split)

"""
Unified Multi-task RL Dataset Construction Tool

Inputs:
- input_json: Path to the raw SFT JSON file containing dialogues and image paths.
  Default: /data/colosseum_dataset/mixed_sft_v1_0.2_cleaned.json

Outputs:
- output_path: A single, shuffled parquet file containing the multi-task dataset.
  Default: /data/colosseum_dataset/rl_elite_all_tasks_fixed.parquet

Main Functions & Processing Methods:
1. Data Extraction & Formatting: Parses the input JSON, extracts valid task names
   from the image paths, flattens the conversational history into a single string
   with <image> tags, and identifies correction samples.
2. Task Normalization: Strips numeric suffixes from task names to group them.
3. True Sample Deduplication: Filters correction samples and drops duplicates 
   based on the prompt string.
4. Priority-based False Sampling: For each task, samples background frames 
   (target = true_count * false_multiplier), prioritizing normal frames from 
   failed trajectories over expert trajectories.
5. Aggregation: Merges the balanced data across all tasks, shuffles, and exports.
"""

import pandas as pd
import json
import re
import argparse

def load_and_parse_json(json_path):
    print(f"Reading JSON from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rl_data = []
    error_count = 0
    
    print("Extracting features and formatting prompts...")
    for item in data:
        messages = item['messages']
        assistant_content = messages[-1]['content']
        is_correction = '"correction_needed": true' in assistant_content.lower()
        
        img_paths = item.get('image_paths', [])
        # num_images = len(img_paths)
        
        try:
            parts = img_paths[0].split('/')
            ep_folder_idx = next(i for i, part in enumerate(parts) if part.lower() == 'episodes')
            task_name = parts[ep_folder_idx - 2] 
            ep_id = int(re.search(r"episode(\d+)", parts[ep_folder_idx + 1]).group(1))
            frame_id = int(re.search(r"(\d+)\.png", parts[-1]).group(1))
        except Exception:
            task_name, ep_id, frame_id = "unknown", 0, 0
            error_count += 1

        prompt_messages = messages[:-1]
        # prompt_str = ""
        history_elements = []
        
        # if num_images > 0:
        #     prompt_str += "<image>\n" * num_images
            
        for m in prompt_messages:
        #     role = "User" if m['role'] == 'user' else "Assistant"
            content = m['content']
            if isinstance(content, list):
                content = "\n".join([part['text'] for part in content if part.get('type') == 'text'])
        #         texts = [part['text'] for part in content if part.get('type') == 'text' and part.get('text')]
        #         content = "\n".join(texts)
            
        #     content = str(content).replace('<image>\n', '').replace('<image>', '').strip()
        #     prompt_str += f"[{role}]: {content}\n\n"
            
        # prompt_str += "[Assistant]: "  
            clean_content = str(content).replace('<image>\n', '').replace('<image>', '').strip()
             
            history_elements.append(clean_content)
 
        num_images = len(img_paths)
        image_placeholders = "<image>\n" * num_images

        full_problem = image_placeholders + "\n\n".join(history_elements)

        rl_data.append({
            "problem": full_problem,
            "images": img_paths,
            "answer": assistant_content,
            "extra_info": {
                "task": task_name,
                "episode_id": ep_id,
                "frame_id": frame_id,
                "baseline_progress": 0.0,
                "is_correction_sample": is_correction
            }
        })
        # rl_data.append({
        #     "problem": prompt_str,
        #     "images": img_paths,
        #     "answer": assistant_content,
            # "extra_info": {
            #     "task": task_name,
            #     "episode_id": ep_id,
            #     "frame_id": frame_id,
            #     "baseline_progress": 0.0,
            #     "is_correction_sample": is_correction
            # }
        # })
        
    print(f"Extraction complete. Total samples: {len(rl_data)}, Errors: {error_count}")
    return pd.DataFrame(rl_data)


def main():
    parser = argparse.ArgumentParser(description="Unified Multi-task RL Dataset Construction Tool")
    parser.add_argument("--input_json", type=str, default="/data/colosseum_dataset/mixed_sft_v1_0.2_cleaned.json")
    parser.add_argument("--output_path", type=str, default="/data/colosseum_dataset/rl_elite_all_tasks_fixed.parquet")
    parser.add_argument("--false_multiplier", type=float, default=1.5)
    # Arguments for validation split
    parser.add_argument("--do_val_split", action="store_true", help="Enable 95/5 train/validation split")
    parser.add_argument("--val_output_path", type=str, default="/data/colosseum_dataset/rl_val_split.parquet")
    args = parser.parse_args()

    print(f"\n{'='*60}\nStarting Unified Pipeline | Target Ratio: 1:{args.false_multiplier}\n{'='*60}")

    # 1. Parse JSON directly to DataFrame
    df_all = load_and_parse_json(args.input_json)

    # 2. Preprocess and normalize tasks
    def normalize_task(info):
        t = info.get('task', 'unknown')
        return re.sub(r'_\d+$', '', t)

    df_all['base_task'] = df_all['extra_info'].apply(normalize_task)
    df_all['is_true'] = df_all['extra_info'].apply(lambda x: x.get('is_correction_sample', False))
    # df_all['is_from_fail'] = df_all['images'].apply(lambda x: 'failed_eval_demos' in str(x[0]) or 'failure' in str(x[0]))
    df_all['is_from_fail'] = df_all['images'].apply(
        lambda x: len(x) > 0 and ('failed_eval_demos' in str(x[0]) or 'failure' in str(x[0]))
    )

    all_base_tasks = sorted([t for t in df_all['base_task'].unique() if t != 'unknown'])
    print(f"Identified {len(all_base_tasks)} base tasks after normalization.")
 
    # 3. Process each task
    task_dfs = []
    summary_stats = []

    for task in all_base_tasks:
        df_task = df_all[df_all['base_task'] == task].copy()
        
        # A. True samples: deduplicate globally
        true_raw = df_task[df_task['is_true'] == True]
        true_elite = true_raw.drop_duplicates(subset=['problem'])
        num_true = len(true_elite)

        if num_true == 0:
            print(f"  [Skip] Task {task:<30} ... No true samples found.")
            continue

        # B. False samples prioritized sampling
        num_false_target = int(num_true * args.false_multiplier)
        
        fail_normal = df_task[(df_task['is_true'] == False) & (df_task['is_from_fail'] == True)]
        expert_normal = df_task[(df_task['is_true'] == False) & (df_task['is_from_fail'] == False)]

        if len(fail_normal) >= num_false_target:
            false_elite = fail_normal.sample(n=num_false_target, random_state=42)
            from_fail = num_false_target
        else:
            needed = num_false_target - len(fail_normal)
            expert_sample = expert_normal.sample(n=min(needed, len(expert_normal)), random_state=42)
            false_elite = pd.concat([fail_normal, expert_sample])
            from_fail = len(fail_normal)

        task_dfs.append(pd.concat([true_elite, false_elite]))
        
        summary_stats.append({
            "Task": task,
            "True": num_true,
            "False": len(false_elite),
            "FromFail": from_fail
        })
        print(f"  [Done] Task {task:<30}: True={num_true:<4} | False={len(false_elite):<4} (From fail: {from_fail})")

    # 4. Merge and save 
    if not task_dfs:
        print("\nError: No data was generated. Check your inputs.")
        return

    final_df = pd.concat(task_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Drop temporary processing columns
    cols_to_drop = [c for c in ['base_task', 'is_true', 'is_from_fail', 'temp_task'] if c in final_df.columns]
    final_df = final_df.drop(columns=cols_to_drop)
    
    # Final Summary Data Computation
    total_true = sum(s['True'] for s in summary_stats)
    total_false = sum(s['False'] for s in summary_stats)
    actual_ratio = round(total_false / total_true, 2) if total_true > 0 else 0

    print(f"\n{'='*20} Final Pipeline Report {'='*20}")
    print(f"Total True Samples: {total_true}")
    print(f"Total False Samples: {total_false} (Ratio 1:{actual_ratio})")

    # Optional 95/5 Split or Full Save
    if args.do_val_split:
        print("Executing 95/5 Train/Validation split...")
        split_idx = int(len(final_df) * 0.95)
        
        train_df = final_df.iloc[:split_idx].reset_index(drop=True)
        val_df = final_df.iloc[split_idx:].reset_index(drop=True)
        
        train_df.to_parquet(args.output_path)
        val_df.to_parquet(args.val_output_path)
        
        print(f"Train Dataset Size: {len(train_df)} rows -> Saved to {args.output_path}")
        print(f"Val Dataset Size: {len(val_df)} rows -> Saved to {args.val_output_path}\n")
    else:
        final_df.to_parquet(args.output_path)
        print(f"Final Dataset Size: {len(final_df)} rows -> Saved to {args.output_path}\n")

if __name__ == "__main__":
    main()
# python data_preparation/build_rldata_single_task.py --task place_wine_at_rack_location

"""
Single-Task RL Data Preparation

Main Function:
Extracts and balances correction (true) and background (false) samples for a 
specific, single task. Highly recommended for debugging, sanity checks, 
and ablation studies prior to full multi-task training.
"""

import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Single-Task RL Data Preparation Tool")
    parser.add_argument("--task", type=str, required=True, help="Target task name (e.g., hockey)")
    parser.add_argument("--false_multiplier", type=float, default=1.5, help="Ratio of false to true samples")
    parser.add_argument("--ref_path", type=str, default="/data/colosseum_dataset/rl_train_hockey.parquet")
    parser.add_argument("--output_path", type=str, default="", help="Custom output path. Defaults to auto-generated name.")
    parser.add_argument("--do_val_split", action="store_true", help="Enable 95/5 train/validation split")
    args = parser.parse_args()

    print(f"\n{'='*60}\nStarting Single-Task Extraction | Target Task: {args.task}\n{'='*60}")

    # 1. Load Data
    train_in = "/data/colosseum_dataset/rl_train_split.parquet"
    val_in = "/data/colosseum_dataset/rl_val_split.parquet"
    df_all = pd.concat([pd.read_parquet(train_in), pd.read_parquet(val_in)], ignore_index=True)
    print(f"Loaded {len(df_all)} total rows from source datasets.")

    # 2. Preprocess
    df_all['temp_task'] = df_all['extra_info'].apply(lambda x: x.get('task', 'unknown'))
    df_all['is_true'] = df_all['extra_info'].apply(lambda x: x.get('is_correction_sample', False))
    
    # FIX: Safe length check to prevent IndexError
    df_all['is_from_fail'] = df_all['images'].apply(
        lambda x: len(x) > 0 and ('failed_eval_demos' in str(x[0]) or 'failure' in str(x[0]))
    )

    # 3. Filter Task
    df_task = df_all[df_all['temp_task'].str.startswith(args.task)].copy()
    if len(df_task) == 0:
        print(f"Error: No data found for task '{args.task}'.")
        return

    print(f"Executing elite sampling strategy for task '{args.task}'...")

    # 4. Elite Sampling Logic
    true_raw = df_task[df_task['is_true'] == True]
    true_elite = true_raw.drop_duplicates(subset=['problem'])
    num_true = len(true_elite)

    if num_true == 0:
        print(f"Warning: No true correction samples found for task '{args.task}'. Proceeding with caution.")
        num_false_target = 0
        false_elite = pd.DataFrame()
        fail_normal_pool_size = 0
    else:
        num_false_target = int(num_true * args.false_multiplier)
        
        fail_normal_pool = df_task[(df_task['is_true'] == False) & (df_task['is_from_fail'] == True)]
        expert_normal_pool = df_task[(df_task['is_true'] == False) & (df_task['is_from_fail'] == False)]
        fail_normal_pool_size = len(fail_normal_pool)

        if fail_normal_pool_size >= num_false_target:
            false_elite = fail_normal_pool.sample(n=num_false_target, random_state=42)
        else:
            num_expert_needed = num_false_target - fail_normal_pool_size
            print(f"  Note: Insufficient fail frames ({fail_normal_pool_size}). Supplementing with {num_expert_needed} expert frames.")
            expert_sample = expert_normal_pool.sample(n=min(num_expert_needed, len(expert_normal_pool)), random_state=42)
            false_elite = pd.concat([fail_normal_pool, expert_sample])

    # 5. Merge and Format
    final_df = pd.concat([true_elite, false_elite]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    cols_to_drop = [c for c in ['temp_task', 'is_true', 'is_from_fail'] if c in final_df.columns]
    final_df = final_df.drop(columns=cols_to_drop)
    
    if os.path.exists(args.ref_path):
        ref_cols = set(pd.read_parquet(args.ref_path).columns)
        new_cols = set(final_df.columns)
        if ref_cols == new_cols:
            print("  Verification: Schema perfectly matches the reference dataset.")
        else:
            print(f"  Warning: Schema mismatch! Missing: {ref_cols - new_cols}, Extra: {new_cols - ref_cols}")

    # 6. Report and Save
    print(f"\n{'='*20} Final Task Dataset Report {'='*20}")
    print(f"Task: {args.task}")
    print(f"Unique True Samples: {num_true}")
    print(f"Sampled False Samples: {len(false_elite)} (From fail pool: {min(fail_normal_pool_size, num_false_target)})")

    # Dynamic output naming and 95/5 Split
    if args.do_val_split:
        print("\nExecuting 95/5 Train/Validation split...")
        split_idx = int(len(final_df) * 0.95)
        
        train_df = final_df.iloc[:split_idx].reset_index(drop=True)
        val_df = final_df.iloc[split_idx:].reset_index(drop=True)
        
        train_out = args.output_path if args.output_path else f"/data/colosseum_dataset/rl_train_{args.task}.parquet"
        val_out = f"/data/colosseum_dataset/rl_val_{args.task}.parquet"
        
        train_df.to_parquet(train_out)
        val_df.to_parquet(val_out)
        
        print(f"Train Size: {len(train_df)} rows -> Saved to {train_out}")
        print(f"Val Size:   {len(val_df)} rows -> Saved to {val_out}\n")
    else:
        out_path = args.output_path if args.output_path else f"/data/colosseum_dataset/rl_elite_{args.task}_ultimate.parquet"
        final_df.to_parquet(out_path)
        print(f"\nFinal Dataset Size: {len(final_df)} rows -> Saved to {out_path}\n")

if __name__ == "__main__":
    main()
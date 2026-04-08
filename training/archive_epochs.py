import os
import shutil
import time
import re

# --- 配置区 ---
CHECKPOINT_DIR = "/home/zxn/EasyR1-main/checkpoints/easy_r1/momagraph_dapo_multitask_baseline"
ARCHIVE_DIR = os.path.join(CHECKPOINT_DIR, "epoch_archive")
STEPS_PER_EPOCH = 110  # 刚才算出的步长
# --------------

os.makedirs(ARCHIVE_DIR, exist_ok=True)
print(f"📡 检查点护航员启动！目标目录: {CHECKPOINT_DIR}")

archived_steps = set()

while True:
    try:
        # 扫描 global_step_XXX 文件夹
        folders = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("global_step_")]
        
        for folder in folders:
            step_num = int(re.search(r'\d+', folder).group())
            
            # 判断是否是 Epoch 的倍数，且还没归档过
            if step_num % STEPS_PER_EPOCH == 0 and step_num not in archived_steps:
                src_path = os.path.join(CHECKPOINT_DIR, folder)
                dst_path = os.path.join(ARCHIVE_DIR, f"epoch_{step_num // STEPS_PER_EPOCH}_step_{step_num}")
                
                # 检查文件夹是否已经写入完成（通过检查是否有特定的标志文件，或简单等待）
                if os.path.exists(os.path.join(src_path, "actor")):
                    print(f"📦 发现 Epoch 节点: Step {step_num}，正在归档...")
                    shutil.copytree(src_path, dst_path)
                    archived_steps.add(step_num)
                    print(f"✅ 已永久保留至: {dst_path}")
                    
    except Exception as e:
        print(f"⚠️ 扫描中遇到错误: {e}")
        
    time.sleep(60) # 每分钟扫描一次
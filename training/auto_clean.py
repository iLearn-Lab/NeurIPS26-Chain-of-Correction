import os
import time
import shutil

# 替换成你的真实输出路径
CHECKPOINT_DIR = "/data/colosseum_dataset/checkpoints/sft/v1_oversample"
KEEP_LATEST_N = 2  # 只保留最新的 2 个 step checkpoint

print(f"🧹 开始后台巡逻清理: {CHECKPOINT_DIR}")

while True:
    if os.path.exists(CHECKPOINT_DIR):
        # 1. 找出所有以 'checkpoint-' 开头，并且后面全都是数字的文件夹 (排除 epoch 和 final)
        step_ckpts = []
        for name in os.listdir(CHECKPOINT_DIR):
            if name.startswith("checkpoint-") and name.split("-")[1].isdigit():
                step_ckpts.append(name)
        
        # 2. 按步数大小排序
        step_ckpts.sort(key=lambda x: int(x.split("-")[1]))
        
        # 3. 如果数量超过保留上限，开始删！
        if len(step_ckpts) > KEEP_LATEST_N:
            to_delete = step_ckpts[:-KEEP_LATEST_N]
            for ckpt in to_delete:
                ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
                try:
                    shutil.rmtree(ckpt_path)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🗑️ 已删除过期检查点: {ckpt}")
                except Exception as e:
                    print(f"删除 {ckpt} 失败: {e}")
                    
    # 每 30 分钟 (1800 秒) 醒来检查一次
    time.sleep(1800)
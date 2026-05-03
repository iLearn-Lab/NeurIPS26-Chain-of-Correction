#!/usr/bin/env python3
"""
SFT Trajectory Training Script for Multimodal CoT.

Optimized for Qwen2.5-VL-3B/7B. This script bridges structured JSON 
trajectories (Perception -> Logic -> Action) into a format compatible 
with the native Qwen Vision-Language Processor.

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=4 \
        --master_port=29516 \
        scripts/train_sft_cot.py \
        --config configs/sft_config.yaml \
        --data-path /data/colosseum_dataset/mixed_sft_v1.json \
        --output-dir /data/checkpoints/sft/v1 \
        --run-name corl-production-sft
"""

import argparse
import logging
import os
import sys
import json
import glob
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("corl.sft_train")

# =========================================================================
# 1. Dataset Adapter: Bridging CoT JSON to VLM Vision Blocks
# =========================================================================
class CoRLQwenDataset(Dataset):
    """
    Adapter to convert multi-turn CoT JSON files into structured VLM inputs.
    
    Transforms raw text containing <image> placeholders into standard 
    Qwen2.5-VL Vision Blocks with local file protocol support.
    """
    def __init__(self, data_path_str):
        self.data = []
        loaded_files_count = 0
         
        paths = [p.strip() for p in data_path_str.split(',') if p.strip()]
        
        for path in paths:
            if os.path.isdir(path):
                # Recursively discover all JSON files in the directory
                json_files = glob.glob(os.path.join(path, "*.json"))
                for json_file in json_files:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        self.data.extend(json.load(f))
                        loaded_files_count += 1
            elif os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.data.extend(json.load(f))
                    loaded_files_count += 1
            else:
                logger.warning(f"Path not found or invalid: {path}")

        logger.info(f"Dataset Initialized: Loaded {len(self.data)} samples from {loaded_files_count} files.")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # Deep copy to prevent accidental mutation of the in-memory cache
        import copy
        msgs = copy.deepcopy(item["messages"]) 
        image_paths = item.get("image_paths", [])
        
        prompt = msgs[:-1]  
        ground_truth = msgs[-1]["content"]  
        
        # Format Alignment: Inject structured Vision Blocks into the final user turn
        if len(image_paths) > 0:
            last_user_idx = -1
            for i in range(len(prompt) - 1, -1, -1):
                if prompt[i]["role"] == "user":
                    last_user_idx = i
                    break
                    
            if last_user_idx != -1:
                original_text = prompt[last_user_idx]["content"]
                # Strip raw <image> tags as they will be replaced by structured blocks
                cleaned_text = original_text.replace("<image>\n", "").replace("<image>", "").lstrip()
                 
                blocks = []
                for img_path in image_paths:
                    # Qwen2.5-VL utilizes the 'file://' protocol for efficient local image loading
                    blocks.append({"type": "image", "image": f"file://{img_path}"})
                
                blocks.append({"type": "text", "text": cleaned_text})
                
                # Replace string content with the structured multimodal block list
                prompt[last_user_idx]["content"] = blocks
        
        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "images": image_paths 
        }

# =========================================================================
# 2. Main Training Orchestration
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoRL SFT-Only Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data-path", type=str, required=True, help="Comma-separated list of paths to JSON files or directories")
    parser.add_argument("--output-dir", type=str, default="./outputs_sft", help="Output directory")
    parser.add_argument("--run-name", type=str, default="corl-3b-sft-cot", help="WandB run name for this experiment")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to checkpoint or 'True' to find latest")
    args = parser.parse_args()

    from configs.base_config import ProductionConfig
    from models.qwen_vlm import ProductionCoRLModel
    from distributed.utils import setup_distributed, is_main_process, cleanup_distributed
    from training.sft_trainer import SFTTrainer

    try:
        import wandb
    except ImportError:
        wandb = None

    # Setup distributed
    is_distributed = setup_distributed()
    rank = torch.distributed.get_rank() if is_distributed else 0

    # Load config
    config = ProductionConfig.from_yaml(args.config)
    logger.info("[Rank %d] Config loaded from %s", rank, args.config)

    # Init wandb (main process only)
    if is_main_process() and wandb:
        try:
            wandb.init(project="corl-production", name=args.run_name)
        except Exception as e:
            logger.warning("wandb init failed: %s. Continuing without wandb.", e)
            wandb = None

    # Init model
    logger.info("[Rank %d] Loading model from %s ...", rank, config.model.path)
    model = ProductionCoRLModel(config.model)
    if not model.weights_loaded:
        logger.error("Model weights not loaded. Cannot proceed.")
        sys.exit(1)
    logger.info("[Rank %d] Model loaded.", rank)

    # Move model to local GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    # Store processor reference before DDP wrapping
    processor = model.processor

    # DDP wrap (full model replica per GPU — fits in 80GB A100)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], static_graph=True)
        # Re-attach processor so trainers can access it
        model.processor = processor
        logger.info("[Rank %d] DDP wrapping complete.", rank)

    # Output dir
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Dataset and Trainer
    logger.info("=== Starting SFT Training on CoT Data ===")
     
    sft_dataset = CoRLQwenDataset(args.data_path)
    logger.info("[Rank %d] Ready to train on %d samples.", rank, len(sft_dataset))
    
    sft_trainer = SFTTrainer(
        model=model, 
        dataset=sft_dataset, 
        config=config,
        output_dir=str(output_dir)
    )
    
    # sft_checkpoint = sft_trainer.train()
    sft_checkpoint = sft_trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info("SFT complete. Checkpoint: %s", sft_checkpoint)

    # Cleanup
    if is_main_process() and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass
    if is_distributed:
        cleanup_distributed()

    logger.info("Training script finished successfully!")



if __name__ == "__main__":
    main()
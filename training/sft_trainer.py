"""Production SFT Trainer for Phase 1 (cold start) of the pipeline.

Implements supervised fine-tuning for Qwen2.5-VL with:
- Native multi-image processing via Qwen2.5-VL processor
- Chat-template formatting with label masking (only supervise on responses)
- Distributed training with DistributedSampler
- AdamW optimiser with cosine-annealing LR schedule
- Gradient accumulation and clipping
- wandb logging and periodic checkpointing
"""

import os
import logging
from typing import Dict, List, Any, Optional 
import bitsandbytes as bnb

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from configs.base_config import ProductionConfig
from models.qwen_vlm import convert_prompt_to_qwen_messages
from distributed.utils import get_world_size, get_rank, is_main_process

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


class SFTTrainer:
    """Supervised Fine-Tuning trainer for chain-of-thought reasoning (Phase 1).

    The trainer tokenises each sample on-the-fly in ``_process_batch``
    (not in the DataLoader workers) so the Qwen2.5-VL processor stays
    on the main thread and num_workers > 0 is safe for data loading.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        config: ProductionConfig,
        output_dir: str = "./outputs",
    ):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.output_dir = output_dir

        # Keep a direct reference to the processor (works even after FSDP wrap)
        self.processor = getattr(model, "processor", None)

        sft = config.training.sft
        # self.max_steps = sft.max_steps
        self.epochs = getattr(sft, "epochs", 4)
        self.batch_size = sft.batch_size
        self.learning_rate = sft.learning_rate
        self.warmup_steps = getattr(sft, "warmup_steps", 500)
        self.grad_accumulation_steps = sft.grad_accumulation_steps
        self.max_grad_norm = config.training.max_grad_norm

        # Device — DDP models are already on GPU; standalone models need moving
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if not isinstance(self.model, DDP):
            self.model = self.model.to(self.device)

        # DataLoader -- collate returns raw dicts; tokenisation happens later
        world_size = get_world_size()
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=get_rank())
            if world_size > 1
            else None
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=_identity_collate,
            num_workers=config.data.num_workers,
            prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
            drop_last=True,
        )

        # 动态计算总步数 
        batches_per_epoch = len(self.dataloader)
        self.total_global_steps = batches_per_epoch * self.epochs
        self.total_optim_steps = self.total_global_steps // self.grad_accumulation_steps

        if is_main_process():
            logger.info(f"Dataset length: {len(dataset)} | Batches per epoch: {batches_per_epoch}")
            logger.info(f"Training for {self.epochs} epochs. Total Optim Steps: {self.total_optim_steps}")

        # Optimiser
        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(), lr=self.learning_rate, weight_decay=0.01,
        # )
        self.optimizer = bnb.optim.AdamW8bit(
            self.model.parameters(), lr=self.learning_rate, weight_decay=0.01,
        ) 
        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_optim_steps, eta_min=self.learning_rate * 0.1,
        )

        self.global_step = 0
        self.optimizer_step = 0

        # wandb (main process only)
        self._wandb_ok = False
        if is_main_process() and wandb is not None:
            try:
                wandb.init(
                    project="corl-sft",
                    config={
                        "epochs": self.epochs,
                        "total_optim_steps": self.total_optim_steps,
                        "batch_size": self.batch_size,
                        "learning_rate": self.learning_rate,
                        "warmup_steps": self.warmup_steps,
                        "grad_accumulation_steps": self.grad_accumulation_steps,
                    },
                    reinit=True,
                )
                self._wandb_ok = True
            except Exception:
                logger.warning("wandb init failed; continuing without logging.")

    # ------------------------------------------------------------------
    # Batch processing (tokenisation + label masking)
    # ------------------------------------------------------------------

    def _process_batch(self, raw_batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Tokenise a raw batch of dataset samples.

        For each sample we build the *full* conversation (prompt + assistant
        response) and a *prompt-only* version.  The prompt-only token length
        is used to mask labels so that the loss is computed only on the
        assistant response tokens.
        """
        if self.processor is None:
            # Fallback (no weights)
            n = len(raw_batch)
            return {
                "input_ids": torch.zeros(n, 50, dtype=torch.long),
                "attention_mask": torch.ones(n, 50, dtype=torch.long),
                "labels": torch.zeros(n, 50, dtype=torch.long),
            }

        full_texts: List[str] = []
        prompt_texts: List[str] = []
        all_images: List = []

        from PIL import Image

        for sample in raw_batch:
            images = sample["images"]  # 这是一个路径字符串列表
            prompt = sample["prompt"]
            ground_truth = sample["ground_truth"]

            # ================= 添加的图像读取逻辑 =================
            pil_images = []
            for img_path in images:
                try:
                    pil_images.append(Image.open(img_path).convert("RGB"))
                except Exception as e:
                    logger.error(f"无法读取图片 {img_path}: {e}")
            all_images.append(pil_images)
            # ====================================================== 

            # -- full conversation (prompt + assistant response) --
            full_msgs = list(prompt) + [
                {"role": "assistant", "content": ground_truth}
            ]
            qwen_full = convert_prompt_to_qwen_messages(full_msgs)
            full_text = self.processor.apply_chat_template(
                qwen_full, tokenize=False, add_generation_prompt=False,
            )
            full_texts.append(full_text)

            # -- prompt only (to measure prompt length for masking) --
            qwen_prompt = convert_prompt_to_qwen_messages(list(prompt))
            prompt_text = self.processor.apply_chat_template(
                qwen_prompt, tokenize=False, add_generation_prompt=True,
            )
            prompt_texts.append(prompt_text) 

        # Tokenise full conversations
        inputs = self.processor(
            text=full_texts,
            images=all_images if any(all_images) else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_seq_len,
        )

        # =====================================================================
        #  对齐检查：仅在第一步打印第一条样本的全链路数据流
        # =====================================================================
        if self.global_step == 0 and is_main_process():
            logger.info("\n" + "▼"*60)
            logger.info("🚦 [TRAIN 对齐核查] 首条样本的全链路数据流拆解")
            logger.info("▼"*60)
            
            # 1. 检查图片是否被正常读取为 PIL 对象
            num_imgs = len(all_images[0]) if all_images and len(all_images) > 0 else 0
            logger.info(f"▶ [1. 物理图片]: 成功读取了 {num_imgs} 张 PIL.Image 对象")
            
            # 2. 检查应用 Chat Template 后的纯文本（看看历史还有没有残留的 <image>）
            logger.info(f"\n▶ [2. 应用 Template 后的文本 (full_texts[0])]:\n{full_texts[0]}")
            
            # 3. 最关键的一步！把 input_ids 解码回人类可读文本，看看模型底层到底看到了什么
            # 注意：这里 skip_special_tokens=False，为了暴露出所有的视觉占位符
            decoded_ids = self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            logger.info(f"\n▶ [3. 模型底层的真实 Input Tokens (关注 <|image_pad|> 的位置)]:\n{decoded_ids}")
            
            logger.info("▲"*60 + "\n")
        # =====================================================================
 
        if self.global_step == 0 and is_main_process():
            logger.info(f"Checking Batch Data:")
            logger.info(f"- Pixel Values Shape: {inputs.get('pixel_values', torch.empty(0)).shape}")
            logger.info(f"- Input IDs Length: {inputs['input_ids'].shape[1]}")
            tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][:20])
            logger.info(f"- First 20 Tokens: {tokens}")

        # 使用完整的 processor (含图像) 来计算 Prompt 的真实长度
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=all_images if any(all_images) else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_seq_len,
        )
        
        # 通过 attention_mask 求和来获取每个 prompt 的准确 token 数量
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)

        # Build labels: -100 for prompt tokens and padding
        labels = inputs["input_ids"].clone()
        for i, plen in enumerate(prompt_lengths):
            # 确保只 mask 掉真正的 prompt 长度
            labels[i, :plen] = -100
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels

        # 仅在主进程的第 0 步打印前 2 条样本的掩码情况
        if get_rank() == 0 and self.global_step == 0:
            for i in range(min(2, len(raw_batch))):
                ids = inputs["input_ids"][i]
                lbs = labels[i]
                tokens = self.processor.tokenizer.convert_ids_to_tokens(ids)
                debug_str = f"\n--- Sample {i} Masking Check ---"
                for t, l in zip(tokens, lbs):
                    # 绿色表示计算 Loss，红色表示被屏蔽
                    color = "\033[92m" if l != -100 else "\033[91m"
                    debug_str += f"{color}{t}\033[0m "
                print(debug_str + "\n") 
        return inputs

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Optional[str]:
        """Run the full SFT training loop.  Returns final checkpoint path."""
        self.model.train()
        os.makedirs(self.output_dir, exist_ok=True)

        # data_iter = iter(self.dataloader)
        accum_loss = 0.0

        for epoch in range(self.epochs):
            if is_main_process():
                logger.info(f"\n=======================================================")
                logger.info(f"🚀 STARTING EPOCH {epoch + 1} / {self.epochs}")
                logger.info(f"=======================================================\n")
 
            if isinstance(self.dataloader.sampler, DistributedSampler):
                self.dataloader.sampler.set_epoch(epoch)
 
            for raw_batch in self.dataloader: 

                if self.global_step < 3:
                    logger.info("step=%d: processing batch (%d samples)...", self.global_step, len(raw_batch))

                batch = self._process_batch(raw_batch)
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                if self.global_step < 3:
                    logger.info("step=%d: forward pass (input_ids shape=%s)...", self.global_step, batch.get("input_ids", torch.zeros(0)).shape)

                outputs = self.model(**batch)
                loss = outputs.loss / self.grad_accumulation_steps

                if self.global_step < 3:
                    logger.info("step=%d: backward pass (loss=%.4f)...", self.global_step, loss.item())

                loss.backward()
                accum_loss += loss.item()

                if (self.global_step + 1) % self.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.optimizer_step += 1

                    if is_main_process():
                        lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            "Epoch=%d  optim_step=%d  global_step=%d  loss=%.4f  lr=%.2e",
                            epoch + 1, self.optimizer_step, self.global_step, accum_loss, lr,
                        )
                        if self._wandb_ok:
                            wandb.log(
                                {"loss": accum_loss, "lr": lr,
                                 "optimizer_step": self.optimizer_step, "epoch": epoch + 1},
                                step=self.global_step,
                            )
                    accum_loss = 0.0

                self.global_step += 1

                # 每 500 步保存一次 Checkpoint
                if self.global_step > 0 and self.global_step % 500 == 0:
                    self._save_checkpoint(
                        os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
                    )

            # 在每个 Epoch 结束时，额外存一个带有 Epoch 标记的 Checkpoint
            self._save_checkpoint(
                os.path.join(self.output_dir, f"checkpoint-epoch-{epoch + 1}")
            )

        # 所有 Epoch 跑完后保存最终模型
        final = os.path.join(self.output_dir, "checkpoint-final")
        self._save_checkpoint(final)
        return final


    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: str) -> None:
        if not is_main_process():
            return
        
        # 把 path 当作一个文件夹来创建
        os.makedirs(path, exist_ok=True)
        
        # 剥离 DDP 的 wrapper 获取最原始的模型
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        # 使用 Hugging Face 原生的 save_pretrained，这样你可以直接从这个目录 load 模型
        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(path)
            # 顺便把 processor (Tokenizer 和 ImageProcessor) 也存下来
            if hasattr(self, "processor") and self.processor:
                self.processor.save_pretrained(path)
        else:
            torch.save(model_to_save.state_dict(), os.path.join(path, "pytorch_model.bin"))
            
        # 保存训练状态，防止中断后要断点续训
        state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "optim_step": self.optimizer_step,
        }
        torch.save(state, os.path.join(path, "trainer_state.pt"))
        
        logger.info("Saved Hugging Face checkpoint to %s", path)


# ---------------------------------------------------------------------------
# Collate helper
# ---------------------------------------------------------------------------

def _identity_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pass samples through unchanged (tokenisation happens in train loop)."""
    return batch
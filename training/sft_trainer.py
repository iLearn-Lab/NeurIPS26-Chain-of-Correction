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
import glob
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

        # Calculate training step budget
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
            images = sample["images"]
            prompt = sample["prompt"]
            ground_truth = sample["ground_truth"]

            # Load and convert images to RGB
            pil_images = []
            for img_path in images:
                try:
                    pil_images.append(Image.open(img_path).convert("RGB"))
                except Exception as e:
                    logger.error(f"Image load failed for {img_path}: {e}")
            all_images.append(pil_images)

            # Construct Full Conversation (Prompt + Response)
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

        # First-step data flow inspection (Master rank only)
        if self.global_step == 0 and is_main_process():
            logger.info("\n" + "▼" * 60)
            logger.info("TRAINING PIPELINE AUDIT: First Sample Analysis")
            logger.info("▼" * 60)
            num_imgs = len(all_images[0]) if all_images else 0
            logger.info(f"Step 1: Loaded {num_imgs} PIL.Image objects.")
            logger.info(f"Step 2: Template applied. Raw text preview:\n{full_texts[0][:500]}...")
            decoded_ids = self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
            logger.info(f"Step 3: Decoded Input IDs (Visual padding markers included):\n{decoded_ids[:1000]}...")
            logger.info("▲" * 60 + "\n")
        # =====================================================================
 
        if self.global_step == 0 and is_main_process():
            logger.info(f"Checking Batch Data:")
            logger.info(f"- Pixel Values Shape: {inputs.get('pixel_values', torch.empty(0)).shape}")
            logger.info(f"- Input IDs Length: {inputs['input_ids'].shape[1]}")
            tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][:20])
            logger.info(f"- First 20 Tokens: {tokens}")

        # Calculate accurate prompt lengths (accounting for visual tokens)
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=all_images if any(all_images) else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.model.max_seq_len,
        )
         
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)

        # Build labels: -100 for prompt tokens and padding
        labels = inputs["input_ids"].clone()
        for i, plen in enumerate(prompt_lengths):
            labels[i, :plen] = -100
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels
 
        if get_rank() == 0 and self.global_step == 0:
            for i in range(min(2, len(raw_batch))):
                ids = inputs["input_ids"][i]
                lbs = labels[i]
                tokens = self.processor.tokenizer.convert_ids_to_tokens(ids)
                debug_str = f"\n--- Sample {i} Masking Check ---"
                for t, l in zip(tokens, lbs): 
                    color = "\033[92m" if l != -100 else "\033[91m"
                    debug_str += f"{color}{t}\033[0m "
                print(debug_str + "\n") 
        return inputs

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Optional[str]:
        """Run the full SFT training loop.  Returns final checkpoint path."""

        # Handle training resumption logic
        if resume_from_checkpoint:
            checkpoint_path = None
            if resume_from_checkpoint.lower() == "true":
                checkpoints = glob.glob(os.path.join(self.output_dir, "checkpoint-*"))
                checkpoints = [c for c in checkpoints if c.split('-')[-1].isdigit()]
                if checkpoints:
                    checkpoint_path = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
            else:
                checkpoint_path = resume_from_checkpoint

            if checkpoint_path and os.path.exists(checkpoint_path):
                if is_main_process():
                    logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
                 
                model_to_load = self.model.module if hasattr(self.model, "module") else self.model
                 
                state_dict_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                if not os.path.exists(state_dict_path): 
                     state_dict_path = os.path.join(checkpoint_path, "model.safetensors")
                 
                state_path = os.path.join(checkpoint_path, "trainer_state.pt")
                if os.path.exists(state_path): 
                    state = torch.load(state_path, map_location="cpu", weights_only=False)
                    self.optimizer.load_state_dict(state["optimizer"])
                    self.scheduler.load_state_dict(state["scheduler"])
                    self.global_step = state["global_step"]
                    self.optimizer_step = state["optim_step"]
                    del state
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache() 
                     
            else:
                if is_main_process():
                    logger.warning(f"{resume_from_checkpoint} not found.")


        self.model.train()
        os.makedirs(self.output_dir, exist_ok=True)

        # data_iter = iter(self.dataloader)
        accum_loss = 0.0

        for epoch in range(self.epochs):
            if is_main_process():
                logger.info(f"\n=======================================================")
                logger.info(f" STARTING EPOCH {epoch + 1} / {self.epochs}")
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
 
                if self.global_step > 0 and self.global_step % 500 == 0:
                    self._save_checkpoint(
                        os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
                    )
 
            self._save_checkpoint(
                os.path.join(self.output_dir, f"checkpoint-epoch-{epoch + 1}")
            )
 
        final = os.path.join(self.output_dir, "checkpoint-final")
        self._save_checkpoint(final)
        return final


    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: str) -> None:
        if not is_main_process():
            return
         
        os.makedirs(path, exist_ok=True)
         
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
         
        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(path) 
            if hasattr(self, "processor") and self.processor:
                self.processor.save_pretrained(path)
        else:
            torch.save(model_to_save.state_dict(), os.path.join(path, "pytorch_model.bin"))
             
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
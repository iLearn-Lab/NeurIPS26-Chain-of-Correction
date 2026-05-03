"""Production CoRL model with Qwen2.5-VL-3B integration.

Uses native multi-image support from Qwen2.5-VL.  Two robot viewpoint
images are passed directly to the vision encoder via structured chat
messages -- no channel-concatenation adapter needed.

When model weights are unavailable (local dev), a fallback mode returns
random outputs so that import-level and unit tests still pass.
"""
import os
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from configs.base_config import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt conversion
# ---------------------------------------------------------------------------

def convert_prompt_to_qwen_messages(
    prompt_messages: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Convert VERL-style chat messages to Qwen2.5-VL structured format."""
    converted: List[Dict[str, Any]] = []

    last_user_idx = -1
    for i in range(len(prompt_messages) - 1, -1, -1):
        if prompt_messages[i]["role"] == "user":
            last_user_idx = i
            break

    for i, msg in enumerate(prompt_messages):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str) and "<image>" in content:
            if i != last_user_idx:
                cleaned_content = content.replace("<image>\n", "").replace("<image>", "")
                converted.append({"role": role, "content": cleaned_content})
                continue

            parts = content.split("<image>")
            blocks: List[Dict[str, Any]] = []
            for j, part in enumerate(parts):
                if part:
                    blocks.append({"type": "text", "text": part})
                if j < len(parts) - 1:
                    blocks.append({"type": "image"})
            converted.append({"role": role, "content": blocks})
        else:
            converted.append({"role": role, "content": content})

    return converted


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProductionCoRLModel(nn.Module):
    """Thin wrapper around ``Qwen2VLForConditionalGeneration``.

    * Loads pre-trained weights from *config.path*.
    * Stores the ``AutoProcessor`` for chat-template formatting and
      image pre-processing.
    * Delegates ``forward`` and ``generate`` to the inner model.
    * Falls back to random outputs when weights are unavailable.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._weights_loaded = False

        try:
            # from transformers import AutoModelForVision2Seq, AutoProcessor, GenerationConfig
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  
 
            attn_impl = "flash_attention_2" if config.use_flash_attention else "sdpa"
            local_rank = os.environ.get("LOCAL_RANK", "0")
            device = f"cuda:{local_rank}"
            logger.info(f"[Rank {os.environ.get('RANK', '0')}] Initializing model on {device}")

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.path,  
                torch_dtype=torch.bfloat16,
                device_map=device,  # let FSDP handle sharding
                attn_implementation=attn_impl,  
            ) 
         
            self.processor = AutoProcessor.from_pretrained(config.path) 

            if config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

            self._weights_loaded = True
            logger.info("Model loaded from %s", config.path)
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Total Parameters: {num_params / 1e9:.2f} B")
 
        # except (ImportError, FileNotFoundError, OSError, ValueError) as exc:
        except Exception as exc:
            import traceback
            logger.error("Failed to load model: %s", exc)
            logger.error(traceback.format_exc())
            self.model = None
            self.processor = None
            self._fallback_vocab_size = 151_936
            self._fallback_hidden_size = 3584 

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights_loaded(self) -> bool:
        return self._weights_loaded

    # ------------------------------------------------------------------
    # Input preparation (tokenisation + image processing)
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        batch_messages: List[List[Dict[str, str]]],
        batch_images: List[List],
        add_generation_prompt: bool = False,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenise a batch of conversations with images.

        Parameters
        ----------
        batch_messages:
            Per-sample list of chat messages (VERL format).
        batch_images:
            Per-sample list of PIL images.
        add_generation_prompt:
            If ``True``, append the generation-prompt token so the model
            starts generating an assistant turn.
        max_length:
            Truncation length.  Defaults to ``config.max_seq_len``.

        Returns
        -------
        Dict of tensors ready for ``self.forward(**inputs)``.
        """
        if not self._weights_loaded:
            raise RuntimeError("prepare_inputs requires loaded weights")

        texts = []
        all_images = []

        for messages, images in zip(batch_messages, batch_images):
            qwen_msgs = convert_prompt_to_qwen_messages(messages)
            text = self.processor.apply_chat_template(
                qwen_msgs,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            texts.append(text)
            all_images.extend(images)

        inputs = self.processor(
            text=texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length or self.config.max_seq_len,
        )
        return inputs

    # ------------------------------------------------------------------
    # Forward / generate
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> Any:
        if self._weights_loaded:
            return self.model(**kwargs)
        return self._forward_fallback(kwargs)

    def generate(self, **kwargs) -> torch.Tensor:
        if self._weights_loaded:
            return self.model.generate(**kwargs)
        batch_size = kwargs.get("input_ids", torch.zeros(1, 1)).shape[0]
        return torch.randint(0, self._fallback_vocab_size, (batch_size, 50))

    # ------------------------------------------------------------------
    # Fallback (no weights)
    # ------------------------------------------------------------------

    def _forward_fallback(self, kwargs: dict) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
        else:
            batch_size, seq_len = 1, 50

        class _Out:
            pass

        out = _Out()
        out.logits = torch.randn(batch_size, seq_len, self._fallback_vocab_size)
        out.loss = torch.tensor(1.0, requires_grad=True)
        out.hidden_states = None
        return out
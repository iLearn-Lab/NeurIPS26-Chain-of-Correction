"""Distributed training utility functions.

Provides helpers to initialise and query the PyTorch distributed
process group.  All functions degrade gracefully when running outside
of a distributed context (single-GPU / CPU development mode).
"""

import os
import logging
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed() -> bool:
    """Initialise the distributed process group from environment variables.

    Expects the standard ``torchrun`` / ``torch.distributed.launch``
    environment variables: ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, and
    ``MASTER_ADDR`` / ``MASTER_PORT``.

    Returns
    -------
    bool
        ``True`` if the process group was successfully initialised,
        ``False`` otherwise (e.g. env vars missing or CUDA unavailable).
    """
    if dist.is_initialized():
        logger.info("Distributed process group already initialised.")
        return True

    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    local_rank = os.environ.get("LOCAL_RANK")

    if rank is None or world_size is None or local_rank is None:
        logger.info(
            "Distributed environment variables (RANK, WORLD_SIZE, LOCAL_RANK) "
            "not set. Running in non-distributed mode."
        )
        return False

    rank = int(rank)
    world_size = int(world_size)
    local_rank = int(local_rank)

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Cannot initialise distributed training."
        )
        return False

    try:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        logger.info(
            "Distributed process group initialised: "
            f"rank={rank}, world_size={world_size}, local_rank={local_rank}"
        )
        return True
    except Exception as exc:
        logger.error(f"Failed to initialise distributed process group: {exc}")
        return False


def get_world_size() -> int:
    """Return the world size, defaulting to 1 when not distributed."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Return the global rank, defaulting to 0 when not distributed."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Return ``True`` if this is rank 0 (or non-distributed)."""
    return get_rank() == 0


def cleanup_distributed() -> None:
    """Destroy the distributed process group if it was initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")
"""
GPU and device utilities.
"""

from __future__ import annotations

from typing import Optional

import torch

from utils.logging_utils import get_logger

log = get_logger(__name__)


def get_device(preferred: str = "cuda") -> str:
    """
    Resolve the best available torch device.

    Args:
        preferred: Preferred device string ('cuda', 'mps', 'cpu').

    Returns:
        Resolved device string.
    """
    if preferred.startswith("cuda"):
        if torch.cuda.is_available():
            device = preferred
        else:
            log.warning("CUDA not available, falling back to CPU")
            device = "cpu"
    elif preferred == "mps":
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            log.warning("MPS not available, falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"

    log.info(f"Using device: {device}")
    return device


def get_gpu_memory_info() -> Optional[dict[str, float]]:
    """
    Return GPU memory usage in GB.

    Returns:
        Dict with 'allocated', 'reserved', 'total' in GB, or None if no GPU.
    """
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(total - reserved, 2),
    }


def clear_gpu_cache() -> None:
    """Free unused GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.debug("GPU cache cleared")

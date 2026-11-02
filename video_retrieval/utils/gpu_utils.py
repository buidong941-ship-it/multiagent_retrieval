"""
GPU and device utilities.
"""

from __future__ import annotations

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

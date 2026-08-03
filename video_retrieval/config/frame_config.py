"""
Frame extraction configuration.

Supports three extraction strategies:
- fixed_fps:   extract at a fixed frame rate (e.g., 1 fps)
- every_n:     extract every N frames
- shot_detect: PySceneDetect-based shot boundary detection (future)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator

from config.base_config import BASE_DIR, BaseConfig


class ExtractionMode(str, Enum):
    FIXED_FPS = "fixed_fps"
    EVERY_N = "every_n"
    SHOT_DETECT = "shot_detect"
    TRANSNETV2 = "transnetv2"


class FrameExtractionConfig(BaseConfig):
    """Configuration for frame extraction service."""

    # Extraction strategy
    mode: ExtractionMode = Field(
        default=ExtractionMode.TRANSNETV2,
        description="Frame extraction strategy",
    )

    # fixed_fps mode
    fps: float = Field(
        default=5.0,
        ge=0.1,
        le=30.0,
        description="Frames per second to extract (used in fixed_fps mode)",
    )

    # every_n mode
    every_n_frames: int = Field(
        default=30,
        ge=1,
        description="Extract one frame every N frames (used in every_n mode)",
    )

    # TransNetV2 mode settings
    transnet_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="TransNetV2 probability threshold for shot boundary cut detection",
    )
    keyframes_per_shot: int = Field(
        default=10,
        ge=1,
        description="Number of keyframes to extract per detected shot",
    )

    # Shot detection threshold (future)
    shot_threshold: float = Field(
        default=30.0,
        description="ContentDetector threshold for shot detection",
    )

    # Storage
    output_dir: Path = Field(
        default=BASE_DIR / "data" / "frames",
        description="Directory to store extracted frames",
    )
    image_format: str = Field(
        default="jpg",
        pattern="^(jpg|png|webp)$",
        description="Output image format",
    )
    image_quality: int = Field(
        default=85,
        ge=1,
        le=100,
        description="JPEG quality (1-100)",
    )
    max_dimension: Optional[int] = Field(
        default=None,
        description="Resize longest side to this value (None = no resize)",
    )

    # Parallelism
    num_workers: int = Field(
        default=4,
        ge=1,
        description="Number of parallel video processing workers",
    )

    model_config = {"env_prefix": "FRAME_"}

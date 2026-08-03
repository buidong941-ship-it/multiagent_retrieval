"""
Object detection configuration (YOLOv11).

Design Decision:
    - YOLOv11 chosen over YOLOv8 / RT-DETR because:
        * 10-15% mAP improvement on COCO vs YOLOv8
        * Faster inference with same parameter count
        * Native support for instance segmentation + pose
    - Detection results are stored as metadata ONLY.
      YOLO is NEVER re-run during online retrieval.
    - COCO 80-class labels used for BM25 text matching.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field

from config.base_config import BASE_DIR, BaseConfig


class YOLOModelSize(str, Enum):
    NANO = "yolo11n.pt"
    SMALL = "yolo11s.pt"
    MEDIUM = "yolo11m.pt"
    LARGE = "yolo11l.pt"
    XLARGE = "yolo11x.pt"


class DetectionConfig(BaseConfig):
    """Configuration for YOLOv11 object detection."""

    model_size: YOLOModelSize = Field(
        default=YOLOModelSize.LARGE,
        description="YOLOv11 model variant",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to custom YOLO weights (None = auto-download)",
    )
    device: str = Field(
        default="cuda",
        description="torch device for YOLO inference",
    )
    confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence",
    )
    iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS",
    )
    imgsz: int = Field(
        default=640,
        description="Input image size for YOLO",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size for detection inference",
    )
    max_detections: int = Field(
        default=100,
        ge=1,
        description="Maximum detections per image",
    )
    classes: Optional[list[int]] = Field(
        default=None,
        description="Filter to these COCO class IDs (None = all classes)",
    )
    half: bool = Field(
        default=True,
        description="Use FP16 inference",
    )

    model_config = {"env_prefix": "DETECT_"}

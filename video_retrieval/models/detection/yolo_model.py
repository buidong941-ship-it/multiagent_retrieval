"""
YOLOv11 object detection model wrapper.

Architecture:
    - CSP (Cross Stage Partial) backbone with C2PSA attention
    - PAN-FPN neck for multi-scale feature fusion
    - Decoupled head with anchor-free detection
    - Output: (class_id, confidence, x1, y1, x2, y2) per detection

Why YOLOv11 over alternatives:
    - YOLOv8:   YOLOv11 = +5% mAP at same latency
    - RT-DETR:  YOLOv11 is 3× faster inference (real-time requirement)
    - DINO:     YOLOv11 is 10× faster, good enough for offline indexing
    - Grounding DINO: Use GD when open-vocabulary detection needed (future)

CRITICAL Design Decision:
    - YOLO runs ONLY during offline indexing (never at query time).
    - Results stored in SQLite with class_name, confidence, normalized bbox.
    - Query-time object search = SQL filter on class_name column.
    - This reduces online latency from ~50ms → ~1ms per query.

Fine-tuning:
    - Use Ultralytics trainer: model.train(data=custom.yaml, epochs=50)
    - Start from yolo11l.pt pretrained on COCO
    - Add custom classes (logos, Vietnamese signs) with labeled data

Input:
    - Image path, numpy array, or PIL Image
    - Resized to 640×640 internally

Output:
    - List of DetectionResult with normalized bbox coordinates
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config.detection_config import DetectionConfig, YOLOModelSize
from interfaces.base_interfaces import BaseDetector, DetectionResult
from utils.gpu_utils import get_device
from utils.logging_utils import get_logger

log = get_logger(__name__)


class YOLOv11Detector(BaseDetector):
    """
    YOLOv11 object detector wrapping Ultralytics YOLO API.

    Attributes:
        config:  DetectionConfig instance.
        device:  Resolved torch device string.
        _model:  Ultralytics YOLO instance (lazy-loaded).
    """

    def __init__(self, config: DetectionConfig) -> None:
        """
        Initialize YOLOv11Detector.

        Args:
            config: DetectionConfig with model size, thresholds, etc.
        """
        self.config = config
        self.device = get_device(config.device)
        self._model = None

    def _load(self) -> None:
        """Lazy-load YOLO model."""
        if self._model is not None:
            return

        from ultralytics import YOLO  # type: ignore

        model_name = (
            self.config.model_path
            if self.config.model_path
            else self.config.model_size.value  # e.g., "yolo11l.pt"
        )
        log.info(f"Loading YOLOv11: {model_name}")
        self._model = YOLO(model_name)
        log.info(f"YOLOv11 loaded | device={self.device}")

    def detect(self, image_path: str) -> list[DetectionResult]:
        """
        Detect objects in a single image.

        Args:
            image_path: Absolute path to image file.

        Returns:
            List of DetectionResult with normalized bbox [0,1].
        """
        self._load()

        try:
            results = self._model.predict(
                source=image_path,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                device=self.device,
                max_det=self.config.max_detections,
                classes=self.config.classes,
                verbose=False,
            )
        except Exception as exc:
            log.error(f"YOLO detection failed on {image_path}: {exc}")
            return []

        return self._parse_results(results[0])

    def detect_batch(
        self, image_paths: list[str]
    ) -> list[list[DetectionResult]]:
        """
        Detect objects in a batch of images.

        Args:
            image_paths: List of image paths.

        Returns:
            One list of DetectionResult per image.
        """
        self._load()

        try:
            results = self._model.predict(
                source=image_paths,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.imgsz,
                device=self.device,
                max_det=self.config.max_detections,
                classes=self.config.classes,
                verbose=False,
                batch=self.config.batch_size,
            )
        except Exception as exc:
            log.error(f"YOLO batch detection failed: {exc}")
            return [[] for _ in image_paths]

        return [self._parse_results(r) for r in results]

    def _parse_results(self, result) -> list[DetectionResult]:
        """
        Convert Ultralytics result to DetectionResult list.

        Bounding boxes are normalized to [0, 1] by dividing by
        image width/height for resolution-independent storage.
        """
        detections: list[DetectionResult] = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        img_h, img_w = result.orig_shape

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                DetectionResult(
                    class_id=cls_id,
                    class_name=result.names[cls_id],
                    confidence=conf,
                    # Normalize to [0, 1]
                    x1=x1 / img_w,
                    y1=y1 / img_h,
                    x2=x2 / img_w,
                    y2=y2 / img_h,
                )
            )

        return detections

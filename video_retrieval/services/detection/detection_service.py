"""
Object detection service.

Runs YOLOv11 on extracted frames and stores detection results in SQLite.
YOLO is NEVER run during online retrieval — only offline indexing.
"""

from __future__ import annotations

from typing import Optional

from tqdm import tqdm

from config.detection_config import DetectionConfig
from database.metadata.metadata_db import MetadataDatabase
from interfaces.base_interfaces import BaseDetector, DetectionResult, FrameRecord
from models.detection.yolo_model import YOLOv11Detector
from utils.logging_utils import get_logger

log = get_logger(__name__)


class DetectionService:
    """
    Runs YOLO object detection on frames and persists results.

    Design:
        - Processes frames in batches for GPU efficiency.
        - Normalizes bounding boxes to [0, 1].
        - Bulk-inserts detection records into SQLite.
        - Result is stored only once; never re-run at query time.

    Attributes:
        config:   DetectionConfig.
        detector: BaseDetector instance (YOLOv11 by default).
        meta_db:  MetadataDatabase for storing detections.
    """

    def __init__(
        self,
        config: DetectionConfig,
        meta_db: MetadataDatabase,
        detector: Optional[BaseDetector] = None,
    ) -> None:
        """
        Initialize DetectionService.

        Args:
            config:   DetectionConfig.
            meta_db:  MetadataDatabase.
            detector: Optional pre-built detector (DI / testing).
        """
        self.config = config
        self.meta_db = meta_db
        self.detector = detector or YOLOv11Detector(config)

    async def process_frames(
        self,
        frames: list[FrameRecord],
    ) -> None:
        """
        Run detection on all frames and persist results.

        Args:
            frames: List of FrameRecord with frame_path.
        """
        log.info(f"Running YOLO detection on {len(frames)} frames")
        batch_size = self.config.batch_size

        for batch_start in tqdm(
            range(0, len(frames), batch_size),
            desc="Object Detection",
            unit="batch",
        ):
            batch = frames[batch_start : batch_start + batch_size]
            image_paths = [f.frame_path for f in batch]

            try:
                batch_results: list[list[DetectionResult]] = (
                    self.detector.detect_batch(image_paths)
                )
            except Exception as exc:
                log.error(f"YOLO batch detection failed at index {batch_start}: {exc}")
                continue

            # Prepare bulk insert data
            detections_data: list[dict] = []
            for frame, results in zip(batch, batch_results):
                for det in results:
                    detections_data.append(
                        {
                            "frame_id": frame.frame_id,
                            "class_id": det.class_id,
                            "class_name": det.class_name,
                            "confidence": det.confidence,
                            "x1": det.x1,
                            "y1": det.y1,
                            "x2": det.x2,
                            "y2": det.y2,
                        }
                    )

            if detections_data:
                await self.meta_db.upsert_detections_bulk(detections_data)

        log.info("Object detection pipeline complete")

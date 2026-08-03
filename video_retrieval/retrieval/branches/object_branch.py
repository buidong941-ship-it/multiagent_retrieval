"""
Branch D: Object metadata filter retrieval.

Queries SQLite for frames containing the detected objects
extracted from the query. No vector search, pure SQL filter.

This is extremely fast (~1ms) and provides high precision
for queries with explicit object mentions.
"""

from __future__ import annotations

from config.retrieval_config import RetrievalConfig
from database.metadata.metadata_db import MetadataDatabase
from interfaces.base_interfaces import (
    BaseRetrievalBranch,
    ParsedQuery,
    RetrievalResult,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)

# COCO class name normalization (Vietnamese → English)
# Extend this dict for competition-specific classes
VIET_TO_COCO: dict[str, str] = {
    "người": "person",
    "người phụ nữ": "person",
    "người đàn ông": "person",
    "xe đạp": "bicycle",
    "xe máy": "motorcycle",
    "ô tô": "car",
    "xe tải": "truck",
    "chó": "dog",
    "mèo": "cat",
    "ghế": "chair",
    "bàn": "table",
    "điện thoại": "cell phone",
    "laptop": "laptop",
    "cốc": "cup",
    "chai": "bottle",
    "ô / dù": "umbrella",
    "ô": "umbrella",
    "túi xách": "handbag",
    "xe bus": "bus",
}


def normalize_to_coco(obj_name: str) -> str:
    """
    Normalize an object name to COCO class label.

    Args:
        obj_name: Object name from LLM (may be English or Vietnamese).

    Returns:
        COCO class name string.
    """
    cleaned = obj_name.strip().lower()
    return VIET_TO_COCO.get(cleaned, cleaned)


class ObjectDetectionBranch(BaseRetrievalBranch):
    """
    Retrieval Branch D: Object metadata filter.

    Process:
        1. Extract object class names from ParsedQuery.objects.
        2. Normalize to COCO class names.
        3. Query SQLite: find frames containing ALL these classes.
        4. Assign uniform score (confidence-based if available).

    Best for: precise object presence queries.

    Attributes:
        config:  RetrievalConfig.
        meta_db: MetadataDatabase with SQLite.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        meta_db: MetadataDatabase,
    ) -> None:
        self.config = config
        self.meta_db = meta_db

    @property
    def branch_name(self) -> str:
        return "object"

    async def retrieve(
        self,
        query: ParsedQuery,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Filter frames by object presence.

        Args:
            query:  ParsedQuery with objects list.
            top_k:  Maximum frames to return.

        Returns:
            List of RetrievalResult (score=1.0 for all matched frames).
        """
        if not query.objects:
            log.debug("Object branch: no objects in query — skipping")
            return []

        # Normalize object names to COCO
        coco_classes = [normalize_to_coco(obj) for obj in query.objects]
        log.debug(f"Object branch: searching for {coco_classes}")

        # Await the DB call directly since retrieve is now async
        frame_data = await self.meta_db.get_frames_by_objects_async(
            class_names=coco_classes,
            min_confidence=0.3,
        )

        if not frame_data:
            log.info("Object branch: no frames found")
            return []

        # Limit to top_k
        frame_data = frame_data[:top_k]

        results: list[RetrievalResult] = []
        for frame_id, confidence in frame_data:
            parts = frame_id.rsplit("_frame_", 1)
            video_id = parts[0] if len(parts) == 2 else ""
            frame_idx = int(parts[1]) if len(parts) == 2 else 0

            results.append(
                RetrievalResult(
                    frame_id=frame_id,
                    video_id=video_id,
                    frame_idx=frame_idx,
                    timestamp=0.0,
                    frame_path="",
                    score=confidence,  # Use YOLO confidence sum
                    source=self.branch_name,
                )
            )

        log.info(f"Object branch: {len(results)} frames with {coco_classes}")
        return results

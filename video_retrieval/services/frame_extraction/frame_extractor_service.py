"""
Frame extraction service.

Orchestrates video → frame extraction using configured strategy.
Stores frame records into the metadata database.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

from config.frame_config import ExtractionMode, FrameExtractionConfig
from database.metadata.metadata_db import MetadataDatabase
from interfaces.base_interfaces import FrameRecord
from models.shot_detection.transnetv2 import TransNetV2Detector
from utils.logging_utils import get_logger
from utils.video_utils import (
    build_frame_id,
    get_video_metadata,
    iter_frames_every_n,
    iter_frames_fixed_fps,
    iter_frames_by_indices,
    save_frame,
)

log = get_logger(__name__)


class FrameExtractionService:
    """
    Extracts frames from video files and stores metadata.

    Supports:
        - fixed_fps:   Extract at a target FPS rate
        - every_n:     Extract every N-th frame
        - transnetv2:  TransNetV2 deep learning shot boundary keyframe extraction
        - shot_detect: Shot boundary keyframe extraction

    Attributes:
        config:  FrameExtractionConfig.
        db:      MetadataDatabase instance for storing frame records.
    """

    def __init__(
        self,
        config: FrameExtractionConfig,
        db: MetadataDatabase,
    ) -> None:
        self.config = config
        self.db = db
        self.transnet_detector = TransNetV2Detector(
            threshold=getattr(config, "transnet_threshold", 0.5),
            keyframes_per_shot=getattr(config, "keyframes_per_shot", 1),
        )

    async def extract_video(self, video_path: str) -> list[FrameRecord]:
        """
        Extract frames from a single video file.

        Args:
            video_path: Absolute path to video file.

        Returns:
            List of FrameRecord objects for extracted frames.

        Raises:
            FileNotFoundError: If video does not exist.
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_id = video_path_obj.stem
        log.info(f"Extracting frames from: {video_path} | mode={self.config.mode}")

        # Get video metadata
        meta = get_video_metadata(video_path)
        await self.db.upsert_video(
            {
                "video_id": video_id,
                "video_path": str(video_path_obj.resolve()),
                "duration": meta["duration"],
                "fps": meta["fps"],
                "width": meta["width"],
                "height": meta["height"],
                "total_frames": meta["total_frames"],
            }
        )

        # Select extraction iterator
        output_dir = Path(self.config.output_dir) / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.mode == ExtractionMode.FIXED_FPS:
            frame_iter = iter_frames_fixed_fps(video_path, self.config.fps)
        elif self.config.mode == ExtractionMode.EVERY_N:
            frame_iter = iter_frames_every_n(video_path, self.config.every_n_frames)
        elif self.config.mode in (ExtractionMode.TRANSNETV2, ExtractionMode.SHOT_DETECT):
            log.info(f"Running TransNetV2 shot boundary detection on '{video_id}'...")
            preds = self.transnet_detector.predict_video(video_path)
            scenes = self.transnet_detector.predictions_to_scenes(preds)
            kf_indices = self.transnet_detector.select_keyframes(scenes)
            log.info(f"TransNetV2 detected {len(scenes)} scenes -> selected {len(kf_indices)} keyframes.")
            frame_iter = iter_frames_by_indices(video_path, kf_indices)
        else:
            raise NotImplementedError(
                f"Extraction mode '{self.config.mode}' not yet implemented"
            )

        records: list[FrameRecord] = []
        frames_data_batch: list[dict] = []
        BATCH_SIZE = 500

        for frame_idx, timestamp, frame_bgr in frame_iter:
            frame_id = build_frame_id(video_id, frame_idx)
            frame_filename = f"{frame_id}.{self.config.image_format}"
            frame_path = str(output_dir / frame_filename)

            # Save frame to disk
            saved_path = save_frame(
                frame=frame_bgr,
                output_path=frame_path,
                quality=self.config.image_quality,
                max_dimension=self.config.max_dimension,
            )

            record = FrameRecord(
                video_id=video_id,
                frame_id=frame_id,
                frame_idx=frame_idx,
                timestamp=timestamp,
                frame_path=saved_path,
            )
            records.append(record)

            frames_data_batch.append(
                {
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "frame_path": saved_path,
                }
            )

            # Bulk insert in batches to avoid memory overflow
            if len(frames_data_batch) >= BATCH_SIZE:
                await self.db.upsert_frames_bulk(frames_data_batch)
                frames_data_batch.clear()

        # Insert remaining frames
        if frames_data_batch:
            await self.db.upsert_frames_bulk(frames_data_batch)

        log.info(
            f"Extracted {len(records)} frames from '{video_id}' "
            f"| mode={self.config.mode} | saved to {output_dir}"
        )
        return records

    async def extract_directory(self, video_dir: str) -> dict[str, list[FrameRecord]]:
        """
        Extract frames from all videos in a directory.

        Args:
            video_dir: Directory containing video files.

        Returns:
            Dict mapping video_id → list of FrameRecord.
        """
        video_dir_path = Path(video_dir)
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        video_files = [
            f for f in video_dir_path.rglob("*")
            if f.suffix.lower() in video_extensions
        ]

        log.info(f"Found {len(video_files)} videos in {video_dir}")

        results: dict[str, list[FrameRecord]] = {}
        for video_file in sorted(video_files):
            try:
                records = await self.extract_video(str(video_file))
                results[video_file.stem] = records
            except Exception as exc:
                log.error(f"Failed to process video {video_file}: {exc}")

        return results

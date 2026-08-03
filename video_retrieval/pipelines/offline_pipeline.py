"""
Offline Indexing Pipeline.

Orchestrates all offline processing steps:
    1. Frame extraction
    2. Image embedding (SigLIP2 → Milvus)
    3. OCR (PaddleOCR → BM25 + BGE-M3 → Milvus)
    4. Object detection (YOLOv11 → SQLite)

Design:
    - Each step is independent and can be run separately.
    - Checkpointing: if frames already extracted, skip extraction.
    - Configurable via settings (each step can be enabled/disabled).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from config.detection_config import DetectionConfig
from config.embedding_config import EmbeddingConfig
from config.frame_config import FrameExtractionConfig
from config.ocr_config import OCRConfig
from config.settings import Settings
from database.bm25.bm25_index import BM25OcrIndex
from database.faiss.faiss_client import FaissVectorDatabase
from database.metadata.metadata_db import MetadataDatabase
from interfaces.base_interfaces import BaseVectorDatabase, FrameRecord
from services.detection.detection_service import DetectionService

from services.embedding.image_embedding_service import ImageEmbeddingService
from services.frame_extraction.frame_extractor_service import FrameExtractionService
from services.ocr.ocr_service import OCRService
from utils.logging_utils import get_logger

log = get_logger(__name__)


class OfflinePipeline:
    """
    Complete offline video indexing pipeline.

    Attributes:
        settings:       Global Settings object.
        meta_db:        SQLite metadata database.
        vector_db:      BaseVectorDatabase vector database.
        frame_svc:      Frame extraction service.
        embed_svc:      Image embedding service.
        ocr_svc:        OCR service.
        detection_svc:  Object detection service.
    """

    def __init__(
        self,
        settings: Settings,
        meta_db: Optional[MetadataDatabase] = None,
        vector_db: Optional[BaseVectorDatabase] = None,
    ) -> None:
        """
        Initialize OfflinePipeline with optional dependency injection.

        Args:
            settings:  Global settings.
            meta_db:   Pre-built metadata DB (for testing).
            vector_db: Pre-built vector DB (for testing).
        """
        self.settings = settings

        # Build dependencies
        self.meta_db = meta_db or MetadataDatabase()
        self.vector_db = vector_db or FaissVectorDatabase(settings.faiss)

        # Build services
        self.frame_svc = FrameExtractionService(settings.frame, self.meta_db)
        # Build embedding services
        self.embed_svcs = []
        for backend in settings.embedding.active_backends:
            if backend.value == "siglip2":
                collection_name = "clip_embeddings"
                db_id_field = "milvus_clip_id"
            else:
                collection_name = f"{backend.value}_embeddings"
                db_id_field = f"milvus_{backend.value}_id"

            svc = ImageEmbeddingService(
                settings.embedding, self.vector_db, self.meta_db,
                backend=backend, collection_name=collection_name, db_id_field=db_id_field
            )
            self.embed_svcs.append(svc)

        bm25 = BM25OcrIndex(settings.ocr)
        self.ocr_svc = OCRService(
            settings.ocr, self.vector_db, self.meta_db, bm25_index=bm25
        )
        self.detection_svc = DetectionService(settings.detection, self.meta_db)

    async def initialize(self) -> None:
        """Initialize database tables and Milvus collections."""
        log.info("Initializing databases...")
        await self.meta_db.init_db()
        for svc in self.embed_svcs:
            self.vector_db.create_collection_if_not_exists(svc._collection, dim=svc.embedder.embedding_dim)
        self.vector_db.create_collection_if_not_exists("ocr_embeddings", dim=self.settings.ocr.bge_embedding_dim)

        log.info("Databases initialized")

    async def run(
        self,
        video_dir: str,
        run_extraction: bool = True,
        run_embedding: bool = True,
        run_ocr: bool = True,
        run_detection: bool = True,

    ) -> None:
        """
        Run the complete offline indexing pipeline.

        Args:
            video_dir:      Directory containing video files.
            run_extraction: Enable frame extraction step.
            run_embedding:  Enable image embedding step.
            run_ocr:        Enable OCR step.
            run_detection:  Enable object detection step.
        """
        await self.initialize()

        # Step 1: Frame Extraction
        all_frames: dict[str, list[FrameRecord]] = {}
        if run_extraction:
            log.info("=" * 60)
            log.info("STEP 1: Frame Extraction")
            log.info("=" * 60)
            all_frames = await self.frame_svc.extract_directory(video_dir)
            total = sum(len(f) for f in all_frames.values())
            log.info(f"Extracted {total} frames from {len(all_frames)} videos")

        # Flatten all frames into a single list
        flat_frames: list[FrameRecord] = [
            f for frames in all_frames.values() for f in frames
        ]

        # If extraction was skipped, load from DB
        if not flat_frames and (run_embedding or run_ocr or run_detection):
            log.warning("No frames extracted — loading from metadata DB")
            db_frames = await self.meta_db.get_all_frames_async()
            flat_frames = [
                FrameRecord(
                    frame_id=f["frame_id"],
                    video_id=f["video_id"],
                    frame_idx=f["frame_idx"],
                    timestamp=f["timestamp"],
                    frame_path=f["frame_path"],
                )
                for f in db_frames
            ]
            log.info(f"Loaded {len(flat_frames)} frames from database")

        # Step 2: Image Embedding
        if run_embedding and flat_frames:
            log.info("=" * 60)
            log.info("STEP 2: Image Embedding (Dual-Embedding Ensemble)")
            log.info("=" * 60)
            kept_frames = flat_frames
            for i, svc in enumerate(self.embed_svcs):
                try:
                    offset = self.vector_db.count(svc._collection)
                except Exception:
                    offset = 0
                log.info(f"{svc._collection} id_offset = {offset} (resuming from existing vectors)")
                res = await svc.embed_frames(kept_frames, id_offset=offset, perform_dedup=(i == 0))
                if res:
                    kept_frames = res
                else:
                    log.warning(f"{svc._collection} returned no frames. Preserving existing {len(kept_frames)} frames for next steps.")
            flat_frames = kept_frames

        # Step 3: OCR
        if run_ocr and flat_frames:
            log.info("=" * 60)
            log.info("STEP 3: OCR (PaddleOCR + BGE-M3 + BM25)")
            log.info("=" * 60)
            try:
                ocr_offset = self.vector_db.count("ocr_embeddings")
            except Exception:
                ocr_offset = 0
            log.info(f"OCR id_offset = {ocr_offset} (resuming from existing vectors)")
            await self.ocr_svc.process_frames(flat_frames, id_offset=ocr_offset)

        if run_detection and flat_frames:
            log.info("=" * 60)
            log.info("STEP 4: Object Detection (YOLOv11 → SQLite)")
            log.info("=" * 60)
            await self.detection_svc.process_frames(flat_frames)



        log.info("=" * 60)
        log.info("Offline indexing pipeline COMPLETE")
        log.info(f"Total frames indexed: {len(flat_frames)}")
        log.info("=" * 60)

    async def run_single_video(self, video_path: str) -> None:
        """
        Index a single video file incrementally.

        Useful for adding new videos to an existing index.

        Args:
            video_path: Path to video file.
        """
        await self.initialize()

        log.info(f"Indexing single video: {video_path}")

        # Get current vector DB counts for ID offsets (incremental indexing)
        try:
            ocr_offset = self.vector_db.count("ocr_embeddings")
        except Exception:
            ocr_offset = 0
        log.info(f"ID offsets — ocr: {ocr_offset}")

        frames = await self.frame_svc.extract_video(video_path)
        for i, svc in enumerate(self.embed_svcs):
            try:
                offset = self.vector_db.count(svc._collection)
            except Exception:
                offset = 0
            frames = await svc.embed_frames(frames, id_offset=offset, perform_dedup=(i == 0))
        await self.ocr_svc.process_frames(frames, id_offset=ocr_offset)
        await self.detection_svc.process_frames(frames)


        log.info(f"Indexed {len(frames)} frames from {video_path}")

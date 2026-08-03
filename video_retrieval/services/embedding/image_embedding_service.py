"""
Image embedding service.

Encodes extracted frames using SigLIP2 (or OpenCLIP fallback)
and inserts embeddings into Milvus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from config.embedding_config import EmbeddingBackend, EmbeddingConfig
from database.metadata.metadata_db import MetadataDatabase
from interfaces.base_interfaces import BaseEmbedder, BaseVectorDatabase, FrameRecord
from models.embedding.siglip_model import SigLIP2Embedder
from utils.logging_utils import get_logger

log = get_logger(__name__)


def build_embedder(config: EmbeddingConfig, backend: EmbeddingBackend) -> BaseEmbedder:
    """
    Factory function: return the configured embedder.

    Args:
        config: EmbeddingConfig.
        backend: The specific embedding backend to build.

    Returns:
        BaseEmbedder instance.
    """
    if backend == EmbeddingBackend.SIGLIP2:
        return SigLIP2Embedder(config)
    elif backend == EmbeddingBackend.OPENCLIP:
        from models.embedding.openclip_model import OpenCLIPEmbedder  # optional import
        return OpenCLIPEmbedder(config)
    elif backend == EmbeddingBackend.JINA:
        from models.embedding.jina_model import JinaClipEmbedder
        return JinaClipEmbedder(config)
    elif backend == EmbeddingBackend.BEIT3:
        from models.embedding.beit3_model import Beit3Embedder
        return Beit3Embedder(config)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")


class ImageEmbeddingService:
    """
    Encodes video frames into vector embeddings and stores in vector database.

    Design:
        - Processes frames in configurable batches.
        - Assigns int64 IDs based on a running counter.
        - Updates frame records in metadata DB with milvus_clip_id.

    Attributes:
        config:    EmbeddingConfig.
        embedder:  BaseEmbedder instance (SigLIP2 by default).
        vector_db: BaseVectorDatabase instance.
        meta_db:   MetadataDatabase for updating frame records.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        vector_db: BaseVectorDatabase,
        meta_db: MetadataDatabase,
        backend: EmbeddingBackend = EmbeddingBackend.SIGLIP2,
        collection_name: str = "clip_embeddings",
        db_id_field: str = "milvus_clip_id",
        embedder: Optional[BaseEmbedder] = None,
    ) -> None:
        """
        Initialize ImageEmbeddingService.

        Args:
            config:    EmbeddingConfig.
            vector_db: MilvusVectorDatabase for storing vectors.
            meta_db:   MetadataDatabase for updating milvus_clip_id.
            backend:   The backend to use if embedder is not provided.
            collection_name: Name of the vector DB collection.
            db_id_field: Name of the field in SQLite to store the vector ID.
            embedder:  Optional pre-built embedder (for dependency injection).
        """
        self.config = config
        self.vector_db = vector_db
        self.meta_db = meta_db
        self.embedder = embedder or build_embedder(config, backend)
        self._collection = collection_name
        self._db_id_field = db_id_field
        
        # Track last kept embedding per video for frame filtering
        self._last_kept_embeddings: dict[str, np.ndarray] = {}

    def setup_collection(self) -> None:
        """Ensure the Milvus clip collection exists with HNSW index."""
        self.vector_db.create_collection_if_not_exists(self._collection, dim=self.embedder.embedding_dim)

    async def embed_frames(
        self,
        frames: list[FrameRecord],
        id_offset: int = 0,
        perform_dedup: bool = True,
    ) -> list[FrameRecord]:
        """
        Run embedding extraction and insert into Milvus.
        Returns the list of frames that were actually inserted (deduplicated).

        Args:
            frames: List of FrameRecord objects.
            id_offset: Starting integer ID for Milvus (avoids collisions
                       across multiple videos).
            perform_dedup: If True, performs frame filtering based on cosine similarity.
                           Typically only the first embedding pass should dedup to avoid
                           deleting frames already indexed by prior passes.
        """
        self.setup_collection()
        batch_size = self.config.batch_size
        total = len(frames)
        all_kept_frames = []
        kept_count = 0
        skipped_count = 0
        milvus_counter = id_offset  # running counter to avoid ID collisions

        log.info(f"Embedding {total} frames | batch_size={batch_size}")

        for batch_start in tqdm(
            range(0, total, batch_size),
            desc="Embedding frames",
            unit="batch",
        ):
            batch = frames[batch_start : batch_start + batch_size]
            image_paths = [f.frame_path for f in batch]

            # Generate embeddings
            try:
                embeddings = self.embedder.encode_images(image_paths)
            except Exception as exc:
                log.error(f"Embedding failed for batch at index {batch_start}: {exc}")
                continue

            filtered_milvus_ids = []
            filtered_metadata = []
            filtered_embeddings = []
            filtered_frames = []

            for i, emb in enumerate(embeddings):
                frame = batch[i]
                vid = frame.video_id
                
                # Frame filtering: check similarity with last kept embedding for this video if dedup is enabled
                if perform_dedup and self.config.enable_dedup and vid in self._last_kept_embeddings:
                    # Both embeddings are L2-normalized, so dot product == cosine similarity
                    sim = np.dot(emb, self._last_kept_embeddings[vid])
                    if sim > self.config.dedup_threshold:
                        # Skip this frame as it is too similar
                        Path(frame.frame_path).unlink(missing_ok=True)
                        await self.meta_db.delete_frame_async(frame.frame_id)
                        skipped_count += 1
                        continue
                
                # Keep this frame
                self._last_kept_embeddings[vid] = emb
                filtered_embeddings.append(emb)
                filtered_frames.append(frame)
                filtered_milvus_ids.append(milvus_counter)
                milvus_counter += 1
                kept_count += 1
                filtered_metadata.append(
                    {
                        "frame_id": frame.frame_id,
                        "video_id": frame.video_id,
                        "frame_idx": frame.frame_idx,
                        "timestamp": frame.timestamp,
                    }
                )

            if not filtered_embeddings:
                continue

            # Insert into Milvus
            self.vector_db.insert(
                collection_name=self._collection,
                ids=filtered_milvus_ids,
                embeddings=np.vstack(filtered_embeddings),
                metadata=filtered_metadata,
            )

            # Update database id in SQLite
            for frame, milvus_id in zip(filtered_frames, filtered_milvus_ids):
                await self.meta_db.upsert_frame(
                    {"frame_id": frame.frame_id, self._db_id_field: milvus_id}
                )
            all_kept_frames.extend(filtered_frames)

        # Flush collection after all inserts
        self.vector_db.flush(self._collection)
        log.info(f"Embedding complete | kept={kept_count}, skipped(similar)={skipped_count}, total_input={total}")
        return all_kept_frames

    def encode_query(self, query_text: str) -> np.ndarray:
        """
        Encode a single text query into a CLIP embedding.

        Args:
            query_text: Vietnamese or English text query.

        Returns:
            np.ndarray shape (1152,), float32, L2-normalized.
        """
        embeddings = self.embedder.encode_texts([query_text])
        return embeddings[0]

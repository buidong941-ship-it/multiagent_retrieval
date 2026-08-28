"""Resume or repair only the Jina image-embedding collection on Kaggle.

It never extracts frames and never runs CLIP, BEiT3, OCR, or YOLO.  Existing
finite Jina vectors are retained; missing or NaN/Inf vectors are regenerated.
"""

import asyncio
import sys
from pathlib import Path

import numpy as np

# Allow ``python scripts/kaggle_resume_jina.py`` from video_retrieval.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.embedding_config import EmbeddingBackend
from config.settings import get_settings
from database.faiss.faiss_client import FaissVectorDatabase
from database.metadata.metadata_db import MetadataDatabase
from interfaces.base_interfaces import FrameRecord
from services.embedding.image_embedding_service import ImageEmbeddingService


COLLECTION = "jina_embeddings"


def find_invalid_vector_ids(vector_db: FaissVectorDatabase) -> list[int]:
    """Return IDs whose stored FAISS vector contains NaN or Inf."""
    index = vector_db.indices.get(COLLECTION)
    if index is None:
        return []

    invalid_ids: list[int] = []
    for vector_id in vector_db.metadata_store.get(COLLECTION, {}):
        try:
            vector = index.reconstruct(int(vector_id))
        except Exception:
            invalid_ids.append(int(vector_id))
            continue
        if not np.isfinite(vector).all():
            invalid_ids.append(int(vector_id))
    return invalid_ids


async def resume_jina_embeddings() -> None:
    settings = get_settings()
    settings.embedding.active_backends = [EmbeddingBackend.JINA]
    # Resume must not delete frame files/metadata based on visual similarity.
    settings.embedding.enable_dedup = False

    meta_db = MetadataDatabase()
    # Avoid loading CLIP/BEiT3/OCR FAISS indexes into RAM for this Jina-only task.
    vector_db = FaissVectorDatabase(settings.faiss, auto_load=False)
    jina_svc = ImageEmbeddingService(
        settings.embedding,
        vector_db,
        meta_db,
        backend=EmbeddingBackend.JINA,
        collection_name=COLLECTION,
        # Schema intentionally has no ``milvus_jina_id`` column.
        db_id_field=None,
    )

    print("=" * 64)
    print("RESUME / REPAIR: JINA EMBEDDINGS ONLY")
    print("=" * 64)

    await meta_db.init_db()
    jina_svc.setup_collection()

    invalid_ids = find_invalid_vector_ids(vector_db)
    if invalid_ids:
        print(f"Removing {len(invalid_ids):,} invalid Jina vector(s) (NaN/Inf)...")
        vector_db.delete(COLLECTION, invalid_ids)

    existing_frame_ids = set(vector_db.frame_id_to_id.get(COLLECTION, {}))
    db_frames = await meta_db.get_all_frames_async()
    missing_frames = [
        FrameRecord(
            frame_id=frame["frame_id"],
            video_id=frame["video_id"],
            frame_idx=frame["frame_idx"],
            timestamp=frame["timestamp"],
            frame_path=frame["frame_path"],
        )
        for frame in db_frames
        if frame["frame_id"] not in existing_frame_ids
    ]

    existing_ids = vector_db.metadata_store.get(COLLECTION, {})
    next_id = max((int(vector_id) for vector_id in existing_ids), default=-1) + 1

    print(f"Frames trong metadata.db : {len(db_frames):,}")
    print(f"Jina vectors hợp lệ       : {len(existing_frame_ids):,}")
    print(f"Frames cần Jina embed lại : {len(missing_frames):,}")
    print(f"Numeric ID bắt đầu mới    : {next_id:,}")

    if missing_frames:
        await jina_svc.embed_frames(
            missing_frames,
            id_offset=next_id,
            perform_dedup=False,
        )
    else:
        # Persist deletions when there was nothing new to embed.
        vector_db.flush(COLLECTION)

    remaining_invalid = find_invalid_vector_ids(vector_db)
    if remaining_invalid:
        # Never leave NaN vectors in the searchable index.  The next rerun can
        # retry these frames after the underlying Jina inference issue is fixed.
        vector_db.delete(COLLECTION, remaining_invalid)
        vector_db.flush(COLLECTION)
        raise RuntimeError(
            f"Jina produced {len(remaining_invalid):,} NaN/Inf vector(s). "
            "They were removed; fix the Jina inference configuration before retrying."
        )

    final_frame_ids = set(vector_db.frame_id_to_id.get(COLLECTION, {}))
    still_missing_ids = {frame["frame_id"] for frame in db_frames} - final_frame_ids
    if still_missing_ids:
        raise RuntimeError(
            f"Jina resume finished but {len(still_missing_ids):,} frame(s) still have no valid Jina vector. "
            "Check the logged image/model errors and rerun after fixing them."
        )

    # Do not call vector_db.close(): it flushes every auto-loaded collection
    # and may rewrite unrelated BEiT3/CLIP PKLs.
    print("JINA RESUME / REPAIR HOÀN TẤT")


if __name__ == "__main__":
    asyncio.run(resume_jina_embeddings())

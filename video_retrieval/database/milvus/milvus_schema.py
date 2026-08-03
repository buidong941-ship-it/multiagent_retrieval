"""
Milvus collection schemas.

Design Decision:
    - Use Milvus auto_id=False so we control the int64 primary key.
      This lets us map frame_id → milvus_id deterministically.
    - VARCHAR(512) for frame_id stored as scalar field to enable
      hybrid filtering (e.g., filter by video_id prefix).
    - Dynamic fields=True to allow future metadata fields without
      schema migration.
"""

from __future__ import annotations

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from config.database_config import MilvusConfig


def build_clip_schema() -> CollectionSchema:
    """
    Schema for the CLIP (SigLIP2) image embeddings collection.

    Fields:
        milvus_id   — int64 PK (auto_id=False)
        frame_id    — VARCHAR scalar for hybrid filter
        video_id    — VARCHAR scalar for video-level filter
        embedding   — FLOAT_VECTOR (dim=1152 for SigLIP2-so400m)
    """
    fields = [
        FieldSchema(
            name="milvus_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
            description="Unique frame integer ID",
        ),
        FieldSchema(
            name="frame_id",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Globally unique frame string ID",
        ),
        FieldSchema(
            name="video_id",
            dtype=DataType.VARCHAR,
            max_length=255,
            description="Video identifier for filtering",
        ),
        FieldSchema(
            name="frame_idx",
            dtype=DataType.INT64,
            description="0-based frame index within video",
        ),
        FieldSchema(
            name="timestamp",
            dtype=DataType.FLOAT,
            description="Frame timestamp in seconds",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=1152,
            description="SigLIP2-so400m image embedding (L2-normalized)",
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="SigLIP2 image embeddings for video frames",
        enable_dynamic_field=True,
    )


def build_ocr_schema() -> CollectionSchema:
    """
    Schema for OCR text embeddings (BGE-M3).

    Fields:
        milvus_id   — int64 PK
        frame_id    — frame string ID
        video_id    — video string ID
        ocr_text    — raw concatenated OCR text (for display)
        embedding   — FLOAT_VECTOR (dim=1024 for BGE-M3 dense)
    """
    fields = [
        FieldSchema(
            name="milvus_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name="frame_id",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="video_id",
            dtype=DataType.VARCHAR,
            max_length=255,
        ),
        FieldSchema(
            name="ocr_text",
            dtype=DataType.VARCHAR,
            max_length=4096,
            description="Concatenated OCR text for this frame",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024,
            description="BGE-M3 dense embedding of OCR text",
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="BGE-M3 OCR text embeddings for video frames",
        enable_dynamic_field=True,
    )


def build_action_schema() -> CollectionSchema:
    """
    Schema for the Action Retrieval (mean-pooled CLIP embeddings) collection.
    """
    fields = [
        FieldSchema(
            name="milvus_id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
            description="Unique sequence integer ID",
        ),
        FieldSchema(
            name="frame_id",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Globally unique string ID of the center frame",
        ),
        FieldSchema(
            name="video_id",
            dtype=DataType.VARCHAR,
            max_length=255,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=1152,  # SigLIP2 dim
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="Mean-pooled CLIP embeddings for action retrieval",
        enable_dynamic_field=True,
    )


COLLECTION_SCHEMAS: dict[str, CollectionSchema] = {
    "clip_embeddings": build_clip_schema(),
    "ocr_embeddings": build_ocr_schema(),
    "action_embeddings": build_action_schema(),
}

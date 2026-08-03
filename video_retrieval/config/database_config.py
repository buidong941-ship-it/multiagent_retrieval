"""
Milvus vector database configuration.

Design Decision:
    - Two separate Milvus collections:
        1. clip_embeddings: stores SigLIP2 image embeddings
        2. ocr_embeddings:  stores BGE-M3 OCR text embeddings
    - Using HNSW index (not IVF_FLAT) because:
        * No need to train (IVF_FLAT requires cluster training)
        * Better recall at high top-k values
        * Faster search at comparable recall vs IVF_PQ
    - Cosine metric because embeddings are L2-normalized.
"""

from __future__ import annotations

from pydantic import Field

from config.base_config import BaseConfig, BASE_DIR


class MilvusConfig(BaseConfig):
    """Configuration for Milvus vector database."""

    # Connection
    uri: str = Field(
        default=(BASE_DIR / "data" / "indexes" / "milvus.db").as_posix(), 
        description="Milvus connection URI (e.g. ./data/indexes/milvus.db for Lite, or http://localhost:19530 for Docker)"
    )
    user: str = Field(default="", description="Milvus username (if auth enabled)")
    password: str = Field(default="", description="Milvus password (if auth enabled)")
    db_name: str = Field(default="default", description="Milvus database name")
    timeout: float = Field(default=30.0, description="Connection timeout in seconds")

    # Collection names
    clip_collection_name: str = Field(
        default="clip_embeddings",
        description="Collection for SigLIP2 image embeddings",
    )
    ocr_collection_name: str = Field(
        default="ocr_embeddings",
        description="Collection for BGE-M3 OCR text embeddings",
    )
    object_collection_name: str = Field(
        default="object_embeddings",
        description="Collection for object/action metadata embeddings",
    )
    action_collection_name: str = Field(
        default="action_embeddings",
        description="Collection for mean-pooled action embeddings",
    )

    # HNSW index parameters
    # M: number of bi-directional links per node (higher = better recall, more memory)
    # ef_construction: search depth during index build (higher = better recall, slower build)
    hnsw_m: int = Field(default=16, ge=4, le=64, description="HNSW M parameter")
    hnsw_ef_construction: int = Field(
        default=200, ge=8, description="HNSW ef_construction"
    )
    hnsw_ef_search: int = Field(
        default=100, ge=8, description="HNSW ef parameter at search time"
    )

    # Search parameters
    metric_type: str = Field(
        default="COSINE",
        description="Distance metric: COSINE, L2, or IP",
    )
    nprobe: int = Field(
        default=16,
        description="IVF nprobe (fallback if using IVF index)",
    )

    # Embedding dimensions
    clip_dim: int = Field(default=1152, description="SigLIP2-so400m embedding dim")
    ocr_dim: int = Field(default=1024, description="BGE-M3 dense embedding dim")

    model_config = {"env_prefix": "MILVUS_"}


class FaissConfig(BaseConfig):
    """Configuration for FAISS vector database."""

    index_dir: str = Field(
        default=(BASE_DIR / "data" / "indexes" / "faiss").as_posix(),
        description="Directory to store FAISS index files and metadata mappings",
    )

    # Collection names
    clip_collection_name: str = Field(
        default="clip_embeddings",
        description="Collection name for SigLIP2 image embeddings",
    )
    ocr_collection_name: str = Field(
        default="ocr_embeddings",
        description="Collection name for BGE-M3 OCR text embeddings",
    )
    object_collection_name: str = Field(
        default="object_embeddings",
        description="Collection name for object/action metadata embeddings",
    )
    action_collection_name: str = Field(
        default="action_embeddings",
        description="Collection name for action embeddings",
    )

    # Index parameters
    metric_type: str = Field(
        default="IP",
        description="Metric type: IP (Inner Product / Cosine Similarity for normalized vectors) or L2",
    )
    use_gpu: bool = Field(
        default=False,
        description="Whether to use GPU acceleration for FAISS if available",
    )

    # Embedding dimensions
    clip_dim: int = Field(default=1152, description="SigLIP2-so400m embedding dim")
    ocr_dim: int = Field(default=1024, description="BGE-M3 dense embedding dim")

    model_config = {"env_prefix": "FAISS_"}


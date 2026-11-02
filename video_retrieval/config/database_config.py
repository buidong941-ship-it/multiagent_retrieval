"""
FAISS vector database configuration.

Design Decision:
    - Flat Inner Product index on disk (embeddings are L2-normalized → IP = cosine).
    - Optional in-memory IVFFlat conversion for large collections (see FaissConfig.use_ivf).
"""

from __future__ import annotations

from pydantic import Field

from config.base_config import BaseConfig, BASE_DIR


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

    # IVF (Approximate Search) parameters
    use_ivf: bool = Field(
        default=True,
        description="If True, convert FlatIP index to IVFFlat in-memory for faster approximate search",
    )
    ivf_nlist: int = Field(
        default=256,
        description="Number of Voronoi clusters for IVF index. Rule of thumb: sqrt(ntotal). "
                    "Higher = faster search but less recall. Typical: 64-4096.",
    )
    ivf_nprobe: int = Field(
        default=32,
        description="Number of clusters to visit at query time. Higher = better recall, slower. "
                    "Typical: nlist/8 to nlist/4.",
    )
    ivf_min_vectors: int = Field(
        default=10000,
        description="Minimum number of vectors to trigger IVF conversion. "
                    "Below this threshold, FlatIP is used (faster for small datasets).",
    )

    # Embedding dimensions
    clip_dim: int = Field(default=1152, description="SigLIP2-so400m embedding dim")
    ocr_dim: int = Field(default=1024, description="BGE-M3 dense embedding dim")

    model_config = {"env_prefix": "FAISS_"}


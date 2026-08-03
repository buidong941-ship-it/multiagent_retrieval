"""
Embedding model configuration.

Design Decision:
    - SigLIP2 is the default because it supports multilingual text
      natively and has better zero-shot performance than CLIP on
      fine-grained tasks (Google Research, 2024).
    - OpenCLIP is the fallback: more community fine-tunes available.
    - Both produce L2-normalized embeddings → cosine similarity
      is just a dot product after normalization.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field

from config.base_config import BaseConfig


class EmbeddingBackend(str, Enum):
    SIGLIP2 = "siglip2"
    OPENCLIP = "openclip"
    JINA = "jina"
    BEIT3 = "beit3"


class EmbeddingConfig(BaseConfig):
    """Configuration for image/text embedding models."""

    active_backends: list[EmbeddingBackend] = Field(
        default=[EmbeddingBackend.BEIT3, EmbeddingBackend.JINA, EmbeddingBackend.SIGLIP2],
        description="List of active embedding backends to run",
    )

    # SigLIP2 settings
    siglip_model_name: str = Field(
        default="google/siglip2-so400m-patch14-384",
        description="HuggingFace model ID for SigLIP2",
    )

    # BEiT-3 settings
    beit3_model_name: str = Field(
        default="microsoft/beit-base-patch16-224",
        description="HuggingFace model ID for BEiT",
    )

    # Jina CLIP v2 settings
    jina_model_name: str = Field(
        default="jinaai/jina-clip-v2",
        description="HuggingFace model ID for Jina CLIP v2",
    )

    # OpenCLIP settings
    openclip_model_name: str = Field(
        default="ViT-L-14",
        description="OpenCLIP model architecture",
    )
    openclip_pretrained: str = Field(
        default="datacomp_xl_s13b_b90k",
        description="OpenCLIP pretrained weights tag",
    )

    # Runtime
    device: str = Field(
        default="cuda",
        description="torch device: 'cuda', 'cpu', or 'cuda:0'",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for embedding inference",
    )
    embedding_dim: int = Field(
        default=1152,
        description="Output embedding dimension (SigLIP2-so400m = 1152)",
    )
    normalize: bool = Field(
        default=True,
        description="L2-normalize embeddings before storage",
    )
    use_fp16: bool = Field(
        default=True,
        description="Use float16 for faster inference on GPU",
    )
    enable_dedup: bool = Field(
        default=True,
        description="Enable cosine similarity frame deduplication",
    )
    dedup_threshold: float = Field(
        default=0.90,
        description="Cosine similarity threshold for deduplication",
    )

    # Cache
    cache_dir: Optional[str] = Field(
        default=None,
        description="HuggingFace cache directory override",
    )

    model_config = {"env_prefix": "EMBED_"}

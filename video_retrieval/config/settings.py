"""
Master settings — aggregates all module configs into one object.

Usage:
    from config.settings import settings

    settings.embedding.batch_size
    settings.milvus.host
    settings.retrieval.clip_top_k
"""

from __future__ import annotations

from functools import lru_cache

from config.database_config import MilvusConfig, FaissConfig
from config.detection_config import DetectionConfig
from config.embedding_config import EmbeddingConfig
from config.frame_config import FrameExtractionConfig
from config.ocr_config import OCRConfig
from config.retrieval_config import RetrievalConfig


class Settings:
    """Singleton container for all module configurations."""

    def __init__(self) -> None:
        self.frame = FrameExtractionConfig()
        self.embedding = EmbeddingConfig()
        self.ocr = OCRConfig()
        self.detection = DetectionConfig()
        self.milvus = MilvusConfig()
        self.faiss = FaissConfig()
        self.retrieval = RetrievalConfig()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached global Settings instance."""
    return Settings()


# Convenience singleton — import this in other modules.
settings = get_settings()

"""
Abstract interfaces for the Video Retrieval system.

Design Principles:
    - Every concrete implementation must satisfy one of these ABCs.
    - This enables:
        * Dependency Injection (inject any conformant implementation)
        * Testability (mock/stub via interface)
        * Swappability (e.g., swap SigLIP2 → OpenCLIP without touching services)
    - All interfaces use Protocol or ABC — prefer ABC for explicit contracts.

Interfaces defined:
    - BaseEmbedder
    - BaseOCR
    - BaseDetector
    - BaseVectorDatabase
    - BaseBM25Index
    - BaseMetadataDB
    - BaseRetrievalBranch
    - BaseFusion
    - BaseReranker
    - BaseQueryParser
    - BaseIndexingPipeline
    - BaseRetrievalPipeline
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Data Transfer Objects (DTOs)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FrameRecord:
    """Represents a single extracted video frame."""

    video_id: str
    frame_id: str
    frame_idx: int
    timestamp: float
    frame_path: str


@dataclass
class OCRResult:
    """Single OCR detection: text + confidence + bounding box."""

    text: str
    confidence: float
    bbox: list[list[float]]  # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]


@dataclass
class DetectionResult:
    """Single YOLO object detection."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class RetrievalResult:
    """A single retrieval result returned to the user."""

    frame_id: str
    video_id: str
    frame_idx: int
    timestamp: float
    frame_path: str
    score: float
    source: str = ""  # which branch produced this result
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedQuery:
    """Structured output from the LLM query parser."""

    original_query: str
    objects: list[str] = field(default_factory=list)
    ocr_text: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    count: dict[str, int] = field(default_factory=dict)
    translated_query: Optional[str] = None  # English translation for CLIP
    expanded_queries: list[str] = field(default_factory=list)  # Multiple rephrased English queries


# ═══════════════════════════════════════════════════════════════════════════
# Model Interfaces
# ═══════════════════════════════════════════════════════════════════════════


class BaseEmbedder(ABC):
    """
    Abstract interface for image and text embedding models.

    Implementations: SigLIP2Embedder, OpenCLIPEmbedder.
    """

    @abstractmethod
    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode a list of images into embedding vectors.

        Args:
            image_paths: Absolute paths to image files.
            batch_size:  Processing batch size.

        Returns:
            np.ndarray of shape (N, dim), dtype float32, L2-normalized.
        """
        ...

    @abstractmethod
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode a list of text strings into embedding vectors.

        Args:
            texts:      List of text strings.
            batch_size: Processing batch size.

        Returns:
            np.ndarray of shape (N, dim), dtype float32, L2-normalized.
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        ...


class BaseOCR(ABC):
    """Abstract interface for OCR models."""

    @abstractmethod
    def extract(self, image_path: str) -> list[OCRResult]:
        """
        Run OCR on a single image.

        Args:
            image_path: Absolute path to image file.

        Returns:
            List of OCRResult objects.
        """
        ...

    @abstractmethod
    def extract_batch(self, image_paths: list[str]) -> list[list[OCRResult]]:
        """
        Run OCR on a batch of images.

        Args:
            image_paths: List of absolute paths.

        Returns:
            List of lists of OCRResult (one inner list per image).
        """
        ...


class BaseDetector(ABC):
    """Abstract interface for object detection models."""

    @abstractmethod
    def detect(self, image_path: str) -> list[DetectionResult]:
        """
        Run object detection on a single image.

        Args:
            image_path: Absolute path to image file.

        Returns:
            List of DetectionResult objects.
        """
        ...

    @abstractmethod
    def detect_batch(
        self, image_paths: list[str]
    ) -> list[list[DetectionResult]]:
        """
        Run detection on a batch of images.

        Returns:
            List of lists (one inner list per image).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Database Interfaces
# ═══════════════════════════════════════════════════════════════════════════


class BaseVectorDatabase(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def insert(
        self,
        collection_name: str,
        ids: list[int],
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Insert embeddings with metadata into a collection."""
        ...

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        top_k: int,
        filter_expr: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for nearest neighbours.

        Returns:
            List of dicts with keys: id, score, metadata fields.
        """
        ...

    @abstractmethod
    def delete(self, collection_name: str, ids: list[int]) -> None:
        """Delete records by ID."""
        ...

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Return the number of vectors in a collection."""
        ...


class BaseBM25Index(ABC):
    """Abstract interface for BM25 text index."""

    @abstractmethod
    def build(self, documents: list[str], doc_ids: list[str]) -> None:
        """Build/rebuild the index from a list of documents."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Search the BM25 index.

        Returns:
            List of (doc_id, score) tuples sorted by score descending.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        ...


class BaseMetadataDB(ABC):
    """Abstract interface for the metadata database (SQLAlchemy / SQLite)."""

    @abstractmethod
    def get_frame(self, frame_id: str) -> Optional[dict[str, Any]]:
        """Return frame metadata by frame_id."""
        ...

    @abstractmethod
    def get_frames_by_video(self, video_id: str) -> list[dict[str, Any]]:
        """Return all frames for a video, ordered by frame_idx."""
        ...

    @abstractmethod
    def get_neighbouring_frames(
        self,
        video_id: str,
        frame_idx: int,
        window: int,
    ) -> list[dict[str, Any]]:
        """
        Return frames within ±window of a given frame_idx.

        Used by temporal refinement.
        """
        ...

    @abstractmethod
    def get_frames_by_objects(
        self, class_names: list[str], min_confidence: float
    ) -> list[str]:
        """Return frame_ids that contain ALL of the given object classes."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval Interfaces
# ═══════════════════════════════════════════════════════════════════════════


class BaseRetrievalBranch(ABC):
    """
    Abstract interface for a single retrieval branch.

    Each branch:
        1. Receives a ParsedQuery
        2. Performs its specific retrieval strategy
        3. Returns a list of RetrievalResult with normalized scores [0, 1]
    """

    @property
    @abstractmethod
    def branch_name(self) -> str:
        """Unique identifier for this branch (used in fusion weighting)."""
        ...

    @abstractmethod
    async def retrieve(
        self,
        query: ParsedQuery,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Execute the retrieval branch.

        Args:
            query:  Parsed Vietnamese query.
            top_k:  Maximum results to return.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        ...


class BaseFusion(ABC):
    """Abstract interface for multi-branch score fusion."""

    @abstractmethod
    def fuse(
        self,
        branch_results: dict[str, list[RetrievalResult]],
        weights: dict[str, float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Fuse results from multiple branches into a single ranked list.

        Args:
            branch_results: Dict mapping branch_name → results.
            weights:        Dict mapping branch_name → weight.
            top_k:          Maximum fused results to return.

        Returns:
            De-duplicated, score-normalized, weighted list.
        """
        ...


class BaseReranker(ABC):
    """Abstract interface for result re-ranking."""

    @abstractmethod
    def rerank(
        self,
        query: ParsedQuery,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Re-rank candidates using a more expensive model.

        Args:
            query:      Parsed query.
            candidates: Candidate results from fusion.
            top_k:      Final number of results to return.

        Returns:
            Re-ranked list of RetrievalResult.
        """
        ...


class BaseQueryParser(ABC):
    """Abstract interface for Vietnamese query understanding."""

    @abstractmethod
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a Vietnamese query into structured fields.

        Args:
            query: Raw Vietnamese user query string.

        Returns:
            ParsedQuery with extracted objects, ocr, actions, etc.
        """
        ...

"""
BM25 index implementation using rank-bm25.

Design Decision:
    - rank-bm25 (Okapi BM25) chosen over Elasticsearch for:
        * Zero external service dependency during offline indexing
        * Sub-millisecond search on competition-scale data (<100K frames)
        * Picklable → save/load index as a single .pkl file
    - If corpus grows >500K docs, consider migrating to Elasticsearch.
    - Vietnamese tokenization: simple whitespace + punctuation split
      (underthesea word tokenizer can be plugged in for better recall).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from config.ocr_config import OCRConfig
from interfaces.base_interfaces import BaseBM25Index
from utils.logging_utils import get_logger

log = get_logger(__name__)


def _tokenize_vietnamese(text: str) -> list[str]:
    """
    Simple Vietnamese tokenizer.

    Splits on whitespace and punctuation, lowercases tokens.
    For production, replace with underthesea.word_tokenize().

    Args:
        text: Input Vietnamese text string.

    Returns:
        List of lowercase token strings.
    """
    # Remove special characters, keep Vietnamese Unicode
    text = text.lower()
    tokens = re.split(r"[\s\u0000-\u001f\u007f-\u009f\.,;:!?\"\'()\[\]{}<>]+", text)
    return [t for t in tokens if t]


class BM25OcrIndex(BaseBM25Index):
    """
    BM25 index over OCR text extracted from video frames.

    The index maps doc_id (= frame_id) → BM25 relevance score.

    Attributes:
        config:    OCRConfig with BM25 parameters and index path.
        _bm25:     BM25Okapi instance.
        _doc_ids:  List of frame_id strings parallel to BM25 corpus.
    """

    def __init__(self, config: OCRConfig) -> None:
        """
        Initialize BM25OcrIndex.

        Args:
            config: OCRConfig with bm25_index_path, k1, b values.
        """
        self.config = config
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: list[str] = []

    def build(self, documents: list[str], doc_ids: list[str]) -> None:
        """
        Build the BM25 index from a list of text documents.

        Args:
            documents: List of OCR text strings (one per frame).
            doc_ids:   Parallel list of frame_id strings.

        Raises:
            ValueError: If documents and doc_ids have different lengths.
        """
        if len(documents) != len(doc_ids):
            raise ValueError("documents and doc_ids must have same length")

        if BM25Okapi is None:
            raise ImportError("rank-bm25 is not installed. Please install it using `pip install rank-bm25`.")

        log.info(f"Building BM25 index with {len(documents)} documents")
        tokenized = [_tokenize_vietnamese(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized, k1=self.config.bm25_k1, b=self.config.bm25_b)
        self._doc_ids = doc_ids
        log.info("BM25 index built successfully")

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Search the BM25 index.

        Args:
            query:  Query string (will be tokenized).
            top_k:  Maximum number of results to return.

        Returns:
            List of (frame_id, score) sorted by score descending.
            Returns empty list if index not built.
        """
        if self._bm25 is None:
            log.warning("BM25 index not built — returning empty results")
            return []

        query_tokens = _tokenize_vietnamese(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = [
            (self._doc_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0  # Skip zero-score results
        ]

        return results

    def save(self, path: Optional[str] = None) -> None:
        """
        Serialize the BM25 index to disk.

        Args:
            path: Output path. Uses config.bm25_index_path if None.
        """
        save_path = Path(path or self.config.bm25_index_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"bm25": self._bm25, "doc_ids": self._doc_ids}
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

        log.info(f"BM25 index saved to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """
        Load a persisted BM25 index from disk.

        Args:
            path: Source path. Uses config.bm25_index_path if None.

        Raises:
            FileNotFoundError: If index file does not exist.
        """
        load_path = Path(path or self.config.bm25_index_path)
        if not load_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {load_path}")

        with open(load_path, "rb") as f:
            payload = pickle.load(f)

        self._bm25 = payload["bm25"]
        self._doc_ids = payload["doc_ids"]
        log.info(f"BM25 index loaded from {load_path} | docs={len(self._doc_ids)}")

    @property
    def num_docs(self) -> int:
        """Return number of indexed documents."""
        return len(self._doc_ids)

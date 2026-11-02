"""
Branch B: OCR BM25 keyword search.
"""

from __future__ import annotations

from config.retrieval_config import RetrievalConfig
from database.bm25.bm25_index import BM25OcrIndex
from interfaces.base_interfaces import (
    BaseRetrievalBranch,
    ParsedQuery,
    RetrievalResult,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)


class OCRBm25Branch(BaseRetrievalBranch):
    """
    Retrieval Branch B: BM25 keyword search over OCR text.

    Process:
        1. Join query.ocr_text list into a search query.
        2. Tokenize and search the BM25 index.
        3. Normalize BM25 scores to [0, 1].
        4. Return top_k frame results.

    Best for: exact text matches (shop names, phone numbers, addresses).

    Attributes:
        config:  RetrievalConfig.
        bm25:    BM25OcrIndex.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        bm25_index: BM25OcrIndex,
    ) -> None:
        self.config = config
        self.bm25 = bm25_index

    @property
    def branch_name(self) -> str:
        return "ocr_bm25"

    async def retrieve(
        self,
        query: ParsedQuery,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        BM25 search using extracted OCR keywords.

        If no OCR keywords in parsed query, falls back to full original query.

        Args:
            query:  ParsedQuery.
            top_k:  Max results.

        Returns:
            List of RetrievalResult.
        """
        # Use LLM-extracted OCR text, or fall back to full query
        if query.ocr_text:
            search_str = " ".join(query.ocr_text)
        else:
            search_str = query.original_query

        log.debug(f"BM25 search: '{search_str}'")

        raw_results = self.bm25.search(search_str, top_k=top_k)

        if not raw_results:
            return []

        # Normalize BM25 scores to [0, 1]
        max_score = max(score for _, score in raw_results)
        if max_score == 0:
            return []

        results: list[RetrievalResult] = []
        for frame_id, score in raw_results:
            # frame_id format: "{video_id}_frame_{idx:06d}"
            parts = frame_id.rsplit("_frame_", 1)
            video_id = parts[0] if len(parts) == 2 else ""
            frame_idx = int(parts[1]) if len(parts) == 2 else 0

            results.append(
                RetrievalResult(
                    frame_id=frame_id,
                    video_id=video_id,
                    frame_idx=frame_idx,
                    timestamp=0.0,  # Will be filled by metadata lookup
                    frame_path="",
                    score=score / max_score,  # Normalized score
                    source=self.branch_name,
                )
            )

        log.info(f"BM25 branch: {len(results)} results")
        return results

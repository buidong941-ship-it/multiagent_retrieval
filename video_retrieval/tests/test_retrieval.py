"""
Unit tests for the retrieval pipeline.

Tests are isolated using mocks/stubs — no real GPU or DB required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config():
    """Return a mock RetrievalConfig."""
    from config.retrieval_config import RetrievalConfig, LLMProvider
    return RetrievalConfig(
        clip_top_k=10,
        ocr_bm25_top_k=5,
        ocr_embed_top_k=5,
        object_top_k=5,
        temporal_window=0,  # Disable temporal for tests
        final_top_k=10,
        llm_provider=LLMProvider.GEMINI,
        gemini_api_key="test_key",
    )


@pytest.fixture
def sample_parsed_query():
    """Return a sample ParsedQuery."""
    from interfaces.base_interfaces import ParsedQuery
    return ParsedQuery(
        original_query="Một người phụ nữ mặc áo đỏ đứng trước Highlands Coffee",
        objects=["person"],
        ocr_text=["Highlands Coffee"],
        attributes=["red shirt", "female"],
        actions=["standing"],
        relations=["in front of"],
        translated_query="A woman in a red shirt standing in front of Highlands Coffee",
    )


@pytest.fixture
def sample_retrieval_results():
    """Return sample RetrievalResult list."""
    from interfaces.base_interfaces import RetrievalResult
    return [
        RetrievalResult(
            frame_id=f"video_001_frame_{i:06d}",
            video_id="video_001",
            frame_idx=i,
            timestamp=float(i),
            frame_path=f"/data/frames/video_001/frame_{i:06d}.jpg",
            score=1.0 - (i * 0.05),
            source="clip",
        )
        for i in range(10)
    ]


# ── Unit Tests ────────────────────────────────────────────────────────────


class TestBM25Index:
    """Tests for BM25OcrIndex."""

    def test_build_and_search(self):
        from config.ocr_config import OCRConfig
        from database.bm25.bm25_index import BM25OcrIndex

        config = OCRConfig()
        index = BM25OcrIndex(config)

        docs = [
            "Highlands Coffee Việt Nam",
            "Trà sữa The Coffee House",
            "Xe đạp và ô tô trên đường",
        ]
        ids = ["frame_001", "frame_002", "frame_003"]

        index.build(docs, ids)
        results = index.search("Highlands Coffee", top_k=3)

        assert len(results) > 0
        assert results[0][0] == "frame_001"  # Most relevant
        assert results[0][1] > 0  # Positive score

    def test_empty_query(self):
        from config.ocr_config import OCRConfig
        from database.bm25.bm25_index import BM25OcrIndex

        config = OCRConfig()
        index = BM25OcrIndex(config)
        index.build(["some text"], ["frame_001"])

        results = index.search("", top_k=5)
        assert results == []

    def test_empty_index_returns_empty(self):
        from config.ocr_config import OCRConfig
        from database.bm25.bm25_index import BM25OcrIndex

        config = OCRConfig()
        index = BM25OcrIndex(config)

        results = index.search("test query", top_k=5)
        assert results == []


class TestWeightedFusion:
    """Tests for WeightedFusion."""

    def test_deduplication(self, mock_config):
        from interfaces.base_interfaces import RetrievalResult
        from retrieval.fusion.fusion_service import WeightedFusion

        fusion = WeightedFusion(mock_config)

        # Same frame appearing in two branches
        frame_in_both = RetrievalResult(
            frame_id="video_001_frame_000010",
            video_id="video_001",
            frame_idx=10,
            timestamp=1.0,
            frame_path="",
            score=0.9,
            source="clip",
        )
        branch_results = {
            "clip": [frame_in_both],
            "ocr_bm25": [
                RetrievalResult(
                    frame_id="video_001_frame_000010",
                    video_id="video_001",
                    frame_idx=10,
                    timestamp=1.0,
                    frame_path="",
                    score=0.8,
                    source="ocr_bm25",
                )
            ],
        }

        result = fusion.fuse(branch_results, {"clip": 1.0, "ocr_bm25": 0.6}, top_k=10)

        # Should deduplicate
        frame_ids = [r.frame_id for r in result]
        assert len(frame_ids) == len(set(frame_ids))

    def test_weight_ordering(self, mock_config):
        from interfaces.base_interfaces import RetrievalResult
        from retrieval.fusion.fusion_service import WeightedFusion

        fusion = WeightedFusion(mock_config)

        results_a = [
            RetrievalResult("vid_frame_000001", "vid", 1, 0.1, "", 1.0, "clip"),
        ]
        results_b = [
            RetrievalResult("vid_frame_000002", "vid", 2, 0.2, "", 1.0, "ocr_bm25"),
        ]

        fused = fusion.fuse(
            {"clip": results_a, "ocr_bm25": results_b},
            weights={"clip": 2.0, "ocr_bm25": 0.1},
            top_k=10,
        )

        # clip-only frame should score higher due to higher weight
        assert fused[0].frame_id == "vid_frame_000001"


class TestQueryParser:
    """Tests for LLMQueryParser with mocked LLM."""

    def test_parse_fallback_on_error(self, mock_config):
        from query_parser.parser_service import LLMQueryParser

        parser = LLMQueryParser(mock_config)

        # Force LLM to fail
        with patch.object(parser, "_call_llm", side_effect=Exception("API error")):
            result = parser.parse("Test query")

        assert result.original_query == "Test query"
        assert result.objects == []

    def test_parse_with_mock_llm(self, mock_config):
        import json
        from query_parser.parser_service import LLMQueryParser

        parser = LLMQueryParser(mock_config)

        mock_response = json.dumps({
            "objects": ["person", "umbrella"],
            "ocr": ["Highlands Coffee"],
            "attributes": ["red shirt"],
            "actions": ["holding"],
            "relations": ["in front of"],
            "count": {"person": 1},
            "translated_query": "A woman in red holding an umbrella",
        })

        with patch.object(parser, "_call_llm", return_value=mock_response):
            result = parser.parse("Một người phụ nữ...")

        assert "person" in result.objects
        assert "Highlands Coffee" in result.ocr_text
        assert result.translated_query == "A woman in red holding an umbrella"


class TestOCRModel:
    """Tests for PaddleOCR model wrapper."""

    def test_results_to_text(self):
        from interfaces.base_interfaces import OCRResult
        from models.ocr.paddle_ocr_model import PaddleOCRModel

        results = [
            OCRResult(text="Highlands", confidence=0.95, bbox=[]),
            OCRResult(text="Coffee", confidence=0.90, bbox=[]),
        ]
        text = PaddleOCRModel.results_to_text(results)
        assert text == "Highlands Coffee"

    def test_empty_results_to_text(self):
        from models.ocr.paddle_ocr_model import PaddleOCRModel
        text = PaddleOCRModel.results_to_text([])
        assert text == ""


class TestFrameUtils:
    """Tests for video utility functions."""

    def test_build_frame_id(self):
        from utils.video_utils import build_frame_id

        fid = build_frame_id("video_001", 42)
        assert fid == "video_001_frame_000042"

        fid = build_frame_id("vid", 0)
        assert fid == "vid_frame_000000"

    def test_build_frame_id_large_index(self):
        from utils.video_utils import build_frame_id

        fid = build_frame_id("vid", 999999)
        assert fid == "vid_frame_999999"

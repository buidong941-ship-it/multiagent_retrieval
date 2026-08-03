"""
API request and response schemas using Pydantic v2.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Request schemas ──────────────────────────────────────────────────────


class RetrievalRequest(BaseModel):
    """Request body for video frame retrieval."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Vietnamese text query",
        examples=["Một người phụ nữ mặc áo đỏ đứng trước Highlands Coffee"],
    )
    top_k: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of results to return",
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Optional: restrict search to a specific video",
    )
    use_temporal: bool = Field(
        default=True,
        description="Enable temporal refinement",
    )
    mode: str = Field(
        default="fusion",
        description="Retrieval mode: 'fusion', 'clip', 'siglip_jina_sequence', 'siglip_jina_parallel', 'ocr_bm25', 'ocr_embed', 'object', 'action', 'direct_ocr'",
    )


class IndexVideoRequest(BaseModel):
    """Request body for indexing a single video."""

    video_path: str = Field(
        ...,
        description="Absolute path to video file on server",
    )


# ── Response schemas ─────────────────────────────────────────────────────


class FrameResult(BaseModel):
    """Single frame retrieval result."""

    frame_id: str = Field(description="Globally unique frame identifier")
    video_id: str = Field(description="Video identifier")
    frame_idx: int = Field(description="0-based frame index within video")
    timestamp: float = Field(description="Frame timestamp in seconds")
    frame_path: str = Field(description="Path to frame image on server")
    score: float = Field(description="Relevance score [0, 1]")
    source: str = Field(description="Which branch produced this result")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Response for a retrieval request."""

    query: str
    parsed_query: dict[str, Any] = Field(description="LLM-parsed query structure")
    total_results: int
    results: list[FrameResult]
    latency_ms: float = Field(description="Total processing time in milliseconds")


class IndexVideoResponse(BaseModel):
    """Response for a video indexing request."""

    video_id: str
    frames_indexed: int
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    milvus: str
    database: str
    version: str = "1.0.0"

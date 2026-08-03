"""
Retrieval pipeline configuration.

Controls:
- Branch top-k values
- Fusion weights
- Temporal refinement window
- LLM query parser settings
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field, model_validator

from config.base_config import BaseConfig


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class RetrievalConfig(BaseConfig):
    """Configuration for the online retrieval pipeline."""

    # ── Branch A: CLIP (SigLIP2) vector search ──────────────────────────
    clip_top_k: int = Field(
        default=300,
        ge=1,
        description="Top-K frames from CLIP vector search",
    )

    # ── Branch B: OCR BM25 keyword search ───────────────────────────────
    ocr_bm25_top_k: int = Field(
        default=50,
        ge=1,
        description="Top-K frames from OCR BM25 search",
    )

    # ── Branch C: OCR embedding vector search ───────────────────────────
    ocr_embed_top_k: int = Field(
        default=50,
        ge=1,
        description="Top-K frames from OCR embedding search",
    )

    # ── Branch D: Object metadata filter ────────────────────────────────
    object_top_k: int = Field(
        default=100,
        ge=1,
        description="Top-K frames from object metadata filter",
    )

    # ── Fusion weights ───────────────────────────────────────────────────
    # Weights are normalized internally, so only relative values matter.
    weight_clip: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for CLIP branch score",
    )
    weight_clip_aux: float = Field(
        default=0.4,
        ge=0.0,
        description="Weight for auxiliary CLIP branches (attr, scene, action, before, after)",
    )
    weight_ocr_bm25: float = Field(
        default=0.6,
        ge=0.0,
        description="Weight for OCR BM25 branch score",
    )
    weight_ocr_embed: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight for OCR embedding branch score",
    )
    weight_object: float = Field(
        default=0.4,
        ge=0.0,
        description="Weight for object detection branch score",
    )
    weight_action: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight for action embedding branch score",
    )

    # ── Temporal refinement ──────────────────────────────────────────────
    temporal_window: int = Field(
        default=1,
        ge=0,
        description="Number of neighbouring frames to check (±N)",
    )
    temporal_top_k: int = Field(
        default=10,
        ge=1,
        description="How many candidates to apply temporal refinement to",
    )

    # ── Neighbor Reranking ───────────────────────────────────────────────
    rerank_window: int = Field(
        default=1,
        ge=0,
        description="Number of neighbouring frames to sum scores from (±N)",
    )
    rerank_top_k: int = Field(
        default=20,
        ge=1,
        description="How many candidates to apply neighbor reranking to",
    )

    # ── Final results ────────────────────────────────────────────────────
    final_top_k: int = Field(
        default=100,
        ge=1,
        description="Number of results to return to user",
    )

    # ── LLM query parser ─────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GEMINI,
        description="LLM provider for Vietnamese query parsing",
    )
    llm_model: str = Field(
        default="gemini-2.0-flash",
        description="LLM model name (provider-specific)",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0 = deterministic)",
    )
    llm_timeout: float = Field(
        default=60.0,
        description="LLM API call timeout in seconds",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b",
        description="Ollama model for ReAct Agent (e.g. llama3, qwen2.5:7b)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama OpenAI-compatible API base URL",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Gemini API Key for LLM Parsing",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API Key for LLM Parsing",
    )
    model_config = {"env_prefix": "RETRIEVAL_"}

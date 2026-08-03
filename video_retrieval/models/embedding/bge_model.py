"""
BGE-M3 text embedding model (for OCR + audio embeddings).

Architecture:
    - XLM-RoBERTa base architecture
    - Multi-granularity retrieval:
        * Dense retrieval: standard CLS token embedding (dim=1024)
        * Sparse retrieval: learned term weights (like BM25 but neural)
        * ColBERT: token-level late interaction
    - Trained on massive multilingual corpus (100+ languages)

Why BGE-M3 for OCR text:
    - Best multilingual dense retrieval model (as of 2024 MTEB leaderboard)
    - Vietnamese support is first-class (trained on Vietnamese CommonCrawl)
    - 8192 token context → handles long OCR / audio transcripts
    - Sparse retrieval mode acts as a learned BM25 → better than keyword search

Fine-tuning strategy:
    - Fine-tune on (Vietnamese OCR query, relevant frame text) pairs
    - Use InfoNCE loss with in-batch negatives
    - LoRA on Q/K/V projections: 4 GPU hours on 8xA100

Input:
    - Text strings (up to 8192 tokens)

Output:
    - Dense: float32 vector dim=1024, L2-normalized
    - Sparse: dict[token_id, weight] (optional)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config.ocr_config import OCRConfig
from interfaces.base_interfaces import BaseEmbedder
from utils.gpu_utils import get_device
from utils.logging_utils import get_logger

log = get_logger(__name__)


class BGEM3Embedder(BaseEmbedder):
    """
    BGE-M3 multilingual text encoder.

    Used for:
        - Encoding OCR text extracted from frames
        - Encoding audio transcripts
        - Encoding Vietnamese OCR queries at retrieval time

    Attributes:
        config:    OCRConfig instance.
        device:    Resolved torch device.
        _model:    Loaded XLM-RoBERTa model.
        _tokenizer: AutoTokenizer.
    """

    def __init__(self, config: OCRConfig) -> None:
        """
        Initialize BGE-M3 embedder.

        Args:
            config: OCRConfig (contains BGE model name, device, batch size).
        """
        self.config = config
        self.device = get_device(config.bge_device)
        self._model: Optional[AutoModel] = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load model weights on first call."""
        if self._model is not None:
            return

        log.info(f"Loading BGE-M3: {self.config.bge_model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.bge_model_name)
        self._model = (
            AutoModel.from_pretrained(self.config.bge_model_name)
            .to(self.device)
            .eval()
        )
        log.info("BGE-M3 loaded successfully")

    @property
    def embedding_dim(self) -> int:
        return self.config.bge_embedding_dim  # 1024

    def encode_texts(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode texts to L2-normalized dense embeddings.

        Args:
            texts:      List of text strings.
            batch_size: Override config batch_size.

        Returns:
            np.ndarray shape (N, 1024), float32, L2-normalized.
        """
        self._load()
        bs = batch_size or self.config.bge_batch_size
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]

            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.bge_max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self._model(**encoded)
                # CLS token embedding (standard for BGE)
                embs = output.last_hidden_state[:, 0, :]

            embs = embs.detach().cpu().float().numpy()

            if self.config.bge_normalize:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / np.maximum(norms, 1e-8)

            all_embeddings.append(embs)

        return np.vstack(all_embeddings)

    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Not supported — BGE-M3 is text-only."""
        raise NotImplementedError("BGE-M3 is a text-only encoder")

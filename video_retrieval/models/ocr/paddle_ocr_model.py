"""
PaddleOCR model wrapper.

Why PaddleOCR:
    - Superior Vietnamese text recognition (official VI model)
    - PP-OCRv4: state-of-the-art detection (DB++) + recognition (SVTR)
    - 11ms/image on GPU for the standard pipeline
    - Active maintenance with frequent model updates

Architecture:
    - Text Detection: DBNet++ (differentiable binarization)
    - Text Direction: MobileNetV3 classifier
    - Text Recognition: SVTR (Scene Text Recognition Transformer)

Fine-tuning:
    - Detection: fine-tune DB++ on custom scene datasets
    - Recognition: fine-tune SVTR on domain-specific text (logos, signs)
    - Both support PaddlePaddle's official fine-tuning toolkit

Input:
    - BGR numpy array or image path

Output:
    - List of (bbox, (text, confidence)) tuples
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

from config.ocr_config import OCRConfig
from interfaces.base_interfaces import BaseOCR, OCRResult
from utils.logging_utils import get_logger

log = get_logger(__name__)


class PaddleOCRModel(BaseOCR):
    """
    PaddleOCR wrapper implementing BaseOCR interface.

    Attributes:
        config:  OCRConfig with language, confidence threshold, etc.
        _reader: PaddleOCR instance (lazy-loaded).
    """

    def __init__(self, config: OCRConfig) -> None:
        """
        Initialize PaddleOCRModel.

        Args:
            config: OCRConfig instance.
        """
        self.config = config
        self._reader = None
        self._load_failed = False

    def _load(self) -> None:
        """Lazy-load PaddleOCR to avoid slow import at startup."""
        if self._reader is not None:
            return
        if self._load_failed:
            raise RuntimeError("PaddleOCR initialization failed. See earlier CRITICAL log for root cause.")

        # Import here to allow the rest of the system to run without PaddlePaddle
        from paddleocr import PaddleOCR  # type: ignore

        log.info(f"Loading PaddleOCR | lang={self.config.lang}")

        # Snapshot PATH before PaddleOCR modifies it.
        _saved_path = os.environ.get("PATH", "")

        try:
            # Minimal init — compatible with both PaddleOCR 2.x and 3.x
            self._reader = PaddleOCR(lang=self.config.lang)
            log.info("PaddleOCR loaded successfully")
        except Exception as exc:
            self._load_failed = True
            log.error(f"CRITICAL: Failed to initialize PaddleOCR: {exc}")
            raise exc
        finally:
            os.environ["PATH"] = _saved_path

    def extract(self, image_path: str) -> list[OCRResult]:
        """
        Run OCR on a single image file.

        Args:
            image_path: Absolute path to image.

        Returns:
            List of OCRResult filtered by min_confidence.
        """
        self._load()  # Raises RuntimeError immediately if init failed
        results: list[OCRResult] = []
        try:
            # PaddleOCR 3.x uses predict() instead of ocr()
            raw = self._reader.predict(image_path)
        except Exception as exc:
            log.error(f"PaddleOCR failed on {image_path}: {exc}")
            return results

        if not raw or raw[0] is None:
            return results

        # PaddleOCR 3.x returns a list of result dicts per image
        # Each dict has 'rec_text', 'rec_score', 'det_polys'
        for res in raw:
            if res is None:
                continue
            polys  = res.get("det_polys", [])
            texts  = res.get("rec_text",  [])
            scores = res.get("rec_score", [])
            for bbox, text, confidence in zip(polys, texts, scores):
                if confidence < self.config.min_confidence:
                    continue
                results.append(
                    OCRResult(
                        text=text.strip(),
                        confidence=float(confidence),
                        bbox=bbox,
                    )
                )

        return results

    def extract_batch(self, image_paths: list[str]) -> list[list[OCRResult]]:
        """
        Run OCR on multiple images sequentially.

        Note: PaddleOCR doesn't natively batch across images,
        so this is a sequential loop. Use multiprocessing for throughput.

        Args:
            image_paths: List of image paths.

        Returns:
            List of OCR result lists.
        """
        return [self.extract(path) for path in image_paths]

    @staticmethod
    def results_to_text(results: list[OCRResult]) -> str:
        """
        Concatenate OCR results into a single text string.

        Args:
            results: List of OCRResult objects.

        Returns:
            Space-joined text string (for BM25 indexing).
        """
        return " ".join(r.text for r in results if r.text)

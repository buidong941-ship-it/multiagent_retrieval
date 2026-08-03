"""
EasyOCR fallback model for Kaggle environments where PaddlePaddle conflicts with PyTorch.
"""

from typing import Any, Optional

from config.ocr_config import OCRConfig
from interfaces.base_interfaces import BaseOCR, OCRResult
from utils.logging_utils import get_logger
from utils.gpu_utils import get_device

log = get_logger(__name__)

class EasyOCRModel(BaseOCR):
    """
    EasyOCR implementation for text detection and recognition.
    Safe to use alongside PyTorch on Kaggle T4.
    """

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._reader: Optional[Any] = None

    def _load(self) -> None:
        """Lazy load EasyOCR."""
        if self._reader is not None:
            return

        import easyocr

        lang = self.config.lang
        if lang == "vn":
            lang = "vi" # EasyOCR uses 'vi'

        use_gpu = self.config.use_gpu and get_device() == "cuda"
        
        try:
            log.info(f"Loading EasyOCR | lang={lang} | gpu={use_gpu}")
            self._reader = easyocr.Reader([lang], gpu=use_gpu)
        except Exception as e:
            log.error(f"Failed to load EasyOCR: {e}")
            raise e

    def extract(self, image_path: str) -> list[OCRResult]:
        self._load()
        if self._reader is None:
            return []

        try:
            # Returns list of (bbox, text, prob)
            results = self._reader.readtext(image_path)
            
            extracted = []
            for bbox, text, conf in results:
                if conf < self.config.min_confidence:
                    continue
                extracted.append(
                    OCRResult(
                        text=text,
                        confidence=float(conf),
                        bbox=[[float(p[0]), float(p[1])] for p in bbox]
                    )
                )
            return extracted
        except Exception as e:
            log.error(f"EasyOCR failed on {image_path}: {e}")
            return []

    def extract_batch(self, image_paths: list[str]) -> list[list[OCRResult]]:
        """
        Extract text from a batch of images.
        EasyOCR doesn't have a robust built-in batch inference yet, so we loop.
        """
        results = []
        for path in image_paths:
            results.append(self.extract(path))
        return results

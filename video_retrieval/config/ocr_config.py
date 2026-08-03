"""
OCR configuration (PaddleOCR + BGE-M3 embeddings).

Design Decision:
    - PaddleOCR is chosen over EasyOCR / Tesseract because:
        * Superior Vietnamese text recognition (trained on VinText dataset)
        * Built-in text detection + recognition pipeline
        * Active maintenance and ONNX export support
    - BGE-M3 for OCR text embedding because:
        * Multilingual (100+ languages including Vietnamese)
        * Multi-granularity retrieval (dense + sparse + colbert)
        * Best-in-class on BEIR benchmark for cross-lingual retrieval
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from config.base_config import BASE_DIR, BaseConfig


class OCRConfig(BaseConfig):
    """Configuration for PaddleOCR service."""

    engine: str = Field(
        default="paddle",
        description="OCR Engine to use ('paddle' or 'easyocr')",
    )
    # PaddleOCR settings
    lang: str = Field(
        default="vi",
        description="PaddleOCR language code (vi = Vietnamese)",
    )
    # Đường dẫn tới model fine-tuned (None = dùng model mặc định)
    # Set bằng: OCR_DET_MODEL_DIR=... trong .env
    det_model_dir: Optional[str] = Field(
        default=None,
        description="Custom detection model directory (fine-tuned). None = PaddleOCR default.",
    )
    rec_model_dir: Optional[str] = Field(
        default=None,
        description="Custom recognition model directory (fine-tuned). None = PaddleOCR default.",
    )

    use_angle_cls: bool = Field(
        default=True,
        description="Enable text angle classification",
    )
    use_gpu: bool = Field(
        default=True,
        description="Use GPU for PaddleOCR inference",
    )
    gpu_id: int = Field(
        default=0,
        description="GPU device ID",
    )
    det_db_thresh: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Detection binarization threshold",
    )
    det_db_box_thresh: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection box threshold",
    )
    rec_batch_num: int = Field(
        default=6,
        description="Recognition batch size",
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum OCR confidence to keep a result",
    )

    # BGE-M3 embedding settings
    bge_model_name: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace model ID for BGE-M3",
    )
    bge_device: str = Field(
        default="cuda",
        description="torch device for BGE-M3",
    )
    bge_batch_size: int = Field(
        default=32,
        description="Batch size for BGE-M3 text encoding",
    )
    bge_embedding_dim: int = Field(
        default=1024,
        description="BGE-M3 dense embedding dimension",
    )
    bge_normalize: bool = Field(
        default=True,
        description="L2-normalize BGE-M3 embeddings",
    )
    bge_max_length: int = Field(
        default=512,
        description="Max token length for BGE-M3",
    )

    # BM25 settings
    bm25_index_path: str = Field(
        default=(BASE_DIR / "data" / "indexes" / "bm25_ocr.pkl").as_posix(),
        description="Path to persist the BM25 index",
    )
    bm25_k1: float = Field(
        default=1.5,
        description="BM25 k1 parameter (term frequency saturation)",
    )
    bm25_b: float = Field(
        default=0.75,
        description="BM25 b parameter (length normalization)",
    )

    model_config = {"env_prefix": "OCR_"}

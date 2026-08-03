"""
OCR service: runs PaddleOCR on frames, embeds with BGE-M3,
builds BM25 index, and stores results in Milvus + SQLite.
"""

from __future__ import annotations

from typing import Optional

from tqdm import tqdm

from config.ocr_config import OCRConfig
from database.bm25.bm25_index import BM25OcrIndex
from database.metadata.metadata_db import MetadataDatabase
from interfaces.base_interfaces import BaseOCR, BaseVectorDatabase, FrameRecord, OCRResult
from models.embedding.bge_model import BGEM3Embedder
from models.ocr.paddle_ocr_model import PaddleOCRModel
from models.ocr.easyocr_model import EasyOCRModel
from utils.logging_utils import get_logger

log = get_logger(__name__)


class OCRService:
    """
    OCR pipeline: extract text → embed → index.

    Responsibilities:
        1. Run PaddleOCR on each frame image.
        2. Store raw OCR results (text, confidence, bbox) in SQLite.
        3. Generate BGE-M3 embeddings of concatenated OCR text.
        4. Insert OCR embeddings into vector database ocr_embeddings collection.
        5. Build/update BM25 index for keyword search.

    Attributes:
        config:    OCRConfig.
        ocr:       BaseOCR instance.
        embedder:  BGEM3Embedder instance.
        vector_db: BaseVectorDatabase.
        meta_db:   MetadataDatabase.
        bm25:      BM25OcrIndex.
    """

    def __init__(
        self,
        config: OCRConfig,
        vector_db: BaseVectorDatabase,
        meta_db: MetadataDatabase,
        ocr_model: Optional[BaseOCR] = None,
        embedder: Optional[BGEM3Embedder] = None,
        bm25_index: Optional[BM25OcrIndex] = None,
    ) -> None:
        """
        Initialize OCRService.

        Args:
            config:     OCRConfig.
            vector_db:  Milvus client.
            meta_db:    SQLite metadata DB.
            ocr_model:  Optional pre-built OCR model (DI).
            embedder:   Optional pre-built BGE-M3 embedder (DI).
            bm25_index: Optional pre-built BM25 index (DI).
        """
        self.config = config
        self.vector_db = vector_db
        self.meta_db = meta_db

        if ocr_model is not None:
            self.ocr = ocr_model
        elif self.config.engine == "easyocr":
            self.ocr = EasyOCRModel(self.config)
        else:
            self.ocr = PaddleOCRModel(self.config)

        self.embedder = embedder or BGEM3Embedder(config)
        self.bm25 = bm25_index or BM25OcrIndex(config)
        self._collection = "ocr_embeddings"

    def setup_collection(self) -> None:
        """Ensure Milvus OCR collection exists."""
        self.vector_db.create_collection_if_not_exists(self._collection, dim=self.config.bge_embedding_dim)

    async def process_frames(
        self,
        frames: list[FrameRecord],
        id_offset: int = 0,
        skip_empty: bool = True,
    ) -> None:
        """
        Run full OCR pipeline on a list of frames.

        Args:
            frames:      List of FrameRecord with frame_path.
            id_offset:   Starting Milvus int64 ID.
            skip_empty:  If True, skip frames with no OCR text.
        """
        self.setup_collection()
        log.info(f"Running OCR on {len(frames)} frames")

        all_texts: list[str] = []     # for BGE-M3 batch encoding
        all_frame_ids: list[str] = [] # for BM25 doc_ids
        embed_batch: list[dict] = []  # metadata for Milvus
        embed_ids: list[int] = []

        for i, frame in enumerate(tqdm(frames, desc="OCR", unit="frame")):
            try:
                results: list[OCRResult] = self.ocr.extract(frame.frame_path)
            except Exception as exc:
                log.warning(f"OCR failed on {frame.frame_path}: {exc}")
                results = []

            ocr_text = PaddleOCRModel.results_to_text(results)

            # Serialize OCR results to JSON-able format
            ocr_json = [
                {
                    "text": r.text,
                    "confidence": r.confidence,
                    "bbox": r.bbox,
                }
                for r in results
            ]

            # Update SQLite frame record with OCR data
            await self.meta_db.upsert_frame(
                {
                    "frame_id": frame.frame_id,
                    "video_id": frame.video_id,
                    "frame_idx": frame.frame_idx,
                    "timestamp": frame.timestamp,
                    "frame_path": frame.frame_path,
                    "ocr_results": ocr_json,
                    "ocr_text": ocr_text,
                }
            )

            if skip_empty and not ocr_text.strip():
                continue

            all_texts.append(ocr_text)
            all_frame_ids.append(frame.frame_id)
            embed_ids.append(id_offset + i)
            embed_batch.append(
                {
                    "frame_id": frame.frame_id,
                    "video_id": frame.video_id,
                    "ocr_text": ocr_text[:4000],  # Milvus VARCHAR limit
                }
            )

        # Generate embeddings in bulk (DISABLED by user request)
        if all_texts:
            # log.info(f"Generating BGE-M3 embeddings for {len(all_texts)} OCR texts")
            # embeddings = self.embedder.encode_texts(all_texts)

            # self.vector_db.insert(
            #     collection_name=self._collection,
            #     ids=embed_ids,
            #     embeddings=embeddings,
            #     metadata=embed_batch,
            # )
            # self.vector_db.flush(self._collection)

            # Build/update BM25 index
            log.info("Building BM25 index")
            self.bm25.build(all_texts, all_frame_ids)
            self.bm25.save()

        log.info(f"OCR pipeline complete | {len(all_texts)} frames with text")

    def encode_query(self, query_text: str):
        """Encode an OCR query text using BGE-M3."""
        return self.embedder.encode_texts([query_text])[0]

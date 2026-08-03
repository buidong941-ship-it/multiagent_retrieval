import sys
import os
import asyncio

# Tắt các cảnh báo gRPC rác (như too_many_pings) của Milvus
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

from core.logger import get_logger

log = get_logger(__name__)

# Add both the agent root and video_retrieval to Python path safely
agent_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
video_retrieval_root = os.path.join(agent_root, 'video_retrieval')

if agent_root not in sys.path:
    sys.path.insert(0, agent_root)
if video_retrieval_root not in sys.path:
    sys.path.insert(0, video_retrieval_root)

from config.settings import get_settings
from database.metadata.metadata_db import MetadataDatabase
from database.milvus.milvus_client import MilvusVectorDatabase
from database.bm25.bm25_index import BM25OcrIndex
from services.embedding.image_embedding_service import ImageEmbeddingService
from services.ocr.ocr_service import OCRService
from pipelines.online_pipeline import OnlinePipeline

_pipeline = None
_is_initializing = False

async def get_real_pipeline() -> OnlinePipeline:
    """
    Lazy, cross-loop safe singleton for initializing the real video_retrieval databases.
    """
    global _pipeline, _is_initializing
    if _pipeline is not None:
        return _pipeline

    # If another thread/task is initializing, wait for it instead of deadlocking
    while _is_initializing:
        await asyncio.sleep(0.5)
        if _pipeline is not None:
            return _pipeline

    _is_initializing = True
    try:
        log.info("Initializing Real Databases (Milvus, SQLite, SigLIP, BM25)... This may take a few seconds.")
        settings = get_settings()
        
        meta_db = MetadataDatabase()
        await meta_db.init_db()

        vector_db = MilvusVectorDatabase(settings.milvus)
        embed_svc = ImageEmbeddingService(settings.embedding, vector_db, meta_db)

        bm25 = BM25OcrIndex(settings.ocr)
        try:
            bm25.load()
        except Exception as e:
            log.warning(f"BM25 index not found - {e}")

        ocr_svc = OCRService(settings.ocr, vector_db, meta_db, bm25_index=bm25)

        _pipeline = OnlinePipeline(
            settings=settings,
            meta_db=meta_db,
            vector_db=vector_db,
            embed_svc=embed_svc,
            ocr_svc=ocr_svc,
            bm25_index=bm25,
        )
    finally:
        _is_initializing = False
    return _pipeline

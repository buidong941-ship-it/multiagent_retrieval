"""
FastAPI application entry point.

Provides:
    POST /api/v1/retrieve       — query-time retrieval
    POST /api/v1/index/video    — index a single video
    GET  /api/v1/health         — health check
    GET  /api/v1/frame/{id}     — get frame image
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas.schemas import (
    FrameResult,
    HealthResponse,
    IndexVideoRequest,
    IndexVideoResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from config.settings import get_settings
from database.bm25.bm25_index import BM25OcrIndex
from database.metadata.metadata_db import MetadataDatabase
from database.faiss.faiss_client import FaissVectorDatabase
from interfaces.base_interfaces import RetrievalResult
from pipelines.offline_pipeline import OfflinePipeline
from pipelines.online_pipeline import OnlinePipeline
from services.embedding.image_embedding_service import ImageEmbeddingService
from services.ocr.ocr_service import OCRService
from utils.logging_utils import get_logger, setup_logging

log = get_logger(__name__)

# Global pipeline instances (initialized at startup)
_online_pipeline: Optional[OnlinePipeline] = None
_offline_pipeline: Optional[OfflinePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services at startup, clean up at shutdown."""
    global _online_pipeline, _offline_pipeline

    setup_logging(log_level="INFO")
    settings = get_settings()

    log.info("Starting Video Retrieval API...")

    # Build shared infrastructure
    meta_db = MetadataDatabase()
    await meta_db.init_db()

    vector_db = FaissVectorDatabase(settings.faiss)

    embed_svcs = []
    for backend in settings.embedding.active_backends:
        if backend.value == "siglip2":
            collection = "clip_embeddings"
            db_id = "milvus_clip_id"
        else:
            collection = f"{backend.value}_embeddings"
            db_id = f"milvus_{backend.value}_id"
        
        svc = ImageEmbeddingService(settings.embedding, vector_db, meta_db, backend=backend, collection_name=collection, db_id_field=db_id)
        embed_svcs.append(svc)

    bm25 = BM25OcrIndex(settings.ocr)
    try:
        bm25.load()
        log.info(f"BM25 index loaded: {bm25.num_docs} documents")
    except FileNotFoundError:
        log.warning("BM25 index not found — OCR BM25 branch will return empty results")

    ocr_svc = OCRService(settings.ocr, vector_db, meta_db, bm25_index=bm25)

    # Online pipeline
    _online_pipeline = OnlinePipeline(
        settings=settings,
        meta_db=meta_db,
        vector_db=vector_db,
        embed_svcs=embed_svcs,
        ocr_svc=ocr_svc,
        bm25_index=bm25,
    )

    # Offline pipeline (for API-triggered indexing)
    _offline_pipeline = OfflinePipeline(settings, meta_db, vector_db)

    log.info("All services initialized — API ready")
    yield

    log.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Video Frame Retrieval API",
    description="Production-ready Vietnamese video retrieval system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────


@app.post(
    "/api/v1/retrieve",
    response_model=RetrievalResponse,
    summary="Retrieve video frames by Vietnamese text query",
)
async def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    """
    Retrieve the most relevant video frames for a Vietnamese text query.

    Returns a ranked list of frames with frame_id, video_id, timestamp,
    frame_path, and relevance score.
    """
    if _online_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    start_time = time.time()

    try:
        results: list[RetrievalResult] = await _online_pipeline.retrieve(
            query=request.query,
            top_k=request.top_k,
            mode=request.mode,
            use_temporal=request.use_temporal,
        )
    except Exception as exc:
        log.error(f"Retrieval failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.time() - start_time) * 1000

    # Get parsed query for response
    parsed = _online_pipeline.query_parser.parse(request.query)
    
    def ensure_list(val):
        if isinstance(val, list): return val
        if isinstance(val, str): return [val] if val else []
        if not val: return []
        return [str(val)]
        
    parsed_dict = {
        "objects": ensure_list(parsed.objects),
        "ocr": ensure_list(parsed.ocr_text),
        "actions": ensure_list(parsed.actions),
        "attributes": ensure_list(getattr(parsed, "attributes", [])),
        "translated_query": str(parsed.translated_query) if parsed.translated_query else "",
    }

    frame_results = [
        FrameResult(
            frame_id=r.frame_id,
            video_id=r.video_id,
            frame_idx=r.frame_idx,
            timestamp=r.timestamp,
            frame_path=r.frame_path,
            score=r.score,
            source=r.source,
            metadata=r.metadata,
        )
        for r in results
    ]

    return RetrievalResponse(
        query=request.query,
        parsed_query=parsed_dict,
        total_results=len(frame_results),
        results=frame_results,
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/api/v1/index/video",
    response_model=IndexVideoResponse,
    summary="Index a single video file",
)
async def index_video(request: IndexVideoRequest) -> IndexVideoResponse:
    """
    Index a single video file into the system.

    This runs the full offline pipeline:
    frame extraction → embedding → OCR → detection.
    """
    if _offline_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Video not found: {request.video_path}"
        )

    try:
        await _offline_pipeline.run_single_video(str(video_path))
        frame_count = _offline_pipeline.vector_db.count("clip_embeddings")
    except Exception as exc:
        log.error(f"Indexing failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return IndexVideoResponse(
        video_id=video_path.stem,
        frames_indexed=frame_count,
        status="success",
        message=f"Successfully indexed {video_path.name}",
    )


@app.get(
    "/api/v1/frame/{frame_id}",
    summary="Get frame image by frame_id",
)
async def get_frame(frame_id: str):
    """Return the frame image file for a given frame_id."""
    if _online_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    frame_meta = await _online_pipeline.meta_db.get_frame_async(frame_id)
    if not frame_meta:
        raise HTTPException(status_code=404, detail=f"Frame not found: {frame_id}")

    frame_path = Path(frame_meta["frame_path"])
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame image file not found")

    return FileResponse(str(frame_path), media_type="image/jpeg")


@app.get(
    "/api/v1/video/{video_id}",
    summary="Get video file by video_id",
)
async def get_video(video_id: str):
    """Return the video file for a given video_id."""
    if _online_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    video_meta = await _online_pipeline.meta_db.get_video(video_id)
    if not video_meta:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")

    video_path = Path(video_meta["video_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(str(video_path), media_type="video/mp4")


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Check the health of all services."""
    milvus_status = "ok"
    db_status = "ok"

    if _online_pipeline:
        try:
            _online_pipeline.vector_db.count("clip_embeddings")
        except Exception:
            milvus_status = "error"

        try:
            await _online_pipeline.meta_db.get_frame_async("__health_check__")
        except Exception:
            db_status = "error"

    return HealthResponse(
        status="healthy" if milvus_status == "ok" and db_status == "ok" else "degraded",
        milvus=milvus_status,
        database=db_status,
    )
@app.get("/api/debug_search")
async def debug_search():
    if _offline_pipeline is None:
        return {"error": "Pipeline not initialized"}
    try:
        import numpy as np
        client = _offline_pipeline.vector_db._client
        client.load_collection("clip_embeddings")
        res = client.search(
            collection_name="clip_embeddings",
            data=[np.zeros(1152).tolist()],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": 10}},
            limit=1,
            output_fields=["frame_id", "video_id"]
        )
        return {"raw_results": res}
    except Exception as e:
        return {"error": str(e)}

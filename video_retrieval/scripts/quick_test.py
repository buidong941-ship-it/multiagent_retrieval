"""
Script chạy thử nhanh toàn bộ pipeline với 1 video mẫu.

Chạy từng bước có thể bật/tắt bằng flag.

Cách dùng:
    cd video_retrieval

    # Kiểm tra hệ thống trước
    python scripts/check_system.py

    # Chạy thử với 1 video
    python scripts/quick_test.py --video path/to/video.mp4

    # Chỉ test retrieval (đã index sẵn)
    python scripts/quick_test.py --query "Một người đang lái xe máy"

    # Bỏ qua một số bước
    python scripts/quick_test.py --video path/to/video.mp4 --no-ocr --no-detection
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def run_test(args: argparse.Namespace) -> None:
    from config.settings import get_settings
    from utils.logging_utils import setup_logging

    setup_logging(log_level="INFO")
    settings = get_settings()

    print()
    print("=" * 60)
    print("  Video Retrieval System — Quick Test")
    print("=" * 60)

    # ── Khởi tạo database ────────────────────────────────────────
    from database.metadata.metadata_db import MetadataDatabase
    from database.milvus.milvus_client import MilvusVectorDatabase

    print("\n[1] Khởi tạo databases...")
    meta_db   = MetadataDatabase()
    vector_db = MilvusVectorDatabase(settings.milvus)

    try:
        await meta_db.init_db()
        print("  ✓ SQLite metadata DB ready")
    except Exception as e:
        print(f"  ✗ SQLite error: {e}")
        return

    try:
        vector_db.create_collection_if_not_exists("clip_embeddings")
        vector_db.create_collection_if_not_exists("ocr_embeddings")
        count = vector_db.count("clip_embeddings")
        print(f"  ✓ Milvus ready | clip_embeddings: {count} vectors")
    except Exception as e:
        print(f"  ✗ Milvus error: {e}")
        print("    → Khởi động Milvus: docker-compose up -d")
        return

    # ── Offline indexing (nếu có video) ─────────────────────────
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"\n✗ Video không tìm thấy: {args.video}")
            return

        print(f"\n[2] Indexing video: {video_path.name}")

        from pipelines.offline_pipeline import OfflinePipeline
        pipeline = OfflinePipeline(settings, meta_db, vector_db)

        t0 = time.time()
        await pipeline.run(
            video_dir=str(video_path.parent),
            run_extraction=True,
            run_embedding=not args.no_embedding,
            run_ocr=not args.no_ocr,
            run_detection=not args.no_detection,
        )
        elapsed = time.time() - t0
        count = vector_db.count("clip_embeddings")
        print(f"  ✓ Indexing xong trong {elapsed:.1f}s | {count} vectors trong Milvus")

    # ── Online retrieval ─────────────────────────────────────────
    if args.query or args.video:
        query = args.query or "người đi trên đường"

        print(f"\n[3] Retrieval query: \"{query}\"")

        from database.bm25.bm25_index import BM25OcrIndex
        from services.embedding.image_embedding_service import ImageEmbeddingService
        from services.ocr.ocr_service import OCRService
        from pipelines.online_pipeline import OnlinePipeline

        embed_svc = ImageEmbeddingService(settings.embedding, vector_db, meta_db)
        bm25      = BM25OcrIndex(settings.ocr)
        try:
            bm25.load()
        except FileNotFoundError:
            pass  # BM25 rỗng — không sao
        ocr_svc = OCRService(settings.ocr, vector_db, meta_db, bm25_index=bm25)

        online = OnlinePipeline(
            settings=settings,
            meta_db=meta_db,
            vector_db=vector_db,
            embed_svc=embed_svc,
            ocr_svc=ocr_svc,
            bm25_index=bm25,
        )

        t0 = time.time()
        try:
            results = await online.retrieve(query=query, top_k=args.top_k)
        except Exception as e:
            print(f"  ✗ Retrieval lỗi: {e}")
            return
        elapsed = (time.time() - t0) * 1000

        print(f"\n  Kết quả ({len(results)} frames | {elapsed:.0f}ms):")
        print(f"  {'#':<4} {'score':>6}  {'video_id':<20}  {'timestamp':>10}  frame_id")
        print(f"  {'-'*4} {'-'*6}  {'-'*20}  {'-'*10}  {'-'*30}")

        for i, r in enumerate(results[:10], 1):
            print(
                f"  {i:<4} {r.score:>6.3f}  {r.video_id:<20}  "
                f"{r.timestamp:>8.2f}s  {r.frame_id}"
            )

        if len(results) > 10:
            print(f"  ... và {len(results) - 10} kết quả khác")

        # Lưu kết quả ra JSON
        out_path = Path("./test_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "rank": i + 1,
                        "frame_id":   r.frame_id,
                        "video_id":   r.video_id,
                        "timestamp":  r.timestamp,
                        "score":      r.score,
                        "frame_path": r.frame_path,
                        "source":     r.source,
                    }
                    for i, r in enumerate(results)
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n  ✓ Kết quả JSON lưu tại: {out_path.resolve()}")

        # Tạo folder copy ảnh vật lý ra để dễ xem
        import shutil
        out_dir = Path("./test_results_images")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, r in enumerate(results):
            img_path = r.frame_path
            if not img_path:
                img_path = f"data/frames/{r.video_id}/{r.frame_id}.jpg"
            src = Path(img_path)
            if src.exists():
                safe_video = r.video_id.replace("/", "_").replace("\\", "_")
                # Format: 01_0.95_L21_V002_frame_12345.jpg
                dst_name = f"{i+1:02d}_{r.score:.2f}_{safe_video}_{r.frame_id}.jpg"
                shutil.copy2(src, out_dir / dst_name)

        print(f"  ✓ Đã copy {len(results)} ảnh vào thư mục: {out_dir.resolve()}")

    print()
    print("=" * 60)
    print("  Quick test hoàn thành!")
    print("=" * 60)
    print()
    print("  Bước tiếp theo:")
    print("  → Chạy API:   uvicorn api.main:app --reload --port 8000")
    print("  → Gọi API:    curl -X POST http://localhost:8000/api/v1/retrieve \\")
    print(f'               -H "Content-Type: application/json" \\')
    print(f'               -d \'{{"query": "{query}", "top_k": 20}}\'')
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick test cho Video Retrieval System")
    p.add_argument("--video",   default=None,
                   help="Đường dẫn tới video để index thử (tùy chọn)")
    p.add_argument("--query",   default=None,
                   help="Câu query tiếng Việt để thử retrieval")
    p.add_argument("--top-k",  dest="top_k", type=int, default=10,
                   help="Số kết quả trả về (mặc định: 10)")
    p.add_argument("--no-ocr",       action="store_true", help="Bỏ qua bước OCR")
    p.add_argument("--no-detection", action="store_true", help="Bỏ qua bước YOLO")
    p.add_argument("--no-embedding", action="store_true", help="Bỏ qua bước embedding")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.video and not args.query:
        print("Cần ít nhất --video hoặc --query.")
        print("Ví dụ:")
        print("  python scripts/quick_test.py --video myvideo.mp4")
        print('  python scripts/quick_test.py --query "người đi xe máy"')
        sys.exit(1)
    asyncio.run(run_test(args))

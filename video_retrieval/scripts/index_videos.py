"""
CLI script to run the offline indexing pipeline.

Usage:
    python scripts/index_videos.py --video_dir /path/to/videos
    python scripts/index_videos.py --video_dir /path/to/videos --no-detection
    python scripts/index_videos.py --video_path /path/to/single.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Prevent gRPC keepalive ping flood when using Milvus Lite with long idle times
os.environ["GRPC_KEEPALIVE_TIME_MS"] = "600000"
os.environ["GRPC_KEEPALIVE_TIMEOUT_MS"] = "600000"
# Disable Paddle PIR API to fix bug in PaddlePaddle 3.0+ (ConvertPirAttribute2RuntimeAttribute error)
os.environ["FLAGS_enable_pir_api"] = "0"

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from pipelines.offline_pipeline import OfflinePipeline
from utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video Retrieval System — Offline Indexing Pipeline"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing video files to index",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Single video file to index",
    )
    parser.add_argument(
        "--no-extraction",
        action="store_true",
        help="Skip frame extraction (use existing frames)",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Skip CLIP embedding step",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR step",
    )
    parser.add_argument(
        "--no-detection",
        action="store_true",
        help="Skip object detection step",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    setup_logging(log_level=args.log_level)

    settings = get_settings()
    pipeline = OfflinePipeline(settings)

    if args.video_path:
        print(f"Indexing single video: {args.video_path}")
        await pipeline.run_single_video(args.video_path)
    elif args.video_dir:
        print(f"Indexing directory: {args.video_dir}")
        await pipeline.run(
            video_dir=args.video_dir,
            run_extraction=not args.no_extraction,
            run_embedding=not args.no_embedding,
            run_ocr=not args.no_ocr,
            run_detection=not args.no_detection,
        )
    else:
        print("Error: provide --video_dir or --video_path")
        sys.exit(1)

    # Crucial for Milvus Lite: explicitly flush all collections before close
    print("Flushing all collections to disk...")
    try:
        pipeline.vector_db.flush("clip_embeddings")
        pipeline.vector_db.flush("action_embeddings")
    except Exception as e:
        print(f"Flush warning: {e}")

    # Crucial for Milvus Lite: close to stop background threads
    if hasattr(pipeline.vector_db, "close"):
        pipeline.vector_db.close()
        
    print("Waiting 5 seconds for database to flush to disk (Kaggle fix)...")
    import time
    time.sleep(5)

    print("Indexing complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

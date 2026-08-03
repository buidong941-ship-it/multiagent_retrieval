import asyncio
import time
import os
import sys

# Monkey patch os.rename to os.replace on Windows to fix milvus_lite WinError 183
if sys.platform == "win32":
    os.rename = os.replace
from database.milvus.milvus_client import MilvusVectorDatabase
from config.settings import get_settings
from pipelines.offline_pipeline import OfflinePipeline

async def main():
    settings = get_settings()
    pipeline = OfflinePipeline(settings)
    await pipeline.initialize()
    
    # 1. Drop the corrupted collection
    try:
        pipeline.vector_db._connect()
        pipeline.vector_db._client.drop_collection("clip_embeddings")
        print("Dropped corrupted clip_embeddings collection.")
    except Exception as e:
        print(f"Error dropping collection: {e}")
        
    # 2. Run the embedding step locally
    print("Running SigLIP2 embedding extraction...")
    await pipeline.run(
        video_dir="data/videos",
        run_extraction=False,
        run_embedding=True,
        run_ocr=False,
        run_detection=False,
        run_action=False
    )
    
    if hasattr(pipeline.vector_db, "close"):
        pipeline.vector_db.close()
        
    print("Waiting 5 seconds for flush...")
    time.sleep(5)
    print("Fix complete! You can start uvicorn now.")

if __name__ == "__main__":
    asyncio.run(main())

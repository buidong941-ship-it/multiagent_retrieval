import asyncio
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

async def run_queries():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    from config.settings import get_settings
    from utils.logging_utils import setup_logging
    from database.metadata.metadata_db import MetadataDatabase
    from database.faiss.faiss_client import FaissVectorDatabase
    from database.bm25.bm25_index import BM25OcrIndex
    from services.embedding.image_embedding_service import ImageEmbeddingService
    from services.ocr.ocr_service import OCRService
    from pipelines.online_pipeline import OnlinePipeline

    setup_logging(log_level="INFO")
    settings = get_settings()

    print("Initializing databases and services...")
    meta_db = MetadataDatabase()
    vector_db = FaissVectorDatabase(settings.faiss)
    await meta_db.init_db()

    embed_svc = ImageEmbeddingService(settings.embedding, vector_db, meta_db)
    bm25 = BM25OcrIndex(settings.ocr)
    try:
        bm25.load()
    except FileNotFoundError:
        pass
    ocr_svc = OCRService(settings.ocr, vector_db, meta_db, bm25_index=bm25)

    online = OnlinePipeline(
        settings=settings,
        meta_db=meta_db,
        vector_db=vector_db,
        embed_svc=embed_svc,
        ocr_svc=ocr_svc,
        bm25_index=bm25,
    )

    test_cases = [
        {
            "query": "Nhiều người đang đi xuống cầu thang",
            "expected": ["L21_V001_frame_008064", "L21_V001_frame_008088"]
        },
        {
            "query": "lính cứu hỏa và máy bay trực thăng",
            "expected": ["L21_V001_frame_028020", "L21_V001_frame_028098"]
        },
        {
            "query": "một người phụ nữ mặc áo cam đang được phỏng vấn",
            "expected": ["L21_V001_frame_029814", "L21_V001_frame_029844", "L21_V001_frame_029850", "L21_V001_frame_029940", "L21_V001_frame_030006", "L21_V001_frame_030060"]
        },
        {
            "query": "Một phòng triển lảm với nhiều người",
            "expected": ["L21_V001_frame_030528"]
        }
    ]

    for tc in test_cases:
        query = tc["query"]
        expected = tc["expected"]
        print(f"\n==========================================")
        print(f"Query: \"{query}\"")
        print(f"Expected frames: {expected}")
        print(f"==========================================")
        
        t0 = time.time()
        try:
            results = await online.retrieve(query=query, mode="agent", use_temporal=True)
            elapsed = time.time() - t0
            print(f"\nCompleted in {elapsed:.2f}s | Returned {len(results)} results")
            
            # Print ranks of expected frames
            frame_ranks = {r.frame_id: idx + 1 for idx, r in enumerate(results)}
            
            found_count = 0
            for exp in expected:
                rank = frame_ranks.get(exp)
                if rank:
                    print(f"  ✓ {exp} found at rank {rank}")
                    found_count += 1
                else:
                    print(f"  ✗ {exp} NOT found in results")
            
            print(f"Found {found_count}/{len(expected)} expected frames.")
            
            # Print top 5 actual results
            print("\nTop 5 results:")
            for i, r in enumerate(results[:5], 1):
                print(f"  {i}. {r.frame_id} (score={r.score:.4f}, source={r.source})")
                
        except Exception as e:
            print(f"Retrieval failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(run_queries())

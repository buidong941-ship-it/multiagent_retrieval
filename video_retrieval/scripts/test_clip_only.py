import asyncio
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

async def main():
    from config.settings import get_settings
    from utils.logging_utils import setup_logging
    from database.metadata.metadata_db import MetadataDatabase
    from database.faiss.faiss_client import FaissVectorDatabase
    from services.embedding.image_embedding_service import ImageEmbeddingService
    from retrieval.branches.clip_branch import CLIPRetrievalBranch
    from interfaces.base_interfaces import ParsedQuery

    setup_logging(log_level="INFO")
    settings = get_settings()

    meta_db = MetadataDatabase()
    vector_db = FaissVectorDatabase(settings.faiss)
    await meta_db.init_db()
    embed_svc = ImageEmbeddingService(settings.embedding, vector_db, meta_db)
    clip_branch = CLIPRetrievalBranch(settings, embed_svc, vector_db)

    test_cases = [
        {
            "query": "many people going down the stairs",
            "expected": ["L21_V001_frame_008064", "L21_V001_frame_008088"]
        },
        {
            "query": "firefighters and helicopters",
            "expected": ["L21_V001_frame_028020", "L21_V001_frame_028098"]
        },
        {
            "query": "a woman in an orange shirt being interviewed",
            "expected": ["L21_V001_frame_029814", "L21_V001_frame_029844", "L21_V001_frame_029850", "L21_V001_frame_029940", "L21_V001_frame_030006", "L21_V001_frame_030060"]
        },
        {
            "query": "a crowded exhibition room",
            "expected": ["L21_V001_frame_030528"]
        }
    ]

    for tc in test_cases:
        query = tc["query"]
        expected = tc["expected"]
        print(f"\n==========================================")
        print(f"CLIP Only Query: \"{query}\"")
        print(f"==========================================")
        
        parsed = ParsedQuery(original_query=query, translated_query=query)
        results = await clip_branch.retrieve(parsed, top_k=200)
        
        frame_ranks = {r.frame_id: idx + 1 for idx, r in enumerate(results)}
        
        for exp in expected:
            rank = frame_ranks.get(exp)
            if rank:
                print(f"  FOUND: {exp} found at rank {rank} (score={results[rank-1].score:.4f})")
            else:
                print(f"  MISSING: {exp} NOT found in top 200")
                
        print("\nTop 5 CLIP results:")
        for i, r in enumerate(results[:5], 1):
            print(f"  {i}. {r.frame_id} (score={r.score:.4f})")

if __name__ == "__main__":
    asyncio.run(main())

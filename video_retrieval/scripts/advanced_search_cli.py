"""
Advanced Search CLI Tool (Text to BEiT-3 PRF Pipeline).

Demonstrates Pseudo-Relevance Feedback:
1. Search text using SigLIP2.
2. Get Top candidates from FAISS.
3. Extract their BEiT-3 vectors to form an Image Query.
4. Run Neighbor Score Aggregation (Algorithm 2) in BEiT-3 space.
5. (Optional) Run Temporal Frame Pair Selection (Algorithm 4).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from database.faiss.faiss_client import FaissVectorDatabase
from database.metadata.metadata_db import MetadataDatabase
from retrieval.advanced_search import AdvancedSearcher
from services.embedding.image_embedding_service import ImageEmbeddingService
from utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced Video Retrieval Tool (Text to BEiT-3 PRF)")
    
    # Text Queries
    parser.add_argument("--text_query1", type=str, required=True, help="Main query text (e.g., 'a red car')")
    parser.add_argument("--text_query2", type=str, help="Secondary query text for Temporal Pair Selection")
    
    # Execution options
    parser.add_argument("--top_k", type=int, default=100, help="Initial top K from text search")
    parser.add_argument("--prf_k", type=int, default=10, help="Top K candidates to average for BEiT-3 PRF query")
    parser.add_argument("--window", type=int, default=1, help="Index-based Window size for neighbor aggregation (e.g. 1 means 1 extracted frame before and after)")
    parser.add_argument("--gap_c", type=int, default=150, help="Max frame gap for Temporal Pair")
    
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    setup_logging(log_level="INFO")

    print("Initializing databases...")
    settings = get_settings()
    
    meta_db = MetadataDatabase()
    await meta_db.init_db()
    
    vector_db = FaissVectorDatabase(settings.faiss)
    
    searcher = AdvancedSearcher(vector_db, meta_db)
    
    # We need a text encoder (SigLIP2)
    print("Loading SigLIP2 model for text encoding...")
    # Get the SigLIP2 backend configuration
    # Actually ImageEmbeddingService encapsulates it
    from config.embedding_config import EmbeddingBackend
    siglip_svc = ImageEmbeddingService(
        config=settings.embedding,
        vector_db=vector_db,
        meta_db=meta_db,
        backend=EmbeddingBackend.SIGLIP2
    )
    # The embedder inside svc handles encode_texts
    text_encoder = siglip_svc.embedder
    
    print(f"\n--- Running Text-to-BEiT3 Pipeline for Query 1 ---")
    aggregated_results, q1_beit3_emb = await searcher.text_to_beit3_pipeline(
        text_query=args.text_query1,
        text_encoder=text_encoder,
        initial_top_k=args.top_k,
        text_collection="clip_embeddings",
        target_collection="beit3_embeddings",
        prf_k=args.prf_k,
        window=args.window
    )
    
    if not aggregated_results:
        print("No results found or failed to extract BEiT-3 vectors.")
        return
        
    print(f"\nTop 5 results after Neighbor Aggregation (BEiT-3 space):")
    for i, res in enumerate(aggregated_results[:5]):
        print(f"  {i+1}. {res.frame_id} (Score: {res.score:.4f})")
        
    # Temporal Frame Pair Selection (Algorithm 4)
    if args.text_query2 and q1_beit3_emb is not None:
        print(f"\n--- Running Temporal Frame Pair Selection ---")
        
        # We need a BEiT-3 vector for Query 2.
        # We must run another PRF search to get it!
        print(f"Running PRF to generate BEiT-3 query vector for Query 2: '{args.text_query2}'")
        _, q2_beit3_emb = await searcher.text_to_beit3_pipeline(
            text_query=args.text_query2,
            text_encoder=text_encoder,
            initial_top_k=50,
            text_collection="clip_embeddings",
            target_collection="beit3_embeddings",
            prf_k=args.prf_k,
            window=args.window
        )
        
        if q2_beit3_emb is None:
            print("Failed to generate BEiT-3 query vector for Query 2.")
            return
            
        best_focal = aggregated_results[0]
        print(f"\nUsing focal frame {best_focal.frame_id} to search left/right...")
        
        best_pair = await searcher.find_best_frame_pair(
            query1_vector=q1_beit3_emb,
            query2_vector=q2_beit3_emb,
            input_frame=best_focal,
            collection="beit3_embeddings",
            gap_C=args.gap_c,
            sim_threshold=0.1,  
            search_window=300
        )
        
        if best_pair:
            left, right = best_pair
            gap = right['frame_idx'] - left['frame_idx']
            print("\nFound best temporal pair:")
            print(f"  Query 1 match (Left) : {left['frame_id']} (idx: {left['frame_idx']})")
            print(f"  Query 2 match (Right): {right['frame_id']} (idx: {right['frame_idx']})")
            print(f"  Gap (frames): {gap}")
        else:
            print("\nNo valid temporal pair found within the constraints.")


if __name__ == "__main__":
    asyncio.run(main())

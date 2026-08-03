"""
Neighbor-aware Re-ranking Service (Algorithm 2).

Algorithm 2: Neighbor Score Aggregation
    For each retrieved frame candidate:
      - Fetch its temporal neighbors.
      - Compute cosine similarity of each neighbor against the query vector.
      - Sum neighbor scores → total_neighbor_score.
      - Blend neighbor_score with the candidate's original fusion score.
      - Sort descending → neighbor-aware ranking.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np

from config.retrieval_config import RetrievalConfig
from database.metadata.metadata_db import MetadataDatabase
from database.milvus.milvus_client import MilvusVectorDatabase
from interfaces.base_interfaces import ParsedQuery, RetrievalResult
from services.embedding.image_embedding_service import ImageEmbeddingService
from utils.logging_utils import get_logger

log = get_logger(__name__)


class NeighborReranker:
    """
    Reranks candidates based on the semantic score of their temporal neighbors.
    Frames that are part of a continuous relevant scene get boosted.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        meta_db: MetadataDatabase,
        vector_db: MilvusVectorDatabase,
        embedding_svc: ImageEmbeddingService,
    ) -> None:
        self.config = config
        self.meta_db = meta_db
        self.vector_db = vector_db
        self.embedding_svc = embedding_svc

    async def rerank(
        self,
        candidates: list[RetrievalResult],
        query: ParsedQuery,
        top_k_to_rerank: Optional[int] = None,
        window: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Execute Algorithm 2.
        
        Args:
            candidates:      List of RetrievalResult from fusion.
            query:           Parsed user query.
            top_k_to_rerank: Only rerank the top N to keep latency acceptable.
            window:          How many frames before/after to consider.
        
        Returns:
            Re-ranked list of RetrievalResult.
        """
        top_k = top_k_to_rerank if top_k_to_rerank is not None else getattr(self.config, "rerank_top_k", 20)
        win = window if window is not None else getattr(self.config, "rerank_window", 2)
        
        if not candidates or win <= 0:
            return candidates

        top_candidates = candidates[:top_k]
        tail = candidates[top_k:]

        # ── Step 1: Encode query ──────────────────────────────────────────────
        search_text = query.translated_query or query.original_query
        try:
            query_vector = self.embedding_svc.encode_query(search_text)
        except Exception as e:
            log.error(f"Failed to encode query for reranking: {e}")
            return candidates

        # ── Step 2: Fetch neighbors for all top candidates ────────────────────
        log.info(f"NeighborReranker: fetching neighbors for top {len(top_candidates)}...")
        
        async def fetch_nbs(c: RetrievalResult) -> list[dict]:
            return await self.meta_db.get_neighbouring_frames_async(
                video_id=c.video_id,
                frame_idx=c.frame_idx,
                window=win,
            )

        neighbors_list = await asyncio.gather(*(fetch_nbs(c) for c in top_candidates))
        
        # Collect all unique frame_ids we need embeddings for
        needed_frame_ids = set()
        for i, c in enumerate(top_candidates):
            needed_frame_ids.add(c.frame_id)
            for nb in neighbors_list[i]:
                needed_frame_ids.add(nb["frame_id"])
                
        needed_list = list(needed_frame_ids)
        if not needed_list:
            return candidates
            
        # ── Step 3: Batch fetch embeddings from Vector DB ─────────────────────
        def fetch_embeddings_sync():
            return self.vector_db.get_embeddings_by_ids(needed_list, collection_name=self.embedding_svc._collection)

        embeddings_map = await asyncio.to_thread(fetch_embeddings_sync)
        
        # ── Step 4: Aggregate neighbor scores ─────────────────────────────────
        for i, candidate in enumerate(top_candidates):
            max_nb_score = 0.0
            
            for nb in neighbors_list[i]:
                nb_id = nb["frame_id"]
                if nb_id == candidate.frame_id:
                    continue  # exclude the frame itself from neighbor sum
                    
                nb_emb = embeddings_map.get(nb_id)
                if nb_emb is not None:
                    score = float(np.dot(query_vector, nb_emb))
                    if score > 0:
                        # Áp dụng Temporal Decay: Khung hình càng xa thì trọng số điểm lân cận càng giảm
                        distance = abs(nb.get("frame_idx", candidate.frame_idx) - candidate.frame_idx)
                        if distance == 0: distance = 1
                        
                        decayed_score = score / distance
                        if decayed_score > max_nb_score:
                            max_nb_score = decayed_score
            
            # ── Step 5: Boost scores ──────────────────────────────────────────
            # Thay vì trung bình cộng (average), ta áp dụng hệ số nhân (boost)
            # Khung hình gốc KHÔNG BAO GIỜ bị trừ điểm. Chỉ được cộng thêm nếu lân cận tốt.
            # Trọng số phụ thuộc vào độ khớp của neighbor tốt nhất, tối đa khoảng 15%.
            if max_nb_score > 0:
                boost_factor = 1.0 + (0.15 * max_nb_score)
                candidate.score = round(candidate.score * boost_factor, 5)
                candidate.source = f"{candidate.source} + neighbor_rerank"

        # ── Step 6: Sort ──────────────────────────────────────────────────────
        reranked = sorted(top_candidates, key=lambda x: x.score, reverse=True)
        
        log.info(f"NeighborReranker complete. Top score: {reranked[0].score:.4f}")
        return reranked + tail

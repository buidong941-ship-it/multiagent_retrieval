"""
Temporal Refinement module.

Design:
    After fusion, the top-K candidate frames may not be the "best"
    single frame to represent a scene. The actual most-relevant frame
    may be ±N frames away.

Algorithm (Optimized):
    1. Fetch neighbouring frames ±window from metadata DB for ALL candidates.
    2. Collect all unique frame_ids (candidates + neighbours).
    3. Batch query the vector DB to fetch embeddings for all these frame_ids.
    4. Compute cosine similarity against the query vector in numpy.
    5. Replace each candidate with its best scoring neighbour.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict

import numpy as np

from config.retrieval_config import RetrievalConfig
from database.metadata.metadata_db import MetadataDatabase
from interfaces.base_interfaces import BaseVectorDatabase, ParsedQuery, RetrievalResult
from services.embedding.image_embedding_service import ImageEmbeddingService
from utils.logging_utils import get_logger

log = get_logger(__name__)


class TemporalRefinement:
    """
    Refines retrieval results by searching neighbouring frames.
    Uses batch vector-DB embedding lookups to avoid slow CPU encoding.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        meta_db: MetadataDatabase,
        vector_db: BaseVectorDatabase,
        embedding_svc: ImageEmbeddingService,
    ) -> None:
        self.config = config
        self.meta_db = meta_db
        self.vector_db = vector_db
        self.embedding_svc = embedding_svc

    async def refine(
        self,
        candidates: list[RetrievalResult],
        query: ParsedQuery,
    ) -> list[RetrievalResult]:
        """
        Apply temporal refinement to top candidates.
        """
        if self.config.temporal_window <= 0 or not candidates:
            return candidates

        top_candidates = candidates[: self.config.temporal_top_k]
        rest = candidates[self.config.temporal_top_k :]

        search_text = query.translated_query or query.original_query
        query_vector = self.embedding_svc.encode_query(search_text)

        # 1. Fetch ALL neighbours from SQLite concurrently
        log.debug(f"Fetching neighbours for {len(top_candidates)} candidates...")
        neighbour_tasks = [
            self.meta_db.get_neighbouring_frames_async(
                video_id=c.video_id,
                frame_idx=c.frame_idx,
                window=self.config.temporal_window,
            )
            for c in top_candidates
        ]
        # returns list of list of dicts
        all_neighbours = await asyncio.gather(*neighbour_tasks)

        # 2. Collect all unique frame_ids needed (candidates + their neighbours)
        needed_frame_ids = set()
        for i, c in enumerate(top_candidates):
            needed_frame_ids.add(c.frame_id)
            for nb in all_neighbours[i]:
                needed_frame_ids.add(nb["frame_id"])

        needed_list = list(needed_frame_ids)
        if not needed_list:
            return candidates

        # 3. Fetch embeddings from VectorDB 
        embeddings_map = self.vector_db.get_embeddings_by_ids(
            needed_list=needed_list,
            collection_name=self.embedding_svc._collection
        )

        # 4. Score all neighbours and promote best
        refined: list[RetrievalResult] = []
        for i, candidate in enumerate(top_candidates):
            candidate_emb = embeddings_map.get(candidate.frame_id)
            
            # Start with candidate's own score
            best_clip_score = -1.0
            if candidate_emb is not None:
                best_clip_score = float(np.dot(query_vector, candidate_emb))

            best_nb = None
            
            for nb in all_neighbours[i]:
                nb_id = nb["frame_id"]
                nb_emb = embeddings_map.get(nb_id)
                if nb_emb is None:
                    continue
                
                score = float(np.dot(query_vector, nb_emb))
                if score > best_clip_score:
                    best_clip_score = score
                    best_nb = nb

            if best_nb is not None and best_nb["frame_id"] != candidate.frame_id:
                log.debug(
                    f"Temporal: promoted {best_nb['frame_id']} "
                    f"(CLIP: {best_clip_score:.4f}) over {candidate.frame_id}"
                )
                refined.append(
                    RetrievalResult(
                        frame_id=best_nb["frame_id"],
                        video_id=best_nb["video_id"],
                        frame_idx=best_nb["frame_idx"],
                        timestamp=best_nb["timestamp"],
                        frame_path=best_nb.get("frame_path", ""),
                        score=candidate.score,  # Keep the original fusion score to maintain RRF order
                        source=f"{candidate.source} + temporal",
                        metadata=candidate.metadata,
                    )
                )
            else:
                refined.append(candidate)

        # Sort refined results by score
        refined.sort(key=lambda r: r.score, reverse=True)

        return refined + rest

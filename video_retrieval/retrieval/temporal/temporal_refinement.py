"""
Temporal Refinement module.

Design:
    After fusion, the top-K candidate frames may not be the "best"
    single frame to represent a scene. The actual most-relevant frame
    may be ±N frames away.

Algorithm (Optimized):
    1. Fetch neighbouring frames ±window from metadata DB for ALL candidates.
    2. Collect all unique frame_ids (candidates + neighbours).
    3. Batch query Milvus to fetch embeddings for all these frame_ids.
    4. Compute cosine similarity against the query vector in numpy.
    5. Replace each candidate with its best scoring neighbour.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict

import numpy as np

from config.retrieval_config import RetrievalConfig
from database.metadata.metadata_db import MetadataDatabase
from database.milvus.milvus_client import MilvusVectorDatabase
from interfaces.base_interfaces import ParsedQuery, RetrievalResult
from services.embedding.image_embedding_service import ImageEmbeddingService
from utils.logging_utils import get_logger

log = get_logger(__name__)


class TemporalRefinement:
    """
    Refines retrieval results by searching neighbouring frames.
    Uses batch Milvus embedding lookups to avoid slow CPU encoding.
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

        # 3. Fetch embeddings from VectorDB (supports both FAISS and Milvus)
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

    async def find_best_frame_pair(
        self,
        query_1: str,
        query_2: str,
        anchors: list[RetrievalResult],
        gap_c: int = 60,
        similarity_threshold: float = 0.2
    ) -> list[dict]:
        """
        Algorithm 4: Temporal Frame Pair Selection.
        
        Given a set of anchor frames, scans backwards for query_1 and forwards
        for query_2 within gap_c frames. Finds the pair that maximizes combined similarity.
        
        Args:
            query_1: Text query that should happen FIRST.
            query_2: Text query that should happen SECOND.
            anchors: List of candidate frames to anchor the search around.
            gap_c: Maximum frame gap (e.g., 60 frames = 2 seconds at 30fps).
            similarity_threshold: Minimum cosine similarity to consider a frame relevant.
            
        Returns:
            List of dictionaries containing the best pair for each anchor.
        """
        if not anchors:
            return []
            
        q1_vec = self.embedding_svc.encode_query(query_1)
        q2_vec = self.embedding_svc.encode_query(query_2)
        
        # 1. Fetch neighbors for all anchors up to gap_c
        neighbor_tasks = [
            self.meta_db.get_neighbouring_frames_async(
                video_id=a.video_id,
                frame_idx=a.frame_idx,
                window=gap_c,
            )
            for a in anchors
        ]
        all_neighbors = await asyncio.gather(*neighbor_tasks)
        
        # 2. Collect frame IDs and batch fetch embeddings
        needed_frame_ids = set()
        for i, a in enumerate(anchors):
            needed_frame_ids.add(a.frame_id)
            for nb in all_neighbors[i]:
                needed_frame_ids.add(nb["frame_id"])
                
        needed_list = list(needed_frame_ids)
        if not needed_list:
            return []
        # 2. Fetch embeddings for all needed frames
        embeddings_map = self.vector_db.get_embeddings_by_ids(
            needed_list=needed_list,
            collection_name=self.embedding_svc._collection
        )
        
        # 3. Find best pair for each anchor
        best_pairs = []
        
        for i, anchor in enumerate(anchors):
            anchor_idx = anchor.frame_idx
            neighbors = all_neighbors[i]
            
            # Left scan (query 1)
            left_frames = [nb for nb in neighbors if nb["frame_idx"] <= anchor_idx]
            best_left = None
            best_left_score = similarity_threshold
            
            for lf in left_frames:
                emb = embeddings_map.get(lf["frame_id"])
                if emb is not None:
                    sim = float(np.dot(q1_vec, emb))
                    if sim > best_left_score:
                        best_left_score = sim
                        best_left = lf
                        
            # Right scan (query 2)
            right_frames = [nb for nb in neighbors if nb["frame_idx"] >= anchor_idx]
            best_right = None
            best_right_score = similarity_threshold
            
            for rf in right_frames:
                emb = embeddings_map.get(rf["frame_id"])
                if emb is not None:
                    sim = float(np.dot(q2_vec, emb))
                    if sim > best_right_score:
                        best_right_score = sim
                        best_right = rf
                        
            if best_left and best_right:
                # Ensure temporal constraint is met
                actual_gap = best_right["frame_idx"] - best_left["frame_idx"]
                if 0 <= actual_gap <= gap_c:
                    best_pairs.append({
                        "anchor_id": anchor.frame_id,
                        "frame_1": best_left,
                        "score_1": best_left_score,
                        "frame_2": best_right,
                        "score_2": best_right_score,
                        "combined_score": best_left_score + best_right_score,
                        "gap": actual_gap
                    })
                    
        # Sort by combined score descending
        best_pairs.sort(key=lambda x: x["combined_score"], reverse=True)
        return best_pairs

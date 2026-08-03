"""
Advanced Search Module.

Implements:
1. Ensemble Search (Algorithm 3)
2. Neighbor Score Aggregation (Algorithm 2)
3. Temporal Frame Pair Selection (Algorithm 4)
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from database.faiss.faiss_client import FaissVectorDatabase
from database.metadata.metadata_db import MetadataDatabase
from interfaces.base_interfaces import RetrievalResult
from utils.logging_utils import get_logger

log = get_logger(__name__)


class RetrievalItem:
    def __init__(self, frame_id: str, video_id: str, frame_idx: int, score: float, metadata: dict = None):
        self.frame_id = frame_id
        self.video_id = video_id
        self.frame_idx = frame_idx
        self.score = score
        self.metadata = metadata or {}


class AdvancedSearcher:
    """
    Implements advanced retrieval algorithms over FAISS and SQLite.
    """

    def __init__(self, vector_db: FaissVectorDatabase, meta_db: MetadataDatabase):
        self.vector_db = vector_db
        self.meta_db = meta_db
        self._video_frames_cache: Dict[str, List[dict]] = {}

    def get_vector_for_frame(self, frame_id: str, collection: str) -> Optional[np.ndarray]:
        """
        Fetch the raw vector for a frame from FAISS memory.
        """
        pk = self.vector_db.frame_id_to_id.get(collection, {}).get(frame_id)
        if pk is None:
            return None
        return self.vector_db.vectors_store.get(collection, {}).get(pk)

    async def get_neighbors(self, video_id: str, frame_idx: int, window: int = 20) -> list[dict]:
        """
        Helper: Get temporal neighbors in a window using SQLite.
        """
        return await self.meta_db.get_neighbouring_frames_async(video_id, frame_idx, window)

    async def get_neighbors_by_count(self, video_id: str, frame_idx: int, count: int = 1) -> list[dict]:
        """
        Helper: Get exact N extracted frames before and after the focal frame (Index-based window).
        """
        if video_id not in self._video_frames_cache:
            frames = await self.meta_db.get_frames_by_video_async(video_id)
            self._video_frames_cache[video_id] = frames
            
        all_frames = self._video_frames_cache[video_id]
        
        # Find the index of the focal frame
        idx = next((i for i, f in enumerate(all_frames) if f["frame_idx"] == frame_idx), None)
        if idx is None:
            return []
            
        start = max(0, idx - count)
        end = min(len(all_frames), idx + count + 1)
        return all_frames[start:end]

    def compute_score(self, neighbor_frame_id: str, query_vector: np.ndarray, collection: str) -> Optional[float]:
        """
        Compute similarity score for a neighbor frame using cached vectors.
        """
        emb = self.get_vector_for_frame(neighbor_frame_id, collection)
        if emb is None:
            return None
            
        score = float(np.dot(emb, query_vector))
        return score

    async def siglip_jina_pipeline(
        self,
        candidates: list[RetrievalResult],
        text_query: str,
        jina_text_encoder: Any,
        target_collection: str = "jina_embeddings",
        window: int = 1
    ) -> Tuple[list[RetrievalItem], Optional[np.ndarray]]:
        """
        Algorithm 1: Advanced Search Pipeline using SigLIP + Jina.
        Uses a list of pre-retrieved candidates (e.g. from CLIP semantic pipeline) 
        instead of doing raw text search, then reranks them using Jina CLIP v2 text vector.
        
        Returns:
            - Ranked list of RetrievalItem after ensemble scoring.
            - The Jina text vector used (for Algorithm 4).
        """
        if not candidates:
            return [], None
            
        # 1. Chuyển text thành vector bằng Jina Text Encoder
        log.info("[Jina] Bước 1: Đang encode text query...")
        try:
            loop = asyncio.get_running_loop()
            q_text_vec = await loop.run_in_executor(
                None, lambda: jina_text_encoder.encode_texts([text_query])[0]
            )
            log.info("[Jina] Encode text query thành công!")
        except Exception:
            log.exception("[Jina] FAILED to encode text query.")
            return [], None
            
        # 2. Chuyển candidates ban đầu sang RetrievalItem để Algorithm 2 sử dụng
        initial_items = [
            RetrievalItem(
                frame_id=c.frame_id,
                video_id=c.video_id,
                frame_idx=c.frame_idx,
                score=c.score,
                metadata=c.metadata
            )
            for c in candidates
        ]

        # 3. Neighbor Score Aggregation dùng Jina query trên Jina embeddings
        jina_cached = len(self.vector_db.vectors_store.get(target_collection, {}))
        log.info(f"[Jina] Bước 2: Re-rank {len(initial_items)} frames | {jina_cached} Jina vectors có trong bộ nhớ")
        try:
            aggregated = await self.aggregate_neighbor_scores(
                indices=initial_items,
                query_vectors={target_collection: q_text_vec},
                collection=target_collection,
                window=window
            )
            log.info(f"[Jina] Bước 2 xong. Scored {len(aggregated)} frames.")
        except Exception:
            log.exception("[Jina] Neighbor score aggregation FAILED.")
            return [], None

        return aggregated, q_text_vec

    async def aggregate_neighbor_scores(
        self,
        indices: list[RetrievalItem],
        query_vectors: dict[str, np.ndarray],
        collection: str,
        window: int = 20
    ) -> list[RetrievalItem]:
        """
        Algorithm 2: Neighbor Score Aggregation
        """
        if collection not in query_vectors:
            log.warning(f"No query vector for {collection}, returning unaggregated scores.")
            return indices
            
        q_vec = query_vectors[collection]
        aggregated_scores = {}
        clip_scores = {}
        items_dict = {}
        
        for item in indices:
            frame_id = item.frame_id
            video_id = item.video_id
            frame_idx = item.frame_idx
            
            # Save original CLIP score for ensemble
            clip_scores[frame_id] = item.score
            
            neighbors = await self.get_neighbors_by_count(video_id, frame_idx, count=window)
            valid_scores = []
            
            for n in neighbors:
                n_frame_id = n["frame_id"]
                score = self.compute_score(n_frame_id, q_vec, collection)
                if score is not None:
                    valid_scores.append(score)
                    
            # Use max pooling or average pooling instead of raw sum to avoid bias towards longer scenes
            if valid_scores:
                aggregated_scores[frame_id] = sum(valid_scores) / len(valid_scores)
            else:
                aggregated_scores[frame_id] = 0.0
                
            items_dict[frame_id] = item
            
        # Algorithm 3: Ensemble Search (Normalize & Combine)
        clip_vals = list(clip_scores.values()) if clip_scores else [0]
        beit3_vals = list(aggregated_scores.values()) if aggregated_scores else [0]
        
        min_c, max_c = min(clip_vals), max(clip_vals)
        min_b, max_b = min(beit3_vals), max(beit3_vals)
        
        if max_c - min_c < 1e-6: max_c = min_c + 1.0
        if max_b - min_b < 1e-6: max_b = min_b + 1.0
        
        ensemble_scores = {}
        for fid in aggregated_scores:
            norm_clip = (clip_scores[fid] - min_c) / (max_c - min_c)
            norm_beit3 = (aggregated_scores[fid] - min_b) / (max_b - min_b)
            # Tinh chỉnh: Ưu tiên BEiT-3 (thông tin visual temporal) hơn một chút
            ensemble_scores[fid] = 0.4 * norm_clip + 0.6 * norm_beit3
            
        # Sort descending by ensemble score
        ranked = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for fid, ensemble_score in ranked:
            item = items_dict[fid]
            # Save the ensemble score to item.score, and store original in metadata
            item.metadata["clip_score"] = clip_scores[fid]
            item.metadata["beit3_score"] = aggregated_scores[fid]
            item.score = ensemble_score
            results.append(item)
            
        return results

    async def find_best_frame_pair(
        self,
        query1_vector: np.ndarray,
        query2_vector: np.ndarray,
        input_frame: RetrievalItem,
        collection: str,
        gap_C: int = 1000,
        sim_threshold: float = 0.15,
        search_window: int = 1200
    ) -> Optional[Tuple[Tuple[dict, float], Tuple[dict, float]]]:
        """
        Algorithm 4: Temporal Frame Pair Selection.
        
        input_frame is the focal frame from Query 1 pipeline.
        We keep it as the 'left' anchor and scan FORWARD in time for the best Query 2 match.
        """
        video_id = input_frame.video_id
        focal_idx = input_frame.frame_idx

        neighbors = await self.get_neighbors(video_id, focal_idx, window=search_window)
        neighbors.sort(key=lambda x: x["frame_idx"])

        focal_pos = next((i for i, n in enumerate(neighbors) if n["frame_id"] == input_frame.frame_id), None)
        if focal_pos is None:
            return None

        # Score the focal frame itself against Query 1
        focal_score_q1 = self.compute_score(input_frame.frame_id, query1_vector, collection)
        if focal_score_q1 is None or focal_score_q1 < sim_threshold:
            return None

        # Scan RIGHT (forward in time) for best Query 2 match
        best_right = None
        best_right_score = -1.0

        for i in range(focal_pos + 1, len(neighbors)):
            n = neighbors[i]
            gap = n["frame_idx"] - focal_idx
            if gap > gap_C:
                break  # Too far ahead, stop scanning
            if gap <= 0:
                continue

            score = self.compute_score(n["frame_id"], query2_vector, collection)
            if score is not None and score >= sim_threshold and score > best_right_score:
                best_right_score = score
                best_right = (n, score)

        if best_right is None:
            return None

        left = ({"frame_id": input_frame.frame_id, "frame_idx": focal_idx, "video_id": video_id}, focal_score_q1)
        return left, best_right

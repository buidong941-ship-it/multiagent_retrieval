"""
Branch A: CLIP (SigLIP2) vector retrieval.

Uses the SigLIP2 text encoder to embed the (translated) query,
then performs ANN search in Milvus clip_embeddings collection.
Returns top-K frames with cosine similarity scores.
"""

from __future__ import annotations

import numpy as np

from config.database_config import MilvusConfig
from config.retrieval_config import RetrievalConfig
from database.milvus.milvus_client import MilvusVectorDatabase
from interfaces.base_interfaces import (
    BaseRetrievalBranch,
    ParsedQuery,
    RetrievalResult,
)
from services.embedding.image_embedding_service import ImageEmbeddingService
from utils.logging_utils import get_logger

log = get_logger(__name__)


class CLIPRetrievalBranch(BaseRetrievalBranch):
    """
    Retrieval Branch A: SigLIP2 text → Milvus ANN search.

    Process:
        1. Use translated_query (English) for better CLIP alignment.
        2. Encode with SigLIP2 text encoder → 1152-dim vector.
        3. ANN search in Milvus clip_embeddings.
        4. Return top_k results with normalized cosine scores.

    Attributes:
        config:         RetrievalConfig.
        embedding_svc:  ImageEmbeddingService (provides text encoder).
        vector_db:      MilvusVectorDatabase.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        embed_svcs: list[ImageEmbeddingService],
        vector_db: MilvusVectorDatabase,
    ) -> None:
        self.config = config
        self.embed_svcs = embed_svcs
        self.vector_db = vector_db

    @property
    def branch_name(self) -> str:
        return "visual"

    def _rrf_fuse(self, lists_of_results: list[list[RetrievalResult]], top_k: int, k_param: int = 60) -> list[RetrievalResult]:
        scores = {}
        items = {}
        for res_list in lists_of_results:
            for rank, item in enumerate(res_list):
                frame_id = item.frame_id
                if frame_id not in items:
                    items[frame_id] = item
                    scores[frame_id] = 0.0
                scores[frame_id] += 1.0 / (k_param + rank + 1)
        
        fused = []
        for frame_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            item = items[frame_id]
            # Optionally normalize score between 0 and 1 here, but raw RRF is fine.
            item.score = score
            fused.append(item)
        return fused

    async def retrieve(
        self,
        query: ParsedQuery,
        top_k: int,
        mode: str = "fusion"
    ) -> list[RetrievalResult]:
        """
        Execute Visual vector search across all active backends and fuse via RRF.

        Args:
            query:  Parsed Vietnamese query.
            top_k:  Number of results to return.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        if mode in ["siglip_jina_parallel", "fusion", "agent"]:
            queries_to_search = query.expanded_queries if query.expanded_queries else []
            if not queries_to_search:
                queries_to_search = [query.translated_query or query.original_query]
        else:
            # Chế độ cũ (Sequence/PRF hoặc CLIP thuần) chỉ dùng 1 câu dịch duy nhất
            queries_to_search = [query.translated_query or query.original_query]

        log.debug(f"Visual search with {len(queries_to_search)} queries (mode: {mode}).")

        all_results = []
        
        for q_text in queries_to_search:
            is_vietnamese = (q_text == query.original_query)
            
            for svc in self.embed_svcs:
                if svc._collection == "jina_embeddings" and mode not in ["siglip_jina_parallel", "fusion", "agent"]:
                    log.debug(f"Skipping jina_embeddings because mode is '{mode}'")
                    continue

                # Bỏ qua SigLIP nếu đây là câu tiếng Việt gốc (để tránh nhiễu vector rác)
                # Chỉ bỏ qua nếu chúng ta có nhiều hơn 1 câu query (đảm bảo không bao giờ bị trắng kết quả)
                if svc._collection == "clip_embeddings" and is_vietnamese and len(queries_to_search) > 1:
                    continue
                    
                # Không ghép "a photo of" vào câu tiếng Việt vì nó làm mô hình đa ngôn ngữ (Jina) bị lú
                search_text = q_text if is_vietnamese else f"a photo of {q_text}"

                try:
                    query_vector = svc.encode_query(search_text)
                except NotImplementedError:
                    log.debug(f"Skipping {svc._collection} as it does not support text encoding.")
                    continue
                    
                hits = self.vector_db.search(
                    collection_name=svc._collection,
                    query_vector=query_vector,
                    top_k=top_k,
                )

                branch_results = []
                backend_name = svc._collection.replace("_embeddings", "")
                for hit in hits:
                    score = 1.0 - float(hit.get("score", 0.0))
                    branch_results.append(
                        RetrievalResult(
                            frame_id=hit["frame_id"],
                            video_id=hit["video_id"],
                            frame_idx=int(hit.get("frame_idx", 0)),
                            timestamp=float(hit.get("timestamp", 0.0)),
                            frame_path="",
                            score=score,
                            source=f"{self.branch_name}_{backend_name}",
                        )
                    )
                
                branch_results.sort(key=lambda r: r.score, reverse=True)
                if branch_results:
                    all_results.append(branch_results)

        if len(all_results) == 1:
            results = all_results[0][:top_k]
        elif len(all_results) > 1:
            results = self._rrf_fuse(all_results, top_k)
        else:
            results = []

        log.info(f"Visual branch: {len(results)} fused results for top_k={top_k}")
        return results

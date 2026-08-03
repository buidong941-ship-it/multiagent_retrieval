"""
Action retrieval branch.

Uses mean-pooled CLIP embeddings to retrieve actions.
"""

from typing import Any

from database.milvus.milvus_client import MilvusVectorDatabase

from interfaces.base_interfaces import BaseRetrievalBranch, ParsedQuery, RetrievalResult
from services.embedding.image_embedding_service import ImageEmbeddingService
from utils.logging_utils import get_logger

log = get_logger(__name__)


class ActionBranch(BaseRetrievalBranch):
    """
    Retrieves frames using action embeddings (mean-pooled CLIP).
    """

    def __init__(
        self,
        embed_svc: ImageEmbeddingService,
        vector_db: MilvusVectorDatabase,
        collection_name: str = "action_embeddings",
    ) -> None:
        self.embed_svc = embed_svc
        self.vector_db = vector_db
        self.collection_name = collection_name
        self._branch_name = "action"

    @property
    def branch_name(self) -> str:
        return self._branch_name

    async def retrieve(
        self,
        query: ParsedQuery,
        top_k: int = 100,
    ) -> list[RetrievalResult]:
        """
        Search action collection using CLIP text encoder.
        """
        text = query.original_query
        if not text:
            return []

        try:
            query_vector = self.embed_svc.encode_query(text)
        except Exception as exc:
            log.error(f"ActionBranch encoding failed: {exc}")
            return []

        try:
            results = self.vector_db.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                top_k=top_k,
            )
            
            # Map Milvus hits to RetrievalResult
            ret = []
            for hit in results:
                ret.append(
                    RetrievalResult(
                        frame_id=hit.get("frame_id", ""),
                        video_id=hit.get("video_id", ""),
                        frame_idx=hit.get("frame_idx", 0),
                        timestamp=hit.get("timestamp", 0.0),
                        frame_path="",
                        score=hit.get("score", 0.0),
                        source=self.branch_name,
                    )
                )
            return ret
        except Exception as exc:
            log.error(f"ActionBranch search failed: {exc}")
            return []

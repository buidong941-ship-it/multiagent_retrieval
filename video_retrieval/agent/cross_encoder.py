"""
Phase 8: Cross Encoder Reranker
Đánh giá lại candidate kết hợp Image, Text (OCR), Metadata và Query.
"""
from typing import List, Dict, Any
from agent.candidate_manager import Candidate
from utils.logging_utils import get_logger

log = get_logger(__name__)

class CrossEncoderReranker:
    def __init__(self, model):
        self.model = model

    def rerank(self, candidates: List[Candidate], parsed_query: Dict[str, Any]) -> List[Candidate]:
        log.info(f"CrossEncoderReranker reranking {len(candidates)} candidates.")
        
        # Trong thực tế, gọi CrossEncoder VLM hoặc BGE-Reranker ở đây
        # Đây là mock implementation
        for c in candidates:
            # Ví dụ: cộng thêm điểm lexical overlap hoặc cross attention score
            c.metadata['rerank_score'] = c.metadata.get('rrf_score', c.max_score)
            
        reranked = sorted(candidates, key=lambda c: c.metadata.get('rerank_score', 0), reverse=True)
        return reranked

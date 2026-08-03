"""
Phase 7: Fusion Engine
Kết hợp điểm số từ nhiều nhánh bằng RRF (Reciprocal Rank Fusion).
"""
from typing import List, Dict
from agent.candidate_manager import Candidate
from utils.logging_utils import get_logger

log = get_logger(__name__)

class FusionEngine:
    def __init__(self, weights: Dict[str, float] = None, k: int = 60):
        self.weights = weights or {}
        self.k = k

    def fuse(self, candidates: List[Candidate]) -> List[Candidate]:
        log.info(f"FusionEngine processing {len(candidates)} candidates.")
        
        # Sort candidates per source to get ranks
        sources = set()
        for c in candidates:
            sources.update(c.scores.keys())
            
        ranks = {s: {} for s in sources}
        for s in sources:
            sorted_for_source = sorted(
                [c for c in candidates if s in c.scores],
                key=lambda x: x.scores[s],
                reverse=True
            )
            for rank, c in enumerate(sorted_for_source, start=1):
                ranks[s][c.frame_id] = rank

        # Calculate RRF score
        for c in candidates:
            rrf_score = 0.0
            for s in c.scores:
                weight = self.weights.get(s, 1.0)
                rank = ranks[s].get(c.frame_id, 1000)
                rrf_score += weight / (self.k + rank)
            
            c.metadata['rrf_score'] = rrf_score

        # Sort all by RRF score
        fused = sorted(candidates, key=lambda c: c.metadata.get('rrf_score', 0), reverse=True)
        return fused

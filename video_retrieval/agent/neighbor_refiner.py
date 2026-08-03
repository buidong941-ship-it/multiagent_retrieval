"""
Phase 10: Neighbor Refiner
Đánh giá lại các frame lân cận quanh candidate tốt nhất để tìm frame cực trị.
"""
from typing import Dict, Any
from agent.candidate_manager import Candidate
from utils.logging_utils import get_logger

log = get_logger(__name__)

class NeighborRefiner:
    def __init__(self, meta_db):
        self.meta_db = meta_db

    def refine(self, best_candidate: Candidate, parsed_query: Dict[str, Any]) -> Candidate:
        log.info(f"NeighborRefiner refining {best_candidate.frame_id}")
        
        # Trong thực tế:
        # Lấy các frames +- 2 giây quanh best_candidate từ meta_db
        # Đánh giá lại bằng visual embed hoặc OCR (tùy query)
        # Trả về Candidate mới nếu tốt hơn
        
        return best_candidate

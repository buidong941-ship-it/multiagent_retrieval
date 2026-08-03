"""
OCR Search Tool
Tìm kiếm văn bản trong video (BM25, Dense).
"""
from typing import List, Dict, Any
from .base import BaseTool
from interfaces.base_interfaces import ParsedQuery
from utils.logging_utils import get_logger

log = get_logger(__name__)

class OCRSearchTool(BaseTool):
    def __init__(self, branch, top_k: int = 50):
        self.branch = branch
        self.top_k = top_k

    async def forward(self, query: str) -> List[Dict[str, Any]]:
        log.info(f"OCRSearchTool executing for query: '{query}'")
        p = ParsedQuery(original_query=query, ocr_text=[query])
        try:
            results = await self.branch.retrieve(p, self.top_k)
        except Exception as e:
            log.error(f"OCRSearchTool failed: {e}")
            return []
            
        output = []
        if not results:
            return output
            
        max_score = getattr(results[0], "score", 0.0)
        min_score = getattr(results[-1], "score", 0.0)
        
        for r in results:
            score = getattr(r, "score", 0.0)
            
            # Min-Max Normalization: Đưa điểm RRF siêu nhỏ về thang [0, 1]
            if max_score > min_score:
                norm_score = (score - min_score) / (max_score - min_score)
            else:
                norm_score = 1.0
                
            # Bonus nhẹ để nó luôn thắng CLIP (Cosine của CLIP thường < 0.4)
            # Nếu được rank 1 bên OCR, điểm norm sẽ là 1.0 (Ăn đứt mọi thể loại CLIP)
            output.append({
                "frame_id": r.frame_id,
                "video_id": r.video_id,
                "timestamp": r.timestamp,
                "score": norm_score,
                "meta": {
                    "source": "OCRSearch",
                    "frame_idx": r.frame_idx
                }
            })
        return output

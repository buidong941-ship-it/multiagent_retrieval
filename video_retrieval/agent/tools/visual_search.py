"""
Visual Search Tool
Sử dụng cho Visual Search, Scene Search, Attribute Search, Action Search.
Tùy vào branch mà backend sẽ là CLIP/SigLIP.
"""
from typing import List, Dict, Any
from .base import BaseTool
from interfaces.base_interfaces import ParsedQuery
from utils.logging_utils import get_logger

log = get_logger(__name__)

class VisualSearchTool(BaseTool):
    def __init__(self, branch, top_k: int = 50):
        self.branch = branch
        self.top_k = top_k

    async def forward(self, query: str) -> List[Dict[str, Any]]:
        log.info(f"VisualSearchTool executing for query: '{query}'")
        p = ParsedQuery(original_query=query, translated_query=query)
        try:
            results = await self.branch.retrieve(p, self.top_k)
        except Exception as e:
            log.error(f"VisualSearchTool failed: {e}")
            return []
            
        output = []
        for r in results:
            score = getattr(r, "score", 0.0)
            output.append({
                "frame_id": r.frame_id,
                "video_id": r.video_id,
                "timestamp": r.timestamp,
                "score": score,
                "meta": {
                    "source": "VisualSearch",
                    "frame_idx": r.frame_idx
                }
            })
        return output

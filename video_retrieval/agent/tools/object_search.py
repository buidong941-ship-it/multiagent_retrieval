"""
Object Search Tool
Tìm kiếm đối tượng sử dụng YOLO metadata (hoặc open-vocabulary).
"""
from typing import List, Dict, Any
from .base import BaseTool
from interfaces.base_interfaces import ParsedQuery
from utils.logging_utils import get_logger

log = get_logger(__name__)

class ObjectSearchTool(BaseTool):
    def __init__(self, branch, top_k: int = 50):
        self.branch = branch
        self.top_k = top_k

    async def forward(self, query: str) -> List[Dict[str, Any]]:
        log.info(f"ObjectSearchTool executing for query: '{query}'")
        p = ParsedQuery(original_query=query, objects=[query])
        try:
            results = await self.branch.retrieve(p, self.top_k)
        except Exception as e:
            log.error(f"ObjectSearchTool failed: {e}")
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
                    "source": "ObjectSearch",
                    "frame_idx": r.frame_idx
                }
            })
        return output

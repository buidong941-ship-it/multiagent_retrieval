from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseTool(ABC):
    @abstractmethod
    async def forward(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Thực thi tool. 
        Trả về danh sách dictionary, mỗi dict chứa ít nhất:
        - frame_id: str
        - score: float
        - meta: dict (optional metadata)
        """
        pass

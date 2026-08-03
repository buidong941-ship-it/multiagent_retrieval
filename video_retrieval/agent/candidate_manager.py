"""
Phase 6: Candidate Manager
Tầng trung gian giữa Retrieval và Fusion. Thu thập, chuẩn hóa, lưu trữ metadata 
và tạo Candidate Pool thống nhất.
"""
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class Candidate:
    frame_id: str
    video_id: str
    timestamp: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict) # Lưu điểm từ các branch khác nhau (clip, ocr, object...)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def max_score(self) -> float:
        if not self.scores:
            return 0.0
        return max(self.scores.values())

    @property
    def best_source(self) -> str:
        if not self.scores:
            return ""
        return max(self.scores, key=self.scores.get)

class CandidateManager:
    def __init__(self):
        self.candidates: Dict[str, Candidate] = {}

    def add_or_update(self, frame_id: str, source: str, score: float, video_id: str = "", timestamp: float = 0.0, meta: Dict = None):
        if not video_id:
            # parse from frame_id if possible e.g. L21_V001_frame_000390 -> L21_V001
            video_id = frame_id.split("_frame_")[0] if "_frame_" in frame_id else ""
            
        if frame_id not in self.candidates:
            self.candidates[frame_id] = Candidate(
                frame_id=frame_id, 
                video_id=video_id, 
                timestamp=timestamp
            )
        
        # update score from this source, keeping the max if source already exists
        current_score = self.candidates[frame_id].scores.get(source, 0.0)
        if score > current_score:
            self.candidates[frame_id].scores[source] = score
            
        if meta:
            self.candidates[frame_id].metadata.update(meta)

    def get_all(self) -> List[Candidate]:
        return list(self.candidates.values())

    def clear(self):
        self.candidates.clear()

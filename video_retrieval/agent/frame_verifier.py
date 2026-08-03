"""
Phase 9: Frame Verifier
Xác thực lại candidate bằng Vision Language Model (VD: Qwen-VL, GPT-4V).
"""
from typing import List
from agent.candidate_manager import Candidate
from utils.logging_utils import get_logger

log = get_logger(__name__)

class FrameVerifier:
    def __init__(self, vlm_client):
        self.vlm = vlm_client

    def verify(self, candidates: List[Candidate], query: str) -> List[Candidate]:
        log.info(f"FrameVerifier checking {len(candidates)} candidates.")
        
        verified = []
        for c in candidates:
            # Trong thực tế: Tải ảnh từ c.frame_id, đưa cho VLM kèm prompt:
            # "Does this frame show: {query}? Answer Yes/No and confidence."
            
            # Đây là mock implementation, cho qua toàn bộ
            c.metadata['verified'] = True
            verified.append(c)
            
        return verified

"""
Phase 12: Response Generator
Dùng LLM để tổng hợp kết quả và trả lời người dùng dưới dạng tự nhiên.
"""
from typing import List, Dict, Any
from utils.logging_utils import get_logger

log = get_logger(__name__)

class ResponseGenerator:
    def __init__(self, llm_client, model_name: str):
        self.llm = llm_client
        self.model_name = model_name

    def generate(self, original_query: str, final_frame_ids: List[str], parsed_query: Dict[str, Any]) -> str:
        log.info("ResponseGenerator creating final response")
        
        if not final_frame_ids:
            return "Tôi không tìm thấy bất kỳ frame nào phù hợp với yêu cầu của bạn."
            
        # Xây dựng prompt sinh câu trả lời
        frames_str = ", ".join(final_frame_ids[:5])
        prompt = f"""You are a Video Retrieval Assistant.
The user asked: "{original_query}"
The system found the best matching frames: {frames_str}.

Write a short, natural response in Vietnamese telling the user what was found.
Do NOT mention internal tools or scores. Just give the frame IDs and a brief summary of what they likely contain based on the query.
"""
        try:
            if hasattr(self.llm, "generate"):
                response = self.llm.generate(prompt=prompt, model=self.model_name)
            elif callable(self.llm):
                response = self.llm(prompt)
            else:
                response = f"Tôi đã tìm thấy các frame sau: {frames_str}"
            
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            log.error(f"Response generation failed: {e}")
            return f"Đây là những kết quả tốt nhất tôi tìm được: {frames_str}"

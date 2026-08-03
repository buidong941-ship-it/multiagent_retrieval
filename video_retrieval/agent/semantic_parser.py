"""
Phase 1: Semantic Query Parser
Sử dụng LLM để phân tích truy vấn thành biểu diễn ngữ nghĩa có cấu trúc.
"""
import json
from typing import Dict, Any, List
from utils.logging_utils import get_logger

log = get_logger(__name__)

class SemanticParser:
    def __init__(self, llm_client, model_name: str):
        """
        llm_client: Client để gọi API (ví dụ OpenAI client wrapper hoặc hàm _call_ollama)
        """
        self.llm = llm_client
        self.model_name = model_name

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Chuyển truy vấn thành JSON schema.
        """
        log.info(f"Parsing query: {query}")
        prompt = f"""You are a Semantic Query Parser for Video Retrieval.
Parse the following Vietnamese query into a strict JSON structure.
Translate keywords to English for retrieval.

Structure:
{{
  "objects": ["object1", "object2"],
  "attributes": ["color", "size"],
  "ocr_text": ["exact text"],
  "scene": "scene description",
  "events": [
     {{"description": "event 1 English", "type": "primary"}},
     {{"description": "event 2 English", "type": "secondary"}}
  ],
  "temporal_relation": "none|after|before",
  "spatial_relations": ["left", "right"],
  "translated_query": "English caption of the primary event for visual search"
}}

Query: "{query}"
Respond ONLY with the JSON object.
"""
        try:
            # Tùy thuộc vào implement thực tế của llm_client.
            # Dưới đây giả sử llm_client có phương thức sinh text cơ bản.
            if hasattr(self.llm, "generate"):
                response = self.llm.generate(prompt=prompt, model=self.model_name)
            elif callable(self.llm):
                response = self.llm(prompt)
            else:
                response = "{}"
            
            # Trích xuất JSON 
            if isinstance(response, str):
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1:
                    json_str = response[start:end+1]
                    return json.loads(json_str)
            elif isinstance(response, dict):
                return response
                
        except Exception as e:
            log.error(f"Semantic parse failed: {e}")
            
        # Fallback
        return {
            "objects": [], "attributes": [], "ocr_text": [], "scene": "",
            "events": [{"description": query, "type": "primary"}],
            "temporal_relation": "none", "spatial_relations": [],
            "translated_query": query
        }

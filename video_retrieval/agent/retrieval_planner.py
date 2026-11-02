"""
Phase 2: Retrieval Planner
Nhận Structured Query từ Semantic Parser và sinh ra Execution Graph / danh sách Tool.
Hoàn toàn Rule-based.
"""
from typing import Dict, Any, List
from utils.logging_utils import get_logger

log = get_logger(__name__)

class RetrievalPlanner:
    def __init__(self):
        pass

    def create_plan(self, parsed_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Dựa vào rules để quyết định gọi tool nào.
        Trả về danh sách các action cần thực thi.
        """
        plan = []
        log.info(f"Creating plan for query: {parsed_query.get('translated_query')}")
        
        # 1. Luôn có Visual Search cho translated_query (hoặc event primary)
        plan.append({
            "tool": "VisualSearch",
            "args": {"query": parsed_query.get("translated_query", "")}
        })
        
        # 2. Nếu có OCR
        if parsed_query.get("ocr_text"):
            for text in parsed_query["ocr_text"]:
                plan.append({
                    "tool": "OCRSearch",
                    "args": {"query": text}
                })
                
        # 3. Nếu có Objects
        if parsed_query.get("objects"):
            for obj in parsed_query["objects"]:
                plan.append({
                    "tool": "ObjectSearch",
                    "args": {"query": obj}
                })
                
        # 4. Nếu có Scene
        if parsed_query.get("scene"):
            plan.append({
                "tool": "SceneSearch",
                "args": {"query": parsed_query["scene"]}
            })
            
        # 5. Nếu có Attributes
        if parsed_query.get("attributes"):
            for attr in parsed_query["attributes"]:
                plan.append({
                    "tool": "AttributeSearch",
                    "args": {"query": attr}
                })


        log.info(f"Generated Execution Plan: {[step['tool'] for step in plan]}")
        return plan

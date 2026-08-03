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
                
        # 6. Nếu có Temporal Relation
        temporal = parsed_query.get("temporal_relation", "none")
        events = parsed_query.get("events", [])
        if temporal != "none" and len(events) >= 2:
            event1 = events[0]["description"]
            event2 = events[1]["description"]
            if temporal == "after":
                plan.append({
                    "tool": "SequenceSearch", 
                    "args": {"first_query": event1, "then_query": event2, "relation": "after"}
                })
            elif temporal == "before":
                plan.append({
                    "tool": "SequenceSearch",
                    "args": {"first_query": event1, "then_query": event2, "relation": "before"}
                })

        # 7. Nếu có Spatial Relations
        spatial = parsed_query.get("spatial_relations", [])
        if spatial:
            for rel in spatial:
                plan.append({
                    "tool": "SpatialReasoning",
                    "args": {"relation": rel, "objects": parsed_query.get("objects", [])}
                })

        log.info(f"Generated Execution Plan: {[step['tool'] for step in plan]}")
        return plan

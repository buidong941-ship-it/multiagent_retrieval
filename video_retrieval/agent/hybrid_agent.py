"""
Hybrid Retrieval Agent Orchestrator
Điều phối pipeline: Semantic Parsing → Planning → Tools → Fusion, không cần vòng lặp LLM.
"""
from typing import List, Dict, Any
from utils.logging_utils import get_logger

from agent.candidate_manager import CandidateManager
from agent.semantic_parser import SemanticParser
from agent.retrieval_planner import RetrievalPlanner

log = get_logger(__name__)

class HybridRetrievalAgent:
    def __init__(self, config: Dict[str, Any], tools_map: Dict[str, Any], llm_client=None):
        """
        tools_map: Dictionary mapping tên tool (VD: "VisualSearch") sang instance thực thi của tool đó.
        """
        self.config = config
        self.tools = tools_map

        self.candidate_manager = CandidateManager()
        self.parser = SemanticParser(llm_client, model_name=config.get("llm_model", "gemini-2.5-pro"))
        self.planner = RetrievalPlanner()

        # Fusion engine (inject thông qua set_pipelines)
        self.fusion_engine = None

    def set_pipelines(self, fusion=None):
        if fusion: self.fusion_engine = fusion

    async def run(self, query: str, top_k: int = 100) -> List[Any]:
        log.info(f"Hybrid Agent starts processing: '{query}'")

        self.candidate_manager.clear()

        # Phase 1: Semantic Parsing
        parsed_query = self.parser.parse(query)

        # Phase 2: Retrieval Planning
        plan = self.planner.create_plan(parsed_query)

        # Phase 3-5: Execute Retrieval Tools
        for step in plan:
            tool_name = step["tool"]
            args = step["args"]

            tool_instance = self.tools.get(tool_name)
            if not tool_instance:
                log.warning(f"Tool {tool_name} not found in tools_map, skipping.")
                continue

            try:
                # Mỗi tool trả về list các dict [{'frame_id': '...', 'score': 0.9, 'meta': {}}]
                results = await tool_instance.forward(**args)

                # Đưa vào Candidate Manager (Phase 6)
                for res in results:
                    self.candidate_manager.add_or_update(
                        frame_id=res["frame_id"],
                        source=tool_name,
                        score=res.get("score", 0.0),
                        video_id=res.get("video_id", ""),
                        timestamp=res.get("timestamp", 0.0),
                        meta=res.get("meta", {})
                    )
            except Exception as e:
                log.error(f"Error executing {tool_name}: {e}")

        # Lấy toàn bộ candidates
        all_candidates = self.candidate_manager.get_all()
        if not all_candidates:
            log.warning("No candidates retrieved from any tool.")
            return []

        # Phase 7: Fusion
        if self.fusion_engine:
            fused_candidates = self.fusion_engine.fuse(all_candidates)
        else:
            # Fallback: sort by max score
            fused_candidates = sorted(all_candidates, key=lambda c: c.max_score, reverse=True)

        final = fused_candidates[:top_k]
        log.info(f"Hybrid Agent finished. Returned {len(final)} frames.")
        return final

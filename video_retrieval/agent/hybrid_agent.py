"""
Hybrid Retrieval Agent Orchestrator
Điều phối toàn bộ pipeline 12 Phase mà không cần vòng lặp LLM.
"""
from typing import List, Dict, Any
from utils.logging_utils import get_logger
from interfaces.base_interfaces import RetrievalResult

from agent.memory_manager import MemoryManager
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
        
        # Init components
        self.memory = MemoryManager()
        self.candidate_manager = CandidateManager()
        self.parser = SemanticParser(llm_client, model_name=config.get("llm_model", "gemini-2.5-pro"))
        self.planner = RetrievalPlanner()
        
        # Plugins (sẽ được inject thông qua set_pipelines)
        self.fusion_engine = None 
        self.reranker = None
        self.verifier = None
        self.neighbor_refiner = None
        self.response_generator = None

    def set_pipelines(self, fusion=None, reranker=None, verifier=None, refiner=None, responder=None):
        if fusion: self.fusion_engine = fusion
        if reranker: self.reranker = reranker
        if verifier: self.verifier = verifier
        if refiner: self.neighbor_refiner = refiner
        if responder: self.response_generator = responder

    async def run(self, query: str, top_k: int = 100) -> List[Any]:
        log.info(f"Hybrid Agent starts processing: '{query}'")
        
        # Phase 11: Setup Working Memory
        self.memory.clear_working_memory()
        self.memory.set_working("original_query", query)
        self.candidate_manager.clear()

        # Phase 1: Semantic Parsing
        parsed_query = self.parser.parse(query)
        self.memory.set_working("parsed_query", parsed_query)

        # Phase 2: Retrieval Planning
        plan = self.planner.create_plan(parsed_query)
        self.memory.set_working("execution_plan", plan)

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

        # Phase 8: Reranker
        if self.reranker:
            # 1. Chuyển đổi Candidate -> RetrievalResult để tương thích với Reranker cũ
            temp_results = []
            top_k_limit = max(50, top_k)
            for c in fused_candidates[:top_k_limit]:
                # Chỉ đem rerank nếu nguồn tìm kiếm chính là từ CLIP/VisualSearch
                if c.best_source == "VisualSearch":
                    frame_idx = c.metadata.get("frame_idx")
                    if frame_idx is None:
                        frame_idx = int(c.frame_id.split('_frame_')[-1]) if '_frame_' in c.frame_id else 0
                    temp_results.append(RetrievalResult(
                        frame_id=c.frame_id, video_id=c.video_id, 
                        frame_idx=frame_idx, timestamp=c.timestamp, frame_path="",
                        score=c.max_score, source=c.best_source, metadata=c.metadata
                    ))
            
            # 2. Chạy Reranker (await vì hàm này async)
            if temp_results:
                reranked_results = await self.reranker.rerank(temp_results, parsed_query)
                
                # 3. Đồng bộ lại điểm số cho Candidate Pool
                reranked_dict = {r.frame_id: r for r in reranked_results}
                for c in fused_candidates[:top_k_limit]:
                    if c.frame_id in reranked_dict:
                        rr = reranked_dict[c.frame_id]
                        old_score = c.scores.get(c.best_source, 1.0)
                        boost_ratio = (rr.score / old_score) if old_score > 0 else 1.0
                        
                        # Ghi đè điểm VisualSearch bằng điểm mới (đã boost)
                        c.scores[c.best_source] = rr.score
                        c.metadata["rerank_score"] = rr.score
                        
                        if "rrf_score" in c.metadata:
                            c.metadata["rrf_score"] = c.metadata["rrf_score"] * boost_ratio
                            
                        if "neighbor_rerank" in rr.source:
                            c.metadata["rerank_boosted"] = True
                            
                # Sắp xếp lại sau khi cộng điểm (dùng rerank_score -> rrf_score -> max_score)
                fused_candidates.sort(
                    key=lambda x: x.metadata.get('rerank_score', x.metadata.get('rrf_score', x.max_score)), 
                    reverse=True
                )
                
        reranked = fused_candidates[:max(50, top_k)]

        # Phase 9: Verification
        if self.verifier:
            verified = self.verifier.verify(reranked[:top_k], query)
        else:
            verified = reranked[:top_k]

        # Phase 10: Neighbor Refiner
        if self.neighbor_refiner and verified:
            best_candidate = verified[0]
            refined_candidate = self.neighbor_refiner.refine(best_candidate, parsed_query)
            if refined_candidate:
                verified[0] = refined_candidate

        # Phase 12: Response Generation (Optional)
        if self.response_generator:
            response = self.response_generator.generate(query, [c.frame_id for c in verified], parsed_query)
            log.info(f"Agent Response: {response}")

        log.info(f"Hybrid Agent finished. Returned {len(verified)} frames.")
        return verified

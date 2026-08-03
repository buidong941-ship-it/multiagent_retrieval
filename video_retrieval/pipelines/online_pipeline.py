"""
Online Retrieval Pipeline.

Orchestrates:
    1. Query Understanding (LLM)
    2. Multi-branch retrieval (CLIP + BM25 + OCR Embed + Object)
    3. Weighted Fusion / RRF
    4. Temporal Refinement
    5. Metadata enrichment (fill frame_path, timestamp from DB)
    6. Return final ranked results
"""

from __future__ import annotations

import asyncio
from typing import Optional

from config.retrieval_config import RetrievalConfig
from config.settings import Settings
from database.bm25.bm25_index import BM25OcrIndex
from database.metadata.metadata_db import MetadataDatabase
from interfaces.base_interfaces import BaseVectorDatabase, ParsedQuery, RetrievalResult
from query_parser.parser_service import LLMQueryParser
from retrieval.branches.clip_branch import CLIPRetrievalBranch
from retrieval.branches.object_branch import ObjectDetectionBranch
from retrieval.branches.ocr_branches import OCRBm25Branch, OCREmbeddingBranch
from retrieval.fusion.fusion_service import RRFFusion, WeightedFusion, SmaxWeightedFusion
from retrieval.temporal.temporal_refinement import TemporalRefinement
from retrieval.reranker.neighbor_reranker import NeighborReranker
from services.embedding.image_embedding_service import ImageEmbeddingService
from services.ocr.ocr_service import OCRService
from utils.logging_utils import get_logger

from agent.hybrid_agent import HybridRetrievalAgent
from agent.tools.visual_search import VisualSearchTool
from agent.tools.ocr_search import OCRSearchTool
from agent.tools.object_search import ObjectSearchTool
from agent.tools.spatial_reasoning import SpatialReasoningTool
from agent.tools.temporal_tools import SequenceSearchTool
from agent.fusion_engine import FusionEngine
from agent.cross_encoder import CrossEncoderReranker
from agent.frame_verifier import FrameVerifier
from agent.neighbor_refiner import NeighborRefiner
from agent.response_generator import ResponseGenerator


log = get_logger(__name__)


class OnlinePipeline:
    """
    End-to-end online retrieval pipeline.

    Given a Vietnamese query, returns the most relevant video frames
    with their metadata (frame_id, video_id, timestamp, score).

    Attributes:
        settings:      Global Settings.
        query_parser:  LLMQueryParser.
        clip_branch:   CLIPRetrievalBranch.
        bm25_branch:   OCRBm25Branch.
        ocr_branch:    OCREmbeddingBranch.
        obj_branch:    ObjectDetectionBranch.
        fusion:        SmaxWeightedFusion (Default) or RRFFusion.
        temporal:      TemporalRefinement.
        reranker:      NeighborReranker.
        meta_db:       MetadataDatabase for result enrichment.
    """

    def __init__(
        self,
        settings: Settings,
        meta_db: MetadataDatabase,
        vector_db: BaseVectorDatabase,
        embed_svcs: list[ImageEmbeddingService],
        ocr_svc: OCRService,
        bm25_index: BM25OcrIndex,
        use_rrf: bool = False,
    ) -> None:
        """
        Initialize OnlinePipeline.

        Args:
            settings:   Global settings.
            meta_db:    SQLite metadata database.
            vector_db:  Milvus vector database.
            embed_svcs: List of Image embedding services.
            ocr_svc:    OCR service (for OCR embed branch).
            bm25_index: BM25 OCR index (for BM25 branch).
            use_rrf:    If True, use RRFFusion; else WeightedFusion.
        """
        self.settings = settings
        self.meta_db = meta_db
        self.vector_db = vector_db

        cfg = settings.retrieval

        # Query parser
        self.query_parser = LLMQueryParser(cfg)

        # Retrieval branches
        self.clip_branch = CLIPRetrievalBranch(cfg, embed_svcs, vector_db)
        self.bm25_branch = OCRBm25Branch(cfg, bm25_index)
        # self.ocr_branch = OCREmbeddingBranch(cfg, ocr_svc, vector_db)
        self.obj_branch = ObjectDetectionBranch(cfg, meta_db)

        # Fusion
        self.fusion = RRFFusion(cfg) if use_rrf else SmaxWeightedFusion(cfg)

        # Find text-capable embedding service (SigLIP) for temporal & reranker
        text_embed_svc = next((svc for svc in embed_svcs if svc._collection == "clip_embeddings"), embed_svcs[0])

        # Temporal refinement & Reranking
        self.temporal = TemporalRefinement(cfg, meta_db, vector_db, text_embed_svc)
        self.reranker = NeighborReranker(cfg, meta_db, vector_db, text_embed_svc)
        
        # Find Jina embedding service for siglip_jina mode
        self.jina_embed_svc = next((svc for svc in embed_svcs if svc._collection == "jina_embeddings"), None)
        
        # Advanced Searcher (SigLIP + Jina)
        from retrieval.advanced_search import AdvancedSearcher
        self.advanced_searcher = AdvancedSearcher(vector_db, meta_db)
        
        # --- Initialize Hybrid Agent ---
        tools_map = {
            "VisualSearch": VisualSearchTool(self.clip_branch, top_k=cfg.clip_top_k),
            "SceneSearch": VisualSearchTool(self.clip_branch, top_k=cfg.clip_top_k),
            "AttributeSearch": VisualSearchTool(self.clip_branch, top_k=cfg.clip_top_k),
            "OCRSearch": OCRSearchTool(self.bm25_branch, top_k=cfg.ocr_bm25_top_k),
            "ObjectSearch": ObjectSearchTool(self.obj_branch, top_k=cfg.object_top_k),
            "SpatialReasoning": SpatialReasoningTool(self.meta_db),
            "SequenceSearch": SequenceSearchTool(VisualSearchTool(self.clip_branch), self.meta_db)
        }
        # Tạm thời truyền model là ollama fallback hoặc gemini
        self.hybrid_agent = HybridRetrievalAgent(config={"llm_model": cfg.ollama_model}, tools_map=tools_map, llm_client=self.query_parser._call_llm)
        
        # Inject pipelines vao agent
        agent_fusion = FusionEngine(k=60)
        self.hybrid_agent.set_pipelines(fusion=agent_fusion)


    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "fusion",
        use_temporal: bool = True,
    ) -> list[RetrievalResult]:
        """
        Execute the full online retrieval pipeline.

        Args:
            query:  Raw Vietnamese text query from user.
            top_k:  Override final_top_k from config.
            mode:   Retrieval mode: 'fusion', 'clip', 'ocr_bm25', 'ocr_embed', 'object', 'action', 'direct_ocr'.
            use_temporal: Enable temporal refinement.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        cfg = self.settings.retrieval
        final_top_k = top_k or cfg.final_top_k

        log.info(f"Query: '{query}' | Mode: '{mode}'")

        # ── Step 1 & 2: Agent Orchestration OR Multi-branch Retrieval ────
        branch_results = {}
        parsed: ParsedQuery | None = None  # Will be set in non-agent modes
        
        if mode == "siglip_jina_sequence":
            log.info("Running Advanced SigLIP + Jina Sequence (PRF) Pipeline...")
            log.info(f"Active embed_svcs: {[svc._collection for svc in self.clip_branch.embed_svcs]}")
            log.info(f"jina_embed_svc: {self.jina_embed_svc}")
            
            if self.jina_embed_svc is None:
                log.error("Jina embedding service not found. Loaded backends do not include 'jina'. Check EMBED_ACTIVE_BACKENDS in .env.")
                return []
                
            jina_encoder = self.jina_embed_svc.embedder
            
            # Handle Temporal Pair Selection (Algorithm 4) if '||' in query
            if "||" in query:
                parts = query.split("||", 1)
                query1 = parts[0].strip()
                query2 = parts[1].strip()
                log.info(f"Detected temporal pair query. Q1='{query1}', Q2='{query2}'")
                
                # Run Pipeline for Query 1
                cands1 = await self.retrieve(query1, top_k=100, mode="clip", use_temporal=True)
                agg_items1, q1_jina_emb = await self.advanced_searcher.siglip_jina_pipeline(
                    candidates=cands1, text_query=query1, jina_text_encoder=jina_encoder,
                    target_collection="jina_embeddings", window=1
                )
                
                # Extract Jina text vector for Query 2
                loop = asyncio.get_running_loop()
                q2_jina_emb = await loop.run_in_executor(
                    None, lambda: jina_encoder.encode_texts([query2])[0]
                )
                
                final_results = []
                if q1_jina_emb is not None and q2_jina_emb is not None:
                    # Collect valid pairs
                    valid_pairs = []
                    for focal_item in agg_items1[:50]:  # Limit search to top 50 focal frames
                        best_pair = await self.advanced_searcher.find_best_frame_pair(
                            query1_vector=q1_jina_emb,
                            query2_vector=q2_jina_emb,
                            input_frame=focal_item,
                            collection="jina_embeddings",
                        )
                        if best_pair:
                            left, right = best_pair
                            pair_score = left[1] + right[1]
                            valid_pairs.append((focal_item, pair_score, left, right))
                            
                    # Sort valid pairs by pair score descending
                    valid_pairs.sort(key=lambda x: x[1], reverse=True)
                    
                    # Return Focal Frames (Choice A)
                    for focal_item, pair_score, left, right in valid_pairs[:final_top_k]:
                        focal_item.score = pair_score
                        focal_item.metadata["pair_left"] = left[0]["frame_id"]
                        focal_item.metadata["pair_right"] = right[0]["frame_id"]
                        
                        final_results.append(
                            RetrievalResult(
                                frame_id=focal_item.frame_id,
                                video_id=focal_item.video_id,
                                frame_idx=focal_item.frame_idx,
                                timestamp=focal_item.metadata.get("timestamp", 0.0),
                                frame_path="",
                                score=focal_item.score,
                                source="siglip_jina_pair",
                                metadata=focal_item.metadata
                            )
                        )
            else:
                # Standard Algorithm 2 + 3 Execution
                cands = await self.retrieve(query, top_k=100, mode="clip", use_temporal=True)
                agg_items, _ = await self.advanced_searcher.siglip_jina_pipeline(
                    candidates=cands,
                    text_query=query,
                    jina_text_encoder=jina_encoder,
                    target_collection="jina_embeddings",
                    window=1
                )
                
                final_results = []
                for item in agg_items[:final_top_k]:
                    final_results.append(
                        RetrievalResult(
                            frame_id=item.frame_id,
                            video_id=item.video_id,
                            frame_idx=item.frame_idx,
                            timestamp=item.metadata.get("timestamp", 0.0),
                            frame_path="",
                            score=item.score,
                            source="siglip_jina"
                        )
                    )
                    
            # Skip standard temporal refinement because advanced searcher already did neighbor aggregation
            # We just need to enrich frame paths and return
            return await self._enrich_results(final_results)
            
        elif mode == "agent":
            log.info("Step 1: Running Hybrid Agent (Deterministic Pipeline)...")
            
            # HybridRetrievalAgent run is now async and returns a list of Candidate
            final_candidates = await self.hybrid_agent.run(query, top_k=final_top_k)
                
            # Convert to RetrievalResult objects
            agent_results = []
            for c in final_candidates:
                video_id = c.video_id or (c.frame_id.split("_frame_")[0] if "_frame_" in c.frame_id else "")
                
                # Parse frame_idx
                frame_idx = c.metadata.get("frame_idx", 0)
                if not frame_idx:
                    try:
                        idx_str = c.frame_id.split("_frame_")[-1]
                        frame_idx = int(idx_str)
                    except:
                        frame_idx = 0
                    
                # Parse score
                score = c.metadata.get("rerank_score", c.metadata.get("rrf_score", c.max_score))
                
                # Map best source from agent to frontend-compatible labels
                source_map = {
                    "VisualSearch": "clip",
                    "OCRSearch": "ocr_bm25",
                    "ObjectSearch": "object"
                }
                final_source = source_map.get(c.best_source, "smax_fusion")
                
                # Khôi phục nhãn Rerank nếu Candidate này được Reranker cộng điểm
                if c.metadata.get("rerank_boosted"):
                    final_source += " + neighbor_rerank"

                agent_results.append(
                    RetrievalResult(
                        frame_id=c.frame_id,
                        video_id=video_id,
                        frame_idx=frame_idx, 
                        timestamp=c.timestamp or 0.0,
                        frame_path="",
                        score=float(score), 
                        source=final_source,
                        metadata=c.metadata
                    )
                )
            branch_results["agent_success"] = agent_results
            mode = "agent_success"
        
        if mode not in ["agent", "agent_success"]:
            if mode == "direct_ocr":
                log.info("Step 1: Bypassing LLM parser for direct OCR search...")
                parsed = ParsedQuery(
                    original_query=query,
                    ocr_text=[query],
                    translated_query=query,
                )
            else:
                log.info("Step 1: Parsing query with LLM...")
                parsed: ParsedQuery = self.query_parser.parse(query)
                log.info(
                    f"Parsed | objects={parsed.objects} | ocr={parsed.ocr_text} "
                    f"| actions={parsed.actions}"
                )

            log.info("Step 2: Running retrieval branches...")

            if mode in ["fusion", "clip", "siglip_jina_parallel"]:
                branch_results["clip"] = await self.clip_branch.retrieve(parsed, cfg.clip_top_k, mode=mode)
            if mode in ["fusion", "ocr_bm25", "direct_ocr"] and (parsed.ocr_text or parsed.original_query):
                branch_results["ocr_bm25"] = await self.bm25_branch.retrieve(parsed, cfg.ocr_bm25_top_k)
            # if mode in ["fusion", "ocr_embed"] and (parsed.ocr_text or parsed.original_query):
            #     branch_results["ocr_embed"] = await self.ocr_branch.retrieve(parsed, cfg.ocr_embed_top_k)
            if mode == "object" or (mode == "fusion" and parsed.objects):
                if mode == "object" and not parsed.objects:
                    parsed.objects = [query]
                branch_results["object"] = await self.obj_branch.retrieve(parsed, cfg.object_top_k)
            # if mode in ["fusion", "action"]:
            #     branch_results["action"] = await getattr(self, "action_branch").retrieve(parsed, cfg.clip_top_k) if hasattr(self, "action_branch") else []

        # Log branch sizes
        for name, results in branch_results.items():
            log.info(f"  {name}: {len(results)} results")

        # ── Step 3: Fusion ───────────────────────────────────────────────
        if mode in ["fusion", "agent"]:
            log.info("Step 3: Fusing branches...")
            _WEIGHT_ATTR_MAP = {
                "clip": "weight_clip",
                "ocr_bm25": "weight_ocr_bm25",
                "object": "weight_object",
                "clip_after": "weight_clip_aux",
                "clip_before": "weight_clip_aux",
                "clip_attr": "weight_clip_aux",
                "clip_scene": "weight_clip_aux",
                "sequence": "weight_clip",
            }
            weights = {}
            for branch_name in branch_results.keys():
                attr = _WEIGHT_ATTR_MAP.get(branch_name)
                weights[branch_name] = getattr(cfg, attr, 1.0) if attr else 1.0
            fused = self.fusion.fuse(branch_results, weights, top_k=cfg.temporal_top_k * 2)
        elif mode == "agent_success":
            log.info("Step 3: Bypassing manual fusion, using Agent's final answer...")
            fused = branch_results.get("agent_success", [])
        elif mode in ["clip", "siglip_jina_parallel"]:
            fused = branch_results.get("clip", [])
        elif mode in ["ocr_bm25", "direct_ocr"]:
            fused = branch_results.get("ocr_bm25", [])
            limit = cfg.temporal_top_k * 2 if use_temporal else final_top_k
            fused = fused[:limit]
        elif mode == "action":
            fused = branch_results.get("action", [])
        else:
            log.info(f"Step 3: Bypassing fusion, returning raw {mode} results...")
            fused = branch_results.get(mode, [])

        # ── Step 4: Temporal Refinement ──────────────────────────────────
        if use_temporal and cfg.temporal_window > 0 and parsed is not None and mode != "agent_success":
            log.info("Step 4: Temporal refinement...")
            fused = await self.temporal.refine(fused, parsed)
            
        # ── Step 4.5: Neighbor-aware Reranking ───────────────────────────
        if parsed is not None and mode != "agent_success":
            log.info("Step 4.5: Neighbor-aware reranking...")
            fused = await self.reranker.rerank(fused, parsed)

        # ── Step 5: Metadata Enrichment ──────────────────────────────────
        log.info("Step 5: Enriching results with metadata...")
        fused = await self._enrich_results(fused[:final_top_k])

        log.info(f"Final results: {len(fused)} frames returned")
        return fused

    async def _enrich_results(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Fill in frame_path and timestamp from metadata DB.

        Args:
            results: List of RetrievalResult (may have empty frame_path).

        Returns:
            Enriched list with frame_path and timestamp populated.
        """
        enriched = []
        for result in results:
            if result.frame_path and result.timestamp > 0:
                enriched.append(result)
                continue

            frame_meta = await self.meta_db.get_frame_async(result.frame_id)
            if frame_meta:
                enriched.append(
                    RetrievalResult(
                        frame_id=result.frame_id,
                        video_id=result.video_id,
                        frame_idx=result.frame_idx,
                        timestamp=frame_meta.get("timestamp", result.timestamp),
                        frame_path=frame_meta.get("frame_path", result.frame_path),
                        score=result.score,
                        source=result.source,
                        metadata=result.metadata,
                    )
                )
            else:
                enriched.append(result)

        return enriched

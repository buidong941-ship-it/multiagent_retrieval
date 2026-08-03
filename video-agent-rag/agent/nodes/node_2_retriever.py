"""
Node 2: Retriever

Implements Smax-Normalized Weighted Fusion across multiple retrieval models.

Algorithm (from paper):
    Require: query (string), model_configs (list of (model_name, weight, use_flag))
    Ensure:  ranked_results (list of (index, score))

    1.  Initialize score_dict = {}
    2.  Normalize weights: Pwi = 1 for selected models (applied via use_flag)
    3.  for (model_name, w, use_flag) in model_configs:
    4.      if use_flag:
    5.          Load model and processor for model_name
    6.          e ← EncodeText(model_name, query)
    7.          I, S ← Search(model_name_idx, e, M=50)
    8.          Smax ← max(S)
    9.          for (i, s) in (I, S):
    10.             score_dict[i] += (s / Smax) × w
    11.     end
    12. end
    13. ranked_results ← Sort(score_dict, descending)
    14. return ranked_results

Mapping to this codebase:
    - model_name  → branch name: "clip", "ocr_bm25", "object"
    - use_flag    → whether the corresponding query field is non-empty
    - EncodeText  → branch.retrieve() internally (SigLIP2 / BM25 / YOLO)
    - I, S        → (frame_id, score) pairs returned by each branch
    - Smax        → max score within that branch's results
    - score_dict  → accumulated per-frame weighted-normalized score dict
"""

import asyncio
import os
from collections import defaultdict

from agent.state import VideoRetrievalState
from core.logger import get_logger
from data_sources.pipeline_manager import get_real_pipeline, video_retrieval_root
from interfaces.base_interfaces import ParsedQuery, RetrievalResult

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  ALGORITHM: Smax-Normalized Weighted Fusion
# ─────────────────────────────────────────────────────────────

def smax_weighted_fusion(
    branch_results: dict[str, list[RetrievalResult]],
    model_configs: list[tuple[str, float, bool]],
    top_k: int = 200,
) -> list[dict]:
    """
    Smax-Normalized Weighted Fusion.

    Require: query Q, model_configs = [(model_name, weight, use_flag), ...]
    Ensure:  ranked_results = [(frame_id, score), ...]

    For each enabled branch:
        - S    = raw scores from that branch's Search results
        - Smax = max(S) — used to normalize within this branch only
        - Each frame accumulates: score_dict[i] += (s / Smax) × w

    This avoids the scale problem (BM25 scores ≠ cosine similarity scores)
    without using rank-based RRF: each model's contribution is proportional
    to its relative relevance within its own result set, then weighted.

    Args:
        branch_results: Dict mapping branch_name → list[RetrievalResult].
        model_configs:  List of (branch_name, weight, use_flag).
        top_k:          Maximum number of results to return.

    Returns:
        Sorted list of dicts with frame_id, video_id, score, source fields.
    """
    # Step 1: Initialize score_dict (A in pseudocode)
    score_dict: dict[str, float] = defaultdict(float)
    frame_data: dict[str, RetrievalResult] = {}
    frame_sources: dict[str, set[str]] = defaultdict(set)

    # Step 2: Normalize weights (only enabled models count)
    active_configs = [(name, w, flag) for name, w, flag in model_configs if flag]
    total_weight = sum(w for _, w, _ in active_configs) or 1.0

    # Step 3: For each (model_name, w, use_flag) in model_configs
    for model_name, w, use_flag in model_configs:

        # Step 4: if use_flag
        if not use_flag:
            continue

        results = branch_results.get(model_name, [])
        if not results:
            continue

        # Normalize weight relative to total active weight
        normalized_w = w / total_weight

        # Steps 7–8: I, S = Search results; Smax = max(S)
        scores = [r.score for r in results]
        s_max = max(scores) if scores else 1.0
        if s_max == 0.0:
            s_max = 1.0  # Guard against zero division

        # Step 9–10: for (i, s) in (I, S): score_dict[i] += (s / Smax) × w
        for result in results:
            norm_score = (result.score / s_max) * normalized_w
            score_dict[result.frame_id] += norm_score
            frame_sources[result.frame_id].add(model_name)

            # Keep the highest-scoring result object for each frame
            if result.frame_id not in frame_data or result.score > frame_data[result.frame_id].score:
                frame_data[result.frame_id] = result

    # Steps 13–14: ranked_results ← Sort(score_dict, descending)
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    fused = []
    for frame_id, total_score in sorted_items[:top_k]:
        r = frame_data[frame_id]
        fused.append({
            "frame_id": r.frame_id,
            "video_id": r.video_id,
            "frame_idx": getattr(r, "frame_idx", 0),
            "score": round(total_score, 6),
            "source": f"smax_fusion ({', '.join(sorted(frame_sources[frame_id]))})",
            "frame_path": r.frame_path or "",
        })

    log.info(
        f"  Smax fusion: {sum(len(v) for v in branch_results.values())} raw results → "
        f"{len(fused)} unique frames (top_k={top_k})"
    )
    return fused


# ─────────────────────────────────────────────────────────────
#  NODE 2: retrieval_node
# ─────────────────────────────────────────────────────────────

async def retrieval_node(state: VideoRetrievalState) -> dict:
    log.info("--- [NODE] 2_retriever ---")

    clip_query   = state.get("clip_query", "")
    ocr_query    = state.get("ocr_query", "")
    yolo_filters = state.get("yolo_filters", {})

    pipeline = await get_real_pipeline()
    branch_results: dict[str, list] = {}
    tasks: dict[str, object] = {}

    # ── Step 5–6: Load model/processor + EncodeText via branch.retrieve() ──
    if clip_query:
        p_clip = ParsedQuery(original_query=clip_query, translated_query=clip_query)
        tasks["clip"] = pipeline.clip_branch.retrieve(p_clip, top_k=50)

    if ocr_query:
        p_ocr = ParsedQuery(original_query=ocr_query, ocr_text=[ocr_query])
        tasks["ocr_bm25"] = pipeline.bm25_branch.retrieve(p_ocr, top_k=50)

    if yolo_filters:
        objects_list = list(yolo_filters.keys())
        if objects_list:
            p_obj = ParsedQuery(original_query=" ".join(objects_list), objects=objects_list)
            tasks["object"] = pipeline.obj_branch.retrieve(p_obj, top_k=50)

    if tasks:
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for (branch_name, _), result in zip(tasks.items(), results):
            if isinstance(result, Exception):
                log.warning(f"  Branch '{branch_name}' failed — {result}")
                continue
            branch_results[branch_name] = result

    # ── Model configs: (model_name, weight, use_flag) ───────────────────────
    # use_flag = True only when the corresponding query/filter is provided
    model_configs: list[tuple[str, float, bool]] = [
        ("clip",     1.0,  bool(clip_query)),
        ("ocr_bm25", 0.6,  bool(ocr_query)),
        ("object",   0.4,  bool(yolo_filters)),
    ]

    # ── Apply Smax-Normalized Weighted Fusion (Algorithm lines 1–14) ────────
    fused_dicts = smax_weighted_fusion(
        branch_results=branch_results,
        model_configs=model_configs,
        top_k=200,
    )

    # ── Enrich each result with frame_path from SQLite ──────────────────────
    async def enrich(r: dict) -> dict:
        # If frame_path is already absolute and valid, skip DB call
        if r.get("frame_path") and os.path.isabs(r["frame_path"]):
            return r
        meta = await pipeline.meta_db.get_frame_async(r["frame_id"])
        if meta:
            frame_path = meta.get("frame_path", "")
            if frame_path and not os.path.isabs(frame_path):
                frame_path = os.path.join(video_retrieval_root, frame_path)
            r["frame_path"] = frame_path
            r.setdefault("frame_idx", meta.get("frame_idx", 0))
        return r

    final_results = list(await asyncio.gather(*(enrich(r) for r in fused_dicts)))

    log.info(f"  Retrieval complete — {len(final_results)} frames after Smax fusion.")
    return {"retrieved_videos": final_results}

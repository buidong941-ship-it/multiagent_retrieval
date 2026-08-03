"""
Node B: Temporal Resolver

Resolves temporal queries ("before/after/around" a reference event) by:
  1. Fetching metadata of anchor frames (from previous search turn).
  2. Querying Milvus for frames within a temporal window around each anchor.
  3. Assigning a proximity score to each hit.
  4. MERGING temporal results with the semantic results from Node 2:
       - Frames that appear in BOTH sets: blended score (temporal + semantic).
       - Temporal-only frames:  temporal_score * 0.8  (context relevant but no CLIP match)
       - Semantic-only frames:  semantic_score * 0.6  (has CLIP match, but not near anchor)
  5. Sorting the merged list by final blended score descending.

Design decision:
    Prior implementation replaced `retrieved_videos` entirely with temporal proximity
    results, discarding all semantic CLIP scores from Node 2.  This merge step
    preserves both signals so the re-ranker (Node C) has richer data to work with.
"""

import asyncio
import os
from agent.state import VideoRetrievalState
from core.logger import get_logger
from data_sources.pipeline_manager import get_real_pipeline, video_retrieval_root

log = get_logger(__name__)

# Delta in frame indices: 60 frames ≈ 2 seconds at 30fps
_TEMPORAL_DELTA = 60
# Maximum window: 4 seconds in each direction
_TEMPORAL_MAX_DELTA = 120

# Score blend weights for the merge step
_W_TEMPORAL  = 0.5   # weight for temporal proximity score
_W_SEMANTIC  = 0.5   # weight for semantic CLIP score
_W_TEMP_ONLY = 0.8   # penalty for frames only in temporal (no CLIP match)
_W_SEM_ONLY  = 0.6   # penalty for frames only in semantic (not near anchor)


async def temporal_resolver_node(state: VideoRetrievalState) -> dict:
    log.info("--- [NODE] B_temporal_resolver ---")

    temporal_intent  = state.get("temporal_intent", "none")
    anchor_frame_ids = state.get("anchor_frame_ids", [])
    # Semantic results from Node 2 — we keep these and blend them in
    semantic_results: list[dict] = state.get("retrieved_videos", [])

    if not anchor_frame_ids:
        log.warning("  No anchor frame IDs provided — skipping temporal resolution.")
        return {}  # pass-through; semantic results remain unchanged

    pipeline = await get_real_pipeline()
    milvus_collection = pipeline.settings.milvus.clip_collection

    # ── Step 1: Fetch metadata for anchor frames ─────────────────────────
    anchor_metas = []
    for fid in anchor_frame_ids[:3]:  # use top 3 anchors
        try:
            meta = await pipeline.meta_db.get_frame_async(fid)
            if meta:
                anchor_metas.append(meta)
        except Exception as e:
            log.warning(f"  Could not fetch meta for anchor '{fid}': {e}")

    if not anchor_metas:
        log.warning("  No valid anchor metadata found — falling back to semantic results.")
        return {}

    # ── Step 2: Query adjacent frames from Milvus for each anchor ────────
    raw_temporal: list[dict] = []

    for meta in anchor_metas:
        video_id  = meta.get("video_id", "")
        frame_idx = int(meta.get("frame_idx", 0))

        if temporal_intent == "before":
            idx_min = max(0, frame_idx - _TEMPORAL_MAX_DELTA)
            idx_max = max(0, frame_idx - _TEMPORAL_DELTA)
        elif temporal_intent == "after":
            idx_min = frame_idx + _TEMPORAL_DELTA
            idx_max = frame_idx + _TEMPORAL_MAX_DELTA
        else:  # "around"
            idx_min = max(0, frame_idx - _TEMPORAL_DELTA)
            idx_max = frame_idx + _TEMPORAL_DELTA

        filter_expr = (
            f'video_id == "{video_id}" '
            f'and frame_idx >= {idx_min} '
            f'and frame_idx <= {idx_max}'
        )

        log.info(
            f"  Querying {video_id} frame_idx [{idx_min}, {idx_max}] "
            f"(intent={temporal_intent})"
        )

        try:
            hits = pipeline.vector_db.query(
                collection_name=milvus_collection,
                filter_expr=filter_expr,
                output_fields=["frame_id", "video_id", "frame_idx", "timestamp"],
            )
            for hit in hits:
                hit_frame_idx = int(hit.get("frame_idx", 0))
                # Proximity score: 1.0 at delta boundary, 0.0 at max delta boundary
                proximity = 1.0 - abs(hit_frame_idx - frame_idx) / _TEMPORAL_MAX_DELTA
                raw_temporal.append({
                    "video_id":  hit.get("video_id", ""),
                    "frame_id":  hit.get("frame_id", ""),
                    "frame_idx": hit.get("frame_idx", 0),   # needed by re_ranker neighbor lookup
                    "score":     max(0.01, round(proximity, 4)),
                    "frame_path": "",
                })
        except Exception as e:
            log.warning(f"  Temporal Milvus query failed for {video_id}: {e}")

    if not raw_temporal:
        log.warning("  Temporal resolution returned no frames — keeping semantic results.")
        return {}

    # ── Step 3: Enrich temporal hits with frame_path from metadata ────────
    async def enrich(r: dict) -> dict:
        try:
            meta = await pipeline.meta_db.get_frame_async(r["frame_id"])
            frame_path = meta.get("frame_path", "") if meta else ""
            if frame_path and not os.path.isabs(frame_path):
                frame_path = os.path.join(video_retrieval_root, frame_path)
        except Exception:
            frame_path = ""
        return {**r, "frame_path": frame_path}

    enriched_temporal = list(await asyncio.gather(*(enrich(r) for r in raw_temporal)))

    # Deduplicate temporal results by frame_id (keep highest proximity)
    temporal_score_map: dict[str, float] = {}
    temporal_meta_map:  dict[str, dict]  = {}
    for r in sorted(enriched_temporal, key=lambda x: x["score"], reverse=True):
        fid = r["frame_id"]
        if fid not in temporal_score_map:
            temporal_score_map[fid] = r["score"]
            temporal_meta_map[fid]  = r

    # ── Step 4: Merge temporal + semantic results ─────────────────────────
    # Build semantic score map (normalize to [0, 1] first)
    semantic_score_map: dict[str, float] = {}
    semantic_meta_map:  dict[str, dict]  = {}
    if semantic_results:
        raw_sem_scores = [r.get("score", 0.0) for r in semantic_results]
        sem_max = max(raw_sem_scores) if raw_sem_scores else 1.0
        sem_max = sem_max if sem_max > 0 else 1.0
        for r in semantic_results:
            fid = r["frame_id"]
            semantic_score_map[fid] = r.get("score", 0.0) / sem_max
            semantic_meta_map[fid]  = r

    all_frame_ids = set(temporal_score_map.keys()) | set(semantic_score_map.keys())
    merged: list[dict] = []

    for fid in all_frame_ids:
        in_temporal = fid in temporal_score_map
        in_semantic  = fid in semantic_score_map

        if in_temporal and in_semantic:
            # Both signals — blend
            blended = round(
                _W_TEMPORAL * temporal_score_map[fid] +
                _W_SEMANTIC  * semantic_score_map[fid],
                4
            )
            base = temporal_meta_map[fid]   # prefer temporal meta (has frame_path enriched)
            source = "temporal+semantic"
        elif in_temporal:
            # Only temporal — slight penalty (no CLIP match)
            blended = round(temporal_score_map[fid] * _W_TEMP_ONLY, 4)
            base = temporal_meta_map[fid]
            source = "temporal"
        else:
            # Only semantic — lower priority (not near anchor)
            blended = round(semantic_score_map[fid] * _W_SEM_ONLY, 4)
            base = semantic_meta_map[fid]
            source = "semantic"

        merged.append({
            **base,
            "score":  blended,
            "source": source,
        })

    # Sort by blended score descending
    merged.sort(key=lambda x: x["score"], reverse=True)

    both_count = sum(
        1 for fid in all_frame_ids
        if fid in temporal_score_map and fid in semantic_score_map
    )
    log.info(
        f"  Temporal merge complete: {len(merged)} frames total "
        f"({len(temporal_score_map)} temporal | {len(semantic_score_map)} semantic | "
        f"{both_count} in both) — intent={temporal_intent}"
    )

    return {"retrieved_videos": merged}

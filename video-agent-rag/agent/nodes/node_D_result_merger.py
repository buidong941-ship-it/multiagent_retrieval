from agent.state import VideoRetrievalState
from core.logger import get_logger

log = get_logger(__name__)

# Minimum intersection size: if fewer than this many frames overlap, fall back to new results
_MIN_INTERSECTION = 5
# Score blending: how much weight to give the NEW search score vs. the OLD score
_NEW_WEIGHT = 0.6
_OLD_WEIGHT = 0.4


async def result_merger_node(state: VideoRetrievalState) -> dict:
    """
    Intersects new retrieved_videos with previous_retrieved_videos when search_mode == 'intersect'.
    Frames that appear in BOTH result sets get a blended score; unmatched frames are dropped.
    Falls back to new results if the intersection is too small.
    """
    log.info("--- [NODE] D_result_merger ---")

    search_mode = state.get("search_mode", "standard")
    new_results = state.get("retrieved_videos", [])
    prev_results = state.get("previous_retrieved_videos", [])

    # Only intersect when explicitly requested and previous results exist
    if search_mode != "intersect" or not prev_results:
        log.info(
            f"  Mode='{search_mode}' | prev={len(prev_results)} | new={len(new_results)} — "
            f"pass-through (no intersection applied)."
        )
        return {}  # No state change — keep retrieved_videos as-is

    if not new_results:
        log.info("  No new results to intersect with — returning previous results as fallback.")
        return {"retrieved_videos": prev_results}

    # Build a lookup from frame_id → score for both sets
    prev_score_map: dict[str, float] = {
        r["frame_id"]: r.get("score", 0.0) for r in prev_results
    }
    new_score_map: dict[str, float] = {
        r["frame_id"]: r.get("score", 0.0) for r in new_results
    }
    new_meta_map: dict[str, dict] = {
        r["frame_id"]: r for r in new_results
    }

    # --- Intersection: keep frames appearing in both sets ---
    common_frame_ids = set(prev_score_map.keys()) & set(new_score_map.keys())

    log.info(
        f"  Intersection: {len(prev_results)} prev ∩ {len(new_results)} new = "
        f"{len(common_frame_ids)} common frames"
    )

    if len(common_frame_ids) < _MIN_INTERSECTION:
        # Intersection too small → probably different subjects; fall back to new results
        log.warning(
            f"  Intersection too small ({len(common_frame_ids)} < {_MIN_INTERSECTION}) "
            f"— falling back to new results."
        )
        return {}

    # Build blended result list for common frames
    intersected = []
    for fid in common_frame_ids:
        new_score = new_score_map[fid]
        old_score = prev_score_map[fid]
        blended_score = round(_NEW_WEIGHT * new_score + _OLD_WEIGHT * old_score, 4)

        entry = dict(new_meta_map[fid])  # copy metadata from new results (has fresh frame_path etc.)
        entry["score"] = blended_score
        intersected.append(entry)

    # Sort by blended score descending
    intersected.sort(key=lambda x: x["score"], reverse=True)

    log.info(
        f"  Merged {len(intersected)} frames. "
        f"Top blended score: {intersected[0]['score'] if intersected else 0:.4f}"
    )
    return {"retrieved_videos": intersected}

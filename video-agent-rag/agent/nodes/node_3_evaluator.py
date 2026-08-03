from agent.state import VideoRetrievalState
from core.config import settings
from core.logger import get_logger

log = get_logger(__name__)

# Minimum number of results with decent score to consider a search "clear"
_MIN_RESULTS_FOR_CLEAR = 3


async def evaluator_node(state: VideoRetrievalState) -> dict:
    log.info("--- [NODE] 3_evaluator ---")
    results = state.get("retrieved_videos", [])
    is_query_complete = state.get("is_query_complete", True)
    clarification_count = state.get("clarification_count", 0)

    # --- Guard: query too vague from Analyzer ---
    if not is_query_complete and clarification_count < 2:
        log.info("  Query marked INCOMPLETE by Analyzer → forcing clarification.")
        return {"confidence_score": 0.0, "is_clear": False}

    # --- No results at all ---
    if not results:
        log.info("  No results returned → is_clear=False")
        return {"confidence_score": 0.0, "is_clear": False}

    # --- Multi-dimensional scoring ---
    top_score = results[0].get("score", 0.0)

    # How many results are above 50% of the top score (measures result quality spread)
    strong_results = sum(
        1 for r in results if r.get("score", 0.0) >= top_score * 0.5
    )

    # Normalize coverage bonus: more strong results = higher confidence
    coverage_bonus = min(strong_results / _MIN_RESULTS_FOR_CLEAR, 1.0) * 0.2
    composite_score = top_score + coverage_bonus

    # Hard cap clarification at 2: if user has answered twice, always show results
    is_clear = (composite_score >= settings.threshold) or (clarification_count >= 2)

    log.info(
        f"  top_score={top_score:.4f} | strong_results={strong_results} | "
        f"composite={composite_score:.4f} | clarification_count={clarification_count} | "
        f"is_clear={is_clear}"
    )

    return {"confidence_score": composite_score, "is_clear": is_clear}

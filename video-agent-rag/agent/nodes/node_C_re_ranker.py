"""
Node C: Re-Ranker

Pipeline:
  1. AGGREGATENEIGHBORSCORES (temporal neighborhood scoring)
     - For each retrieved frame (index I), fetch its temporal neighbors.
     - Compute cosine similarity of each neighbor against the query vector Q.
     - Sum neighbor scores → total_score per candidate key.
     - Sort descending → neighbor-aware ranking.

  2. Score blending (no LLM)
     - 60% neighbor aggregate score (temporal coherence signal)
     - 40% original retrieval score (Smax-fusion / RRF)
     - Faster, deterministic, and avoids LLM hallucination on score assignment.
"""

import asyncio
import numpy as np

from agent.state import VideoRetrievalState
from core.logger import get_logger
from data_sources.pipeline_manager import get_real_pipeline


log = get_logger(__name__)

# Only re-rank the top N to keep latency acceptable
_TOP_K_RERANK = 15

# Temporal window: how many frames before/after to consider as neighbors
_TEMPORAL_WINDOW = 2


# ─────────────────────────────────────────────────────────────
#  HELPER: Fetch embeddings for a set of frame_ids from Milvus
# ─────────────────────────────────────────────────────────────

def _fetch_embeddings(vector_db, frame_ids: list[str]) -> dict[str, np.ndarray]:
    """
    Batch-query Milvus to get embeddings for a list of frame_ids.

    Returns:
        dict mapping frame_id → np.ndarray (shape: (dim,))
    """
    if not frame_ids:
        return {}

    embeddings_map: dict[str, np.ndarray] = {}
    chunk_size = 500

    vector_db._connect()
    vector_db._client.load_collection("clip_embeddings")

    for i in range(0, len(frame_ids), chunk_size):
        chunk = frame_ids[i: i + chunk_size]
        in_list = ", ".join(f"'{fid}'" for fid in chunk)
        filter_expr = f"frame_id in [{in_list}]"
        try:
            rows = vector_db._client.query(
                collection_name="clip_embeddings",
                filter=filter_expr,
                output_fields=["frame_id", "embedding"],
            )
            for row in rows:
                emb = row.get("embedding")
                if emb:
                    embeddings_map[row["frame_id"]] = np.array(emb, dtype=np.float32)
        except Exception as e:
            log.warning(f"  Milvus batch query error: {e}")

    return embeddings_map


# ─────────────────────────────────────────────────────────────
#  ALGORITHM: AGGREGATENEIGHBORSCORES(I, Q)
# ─────────────────────────────────────────────────────────────

async def aggregate_neighbor_scores(
    candidates: list[dict],
    query_vector: np.ndarray,
    meta_db,
    vector_db,
    window: int = _TEMPORAL_WINDOW,
) -> dict[str, float]:
    """
    Implements the AGGREGATENEIGHBORSCORES algorithm.

    Require: Indices I (candidates), Query Q (query_vector)
    1. Initialize aggregated_score A = {}
    2. For each idx in I:
       a. key = idx (frame_id)
       b. neighbors = GETNEIGHBORS(key)  ← SQLite lookup ±window frames
       c. total_score = 0
       d. For each neighbor N in neighbors:
            score = COMPUTESCORE(N, Q)  ← cosine similarity (dot product, L2-normalized)
            if score is not None:
                total_score += score
       e. UPDATESCORES(A, key, total_score)
    3. sorted_scores = SORT(A, descending)
    4. return sorted_scores

    Args:
        candidates:   List of retrieved frame dicts (must have video_id, frame_id, frame_idx).
        query_vector: L2-normalized query embedding (np.ndarray).
        meta_db:      MetadataDatabase for SQLite neighbor lookup.
        vector_db:    MilvusVectorDatabase for embedding fetch.
        window:       Number of frames before/after each candidate to consider.

    Returns:
        Dict mapping frame_id → aggregated neighbor score (float).
    """
    if not candidates:
        return {}

    # Step 1: Initialize A = {}
    aggregated_score: dict[str, float] = {}

    # Step 2b: GETNEIGHBORS for all candidates concurrently
    neighbor_tasks = [
        meta_db.get_neighbouring_frames_async(
            video_id=c["video_id"],
            frame_idx=c.get("frame_idx", 0),
            window=window,
        )
        for c in candidates
    ]
    all_neighbors = await asyncio.gather(*neighbor_tasks, return_exceptions=True)

    # Collect all unique frame_ids (candidates + neighbors) to batch-fetch embeddings
    all_frame_ids: set[str] = set()
    for c in candidates:
        all_frame_ids.add(c["frame_id"])
    for neighbors in all_neighbors:
        if isinstance(neighbors, list):
            for nb in neighbors:
                all_frame_ids.add(nb["frame_id"])

    # Fetch all embeddings in one Milvus batch call
    embeddings_map = await asyncio.to_thread(
        _fetch_embeddings, vector_db, list(all_frame_ids)
    )

    # Step 2: Loop over each candidate (idx in I)
    for i, candidate in enumerate(candidates):
        key = candidate["frame_id"]   # Step 2a: key = idx (as frame_id string)
        neighbors = all_neighbors[i]

        if isinstance(neighbors, Exception):
            log.warning(f"  Neighbor fetch failed for {key}: {neighbors}")
            aggregated_score[key] = 0.0
            continue

        total_score = 0.0  # Step 2c

        # Step 2d: For each neighbor N in neighbors
        for neighbor in neighbors:
            nb_emb = embeddings_map.get(neighbor["frame_id"])
            if nb_emb is None:
                continue  # score = None → skip (algorithm line 9)

            # COMPUTESCORE(N, Q) = cosine similarity = dot product (both L2-normalized)
            score = float(np.dot(query_vector, nb_emb))
            total_score += score  # accumulate

        # Step 2e: UPDATESCORES(A, key, total_score)
        aggregated_score[key] = total_score

    # Step 3: sorted_scores = SORT(A, descending) is done by caller
    return aggregated_score


# ─────────────────────────────────────────────────────────────
#  NODE C: re_ranker_node
# ─────────────────────────────────────────────────────────────

async def re_ranker_node(state: VideoRetrievalState) -> dict:
    log.info("--- [NODE] C_re_ranker ---")

    results = state.get("retrieved_videos", [])
    # Prefer the rewritten query; fall back to raw clip_query
    query = state.get("rewritten_query", "") or state.get("clip_query", "")

    if not results:
        log.info("  No results to re-rank — skipping.")
        return {}

    if not query:
        log.info("  No query available for re-ranking — skipping.")
        return {}

    top_candidates = results[:_TOP_K_RERANK]
    tail = results[_TOP_K_RERANK:]

    # ── STEP 1: AGGREGATENEIGHBORSCORES ─────────────────────
    neighbor_scores: dict[str, float] = {}
    try:
        pipeline = await get_real_pipeline()
        
        # Encode query to vector (embed_svc is stored on the clip_branch)
        query_vector = await asyncio.to_thread(
            pipeline.clip_branch.embedding_svc.encode_query, query
        )

        # Enrich candidates with frame_idx if missing (needed for neighbor lookup)
        enriched_candidates = []
        for c in top_candidates:
            if "frame_idx" not in c:
                meta = await pipeline.meta_db.get_frame_async(c["frame_id"])
                if meta:
                    c = {**c, "frame_idx": meta.get("frame_idx", 0)}
            enriched_candidates.append(c)

        # Run AGGREGATENEIGHBORSCORES(I, Q)
        neighbor_scores = await aggregate_neighbor_scores(
            candidates=enriched_candidates,
            query_vector=query_vector,
            meta_db=pipeline.meta_db,
            vector_db=pipeline.vector_db,
            window=_TEMPORAL_WINDOW,
        )
        top_candidates = enriched_candidates

        # Normalize neighbor scores to [0, 1]
        if neighbor_scores:
            max_ns = max(neighbor_scores.values()) or 1.0
            min_ns = min(neighbor_scores.values())
            span = max_ns - min_ns or 1.0
            neighbor_scores = {
                k: (v - min_ns) / span for k, v in neighbor_scores.items()
            }

        log.info(
            f"  Neighbor aggregation complete. "
            f"Top neighbor score: {max(neighbor_scores.values(), default=0):.4f}"
        )

    except Exception as e:
        log.warning(f"  Neighbor score aggregation failed ({e}) — skipping step 1.")

    # ── STEP 2: Sort by blended score (neighbor + original) — no LLM ────────
    # blend: 60% neighbor aggregate score (temporal context signal)
    #        40% original retrieval score (CLIP / RRF / Smax-fusion score)
    for r in top_candidates:
        frame_id   = r["frame_id"]
        original   = r.get("score", 0.0)
        neighbor_s = neighbor_scores.get(frame_id, 0.0)

        r["score"]          = round(0.6 * neighbor_s + 0.4 * original, 4)
        r["neighbor_score"] = round(neighbor_s, 4)

    reranked = sorted(top_candidates, key=lambda x: x["score"], reverse=True)
    final    = reranked + tail

    log.info(
        f"  Re-ranked {len(reranked)} candidates (neighbor aggregate only, no LLM). "
        f"Top score: {reranked[0]['score'] if reranked else 0:.4f}"
    )
    return {"retrieved_videos": final}

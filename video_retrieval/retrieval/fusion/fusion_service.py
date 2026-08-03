"""
Multi-branch score fusion.

Design: Reciprocal Rank Fusion (RRF) + Weighted Score Normalization.

Why RRF over simple weighted sum:
    - RRF is robust to score scale differences across branches
      (BM25 scores are in [0, ∞), Milvus COSINE in [0, 1])
    - Simple normalization is sufficient ONLY when all branches
      have comparable score distributions
    - RRF: score(d) = Σ weight_i / (k + rank_i(d))

Strategy:
    1. Min-max normalize each branch's scores to [0, 1]
    2. Compute weighted sum across branches
    3. De-duplicate by frame_id (keep highest combined score)
    4. Sort by final score descending
    5. Return top_k
"""

from __future__ import annotations

from collections import defaultdict

from config.retrieval_config import RetrievalConfig
from interfaces.base_interfaces import BaseFusion, ParsedQuery, RetrievalResult
from utils.logging_utils import get_logger

log = get_logger(__name__)

# RRF constant k (higher = less impact of rank position)
RRF_K = 60


class WeightedFusion(BaseFusion):
    """
    Weighted score fusion with min-max normalization and deduplication.

    Attributes:
        config: RetrievalConfig with weight_clip, weight_ocr_bm25, etc.
    """

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    def fuse(
        self,
        branch_results: dict[str, list[RetrievalResult]],
        weights: dict[str, float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Fuse branch results with weighted normalized scores.

        Args:
            branch_results: Dict mapping branch_name → list of results.
            weights:        Dict mapping branch_name → weight.
            top_k:          Number of results to return.

        Returns:
            Deduplicated, sorted list of RetrievalResult.
        """
        # Accumulate per-frame data
        frame_scores: dict[str, float] = defaultdict(float)
        frame_data: dict[str, RetrievalResult] = {}
        frame_sources: dict[str, set[str]] = defaultdict(set)

        for branch_name, results in branch_results.items():
            if not results:
                continue

            weight = weights.get(branch_name, 1.0)
            if weight == 0.0:
                continue

            # Min-max normalize scores for this branch
            scores = [r.score for r in results]
            min_s = min(scores)
            max_s = max(scores)
            score_range = max_s - min_s if max_s > min_s else 1.0

            for result in results:
                norm_score = (result.score - min_s) / score_range
                frame_scores[result.frame_id] += weight * norm_score
                frame_sources[result.frame_id].add(branch_name)

                # Store frame data (prefer higher-score source)
                if result.frame_id not in frame_data:
                    frame_data[result.frame_id] = result
                elif norm_score > (frame_data[result.frame_id].score):
                    frame_data[result.frame_id] = result

        # Assign final fused scores
        fused: list[RetrievalResult] = []
        for frame_id, total_score in frame_scores.items():
            result = frame_data[frame_id]
            fused.append(
                RetrievalResult(
                    frame_id=result.frame_id,
                    video_id=result.video_id,
                    frame_idx=result.frame_idx,
                    timestamp=result.timestamp,
                    frame_path=result.frame_path,
                    score=total_score,
                    source=f"fusion ({', '.join(sorted(frame_sources[frame_id]))})",
                    metadata=result.metadata,
                )
            )

        # Sort by fused score descending
        fused.sort(key=lambda r: r.score, reverse=True)

        log.info(
            f"Fusion: {sum(len(v) for v in branch_results.values())} candidates → "
            f"{len(fused)} unique → returning top {top_k}"
        )

        return fused[:top_k]


class RRFFusion(BaseFusion):
    """
    Reciprocal Rank Fusion — more robust than weighted score fusion.

    Each branch contributes: weight / (k + rank)
    where rank is 1-indexed position in sorted results.

    Attributes:
        config: RetrievalConfig.
        k:      RRF smoothing constant (default 60).
    """

    def __init__(self, config: RetrievalConfig, k: int = RRF_K) -> None:
        self.config = config
        self.k = k

    def fuse(
        self,
        branch_results: dict[str, list[RetrievalResult]],
        weights: dict[str, float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Apply Reciprocal Rank Fusion.

        Args:
            branch_results: Dict branch_name → results (already sorted by score).
            weights:        Dict branch_name → weight.
            top_k:          Final number of results.

        Returns:
            RRF-fused, deduplicated results.
        """
        frame_rrf: dict[str, float] = defaultdict(float)
        frame_data: dict[str, RetrievalResult] = {}
        frame_sources: dict[str, set[str]] = defaultdict(set)

        for branch_name, results in branch_results.items():
            weight = weights.get(branch_name, 1.0)
            if weight == 0.0 or not results:
                continue

            for rank, result in enumerate(results, start=1):
                rrf_score = weight / (self.k + rank)
                frame_rrf[result.frame_id] += rrf_score
                frame_sources[result.frame_id].add(branch_name)

                if result.frame_id not in frame_data:
                    frame_data[result.frame_id] = result

        fused = [
            RetrievalResult(
                frame_id=fid,
                video_id=frame_data[fid].video_id,
                frame_idx=frame_data[fid].frame_idx,
                timestamp=frame_data[fid].timestamp,
                frame_path=frame_data[fid].frame_path,
                score=score,
                source=f"rrf ({', '.join(sorted(frame_sources[fid]))})",
            )
            for fid, score in frame_rrf.items()
        ]

        fused.sort(key=lambda r: r.score, reverse=True)
        log.info(f"RRF Fusion → {len(fused)} unique frames → top {top_k}")
        return fused[:top_k]


class SmaxWeightedFusion(BaseFusion):
    """
    Smax-Normalized Weighted Fusion (Algorithm 3).

    For each branch:
        - Smax = max(score)
        - Each frame accumulates: weight * (score / Smax)
    
    This avoids the scale problem without using rank-based RRF, preserving
    relative score distances within a branch's result set.

    Attributes:
        config: RetrievalConfig.
    """

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    def fuse(
        self,
        branch_results: dict[str, list[RetrievalResult]],
        weights: dict[str, float],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Apply Smax-Normalized Weighted Fusion.
        """
        score_dict: dict[str, float] = defaultdict(float)
        frame_data: dict[str, RetrievalResult] = {}
        frame_sources: dict[str, set[str]] = defaultdict(set)

        # Normalize weights so they sum to 1 (only for active branches)
        active_weights = {
            b: w for b, w in weights.items() 
            if b in branch_results and branch_results[b] and w > 0
        }
        total_weight = sum(active_weights.values()) or 1.0

        for branch_name, results in branch_results.items():
            if branch_name not in active_weights:
                continue

            normalized_w = active_weights[branch_name] / total_weight

            scores = [r.score for r in results]
            
            # Smax logic with Base Thresholds:
            # Instead of always dividing by max(scores) which scales even terrible results to 100%,
            # we enforce a minimum 'good' threshold. If the max score is below this threshold,
            # it will not be fully scaled up to 1.0.
            max_s = max(scores) if scores else 0.0
            
            if branch_name.startswith("clip") or branch_name in ["action", "sequence"]:
                # A good CLIP score is typically > 0.30
                s_max = max(max_s, 0.30)
            elif branch_name in ["ocr_bm25", "object"]:
                # A good BM25 score is typically > 10.0
                s_max = max(max_s, 10.0)
            else:
                s_max = max(max_s, 1.0)

            for result in results:
                # Calculate normalized score bounded roughly [0, 1]
                norm_score = (result.score / s_max) * normalized_w
                score_dict[result.frame_id] += norm_score
                frame_sources[result.frame_id].add(branch_name)

                # Keep the highest-scoring result object for metadata
                if result.frame_id not in frame_data or norm_score > frame_data[result.frame_id].score:
                    frame_data[result.frame_id] = result

        fused = [
            RetrievalResult(
                frame_id=fid,
                video_id=frame_data[fid].video_id,
                frame_idx=frame_data[fid].frame_idx,
                timestamp=frame_data[fid].timestamp,
                frame_path=frame_data[fid].frame_path,
                score=score,
                source=f"smax_fusion ({', '.join(sorted(frame_sources[fid]))})",
                metadata=frame_data[fid].metadata,
            )
            for fid, score in score_dict.items()
        ]

        fused.sort(key=lambda r: r.score, reverse=True)
        log.info(f"Smax Fusion → {len(fused)} unique frames → top {top_k}")
        return fused[:top_k]

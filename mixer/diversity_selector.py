"""
Mixer & Diversity Selection Stage (Section 5.7).
Implements:
1. Maximal Marginal Relevance (MMR) trade-off between Heavy Ranker score and content redundancy.
2. Hard author fatigue constraint: max 2 posts per author in top-K feed.
3. In-network vs out-of-network candidate balance.
"""

from typing import Dict, List, Optional
import numpy as np

from config import DEFAULT_MMR_LAMBDA, FEED_TOP_K, MAX_POSTS_PER_AUTHOR, TARGET_IN_NETWORK_RATIO
from data.schemas import ScoredCandidate


class DiversitySelector:
    """Selects diverse and balanced feed from scored candidates using MMR."""

    def __init__(self, content_model):
        self.content_model = content_model

    def select(
        self,
        scored_candidates: List[ScoredCandidate],
        top_k: int = FEED_TOP_K,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        max_per_author: int = MAX_POSTS_PER_AUTHOR,
        balance_ratio: bool = True,
    ) -> List[ScoredCandidate]:
        """
        Runs greedy MMR selection with author frequency constraints and in/out ratio balance.
        """
        if not scored_candidates:
            return []

        # Normalize candidate scores into [0, 1] for balanced MMR computation
        raw_scores = [c.final_score for c in scored_candidates]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        score_range = max(1e-5, max_s - min_s)
        norm_scores = {id(c): (c.final_score - min_s) / score_range for c in scored_candidates}

        target_in = int(top_k * TARGET_IN_NETWORK_RATIO) if balance_ratio else top_k
        target_out = top_k - target_in if balance_ratio else top_k

        selected: List[ScoredCandidate] = []
        author_counts: Dict[int, int] = {}
        remaining = list(scored_candidates)
        count_in = 0
        count_out = 0

        while remaining and len(selected) < top_k:
            # Check availability per pool satisfying author limit
            avail_in = [
                c for c in remaining
                if c.hydrated.candidate.source == "in_network"
                and author_counts.get(c.hydrated.candidate.post.author_id, 0) < max_per_author
            ]
            avail_out = [
                c for c in remaining
                if c.hydrated.candidate.source != "in_network"
                and author_counts.get(c.hydrated.candidate.post.author_id, 0) < max_per_author
            ]

            # Determine pool to select from based on ratio balancer
            if balance_ratio and (avail_in or avail_out):
                if count_in < target_in and avail_in and (count_out >= target_out or not avail_out or len(selected) % 2 == 0):
                    eligible = avail_in
                elif count_out < target_out and avail_out:
                    eligible = avail_out
                elif avail_in:
                    eligible = avail_in
                else:
                    eligible = avail_out
            else:
                eligible = [
                    c for c in remaining
                    if author_counts.get(c.hydrated.candidate.post.author_id, 0) < max_per_author
                ]

            if not eligible:
                break

            best_cand: Optional[ScoredCandidate] = None
            best_mmr_score = -float("inf")

            for cand in eligible:
                p_id = cand.hydrated.candidate.post.post_id

                # MMR score computation
                if not selected or mmr_lambda >= 0.999:
                    mmr_val = norm_scores[id(cand)]
                else:
                    max_sim = 0.0
                    if self.content_model is not None:
                        for sel in selected:
                            sel_id = sel.hydrated.candidate.post.post_id
                            sim = self.content_model.similarity_between_posts(p_id, sel_id)
                            if sim > max_sim:
                                max_sim = sim
                    mmr_val = mmr_lambda * norm_scores[id(cand)] - (1.0 - mmr_lambda) * max_sim

                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_cand = cand

            if best_cand is None:
                break

            selected.append(best_cand)
            remaining.remove(best_cand)
            auth_id = best_cand.hydrated.candidate.post.author_id
            author_counts[auth_id] = author_counts.get(auth_id, 0) + 1

            if best_cand.hydrated.candidate.source == "in_network":
                count_in += 1
            else:
                count_out += 1

        return selected

"""
Recommendation Pipeline Orchestrator (Section 2, 5).
Coordinates end-to-end execution across all stages:
Candidate Sourcing -> Hydration -> Filters -> Heavy Ranker -> Diversity Mixer -> Explainer.
Collects real-time funnel metrics for pipeline observability and interactive inspection.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from config import DEFAULT_MMR_LAMBDA, FEED_TOP_K
from data.schemas import Post, ScoredCandidate, User
from mixer.diversity_selector import DiversitySelector
from pipeline.candidate_retrieval import CandidateRetriever
from pipeline.explainer import Explainer
from pipeline.filters import VisibilityFilterChain
from pipeline.heavy_ranker import HeavyRanker
from pipeline.hydrator import FeatureHydrator


class RecommendationOrchestrator:
    """End-to-End Orchestrator for the Mini-X Recommendation Pipeline."""

    def __init__(
        self,
        users: List[User],
        posts: List[Post],
        follows: List[tuple],
        interactions: List[Dict],
        als_model,
        item_knn_model,
        content_model,
        heavy_ranker: HeavyRanker,
        shilling_detector=None,
    ):
        self.users = users
        self.posts = posts
        self.user_map = {u.user_id: u for u in users}
        self.post_map = {p.post_id: p for p in posts}
        self.interactions = interactions

        # Pipeline stages
        self.retriever = CandidateRetriever(posts, als_model, item_knn_model, content_model)
        self.hydrator = FeatureHydrator(users, posts, follows, interactions, als_model, content_model)
        self.filter_chain = VisibilityFilterChain(shilling_detector=shilling_detector)
        self.heavy_ranker = heavy_ranker
        self.diversity_selector = DiversitySelector(content_model)
        self.explainer = Explainer()

        # Cache seen posts per user
        self.user_seen_posts: Dict[int, Set[int]] = {u.user_id: set() for u in users}
        self.user_interaction_counts: Dict[int, int] = {u.user_id: 0 for u in users}
        for it in interactions:
            u_id = it["user_id"]
            p_id = it["post_id"]
            if u_id in self.user_seen_posts:
                self.user_seen_posts[u_id].add(p_id)
                self.user_interaction_counts[u_id] = self.user_interaction_counts.get(u_id, 0) + 1

    def generate_feed(
        self,
        user_id: int,
        top_k: int = FEED_TOP_K,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        w_like: Optional[float] = None,
        w_reply: Optional[float] = None,
        w_skip: Optional[float] = None,
        lambda_age: Optional[float] = None,
        defense_active: bool = False,
        ignore_seen: bool = False,
        retrieval_mode: str = "all",
        department_filter: Optional[str] = None,
    ) -> Tuple[List[ScoredCandidate], Dict[str, Any]]:
        """
        Executes full recommendation funnel for user_id and returns feed + funnel telemetry.
        """
        start_time = time.perf_counter()
        user = self.user_map.get(user_id)
        if user is None:
            raise ValueError(f"User ID {user_id} not found in database.")

        now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
        user_history = [it for it in self.interactions if it["user_id"] == user_id]
        user_interaction_count = self.user_interaction_counts.get(user_id, 0)

        # -------------------------------------------------------------
        # Stage 1: Candidate Sourcing
        # -------------------------------------------------------------
        raw_candidates = self.retriever.retrieve(user, user_history, now=now, retrieval_mode=retrieval_mode)
        retrieved_count = len(raw_candidates)

        # -------------------------------------------------------------
        # Stage 2: Feature Hydration
        # -------------------------------------------------------------
        hydrated_candidates = self.hydrator.hydrate_batch(raw_candidates, user, now=now)
        hydrated_count = len(hydrated_candidates)

        # -------------------------------------------------------------
        # Stage 3: Visibility & Safety Filters
        # -------------------------------------------------------------
        seen_set = set() if ignore_seen else self.user_seen_posts.get(user_id, set())
        filtered_candidates, filter_stats = self.filter_chain.apply_filters(
            hydrated_candidates=hydrated_candidates,
            user=user,
            seen_post_ids=seen_set,
            defense_active=defense_active,
            department_filter=department_filter,
        )
        filtered_count = len(filtered_candidates)

        # If strict filtering left too few candidates, retry without seen_post_ids filter
        if filtered_count < top_k and not ignore_seen:
            filtered_candidates, filter_stats = self.filter_chain.apply_filters(
                hydrated_candidates=hydrated_candidates,
                user=user,
                seen_post_ids=set(),
                defense_active=defense_active,
                department_filter=department_filter,
            )
            filtered_count = len(filtered_candidates)

        # -------------------------------------------------------------
        # Stage 4: Multi-Signal Heavy Ranking
        # -------------------------------------------------------------
        scored_candidates = self.heavy_ranker.score_candidates(
            hydrated_candidates=filtered_candidates,
            user=user,
            user_interaction_count=user_interaction_count,
            w_like=w_like,
            w_reply=w_reply,
            w_skip=w_skip,
            lambda_age=lambda_age,
        )
        ranked_count = len(scored_candidates)

        # -------------------------------------------------------------
        # Stage 5: Diversity & Selection Mixer (MMR)
        # -------------------------------------------------------------
        final_feed = self.diversity_selector.select(
            scored_candidates=scored_candidates,
            top_k=top_k,
            mmr_lambda=mmr_lambda,
            balance_ratio=(retrieval_mode == "all"),
        )

        # -------------------------------------------------------------
        # Stage 6: Explainability Layer
        # -------------------------------------------------------------
        final_feed = self.explainer.explain_feed(final_feed, user)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        funnel_stats = {
            "corpus_count": len(self.posts),
            "retrieved_count": retrieved_count,
            "hydrated_count": hydrated_count,
            "filtered_count": filtered_count,
            "ranked_count": ranked_count,
            "feed_count": len(final_feed),
            "latency_ms": elapsed_ms,
            "filter_breakdown": filter_stats,
            "is_cold_start": user_interaction_count < 3,
            "user_interactions": user_interaction_count,
        }

        return final_feed, funnel_stats

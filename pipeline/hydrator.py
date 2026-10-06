"""
Feature Hydration & Enrichment Stage (Section 5.2, 5.5).
Enriches candidate identifiers with real-time metadata, author statistics,
social velocity, and constructs the 12-dimensional feature vector for Heavy Ranking.
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from data.schemas import CandidatePost, HydratedCandidate, Post, User


class FeatureHydrator:
    """Enriches candidate posts with user-item graph and engagement features."""

    def __init__(
        self,
        users: List[User],
        posts: List[Post],
        follows: List[tuple],
        interactions: List[Dict],
        als_model,
        content_model,
    ):
        self.user_map = {u.user_id: u for u in users}
        self.post_map = {p.post_id: p for p in posts}
        self.als_model = als_model
        self.content_model = content_model

        # Precompute author follower counts
        self.author_followers: Dict[int, int] = {u.user_id: 0 for u in users}
        for src, dst in follows:
            self.author_followers[dst] = self.author_followers.get(dst, 0) + 1

        # Precompute user-author interaction counts
        self.user_author_history: Dict[tuple, int] = {}
        # Precompute user historical dwell means
        user_dwells: Dict[int, List[float]] = {u.user_id: [] for u in users}
        for it in interactions:
            u_id = it["user_id"]
            p_id = it["post_id"]
            if p_id in self.post_map:
                author_id = self.post_map[p_id].author_id
                key = (u_id, author_id)
                self.user_author_history[key] = self.user_author_history.get(key, 0) + 1
            user_dwells[u_id].append(float(it.get("dwell_seconds", 5.0)))

        self.user_dwell_means = {
            u_id: (float(np.mean(dwells)) if dwells else 5.0)
            for u_id, dwells in user_dwells.items()
        }

        # Precompute post popularity percentiles
        all_likes = np.array([p.likes_count for p in posts], dtype=float)
        self.max_likes = max(1.0, float(np.max(all_likes)))

    def hydrate_candidate(
        self,
        candidate: CandidatePost,
        user: User,
        now: Optional[datetime] = None,
    ) -> HydratedCandidate:
        """Hydrate a single candidate post with graph and interaction signals."""
        if now is None:
            now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

        post = candidate.post
        author_id = post.author_id

        # 1. Author statistics
        author_follower_count = self.author_followers.get(author_id, 0)
        author_reputation = math.log10(1.0 + author_follower_count)

        # 2. User-Author historical affinity
        user_author_interactions = self.user_author_history.get((user.user_id, author_id), 0)

        # 3. Post recency
        age_hours = max(0.01, (now - post.created_at).total_seconds() / 3600.0)

        # 4. Construct 12-dim feature vector
        # Feature 1: als_latent_dot_product
        als_dot = 0.0
        if self.als_model is not None:
            als_dot = self.als_model.predict_score(user.user_id, post.post_id)

        # Feature 2: content_tfidf_cosine
        content_cos = 0.0
        if self.content_model is not None:
            content_cos = self.content_model.similarity_user_post(user, post.post_id)

        # Feature 3: is_in_network
        is_in_network = 1.0 if author_id in user.followed_user_ids else 0.0

        # Feature 4: author_reputation_score
        rep_score = float(author_reputation)

        # Feature 5: user_author_historical_likes
        user_auth_hist = float(min(10, user_author_interactions))

        # Feature 6: post_total_likes (log1p)
        post_likes_log = math.log1p(post.likes_count)

        # Feature 7: post_total_replies (log1p)
        post_replies_log = math.log1p(post.replies_count)

        # Feature 8: post_age_hours
        age_h = float(age_hours)

        # Feature 9: tag_overlap_jaccard
        u_tags = set(user.preferred_tags)
        p_tags = set(post.tags)
        union_len = len(u_tags | p_tags)
        tag_jaccard = (len(u_tags & p_tags) / union_len) if union_len > 0 else 0.0

        # Feature 10: user_dwell_mean
        dwell_mean = float(self.user_dwell_means.get(user.user_id, 5.0))

        # Feature 11: item_popularity_percentile
        pop_percentile = float(post.likes_count / self.max_likes)

        # Feature 12: department_match
        dept_match = 1.0 if user.department == post.department else 0.0

        feature_vector = [
            float(als_dot),
            float(content_cos),
            float(is_in_network),
            float(rep_score),
            float(user_auth_hist),
            float(post_likes_log),
            float(post_replies_log),
            float(age_h),
            float(tag_jaccard),
            float(dwell_mean),
            float(pop_percentile),
            float(dept_match),
        ]

        return HydratedCandidate(
            candidate=candidate,
            author_follower_count=author_follower_count,
            user_author_interaction_history=user_author_interactions,
            post_age_hours=round(age_hours, 2),
            author_reputation_score=round(author_reputation, 3),
            feature_vector=feature_vector,
        )

    def hydrate_batch(
        self,
        candidates: List[CandidatePost],
        user: User,
        now: Optional[datetime] = None,
    ) -> List[HydratedCandidate]:
        """Hydrate an entire candidate pool."""
        return [self.hydrate_candidate(cand, user, now) for cand in candidates]

"""
Candidate Retrieval Stage (Section 5.3).
Two-Tower Candidate Sourcing:
- Tower 1 (In-Network / Thunder-style): Follow graph posts from the past 7 days
- Tower 2 (Out-of-Network / Phoenix-style):
    - Engine A: Implicit ALS Latent Retrieval
    - Engine B: Item-kNN Latent Similarities
    - Engine C: Content TF-IDF Centroid Search
    - Exploration / Cold-Start Tag Matching
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from config import (
    LIMIT_ALS_LATENT,
    LIMIT_CONTENT_TFIDF,
    LIMIT_IN_NETWORK,
    LIMIT_ITEM_KNN,
    LIMIT_TRENDING,
)
from data.schemas import CandidatePost, Post, User


class CandidateRetriever:
    """Pools candidate posts across In-Network and Out-of-Network retrieval engines."""

    def __init__(self, posts: List[Post], als_model, item_knn_model, content_model):
        self.posts = posts
        self.post_map: Dict[int, Post] = {p.post_id: p for p in posts}
        self.als_model = als_model
        self.item_knn_model = item_knn_model
        self.content_model = content_model

    def retrieve(
        self,
        user: User,
        user_interaction_history: Optional[List[Dict]] = None,
        now: Optional[datetime] = None,
        retrieval_mode: str = "all",
    ) -> List[CandidatePost]:
        """
        Executes multi-engine retrieval and returns deduplicated CandidatePosts.
        retrieval_mode: 'all', 'in_network', or 'out_network'.
        """
        if now is None:
            now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

        candidates_dict: Dict[int, CandidatePost] = {}
        followed_set = set(user.followed_user_ids)

        # -------------------------------------------------------------
        # 1. In-Network Tower (Earlybird / Thunder Proxy)
        # -------------------------------------------------------------
        if retrieval_mode in ("all", "in_network") and followed_set:
            in_network_posts = []
            for post in self.posts:
                if post.author_id in followed_set and post.author_id != user.user_id:
                    # Calculate age in hours
                    age_hours = max(0.1, (now - post.created_at).total_seconds() / 3600.0)
                    if age_hours <= 168.0:  # Within past 7 days
                        # Ranking formula: likes + 2*replies + 10 / (1 + age_hours)
                        score = float(post.likes_count + 2.0 * post.replies_count + (10.0 / (1.0 + age_hours)))
                        in_network_posts.append((post, score))

            in_network_posts.sort(key=lambda x: x[1], reverse=True)
            for post, score in in_network_posts[:LIMIT_IN_NETWORK]:
                candidates_dict[post.post_id] = CandidatePost(
                    post=post,
                    source="in_network",
                    retrieval_score=score,
                )

        # -------------------------------------------------------------
        # 2. Out-of-Network Tower (Phoenix Proxy)
        # -------------------------------------------------------------
        if retrieval_mode in ("all", "out_network"):
            exclude_from_oon = set(candidates_dict.keys()) | {p.post_id for p in self.posts if p.author_id in followed_set or p.author_id == user.user_id}

            # Engine A: Implicit ALS Latent Retrieval
            if self.als_model is not None and user.user_id < self.als_model.num_users:
                als_results = self.als_model.get_top_candidates(
                    user_id=user.user_id,
                    top_n=LIMIT_ALS_LATENT,
                    exclude_ids=exclude_from_oon,
                )
                for p_id, score in als_results:
                    if p_id in self.post_map and p_id not in candidates_dict:
                        candidates_dict[p_id] = CandidatePost(
                            post=self.post_map[p_id],
                            source="als_latent",
                            retrieval_score=score,
                        )

            # Engine B: Item-kNN Latent Similarities
            recent_liked_ids = []
            if user_interaction_history:
                liked = [it["post_id"] for it in user_interaction_history if it.get("liked", False)]
                recent_liked_ids = liked[-3:]

            if self.item_knn_model is not None and recent_liked_ids:
                knn_candidates: Dict[int, float] = {}
                for liked_id in recent_liked_ids:
                    sim_items = self.item_knn_model.get_similar_items(liked_id, top_n=5)
                    for sim_id, sim_score in sim_items:
                        if sim_id not in candidates_dict and sim_id not in exclude_from_oon:
                            knn_candidates[sim_id] = max(knn_candidates.get(sim_id, 0.0), sim_score)

                sorted_knn = sorted(knn_candidates.items(), key=lambda x: x[1], reverse=True)[:LIMIT_ITEM_KNN]
                for p_id, score in sorted_knn:
                    if p_id in self.post_map and p_id not in candidates_dict:
                        candidates_dict[p_id] = CandidatePost(
                            post=self.post_map[p_id],
                            source="item_knn",
                            retrieval_score=score,
                        )

            # Engine C: Content TF-IDF Retrieval
            if self.content_model is not None:
                content_results = self.content_model.get_top_candidates(
                    user=user,
                    top_n=LIMIT_CONTENT_TFIDF,
                    exclude_ids=set(candidates_dict.keys()),
                )
                for p_id, score in content_results:
                    if p_id in self.post_map and p_id not in candidates_dict:
                        candidates_dict[p_id] = CandidatePost(
                            post=self.post_map[p_id],
                            source="content_tfidf",
                            retrieval_score=score,
                        )

            # Engine D: Trending / Social Velocity (Viral & High Engagement Out-of-Network)
            trending_posts = []
            for post in self.posts:
                if post.post_id not in candidates_dict and post.author_id not in followed_set and post.author_id != user.user_id:
                    age_hours = max(0.1, (now - post.created_at).total_seconds() / 3600.0)
                    if age_hours <= 168.0:
                        v_score = float(post.likes_count + 2.0 * post.replies_count + (10.0 / (1.0 + age_hours)))
                        trending_posts.append((post, v_score))
            trending_posts.sort(key=lambda x: x[1], reverse=True)
            for post, score in trending_posts[:LIMIT_TRENDING]:
                candidates_dict[post.post_id] = CandidatePost(
                    post=post,
                    source="trending_velocity",
                    retrieval_score=score,
                )

        # -------------------------------------------------------------
        # 3. Exploration / Cold-Start Safety Net
        # If user has very few candidates (< 50), add department & tag matching posts
        # -------------------------------------------------------------
        if len(candidates_dict) < 50:
            for post in self.posts:
                if post.post_id not in candidates_dict and post.author_id != user.user_id:
                    if retrieval_mode == "in_network" and post.author_id not in followed_set:
                        continue
                    if retrieval_mode == "out_network" and post.author_id in followed_set:
                        continue
                    has_tag = any(t in user.preferred_tags for t in post.tags)
                    same_dept = post.department == user.department
                    if has_tag or same_dept:
                        score = 1.0 if has_tag and same_dept else 0.5
                        candidates_dict[post.post_id] = CandidatePost(
                            post=post,
                            source="cold_tag",
                            retrieval_score=score,
                        )
                if len(candidates_dict) >= 120:
                    break

        # Final fallback if still empty (e.g. 0 follows, 0 tags)
        if len(candidates_dict) == 0:
            for post in sorted(self.posts, key=lambda p: p.likes_count, reverse=True)[:30]:
                if post.author_id != user.user_id:
                    candidates_dict[post.post_id] = CandidatePost(
                        post=post,
                        source="cold_tag",
                        retrieval_score=float(post.likes_count),
                    )

        return list(candidates_dict.values())

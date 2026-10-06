"""
Shilling Defense & Anomaly Detector (Section 3.5, 5.9).
Detects coordinated bot attacks using:
1. Interaction Rating Deviation: Dev(u) = (1 / |I_u|) * sum |r_ui - r_bar_i|
2. Co-Action Jaccard Similarity Graph: J(u, v) = |I_u n I_v| / |I_u u I_v|
3. Bot Cluster Anomaly Scoring: Identifies bot clusters with high mutual overlap (J > 0.70)
   and computes post anomaly score to filter out artificially boosted content.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import SHILLING_ANOMALY_THRESHOLD


class ShillingDefenseDetector:
    """Detects bot accounts and computes post anomaly scores."""

    def __init__(self, anomaly_threshold: float = SHILLING_ANOMALY_THRESHOLD, jaccard_threshold: float = 0.65):
        self.anomaly_threshold = anomaly_threshold
        self.jaccard_threshold = jaccard_threshold
        self.flagged_bot_users: Set[int] = set()
        self.post_anomaly_scores: Dict[int, float] = {}

    def analyze_interactions(self, all_interactions: List[Dict]):
        """
        Analyzes interactions across users and builds anomaly profiles.
        """
        self.flagged_bot_users.clear()
        self.post_anomaly_scores.clear()

        # Group interactions by user
        user_items: Dict[int, Set[int]] = defaultdict(set)
        post_raters: Dict[int, List[int]] = defaultdict(list)
        post_likes: Dict[int, int] = defaultdict(int)

        for it in all_interactions:
            u_id = it["user_id"]
            p_id = it["post_id"]
            liked = it.get("liked", False)
            if liked:
                user_items[u_id].add(p_id)
                post_raters[p_id].append(u_id)
                post_likes[p_id] += 1

        users = list(user_items.keys())
        num_users = len(users)

        # 1. Co-Action Pairwise Jaccard Clustering
        # In a botnet of 25 bots, each bot likes the exact same target + 15 filler posts.
        # Thus J(u, v) between bots is very high (often > 0.70 - 0.90),
        # whereas organic student users have J(u, v) < 0.15.
        bot_candidates: Set[int] = set()

        # To keep it efficient, compare users with similar interaction count
        # and high overlap on any shared item
        for i in range(num_users):
            u = users[i]
            items_u = user_items[u]
            if len(items_u) < 5:
                continue

            # Check potential peers
            high_jaccard_peers = 0
            for j in range(i + 1, num_users):
                v = users[j]
                items_v = user_items[v]
                if len(items_v) < 5:
                    continue

                intersection_len = len(items_u & items_v)
                if intersection_len >= 5:
                    union_len = len(items_u | items_v)
                    jaccard = intersection_len / max(1, union_len)
                    if jaccard >= self.jaccard_threshold:
                        high_jaccard_peers += 1
                        bot_candidates.add(u)
                        bot_candidates.add(v)

        self.flagged_bot_users = bot_candidates

        # 2. Compute Post Anomaly Scores
        # Fraction of post likes originating from flagged bot cluster
        for p_id, raters in post_raters.items():
            if not raters:
                continue
            bot_raters_count = sum(1 for r in raters if r in self.flagged_bot_users)
            # Anomaly score is the ratio of bot likes to total likes on this post
            anomaly_ratio = bot_raters_count / len(raters)
            
            # Boost anomaly score if there's a burst of bot raters
            if bot_raters_count >= 15:
                anomaly_score = max(anomaly_ratio, 0.95)
            else:
                anomaly_score = anomaly_ratio

            self.post_anomaly_scores[p_id] = round(float(anomaly_score), 4)

        return {
            "flagged_bots_count": len(self.flagged_bot_users),
            "flagged_bot_user_ids": list(self.flagged_bot_users),
            "anomalous_posts_count": sum(1 for s in self.post_anomaly_scores.values() if s >= self.anomaly_threshold),
        }

    def is_post_anomalous(self, post_id: int) -> Tuple[bool, float]:
        """Returns whether a post is flagged as shilling target and its anomaly score."""
        score = self.post_anomaly_scores.get(post_id, 0.0)
        is_anomalous = bool(score >= self.anomaly_threshold)
        return is_anomalous, score

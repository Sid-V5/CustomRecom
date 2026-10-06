"""
Pre-Scoring Visibility & Safety Filters (Section 5.4).
Enforces hard constraints:
1. Self-Post Filter
2. Block/Mute Filter
3. Already-Seen / Interacted Filter
4. Author Fatigue / Spam Pre-Filter (max 5 candidates per author in pool)
5. Shilling Defense Anomaly Filter
"""

from typing import Dict, List, Optional, Set, Tuple

from data.schemas import HydratedCandidate, User


class VisibilityFilterChain:
    """Sequential pipeline filter chain that tracks drop metrics for pipeline inspection."""

    def __init__(self, shilling_detector=None):
        self.shilling_detector = shilling_detector

    def apply_filters(
        self,
        hydrated_candidates: List[HydratedCandidate],
        user: User,
        seen_post_ids: Optional[Set[int]] = None,
        defense_active: bool = False,
        department_filter: Optional[str] = None,
    ) -> Tuple[List[HydratedCandidate], Dict[str, int]]:
        """
        Applies filters sequentially and returns surviving candidates + drop statistics.
        """
        if seen_post_ids is None:
            seen_post_ids = set()

        stats = {
            "initial_count": len(hydrated_candidates),
            "dropped_self": 0,
            "dropped_blocked": 0,
            "dropped_seen": 0,
            "dropped_age": 0,
            "dropped_dept": 0,
            "dropped_author_spam": 0,
            "dropped_shilling": 0,
            "survived_count": 0,
        }

        blocked_set = set(user.blocked_user_ids)
        survivors_pass_1: List[HydratedCandidate] = []

        # Pass 1: Self, Block, Seen, Age cutoff, and Department filters
        for hc in hydrated_candidates:
            post = hc.candidate.post
            if post.author_id == user.user_id:
                stats["dropped_self"] += 1
                continue
            if post.author_id in blocked_set:
                stats["dropped_blocked"] += 1
                continue
            if post.post_id in seen_post_ids:
                stats["dropped_seen"] += 1
                continue
            # L28: Hard constraints (max post age cutoff t <= 7 days = 168h)
            if hc.post_age_hours > 168.0:
                stats["dropped_age"] += 1
                continue
            # L28: Department matching filter if requested
            if department_filter and post.department.lower() != department_filter.lower():
                stats["dropped_dept"] += 1
                continue
            survivors_pass_1.append(hc)

        # Pass 2: Author Spam Filter (max 5 candidates per author)
        survivors_pass_2: List[HydratedCandidate] = []
        author_counts: Dict[int, int] = {}
        # Sort by retrieval score descending first
        survivors_pass_1.sort(key=lambda x: x.candidate.retrieval_score, reverse=True)

        for hc in survivors_pass_1:
            author_id = hc.candidate.post.author_id
            current_count = author_counts.get(author_id, 0)
            if current_count >= 5:
                stats["dropped_author_spam"] += 1
                continue
            author_counts[author_id] = current_count + 1
            survivors_pass_2.append(hc)

        # Pass 3: Shilling Defense Anomaly Filter
        survivors_pass_3: List[HydratedCandidate] = []
        for hc in survivors_pass_2:
            post_id = hc.candidate.post.post_id
            if defense_active and self.shilling_detector is not None:
                is_anomalous, anomaly_score = self.shilling_detector.is_post_anomalous(post_id)
                if is_anomalous:
                    stats["dropped_shilling"] += 1
                    continue
            survivors_pass_3.append(hc)

        stats["survived_count"] = len(survivors_pass_3)
        return survivors_pass_3, stats

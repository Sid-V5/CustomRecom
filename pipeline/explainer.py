"""
Explainability Engine (Section 5.8).
Decomposes scoring signals and generates human-interpretable "Why this post?"
explanations, confidence scores, and feature weight breakdowns for UI display.
"""

from typing import Any, Dict, List
from data.schemas import ScoredCandidate, User


class Explainer:
    """Assigns transparent reasoning badges and attribution weights to recommended posts."""

    def explain(self, scored_candidate: ScoredCandidate, user: User) -> Dict[str, Any]:
        """Generate structured explanation metadata for a scored candidate."""
        hc = scored_candidate.hydrated
        post = hc.candidate.post
        source = hc.candidate.source
        feat = hc.feature_vector or [0.0] * 12

        # Features:
        # 0: als_dot, 1: content_cos, 2: is_in_network, 3: author_rep, 4: user_auth_hist,
        # 5: likes_log, 6: replies_log, 7: age_h, 8: tag_jaccard, 9: dwell_mean,
        # 10: pop_percentile, 11: dept_match
        als_dot = feat[0]
        content_cos = feat[1]
        is_in_network = feat[2] > 0.5
        tag_jaccard = feat[8]
        dept_match = feat[11] > 0.5

        # Shared tags
        matched_tags = [t for t in post.tags if t in user.preferred_tags]

        # Determine dominant explanation signal
        primary_reason = "Recommended for your campus feed"
        badge_type = "discovery"
        badge_color = "indigo"

        if is_in_network or source == "in_network":
            primary_reason = "From an account you follow"
            badge_type = "network"
            badge_color = "blue"
        elif matched_tags:
            primary_reason = f"Matches your interest in {matched_tags[0]}"
            badge_type = "topic"
            badge_color = "emerald"
        elif source == "trending_velocity" or post.replies_count >= 5 or post.likes_count >= 20:
            primary_reason = "Trending campus discussion"
            badge_type = "trending"
            badge_color = "rose"
        elif source == "als_latent" or als_dot > 0.2:
            primary_reason = "Students with similar interests liked this"
            badge_type = "collaborative"
            badge_color = "purple"
        elif dept_match and post.likes_count > 10:
            primary_reason = f"Popular in {user.department}"
            badge_type = "community"
            badge_color = "amber"

        # Signal attribution breakdown
        raw_conf = min(99.0, max(25.0, scored_candidate.p_like * 100.0))
        
        # Breakdown of scoring components
        total_signal = max(0.001, scored_candidate.p_like + scored_candidate.p_reply + (0.5 if is_in_network else 0.1))
        network_weight = (0.5 if is_in_network else 0.05) / total_signal
        topic_weight = max(0.1, content_cos) / total_signal
        social_weight = scored_candidate.p_like / total_signal
        discussion_weight = scored_candidate.p_reply / total_signal

        norm = network_weight + topic_weight + social_weight + discussion_weight
        feature_weights = {
            "network_affinity": round((network_weight / norm) * 100, 1),
            "topic_alignment": round((topic_weight / norm) * 100, 1),
            "peer_collaborative": round((social_weight / norm) * 100, 1),
            "discussion_velocity": round((discussion_weight / norm) * 100, 1),
        }

        return {
            "primary_reason": primary_reason,
            "badge_type": badge_type,
            "badge_color": badge_color,
            "confidence_score": round(raw_conf, 1),
            "source": source,
            "matched_tags": matched_tags,
            "feature_weights": feature_weights,
            "probabilities": {
                "p_like": round(scored_candidate.p_like * 100, 1),
                "p_reply": round(scored_candidate.p_reply * 100, 1),
                "p_skip": round(scored_candidate.p_skip * 100, 1),
            },
        }

    def explain_feed(self, candidates: List[ScoredCandidate], user: User) -> List[ScoredCandidate]:
        """Attach explanations to all candidates in feed."""
        for cand in candidates:
            cand.explanation = self.explain(cand, user)
        return candidates

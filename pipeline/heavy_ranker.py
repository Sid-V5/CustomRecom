"""
Heavy Ranker & Cold-Start Routing Stage (Section 5.5).
Implements:
1. Cold-Start Switching: Routes users with < 3 interactions to Content + Knowledge rules.
2. Warm User Calibrated Multi-Action Ranker: Uses 12-dim feature vector and calibrated
   Logistic Regression models to predict P(like), P(reply), and P(skip).
3. Configurable composite utility score with recency exponential decay.
"""

import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACTS_DIR,
    COLD_USER_THRESHOLD,
    DEFAULT_LAMBDA_AGE,
    DEFAULT_WEIGHT_LIKE,
    DEFAULT_WEIGHT_REPLY,
    DEFAULT_WEIGHT_SKIP,
)
from data.schemas import HydratedCandidate, ScoredCandidate, User


class HeavyRanker:
    """Multi-Signal Heavy Ranker with Cold-Start switching."""

    def __init__(
        self,
        w_like: float = DEFAULT_WEIGHT_LIKE,
        w_reply: float = DEFAULT_WEIGHT_REPLY,
        w_skip: float = DEFAULT_WEIGHT_SKIP,
        lambda_age: float = DEFAULT_LAMBDA_AGE,
    ):
        self.w_like = w_like
        self.w_reply = w_reply
        self.w_skip = w_skip
        self.lambda_age = lambda_age

        # Multi-Action Calibrated Classifiers
        self.model_like: Optional[CalibratedClassifierCV] = None
        self.model_reply: Optional[CalibratedClassifierCV] = None
        self.model_skip: Optional[CalibratedClassifierCV] = None
        self.is_trained: bool = False

    def train(self, X: np.ndarray, y_like: np.ndarray, y_reply: np.ndarray, y_skip: np.ndarray):
        """Train calibrated logistic models on 12-dimensional hydrated feature matrix."""
        # Model 1: P(like)
        pipe_like = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, C=1.0))])
        self.model_like = CalibratedClassifierCV(estimator=pipe_like, method="sigmoid", cv=3)
        self.model_like.fit(X, y_like)

        # Model 2: P(reply)
        pipe_reply = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, C=1.0))])
        self.model_reply = CalibratedClassifierCV(estimator=pipe_reply, method="sigmoid", cv=3)
        self.model_reply.fit(X, y_reply)

        # Model 3: P(skip)
        pipe_skip = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, C=1.0))])
        self.model_skip = CalibratedClassifierCV(estimator=pipe_skip, method="sigmoid", cv=3)
        self.model_skip.fit(X, y_skip)

        self.is_trained = True
        return self

    def score_candidates(
        self,
        hydrated_candidates: List[HydratedCandidate],
        user: User,
        user_interaction_count: int,
        w_like: Optional[float] = None,
        w_reply: Optional[float] = None,
        w_skip: Optional[float] = None,
        lambda_age: Optional[float] = None,
    ) -> List[ScoredCandidate]:
        """Scores candidate posts using either Cold-Start Switch or Warm Calibrated Ranker."""
        wl = w_like if w_like is not None else self.w_like
        wr = w_reply if w_reply is not None else self.w_reply
        ws = w_skip if w_skip is not None else self.w_skip
        lage = lambda_age if lambda_age is not None else self.lambda_age

        scored_list: List[ScoredCandidate] = []

        # -------------------------------------------------------------
        # 1. Cold-Start User Routing Path (< 3 interactions)
        # -------------------------------------------------------------
        if user_interaction_count < COLD_USER_THRESHOLD:
            for hc in hydrated_candidates:
                post = hc.candidate.post
                u_tags = set(user.preferred_tags)
                p_tags = set(post.tags)
                union_len = len(u_tags | p_tags)
                tag_jaccard = (len(u_tags & p_tags) / union_len) if union_len > 0 else 0.0
                dept_match = 1.0 if post.department == user.department else 0.0
                recency_factor = math.exp(-0.05 * hc.post_age_hours)

                # Pure content + knowledge baseline score
                cold_score = 0.50 * tag_jaccard + 0.30 * dept_match + 0.20 * recency_factor

                p_like = float(np.clip(cold_score, 0.01, 0.99))
                p_reply = float(np.clip(cold_score * 0.20, 0.01, 0.50))
                p_skip = float(np.clip(1.0 - cold_score, 0.01, 0.99))

                raw_score = wl * p_like + wr * p_reply - ws * p_skip
                final_score = raw_score * math.exp(-lage * hc.post_age_hours)

                scored_list.append(
                    ScoredCandidate(
                        hydrated=hc,
                        p_like=round(p_like, 4),
                        p_reply=round(p_reply, 4),
                        p_skip=round(p_skip, 4),
                        final_score=round(final_score, 5),
                        explanation={},  # Will be populated by explainer
                    )
                )
            return scored_list

        # -------------------------------------------------------------
        # 2. Warm User Routing Path (Calibrated Multi-Action Ranker)
        # -------------------------------------------------------------
        if not self.is_trained or len(hydrated_candidates) == 0:
            # Fallback heuristic if ranker not yet trained
            for hc in hydrated_candidates:
                p_like = 0.3
                p_reply = 0.05
                p_skip = 0.4
                final_score = wl * p_like + wr * p_reply - ws * p_skip
                scored_list.append(
                    ScoredCandidate(
                        hydrated=hc,
                        p_like=p_like,
                        p_reply=p_reply,
                        p_skip=p_skip,
                        final_score=final_score,
                        explanation={},
                    )
                )
            return scored_list

        X = np.array([hc.feature_vector for hc in hydrated_candidates], dtype=np.float32)

        # Batch predict calibrated probabilities
        p_likes = self.model_like.predict_proba(X)[:, 1]
        p_replies = self.model_reply.predict_proba(X)[:, 1]
        p_skips = self.model_skip.predict_proba(X)[:, 1]

        for idx, hc in enumerate(hydrated_candidates):
            pl = float(p_likes[idx])
            pr = float(p_replies[idx])
            ps = float(p_skips[idx])

            # Social discussion velocity & viral amplification (Section 5.5, 5.9):
            # Posts with massive engagement velocity (e.g. viral promotion or coordinated like-bombing)
            # receive proportional popularity amplification in P(like) and reduction in P(skip)
            likes_count = hc.candidate.post.likes_count
            if likes_count >= 15:
                velocity_boost = min(0.65, 0.025 * likes_count)
                pl = min(0.99, pl + velocity_boost)
                pr = min(0.85, pr + velocity_boost * 0.4)
                ps = max(0.01, ps * (1.0 - velocity_boost))

            raw_score = wl * pl + wr * pr - ws * ps
            decay = math.exp(-lage * hc.post_age_hours)
            final_score = raw_score * decay

            scored_list.append(
                ScoredCandidate(
                    hydrated=hc,
                    p_like=round(pl, 4),
                    p_reply=round(pr, 4),
                    p_skip=round(ps, 4),
                    final_score=round(final_score, 5),
                    explanation={},
                )
            )

        return scored_list

    def save(self, filepath: Optional[Path] = None):
        if filepath is None:
            filepath = ARTIFACTS_DIR / "heavy_ranker.pkl"
        with open(filepath, "wb") as f:
            pickle.dump({
                "model_like": self.model_like,
                "model_reply": self.model_reply,
                "model_skip": self.model_skip,
                "w_like": self.w_like,
                "w_reply": self.w_reply,
                "w_skip": self.w_skip,
                "lambda_age": self.lambda_age,
                "is_trained": self.is_trained,
            }, f)

    def load(self, filepath: Optional[Path] = None):
        if filepath is None:
            filepath = ARTIFACTS_DIR / "heavy_ranker.pkl"
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model_like = data["model_like"]
            self.model_reply = data["model_reply"]
            self.model_skip = data["model_skip"]
            self.w_like = data.get("w_like", DEFAULT_WEIGHT_LIKE)
            self.w_reply = data.get("w_reply", DEFAULT_WEIGHT_REPLY)
            self.w_skip = data.get("w_skip", DEFAULT_WEIGHT_SKIP)
            self.lambda_age = data.get("lambda_age", DEFAULT_LAMBDA_AGE)
            self.is_trained = data["is_trained"]
        return self

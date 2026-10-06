"""
Implicit Alternating Least Squares (iALS) Matrix Factorization & SVD Baseline.
Implements:
- TruncatedSVD baseline (Lecture L11)
- Confidence-weighted Implicit ALS (Lectures L12, L13, L14, L15)
  with C_ui = 1 + alpha * r_ui and L2 regularization.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

from config import ALS_ALPHA, ALS_FACTORS, ALS_ITERATIONS, ALS_REGULARIZATION, ARTIFACTS_DIR


class TruncatedSVDBaseline:
    """Matrix Factorization using TruncatedSVD (Lecture L11)."""

    def __init__(self, n_components: int = ALS_FACTORS):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    def fit(self, R: np.ndarray):
        """Fit SVD on user-item matrix R (num_users, num_items)."""
        # TruncatedSVD fits X = U * Sigma * V^T
        user_factors = self.svd.fit_transform(R)  # U * Sigma
        self.user_factors = user_factors
        self.item_factors = self.svd.components_.T  # V: (num_items, n_components)
        return self

    def predict_score(self, user_id: int, item_id: int) -> float:
        if user_id >= len(self.user_factors) or item_id >= len(self.item_factors):
            return 0.0
        return float(np.dot(self.user_factors[user_id], self.item_factors[item_id]))

    def predict_user(self, user_id: int) -> np.ndarray:
        if user_id >= len(self.user_factors):
            return np.zeros(len(self.item_factors), dtype=np.float32)
        return self.user_factors[user_id] @ self.item_factors.T


class ImplicitALS:
    """
    Implicit Feedback Alternating Least Squares (iALS).
    Optimizes confidence-weighted least squares:
    L_ALS = sum_ui C_ui (p_ui - x_u^T y_i)^2 + lambda (sum_u ||x_u||^2 + sum_i ||y_i||^2)
    where C_ui = 1 + alpha * r_ui and p_ui = 1 if r_ui > 0 else 0.
    """

    def __init__(
        self,
        factors: int = ALS_FACTORS,
        alpha: float = ALS_ALPHA,
        regularization: float = ALS_REGULARIZATION,
        iterations: int = ALS_ITERATIONS,
        seed: int = 42,
    ):
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

        self.user_factors: Optional[np.ndarray] = None  # (num_users, factors)
        self.item_factors: Optional[np.ndarray] = None  # (num_items, factors)
        self.num_users: int = 0
        self.num_items: int = 0

    def fit(self, R: np.ndarray, verbose: bool = False):
        """
        Fit iALS on interaction matrix R (num_users, num_items).
        R contains non-negative implicit interaction weights (e.g., log dwell / likes).
        """
        np.random.seed(self.seed)
        self.num_users, self.num_items = R.shape

        # Initialize factors randomly
        self.user_factors = np.random.normal(0, 0.01, (self.num_users, self.factors)).astype(np.float32)
        self.item_factors = np.random.normal(0, 0.01, (self.num_items, self.factors)).astype(np.float32)

        # Pre-build sparse indices for fast iteration
        # Users' non-zero item indices and ratings
        user_items = [np.where(R[u] > 0)[0] for u in range(self.num_users)]
        user_ratings = [R[u, user_items[u]] for u in range(self.num_users)]

        # Items' non-zero user indices and ratings
        item_users = [np.where(R[:, i] > 0)[0] for i in range(self.num_items)]
        item_ratings = [R[item_users[i], i] for i in range(self.num_items)]

        reg_eye = self.regularization * np.eye(self.factors, dtype=np.float32)

        for it in range(self.iterations):
            # 1. Update user factors X fixing item factors Y
            # Y^T Y is shared across all users
            YtY = self.item_factors.T @ self.item_factors  # (factors, factors)

            for u in range(self.num_users):
                items = user_items[u]
                if len(items) == 0:
                    continue
                r_u = user_ratings[u]
                # Confidence: C_ui = 1 + alpha * r_ui
                c_u_minus_1 = self.alpha * r_u  # non-zero diag of (C_u - I)
                Y_u = self.item_factors[items]  # (|I_u|, factors)

                # A_u = Y^T Y + Y_u^T diag(c_u - 1) Y_u + lambda * I
                A_u = YtY + (Y_u.T * c_u_minus_1) @ Y_u + reg_eye
                # b_u = Y_u^T (1 + c_u_minus_1)
                b_u = (Y_u.T * (1.0 + c_u_minus_1)).sum(axis=1)

                self.user_factors[u] = np.linalg.solve(A_u, b_u)

            # 2. Update item factors Y fixing user factors X
            XtX = self.user_factors.T @ self.user_factors  # (factors, factors)

            for i in range(self.num_items):
                users = item_users[i]
                if len(users) == 0:
                    continue
                r_i = item_ratings[i]
                c_i_minus_1 = self.alpha * r_i
                X_i = self.user_factors[users]  # (|U_i|, factors)

                A_i = XtX + (X_i.T * c_i_minus_1) @ X_i + reg_eye
                b_i = (X_i.T * (1.0 + c_i_minus_1)).sum(axis=1)

                self.item_factors[i] = np.linalg.solve(A_i, b_i)

            if verbose:
                # Approximate loss sample
                print(f"iALS Iteration {it + 1}/{self.iterations} complete")

        return self

    def predict_score(self, user_id: int, item_id: int) -> float:
        """Dot product affinity r_hat = x_u^T y_i."""
        if self.user_factors is None or user_id >= self.num_users or item_id >= self.num_items:
            return 0.0
        return float(np.dot(self.user_factors[user_id], self.item_factors[item_id]))

    def predict_user_scores(self, user_id: int) -> np.ndarray:
        """Compute scores for all items: x_u Y^T."""
        if self.user_factors is None or user_id >= self.num_users:
            return np.zeros(self.num_items, dtype=np.float32)
        return self.user_factors[user_id] @ self.item_factors.T

    def get_top_candidates(
        self, user_id: int, top_n: int = 100, exclude_ids: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """Retrieve top candidate item IDs using latent factor projection."""
        scores = self.predict_user_scores(user_id)
        if exclude_ids:
            for item_id in exclude_ids:
                if item_id < len(scores):
                    scores[item_id] = -1e9

        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > -1e8]

    def save(self, filepath: Optional[Path] = None):
        """Save factor matrices to disk."""
        if filepath is None:
            filepath = ARTIFACTS_DIR / "ials_model.pkl"
        with open(filepath, "wb") as f:
            pickle.dump({
                "user_factors": self.user_factors,
                "item_factors": self.item_factors,
                "factors": self.factors,
                "num_users": self.num_users,
                "num_items": self.num_items,
            }, f)

    def load(self, filepath: Optional[Path] = None):
        """Load factor matrices from disk."""
        if filepath is None:
            filepath = ARTIFACTS_DIR / "ials_model.pkl"
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.user_factors = data["user_factors"]
            self.item_factors = data["item_factors"]
            self.factors = data["factors"]
            self.num_users = data["num_users"]
            self.num_items = data["num_items"]
        return self

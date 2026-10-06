"""
Collaborative Filtering k-Nearest Neighbors (kNN).
Implements User-Based and Item-Based Collaborative Filtering with:
- Centered Cosine (Pearson Correlation) and Vectorized Cosine similarity
- Significance weighting threshold (tau = 5)
- Neighborhood aggregation with rating prediction
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse

from config import KNN_K, SIGNIFICANCE_THRESHOLD_TAU


class UserKNN:
    """
    User-Based Collaborative Filtering (Lecture L6, L7, L9).
    Finds peer users with similar interaction profiles and aggregates their ratings.
    """

    def __init__(self, k: int = KNN_K, tau: float = SIGNIFICANCE_THRESHOLD_TAU, similarity_metric: str = "pearson"):
        self.k = k
        self.tau = tau
        self.similarity_metric = similarity_metric
        self.R: Optional[np.ndarray] = None  # User-Item interaction matrix (num_users, num_items)
        self.user_means: Optional[np.ndarray] = None
        self.global_mean: float = 0.0
        self.num_users: int = 0
        self.num_items: int = 0
        self.sim_matrix: Optional[np.ndarray] = None

    def fit(self, R: np.ndarray):
        """Fit model on user-item matrix R (num_users, num_items)."""
        self.R = np.array(R, dtype=np.float32)
        self.num_users, self.num_items = self.R.shape
        
        # Non-zero counts per user
        mask = (self.R > 0)
        counts = mask.sum(axis=1)
        sums = self.R.sum(axis=1)
        self.user_means = np.zeros(self.num_users, dtype=np.float32)
        nonzero_users = counts > 0
        self.user_means[nonzero_users] = sums[nonzero_users] / counts[nonzero_users]
        
        total_interactions = mask.sum()
        self.global_mean = float(self.R.sum() / max(1, total_interactions))
        
        self._compute_similarity_matrix()
        return self

    def _compute_similarity_matrix(self):
        """Precompute user-user similarity matrix with significance weighting."""
        self.sim_matrix = np.zeros((self.num_users, self.num_users), dtype=np.float32)
        binary_mask = (self.R > 0).astype(np.float32)

        # Co-rated count matrix: intersection |I_uv|
        co_rated = binary_mask @ binary_mask.T  # (U, U)

        if self.similarity_metric == "pearson":
            # Mean-centered rating matrix
            R_centered = np.zeros_like(self.R)
            for u in range(self.num_users):
                idx = self.R[u] > 0
                if np.any(idx):
                    R_centered[u, idx] = self.R[u, idx] - self.user_means[u]

            norms = np.linalg.norm(R_centered, axis=1)  # (U,)
            norms[norms == 0] = 1e-9
            raw_sim = (R_centered @ R_centered.T) / np.outer(norms, norms)
        else:  # Cosine
            norms = np.linalg.norm(self.R, axis=1)
            norms[norms == 0] = 1e-9
            raw_sim = (self.R @ self.R.T) / np.outer(norms, norms)

        # Significance weighting: min(|I_uv|, tau) / tau
        sig_weights = np.clip(co_rated / self.tau, 0.0, 1.0)
        self.sim_matrix = raw_sim * sig_weights
        np.fill_diagonal(self.sim_matrix, 0.0)

    def predict_score(self, user_id: int, item_id: int) -> float:
        """
        Predict rating/interaction score using top-k user neighbors who rated item_id:
        r_hat = user_mean + sum(sim * (r_v - v_mean)) / sum(|sim|)
        """
        if user_id >= self.num_users or item_id >= self.num_items:
            return self.global_mean

        # Find users who have rated this item
        item_raters = np.where(self.R[:, item_id] > 0)[0]
        # Remove target user
        item_raters = item_raters[item_raters != user_id]

        if len(item_raters) == 0:
            return float(self.user_means[user_id]) if self.user_means[user_id] > 0 else self.global_mean

        sims = self.sim_matrix[user_id, item_raters]
        # Take top-k positive or highest magnitude neighbors
        if len(sims) > self.k:
            top_k_indices = np.argsort(sims)[-self.k:]
            chosen_raters = item_raters[top_k_indices]
            chosen_sims = sims[top_k_indices]
        else:
            chosen_raters = item_raters
            chosen_sims = sims

        abs_sim_sum = np.sum(np.abs(chosen_sims))
        if abs_sim_sum < 1e-7:
            return float(self.user_means[user_id]) if self.user_means[user_id] > 0 else self.global_mean

        if self.similarity_metric == "pearson":
            diffs = self.R[chosen_raters, item_id] - self.user_means[chosen_raters]
            pred = self.user_means[user_id] + np.sum(chosen_sims * diffs) / abs_sim_sum
        else:
            pred = np.sum(chosen_sims * self.R[chosen_raters, item_id]) / abs_sim_sum

        return float(pred)

    def recommend(self, user_id: int, top_n: int = 10, candidate_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Return top-N recommended items for user_id using vectorized top-k neighborhood aggregation."""
        if self.R is None or user_id >= self.num_users:
            return []

        user_sims = self.sim_matrix[user_id]  # (num_users,)
        top_k_users = np.argsort(user_sims)[-self.k:]
        chosen_sims = user_sims[top_k_users]  # (k,)
        abs_sum = np.sum(np.abs(chosen_sims))

        if abs_sum < 1e-7:
            scores = np.mean(self.R, axis=0)
        else:
            if self.similarity_metric == "pearson":
                diffs = self.R[top_k_users] - self.user_means[top_k_users, None]
                scores = self.user_means[user_id] + (chosen_sims @ diffs) / abs_sum
            else:
                scores = (chosen_sims @ self.R[top_k_users]) / abs_sum

        if candidate_ids is None:
            scores[self.R[user_id] > 0] = -1e9
            top_indices = np.argsort(scores)[-top_n:][::-1]
            return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > -1e8]
        else:
            cand_arr = np.array(candidate_ids)
            cand_scores = scores[cand_arr]
            top_idx = np.argsort(cand_scores)[-top_n:][::-1]
            return [(int(cand_arr[i]), float(cand_scores[i])) for i in top_idx]


class ItemKNN:
    """
    Item-Based Collaborative Filtering (Lecture L8).
    Computes item-item similarity based on co-engagement cosine similarity.
    Provides candidate expansion for Candidate Retrieval Engine B.
    """

    def __init__(self, k: int = KNN_K, tau: float = SIGNIFICANCE_THRESHOLD_TAU):
        self.k = k
        self.tau = tau
        self.R: Optional[np.ndarray] = None
        self.item_sim_matrix: Optional[np.ndarray] = None
        self.num_users: int = 0
        self.num_items: int = 0

    def fit(self, R: np.ndarray):
        """Fit item-item similarity matrix on interaction matrix R (num_users, num_items)."""
        self.R = np.array(R, dtype=np.float32)
        self.num_users, self.num_items = self.R.shape

        # Transpose so rows are items
        R_item = self.R.T  # (num_items, num_users)
        binary_mask = (R_item > 0).astype(np.float32)

        co_rated = binary_mask @ binary_mask.T  # (I, I)
        norms = np.linalg.norm(R_item, axis=1)  # (I,)
        norms[norms == 0] = 1e-9

        raw_sim = (R_item @ R_item.T) / np.outer(norms, norms)
        sig_weights = np.clip(co_rated / self.tau, 0.0, 1.0)
        self.item_sim_matrix = raw_sim * sig_weights
        np.fill_diagonal(self.item_sim_matrix, 0.0)
        return self

    def get_similar_items(self, item_id: int, top_n: int = 5) -> List[Tuple[int, float]]:
        """Retrieve top-N nearest neighbors for item_id."""
        if self.item_sim_matrix is None or item_id >= self.num_items:
            return []
        sims = self.item_sim_matrix[item_id]
        top_indices = np.argsort(sims)[-top_n:][::-1]
        return [(int(idx), float(sims[idx])) for idx in top_indices if sims[idx] > 0]

    def predict_score(self, user_id: int, item_id: int) -> float:
        """Predict rating/affinity score using item neighbors user has interacted with."""
        if self.item_sim_matrix is None or user_id >= self.num_users or item_id >= self.num_items:
            return 0.0

        user_interacted = np.where(self.R[user_id] > 0)[0]
        if len(user_interacted) == 0:
            return 0.0

        sims = self.item_sim_matrix[item_id, user_interacted]
        if len(sims) > self.k:
            top_k_idx = np.argsort(sims)[-self.k:]
            chosen_items = user_interacted[top_k_idx]
            chosen_sims = sims[top_k_idx]
        else:
            chosen_items = user_interacted
            chosen_sims = sims

        sum_sims = np.sum(np.abs(chosen_sims))
        if sum_sims < 1e-7:
            return 0.0

        pred = np.sum(chosen_sims * self.R[user_id, chosen_items]) / sum_sims
        return float(pred)

    def recommend(self, user_id: int, top_n: int = 10, candidate_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        if self.item_sim_matrix is None or user_id >= self.num_users:
            return []

        user_ratings = self.R[user_id]  # (num_items,)
        sim_sums = np.sum(np.abs(self.item_sim_matrix), axis=1)
        sim_sums[sim_sums < 1e-7] = 1.0
        scores = (user_ratings @ self.item_sim_matrix) / sim_sums

        if candidate_ids is None:
            scores[user_ratings > 0] = -1e9
            top_indices = np.argsort(scores)[-top_n:][::-1]
            return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > -1e8]
        else:
            cand_arr = np.array(candidate_ids)
            cand_scores = scores[cand_arr]
            top_idx = np.argsort(cand_scores)[-top_n:][::-1]
            return [(int(cand_arr[i]), float(cand_scores[i])) for i in top_idx]

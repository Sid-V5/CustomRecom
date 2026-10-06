"""
Content-Based Recommendation Engine (Lectures L18, L22, L24, L25, L29).
Builds TF-IDF representations from post text and tags,
constructs dynamic user centroid interest profiles,
and computes content-based cosine similarities for retrieval and MMR diversity.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import ARTIFACTS_DIR, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
from data.schemas import Post, User


class ContentBasedModel:
    """
    Content-Based Model with TF-IDF Vectorizer and Dynamic User Centroid Profiling.
    """

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b|#\w+",
        )
        self.post_vectors: Optional[np.ndarray] = None  # (num_posts, num_features)
        self.post_id_to_idx: Dict[int, int] = {}
        self.idx_to_post_id: Dict[int, int] = {}
        self.user_profiles: Dict[int, np.ndarray] = {}  # user_id -> centroid vector (1, num_features)

    def fit(self, posts: List[Post]):
        """Fit TF-IDF on corpus of posts and compute item feature vectors."""
        corpus = []
        for idx, post in enumerate(posts):
            self.post_id_to_idx[post.post_id] = idx
            self.idx_to_post_id[idx] = post.post_id
            # Combine content and explicit tag tokens
            tags_text = " ".join(post.tags)
            doc = f"{post.content} {tags_text} {post.department}"
            corpus.append(doc)

        tfidf_sparse = self.vectorizer.fit_transform(corpus)
        self.post_vectors = tfidf_sparse.toarray().astype(np.float32)  # (N, D)
        return self

    def build_user_profiles(self, users: List[User], interactions: List[Dict]):
        """
        Build dynamic centroid profile for each user:
        U_u = sum(w_i * V_i) / ||sum(w_i * V_i)||
        If user is cold-start (no likes), use tag text fallback.
        """
        user_interactions: Dict[int, List[Dict]] = {u.user_id: [] for u in users}
        for inter in interactions:
            u_id = inter["user_id"]
            if u_id in user_interactions:
                user_interactions[u_id].append(inter)

        for user in users:
            u_id = user.user_id
            inters = user_interactions.get(u_id, [])
            liked_inters = [it for it in inters if it.get("liked", False) or it.get("dwell_seconds", 0) > 10.0]

            if len(liked_inters) > 0 and self.post_vectors is not None:
                # Dwell-weighted centroid
                centroid = np.zeros(self.post_vectors.shape[1], dtype=np.float32)
                total_weight = 0.0

                for it in liked_inters:
                    p_id = it["post_id"]
                    if p_id in self.post_id_to_idx:
                        p_idx = self.post_id_to_idx[p_id]
                        dwell = it.get("dwell_seconds", 5.0)
                        reply_bonus = 2.0 if it.get("replied", False) else 1.0
                        w = np.log1p(dwell) * reply_bonus
                        centroid += w * self.post_vectors[p_idx]
                        total_weight += w

                norm = np.linalg.norm(centroid)
                if norm > 1e-7:
                    self.user_profiles[u_id] = (centroid / norm).reshape(1, -1)
                else:
                    self._build_tag_profile(user)
            else:
                self._build_tag_profile(user)

        return self

    def _build_tag_profile(self, user: User):
        """Cold-start fallback: embed user's preferred tags and department."""
        text = f"{' '.join(user.preferred_tags)} {user.department} {user.persona}"
        vec = self.vectorizer.transform([text]).toarray().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-7:
            vec = vec / norm
        self.user_profiles[user.user_id] = vec

    def get_user_profile(self, user: User) -> np.ndarray:
        """Get or lazily create user profile vector."""
        if user.user_id in self.user_profiles:
            return self.user_profiles[user.user_id]
        self._build_tag_profile(user)
        return self.user_profiles[user.user_id]

    def get_post_vector(self, post_id: int) -> Optional[np.ndarray]:
        """Get feature vector for a post."""
        if post_id in self.post_id_to_idx and self.post_vectors is not None:
            return self.post_vectors[self.post_id_to_idx[post_id]]
        return None

    def similarity_between_posts(self, post_id_1: int, post_id_2: int) -> float:
        """Compute cosine similarity between two posts for MMR diversity."""
        v1 = self.get_post_vector(post_id_1)
        v2 = self.get_post_vector(post_id_2)
        if v1 is None or v2 is None:
            return 0.0
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-7 or n2 < 1e-7:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def similarity_user_post(self, user: User, post_id: int) -> float:
        """Compute cosine similarity between user centroid profile and a post."""
        u_vec = self.get_user_profile(user)
        p_vec = self.get_post_vector(post_id)
        if p_vec is None:
            return 0.0
        p_norm = np.linalg.norm(p_vec)
        if p_norm < 1e-7:
            return 0.0
        p_vec_norm = p_vec / p_norm
        return float(np.dot(u_vec.squeeze(), p_vec_norm))

    def get_top_candidates(
        self, user: User, top_n: int = 50, exclude_ids: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """Retrieve top content-matching post candidates for user."""
        if self.post_vectors is None:
            return []

        u_vec = self.get_user_profile(user)  # (1, D)
        # Fast batch matrix-vector cosine
        scores = (self.post_vectors @ u_vec.T).squeeze()  # (N,)

        if exclude_ids:
            for p_id in exclude_ids:
                if p_id in self.post_id_to_idx:
                    scores[self.post_id_to_idx[p_id]] = -1.0

        top_indices = np.argsort(scores)[-top_n:][::-1]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.idx_to_post_id[idx], score))
        return results

    def case_based_query_search(self, query: str, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Case-Based Recommendation (Lecture L29):
        Student enters a homework/project problem description, system finds
        structurally similar campus posts.
        """
        if self.post_vectors is None:
            return []
        q_vec = self.vectorizer.transform([query]).toarray().astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm < 1e-7:
            return []
        q_vec = q_vec / q_norm
        scores = (self.post_vectors @ q_vec.T).squeeze()
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [(self.idx_to_post_id[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def save(self, filepath: Optional[Path] = None):
        if filepath is None:
            filepath = ARTIFACTS_DIR / "content_model.pkl"
        with open(filepath, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "post_vectors": self.post_vectors,
                "post_id_to_idx": self.post_id_to_idx,
                "idx_to_post_id": self.idx_to_post_id,
                "user_profiles": self.user_profiles,
            }, f)

    def load(self, filepath: Optional[Path] = None):
        if filepath is None:
            filepath = ARTIFACTS_DIR / "content_model.pkl"
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.post_vectors = data["post_vectors"]
            self.post_id_to_idx = data["post_id_to_idx"]
            self.idx_to_post_id = data["idx_to_post_id"]
            self.user_profiles = data["user_profiles"]
        return self

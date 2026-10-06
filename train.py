"""
Master Training Script for Mini-X Recommender Pipeline.
Trains and serializes all models:
1. User-kNN & Item-kNN (Pearson/Cosine)
2. Implicit ALS Factorization
3. Content-Based TF-IDF & Dynamic Centroid Profiler
4. PyTorch Two-Tower Neural Network
5. Multi-Action Heavy Ranker (Calibrated P(like), P(reply), P(skip))
6. Shilling Defense Anomaly Detector
Saves all model artifacts to /artifacts directory.
"""

import csv
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import (
    ARTIFACTS_DIR,
    EXPLICIT_HOLDOUT_FILE,
    FOLLOWS_FILE,
    INTERACTIONS_FILE,
    NUM_POSTS,
    NUM_USERS,
    POSTS_FILE,
    USERS_FILE,
)
from data.schemas import Post, User
from models.als_factorization import ImplicitALS
from models.collaborative_knn import ItemKNN, UserKNN
from models.content_based import ContentBasedModel
from models.neural_two_tower import TwoTowerRecommender
from pipeline.heavy_ranker import HeavyRanker
from pipeline.hydrator import FeatureHydrator
from security.shilling_detector import ShillingDefenseDetector


def load_dataset() -> Tuple[List[User], List[Post], List[Tuple[int, int]], List[Dict], List[Dict]]:
    """Load raw synthetic dataset files."""
    print("Loading raw dataset from disk...")

    # 1. Users
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        users_raw = json.load(f)
        users = [User(**u) for u in users_raw]

    # 2. Posts
    with open(POSTS_FILE, "r", encoding="utf-8") as f:
        posts_raw = json.load(f)
        posts = []
        for p in posts_raw:
            p["created_at"] = datetime.fromisoformat(p["created_at"])
            posts.append(Post(**p))

    # 3. Follows
    follows = []
    with open(FOLLOWS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            follows.append((int(row["user_id"]), int(row["followed_user_id"])))

    # 4. Interactions
    interactions = []
    with open(INTERACTIONS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            interactions.append({
                "user_id": int(row["user_id"]),
                "post_id": int(row["post_id"]),
                "dwell_seconds": float(row["dwell_seconds"]),
                "liked": bool(int(row["liked"])),
                "replied": bool(int(row["replied"])),
                "skipped": bool(int(row["skipped"])),
                "explicit_rating": float(row["explicit_rating"]) if row["explicit_rating"] else None,
                "timestamp": row["timestamp"],
            })

    # 5. Explicit holdout
    explicit_holdout = []
    with open(EXPLICIT_HOLDOUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            explicit_holdout.append({
                "user_id": int(row["user_id"]),
                "post_id": int(row["post_id"]),
                "rating": float(row["rating"]),
                "timestamp": row["timestamp"],
            })

    print(f"Loaded {len(users)} users, {len(posts)} posts, {len(follows)} follows, {len(interactions)} interactions.")
    return users, posts, follows, interactions, explicit_holdout


def split_interactions(interactions: List[Dict], train_ratio: float = 0.80) -> Tuple[List[Dict], List[Dict]]:
    """Temporal 80/20 train/test split per user."""
    user_interactions: Dict[int, List[Dict]] = {}
    for it in interactions:
        u_id = it["user_id"]
        if u_id not in user_interactions:
            user_interactions[u_id] = []
        user_interactions[u_id].append(it)

    train_data = []
    test_data = []

    for u_id, items in user_interactions.items():
        # Sort chronologically by timestamp
        items.sort(key=lambda x: x["timestamp"])
        if len(items) <= 3:
            # Keep very short histories in train to test cold-start
            train_data.extend(items)
        else:
            split_idx = int(len(items) * train_ratio)
            train_data.extend(items[:split_idx])
            test_data.extend(items[split_idx:])

    print(f"Split interactions: {len(train_data)} train, {len(test_data)} test.")
    return train_data, test_data


def build_interaction_matrix(interactions: List[Dict], num_users: int = NUM_USERS, num_posts: int = NUM_POSTS) -> np.ndarray:
    """Build continuous implicit interaction matrix R for CF models."""
    R = np.zeros((num_users, num_posts), dtype=np.float32)
    for it in interactions:
        u_id = it["user_id"]
        p_id = it["post_id"]
        if u_id < num_users and p_id < num_posts:
            # Composite implicit feedback formula (Lecture L15)
            # r_ui = log(1 + dwell) + 2*liked + 3*replied
            val = np.log1p(it["dwell_seconds"]) + (2.0 if it["liked"] else 0.0) + (3.0 if it["replied"] else 0.0)
            R[u_id, p_id] = float(val)
    return R


def main():
    start_total = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    users, posts, follows, interactions, explicit_holdout = load_dataset()
    train_interactions, test_interactions = split_interactions(interactions)

    R_train = build_interaction_matrix(train_interactions, num_users=len(users), num_posts=len(posts))

    # -------------------------------------------------------------
    # 1. Train Content-Based Model
    # -------------------------------------------------------------
    print("\n--- Training Content-Based TF-IDF Model ---")
    content_model = ContentBasedModel()
    content_model.fit(posts)
    content_model.build_user_profiles(users, train_interactions)
    content_model.save(ARTIFACTS_DIR / "content_model.pkl")
    print("Content-based model fitted and saved.")

    # -------------------------------------------------------------
    # 2. Train Collaborative kNN (User-kNN and Item-kNN)
    # -------------------------------------------------------------
    print("\n--- Training Collaborative kNN Models ---")
    user_knn = UserKNN(k=20, tau=5.0, similarity_metric="pearson")
    user_knn.fit(R_train)
    with open(ARTIFACTS_DIR / "user_knn.pkl", "wb") as f:
        pickle.dump(user_knn, f)

    item_knn = ItemKNN(k=20, tau=5.0)
    item_knn.fit(R_train)
    with open(ARTIFACTS_DIR / "item_knn.pkl", "wb") as f:
        pickle.dump(item_knn, f)
    print("User-kNN and Item-kNN trained and saved.")

    # -------------------------------------------------------------
    # 3. Train Implicit ALS Factorization
    # -------------------------------------------------------------
    print("\n--- Training Implicit ALS (iALS) Model ---")
    ials = ImplicitALS(factors=32, alpha=40.0, regularization=0.1, iterations=15)
    ials.fit(R_train, verbose=True)
    ials.save(ARTIFACTS_DIR / "ials_model.pkl")
    print("iALS model trained and saved.")

    # -------------------------------------------------------------
    # 4. Train Two-Tower Neural CF in PyTorch
    # -------------------------------------------------------------
    print("\n--- Training Two-Tower Neural CF in PyTorch ---")
    two_tower = TwoTowerRecommender(embed_dim=32)
    dl = two_tower.prepare_data(users, posts, train_interactions, content_model)
    two_tower.train(dl, epochs=8, lr=0.002, verbose=True)
    two_tower.precompute_embeddings(users, posts, content_model)
    two_tower.save(ARTIFACTS_DIR / "two_tower_model.pt")
    print("Two-Tower Neural CF trained, cached, and saved.")

    # -------------------------------------------------------------
    # 5. Train Heavy Ranker Calibrated Classifiers
    # -------------------------------------------------------------
    print("\n--- Training Multi-Signal Heavy Ranker ---")
    hydrator = FeatureHydrator(users, posts, follows, train_interactions, ials, content_model)

    from data.schemas import CandidatePost

    # Assemble training feature rows
    feature_rows = []
    labels_like = []
    labels_reply = []
    labels_skip = []

    user_map = {u.user_id: u for u in users}
    post_map = {p.post_id: p for p in posts}
    all_post_ids = [p.post_id for p in posts]
    user_seen = {u.user_id: set() for u in users}
    for it in train_interactions:
        user_seen[it["user_id"]].add(it["post_id"])

    # 1. Observed multi-action impression logs (Lecture L18/L19)
    sample_train = train_interactions[:12000] if len(train_interactions) > 12000 else train_interactions
    for it in sample_train:
        u = user_map.get(it["user_id"])
        p = post_map.get(it["post_id"])
        if u and p:
            cand = CandidatePost(post=p, source="train_log", retrieval_score=1.0)
            hc = hydrator.hydrate_candidate(cand, u)
            if hc.feature_vector is not None:
                feature_rows.append(hc.feature_vector)
                labels_like.append(1 if it["liked"] else 0)
                labels_reply.append(1 if it["replied"] else 0)
                labels_skip.append(1 if it["skipped"] else 0)

    X_train = np.array(feature_rows, dtype=np.float32)
    y_like = np.array(labels_like, dtype=np.int32)
    y_reply = np.array(labels_reply, dtype=np.int32)
    y_skip = np.array(labels_skip, dtype=np.int32)

    heavy_ranker = HeavyRanker()
    heavy_ranker.train(X_train, y_like, y_reply, y_skip)
    heavy_ranker.save(ARTIFACTS_DIR / "heavy_ranker.pkl")
    print("Heavy Ranker trained and saved.")

    # -------------------------------------------------------------
    # 6. Train Shilling Defense Detector
    # -------------------------------------------------------------
    print("\n--- Initializing Shilling Anomaly Detector ---")
    detector = ShillingDefenseDetector()
    detector.analyze_interactions(train_interactions)
    with open(ARTIFACTS_DIR / "detector.pkl", "wb") as f:
        pickle.dump(detector, f)
    print("Detector initialized and saved.")

    # -------------------------------------------------------------
    # 7. Save train/test split data for evaluation runner
    # -------------------------------------------------------------
    with open(ARTIFACTS_DIR / "split_data.pkl", "wb") as f:
        pickle.dump({
            "train_interactions": train_interactions,
            "test_interactions": test_interactions,
            "explicit_holdout": explicit_holdout,
        }, f)

    elapsed = round(time.time() - start_total, 2)
    print(f"\n==========================================")
    print(f"ALL MODELS TRAINED AND SAVED IN {elapsed}s")
    print(f"Artifacts directory: {ARTIFACTS_DIR}")
    print(f"==========================================")


if __name__ == "__main__":
    main()

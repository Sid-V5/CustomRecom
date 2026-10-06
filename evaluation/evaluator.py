"""
Offline Evaluation & Benchmarking Suite (Section 6).
Implements:
1. Temporal 80/20 train/test split per user and cohort tagging (Warm vs Cold User/Item).
2. Benchmarks all 8 RecSys algorithms across NDCG@10, P@10, R@10, RMSE, Coverage, and ILD.
3. Runs the 6-part Ablation Study matrix.
4. Generates structured JSON reports for the UI and academic grading tables.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from config import (
    ARTIFACTS_DIR,
    BENCHMARK_FILE,
    COLD_ITEM_THRESHOLD,
    COLD_USER_THRESHOLD,
    FEED_TOP_K,
)
from data.schemas import Post, User
from evaluation.metrics import (
    catalog_coverage,
    intra_list_distance,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse_score,
)


class RecsysEvaluator:
    """Evaluates algorithms across ranking accuracy, explicit RMSE, and beyond-accuracy metrics."""

    def __init__(
        self,
        users: List[User],
        posts: List[Post],
        train_interactions: List[Dict],
        test_interactions: List[Dict],
        explicit_holdout: List[Dict],
        content_model,
    ):
        self.users = users
        self.posts = posts
        self.content_model = content_model
        self.train_interactions = train_interactions
        self.test_interactions = test_interactions
        self.explicit_holdout = explicit_holdout

        self.num_users = len(users)
        self.num_posts = len(posts)

        # Build test sets per user (ground truth: liked/engaged in test slice)
        self.ground_truth: Dict[int, Set[int]] = {u.user_id: set() for u in users}
        for it in test_interactions:
            u_id = it["user_id"]
            if it.get("liked", False) or it.get("dwell_seconds", 0.0) >= 8.0:
                self.ground_truth[u_id].add(it["post_id"])

        # Cohort identification
        user_train_counts = {u.user_id: 0 for u in users}
        item_train_counts = {p.post_id: 0 for p in posts}
        for it in train_interactions:
            user_train_counts[it["user_id"]] = user_train_counts.get(it["user_id"], 0) + 1
            item_train_counts[it["post_id"]] = item_train_counts.get(it["post_id"], 0) + 1

        self.cold_users = [u.user_id for u in users if user_train_counts[u.user_id] < COLD_USER_THRESHOLD]
        self.warm_users = [u.user_id for u in users if user_train_counts[u.user_id] > 10]
        self.cold_items = [p.post_id for p in posts if item_train_counts[p.post_id] <= 1]

        # Explicit test slice
        self.explicit_test_pairs = [(r["user_id"], r["post_id"], float(r["rating"])) for r in explicit_holdout]

    def evaluate_model(
        self,
        model_name: str,
        recommend_fn,
        predict_rating_fn=None,
        target_users: Optional[List[int]] = None,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluates a model over a cohort of users.
        recommend_fn(user_id, k) -> List[int]
        """
        if target_users is None:
            # Users who have at least 1 ground truth item in test
            target_users = [u for u in range(self.num_users) if len(self.ground_truth[u]) > 0]

        if not target_users:
            return {"p@10": 0.0, "r@10": 0.0, "ndcg@10": 0.0, "coverage": 0.0, "ild": 0.0, "rmse": 0.0}

        p_list = []
        r_list = []
        ndcg_list = []
        all_recs = []
        ild_list = []

        for u_id in target_users:
            recs = recommend_fn(u_id, top_k)
            gt = self.ground_truth[u_id]
            p_list.append(precision_at_k(recs, gt, k=top_k))
            r_list.append(recall_at_k(recs, gt, k=top_k))
            ndcg_list.append(ndcg_at_k(recs, gt, k=top_k))
            all_recs.append(recs)
            ild_list.append(intra_list_distance(recs, self.content_model))

        coverage = catalog_coverage(all_recs, self.num_posts)

        # RMSE evaluation on explicit slice
        rmse_val = 0.0
        if predict_rating_fn is not None and self.explicit_test_pairs:
            y_true = []
            y_pred = []
            for u_id, p_id, rating in self.explicit_test_pairs:
                if u_id < self.num_users and p_id < self.num_posts:
                    pred = predict_rating_fn(u_id, p_id)
                    y_true.append(rating)
                    y_pred.append(pred)
            rmse_val = rmse_score(y_true, y_pred)

        return {
            "model": model_name,
            "p@10": round(float(np.mean(p_list)), 4),
            "r@10": round(float(np.mean(r_list)), 4),
            "ndcg@10": round(float(np.mean(ndcg_list)), 4),
            "coverage_pct": round(float(coverage), 2),
            "ild_diversity": round(float(np.mean(ild_list)), 4),
            "rmse": round(float(rmse_val), 4) if rmse_val > 0 else None,
        }

    def run_all_benchmarks(
        self,
        user_knn,
        item_knn,
        ials,
        content_model,
        two_tower,
        orchestrator,
    ) -> Dict[str, Any]:
        """
        Executes full benchmark suite across all 8 models, cohorts, and ablations.
        """
        # Top popular posts for Popularity baseline
        post_likes = {p.post_id: p.likes_count for p in self.posts}
        top_popular_ids = [p[0] for p in sorted(post_likes.items(), key=lambda x: x[1], reverse=True)[:50]]

        # Baseline 1: Random
        np.random.seed(42)
        all_post_ids = [p.post_id for p in self.posts]

        def rec_random(u_id, k):
            return list(np.random.choice(all_post_ids, size=k, replace=False))

        # Baseline 2: Popularity
        def rec_popular(u_id, k):
            return top_popular_ids[:k]

        # Baseline 3: User-kNN
        def rec_user_knn(u_id, k):
            return [x[0] for x in user_knn.recommend(u_id, top_n=k)]

        # Baseline 4: Item-kNN
        def rec_item_knn(u_id, k):
            return [x[0] for x in item_knn.recommend(u_id, top_n=k)]

        # Baseline 5: iALS
        def rec_ials(u_id, k):
            return [x[0] for x in ials.get_top_candidates(u_id, top_n=k)]

        # Baseline 6: Content-Based
        def rec_content(u_id, k):
            u = self.users[u_id]
            return [x[0] for x in content_model.get_top_candidates(u, top_n=k)]

        # Baseline 7: Two-Tower Neural CF
        def rec_two_tower(u_id, k):
            return [x[0] for x in two_tower.recommend(u_id, top_n=k)]

        # Baseline 8: Mini-X Hybrid Pipeline
        def rec_hybrid(u_id, k):
            feed, _ = orchestrator.generate_feed(u_id, top_k=k, ignore_seen=True)
            return [sc.hydrated.candidate.post.post_id for sc in feed]

        # -------------------------------------------------------------
        # 1. Main Algorithms Benchmark (Warm Users)
        # -------------------------------------------------------------
        eval_users = [u for u in self.warm_users if len(self.ground_truth[u]) > 0][:50]
        if not eval_users:
            eval_users = [u for u in range(self.num_users) if len(self.ground_truth[u]) > 0][:50]

        benchmarks = [
            self.evaluate_model("Random Baseline", rec_random, target_users=eval_users),
            self.evaluate_model("Popularity Baseline", rec_popular, target_users=eval_users),
            self.evaluate_model("User-kNN CF", rec_user_knn, predict_rating_fn=user_knn.predict_score, target_users=eval_users),
            self.evaluate_model("Item-kNN CF", rec_item_knn, predict_rating_fn=item_knn.predict_score, target_users=eval_users),
            self.evaluate_model("Matrix Factorization (iALS)", rec_ials, target_users=eval_users),
            self.evaluate_model("Content-Based TF-IDF", rec_content, target_users=eval_users),
            self.evaluate_model("Two-Tower Neural CF", rec_two_tower, target_users=eval_users),
            self.evaluate_model("Mini-X Hybrid Pipeline (Ours)", rec_hybrid, target_users=eval_users),
        ]

        # -------------------------------------------------------------
        # 2. Cold-Start Cohort Benchmark (Cold Users)
        # -------------------------------------------------------------
        # For cold users, evaluate CF vs Content vs Hybrid
        cold_eval_users = self.cold_users[:40]

        cold_benchmarks = []
        if cold_eval_users:
            # Synthetic cold-start ground truth: tag-matching posts
            cold_gt = {}
            for u_id in cold_eval_users:
                u = self.users[u_id]
                cold_gt[u_id] = {p.post_id for p in self.posts if any(t in u.preferred_tags for t in p.tags)}

            orig_gt = self.ground_truth
            self.ground_truth = cold_gt

            cold_benchmarks = [
                self.evaluate_model("User-kNN CF (Cold)", rec_user_knn, target_users=cold_eval_users),
                self.evaluate_model("iALS (Cold)", rec_ials, target_users=cold_eval_users),
                self.evaluate_model("Content-Based (Cold)", rec_content, target_users=cold_eval_users),
                self.evaluate_model("Mini-X Hybrid Switch (Cold)", rec_hybrid, target_users=cold_eval_users),
            ]
            self.ground_truth = orig_gt

        # -------------------------------------------------------------
        # 3. Ablation Experiments Matrix (Section 6.4)
        # -------------------------------------------------------------
        # Ablation 1: In-Network Only (Follow graph only)
        def rec_ablation_in_net(u_id, k):
            feed, _ = orchestrator.generate_feed(u_id, top_k=k, mmr_lambda=1.0, ignore_seen=True, retrieval_mode="in_network")
            return [sc.hydrated.candidate.post.post_id for sc in feed]

        # Ablation 2: Out-of-Network Only (No follow graph)
        def rec_ablation_oon(u_id, k):
            feed, _ = orchestrator.generate_feed(u_id, top_k=k, mmr_lambda=1.0, ignore_seen=True, retrieval_mode="out_network")
            return [sc.hydrated.candidate.post.post_id for sc in feed]

        # Ablation 3: Full Pipeline without MMR Diversity (lambda=1.0)
        def rec_ablation_no_mmr(u_id, k):
            feed, _ = orchestrator.generate_feed(u_id, top_k=k, mmr_lambda=1.0, ignore_seen=True)
            return [sc.hydrated.candidate.post.post_id for sc in feed]

        # Ablation 4: Full Pipeline with MMR Diversity (lambda=0.7)
        def rec_ablation_mmr(u_id, k):
            feed, _ = orchestrator.generate_feed(u_id, top_k=k, mmr_lambda=0.7, ignore_seen=True)
            return [sc.hydrated.candidate.post.post_id for sc in feed]

        # Setup Shilling Attack for Ablation 5 and 6
        from security.shilling_simulator import ShillingAttackSimulator
        sim = ShillingAttackSimulator(num_bots=25)
        sim.inject_attack(42, attack_type="bandwagon", posts=self.posts)
        if orchestrator.filter_chain.shilling_detector is not None:
            all_inter = list(self.train_interactions) + sim.bot_interactions
            orchestrator.filter_chain.shilling_detector.analyze_interactions(all_inter)

        # Ablation 5: Under Shilling Attack (Unprotected)
        def rec_ablation_attack_unprotected(u_id, k):
            feed, _ = orchestrator.generate_feed(u_id, top_k=k, mmr_lambda=0.7, ignore_seen=True, defense_active=False)
            return [sc.hydrated.candidate.post.post_id for sc in feed]

        # Ablation 6: Under Shilling Attack (With Defense Filter Active)
        def rec_ablation_attack_defended(u_id, k):
            feed, _ = orchestrator.generate_feed(u_id, top_k=k, mmr_lambda=0.7, ignore_seen=True, defense_active=True)
            return [sc.hydrated.candidate.post.post_id for sc in feed]

        ablations = [
            self.evaluate_model("Ablation 1: In-Network Only (Follow Graph Only)", rec_ablation_in_net, target_users=eval_users),
            self.evaluate_model("Ablation 2: Out-of-Network Only (No Follow Graph)", rec_ablation_oon, target_users=eval_users),
            self.evaluate_model("Ablation 3: Full Pipeline (No MMR, lambda=1.0)", rec_ablation_no_mmr, target_users=eval_users),
            self.evaluate_model("Ablation 4: Full Pipeline (MMR Diversity, lambda=0.7)", rec_ablation_mmr, target_users=eval_users),
            self.evaluate_model("Ablation 5: Under Shilling Attack (Unprotected)", rec_ablation_attack_unprotected, target_users=eval_users),
            self.evaluate_model("Ablation 6: Under Shilling Attack (Defense Active)", rec_ablation_attack_defended, target_users=eval_users),
        ]

        # Restore organic state after ablations
        sim.reset_attack(posts=self.posts)
        if orchestrator.filter_chain.shilling_detector is not None:
            orchestrator.filter_chain.shilling_detector.analyze_interactions(self.train_interactions)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "main_benchmarks": benchmarks,
            "cold_start_benchmarks": cold_benchmarks,
            "ablations": ablations,
            "metadata": {
                "total_users": self.num_users,
                "total_posts": self.num_posts,
                "warm_users_evaluated": len(eval_users),
                "cold_users_evaluated": len(cold_eval_users),
            },
        }

        # Save to disk
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

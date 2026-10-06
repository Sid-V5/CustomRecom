"""
Standalone CLI Runner for RecSys Evaluation & Academic Benchmarking.
Usage:
    python evaluate.py
Evaluates all 8 algorithms, Cold-Start cohorts, and the 6-stage Ablation Matrix.
Prints formatted tables and saves summary JSON to /artifacts/benchmark_summary.json.
"""

import pickle
import sys
from pathlib import Path

from config import ARTIFACTS_DIR, BENCHMARK_FILE
from evaluation.evaluator import RecsysEvaluator
from models.als_factorization import ImplicitALS
from models.content_based import ContentBasedModel
from models.neural_two_tower import TwoTowerRecommender
from pipeline.heavy_ranker import HeavyRanker
from pipeline.orchestrator import RecommendationOrchestrator
from train import load_dataset


def print_table(title: str, headers: list, rows: list):
    """Pretty prints a formatted ASCII table."""
    print(f"\n{'=' * 85}")
    print(f" {title.upper()}")
    print(f"{'=' * 85}")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)

    for row in rows:
        row_line = " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        print(row_line)
    print(f"{'=' * 85}\n")


def main():
    print("Initializing Mini-X Offline Evaluation Suite...")

    # 1. Load dataset & saved split
    users, posts, follows, interactions, explicit_holdout = load_dataset()

    split_file = ARTIFACTS_DIR / "split_data.pkl"
    if not split_file.exists():
        print("Split data not found. Please run 'python train.py' first.")
        sys.exit(1)

    with open(split_file, "rb") as f:
        split_data = pickle.load(f)
    train_interactions = split_data["train_interactions"]
    test_interactions = split_data["test_interactions"]

    # 2. Load trained models
    print("Loading model artifacts from disk...")
    with open(ARTIFACTS_DIR / "user_knn.pkl", "rb") as f:
        user_knn = pickle.load(f)

    with open(ARTIFACTS_DIR / "item_knn.pkl", "rb") as f:
        item_knn = pickle.load(f)

    ials = ImplicitALS().load(ARTIFACTS_DIR / "ials_model.pkl")
    content_model = ContentBasedModel().load(ARTIFACTS_DIR / "content_model.pkl")
    two_tower = TwoTowerRecommender().load(ARTIFACTS_DIR / "two_tower_model.pt")
    heavy_ranker = HeavyRanker().load(ARTIFACTS_DIR / "heavy_ranker.pkl")

    with open(ARTIFACTS_DIR / "detector.pkl", "rb") as f:
        detector = pickle.load(f)

    # 3. Instantiate orchestrator
    orchestrator = RecommendationOrchestrator(
        users=users,
        posts=posts,
        follows=follows,
        interactions=train_interactions,
        als_model=ials,
        item_knn_model=item_knn,
        content_model=content_model,
        heavy_ranker=heavy_ranker,
        shilling_detector=detector,
    )

    # 4. Instantiate evaluator
    evaluator = RecsysEvaluator(
        users=users,
        posts=posts,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        explicit_holdout=explicit_holdout,
        content_model=content_model,
    )

    print("Executing offline benchmarking over Warm and Cold cohorts...")
    results = evaluator.run_all_benchmarks(
        user_knn=user_knn,
        item_knn=item_knn,
        ials=ials,
        content_model=content_model,
        two_tower=two_tower,
        orchestrator=orchestrator,
    )

    # -------------------------------------------------------------
    # Format and Print Academic Comparison Tables
    # -------------------------------------------------------------
    headers_main = ["Algorithm", "P@10", "R@10", "NDCG@10", "Coverage (%)", "ILD (Diversity)", "RMSE"]
    rows_main = []
    for m in results["main_benchmarks"]:
        rmse_str = f"{m['rmse']:.4f}" if m.get("rmse") is not None else "N/A"
        rows_main.append([
            m["model"],
            f"{m['p@10']:.4f}",
            f"{m['r@10']:.4f}",
            f"{m['ndcg@10']:.4f}",
            f"{m['coverage_pct']:.2f}%",
            f"{m['ild_diversity']:.4f}",
            rmse_str,
        ])
    print_table("Table 1: Main Recommendation Benchmarks (Warm Cohort)", headers_main, rows_main)

    headers_cold = ["Model (Cold-Start Cohort)", "P@10", "R@10", "NDCG@10", "Coverage (%)", "ILD"]
    rows_cold = []
    for m in results["cold_start_benchmarks"]:
        rows_cold.append([
            m["model"],
            f"{m['p@10']:.4f}",
            f"{m['r@10']:.4f}",
            f"{m['ndcg@10']:.4f}",
            f"{m['coverage_pct']:.2f}%",
            f"{m['ild_diversity']:.4f}",
        ])
    print_table("Table 2: Cold-Start User Breakdown (< 3 interactions)", headers_cold, rows_cold)

    headers_abl = ["Ablation Configuration", "P@10", "R@10", "NDCG@10", "Coverage (%)", "ILD"]
    rows_abl = []
    for m in results["ablations"]:
        rows_abl.append([
            m["model"],
            f"{m['p@10']:.4f}",
            f"{m['r@10']:.4f}",
            f"{m['ndcg@10']:.4f}",
            f"{m['coverage_pct']:.2f}%",
            f"{m['ild_diversity']:.4f}",
        ])
    print_table("Table 3: Multi-Stage Hybrid Ablation Analysis", headers_abl, rows_abl)

    print(f"Benchmark report saved to: {BENCHMARK_FILE}")


if __name__ == "__main__":
    main()

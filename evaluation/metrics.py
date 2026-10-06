"""
Evaluation Metrics for Recommendation Systems (Lectures L21, L35).
Implements:
- Ranking metrics: Precision@K, Recall@K, NDCG@K
- Error metrics: RMSE, MAE
- Beyond-accuracy metrics: Catalog Coverage (%), Intra-List Distance (ILD / Diversity)
"""

import math
from typing import List, Optional, Set
import numpy as np


def precision_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """Precision@K = |Recommended@K n Relevant| / K."""
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / float(k)


def recall_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """Recall@K = |Recommended@K n Relevant| / |Relevant|."""
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / float(len(relevant))


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain at rank K.
    DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K
    """
    if not relevant or k <= 0:
        return 0.0

    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        rel = 1.0 if item in relevant else 0.0
        dcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 2.0)

    # Ideal DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum((math.pow(2.0, 1.0) - 1.0) / math.log2(i + 2.0) for i in range(ideal_hits))

    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def rmse_score(y_true: List[float], y_pred: List[float]) -> float:
    """Root Mean Square Error on ratings."""
    if not y_true:
        return 0.0
    arr_true = np.array(y_true, dtype=np.float32)
    arr_pred = np.array(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((arr_true - arr_pred) ** 2)))


def mae_score(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Error on ratings."""
    if not y_true:
        return 0.0
    arr_true = np.array(y_true, dtype=np.float32)
    arr_pred = np.array(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(arr_true - arr_pred)))


def catalog_coverage(all_recommendations: List[List[int]], total_items: int) -> float:
    """
    Catalog Coverage (%):
    Coverage = |Union of Recs| / |Catalog| * 100%
    """
    if total_items <= 0:
        return 0.0
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)
    return (len(unique_items) / float(total_items)) * 100.0


def intra_list_distance(recommendations: List[int], content_model) -> float:
    """
    Intra-List Distance (ILD / Diversity):
    ILD(L) = (2 / (|L|(|L|-1))) * sum_{i < j} (1 - Sim(i, j))
    """
    L = len(recommendations)
    if L <= 1 or content_model is None:
        return 0.0

    total_dist = 0.0
    pairs = 0
    for i in range(L):
        for j in range(i + 1, L):
            sim = content_model.similarity_between_posts(recommendations[i], recommendations[j])
            dist = 1.0 - max(0.0, min(1.0, sim))
            total_dist += dist
            pairs += 1

    return total_dist / float(pairs) if pairs > 0 else 0.0

"""
FastAPI REST API Routes (Section 7.1).
Exposes endpoints for user profiles, personalized feeds, real-time logging,
shilling attack injection/defense, and offline evaluation benchmark metrics.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from config import ARTIFACTS_DIR, BENCHMARK_FILE, FEED_TOP_K
from data.schemas import Post, User
from models.als_factorization import ImplicitALS
from models.collaborative_knn import ItemKNN, UserKNN
from models.content_based import ContentBasedModel
from pipeline.heavy_ranker import HeavyRanker
from pipeline.orchestrator import RecommendationOrchestrator
from security.shilling_detector import ShillingDefenseDetector
from security.shilling_simulator import ShillingAttackSimulator
from train import load_dataset

router = APIRouter(prefix="/api")

# Global In-Memory State
state: Dict[str, Any] = {}


def initialize_app_state():
    """Load dataset, models, and orchestrator into memory."""
    print("Loading datasets and model artifacts for FastAPI...")
    users, posts, follows, interactions, explicit_holdout = load_dataset()

    # Load models
    ials = ImplicitALS().load(ARTIFACTS_DIR / "ials_model.pkl")
    content_model = ContentBasedModel().load(ARTIFACTS_DIR / "content_model.pkl")

    with open(ARTIFACTS_DIR / "item_knn.pkl", "rb") as f:
        item_knn = pickle.load(f)

    with open(ARTIFACTS_DIR / "user_knn.pkl", "rb") as f:
        user_knn = pickle.load(f)

    heavy_ranker = HeavyRanker().load(ARTIFACTS_DIR / "heavy_ranker.pkl")

    with open(ARTIFACTS_DIR / "detector.pkl", "rb") as f:
        detector: ShillingDefenseDetector = pickle.load(f)

    simulator = ShillingAttackSimulator()

    orchestrator = RecommendationOrchestrator(
        users=users,
        posts=posts,
        follows=follows,
        interactions=interactions,
        als_model=ials,
        item_knn_model=item_knn,
        content_model=content_model,
        heavy_ranker=heavy_ranker,
        shilling_detector=detector,
    )

    state["users"] = users
    state["posts"] = posts
    state["follows"] = follows
    state["interactions"] = interactions
    state["user_map"] = {u.user_id: u for u in users}
    state["post_map"] = {p.post_id: p for p in posts}
    state["orchestrator"] = orchestrator
    state["simulator"] = simulator
    state["detector"] = detector
    state["content_model"] = content_model
    state["user_knn"] = user_knn
    state["item_knn"] = item_knn
    state["ials"] = ials

    print("FastAPI state initialization complete.")


def ensure_state_initialized():
    """Ensure state is loaded even if lifespan was not triggered."""
    if "orchestrator" not in state:
        initialize_app_state()


# Request models
class InteractionRequest(BaseModel):
    user_id: int
    post_id: int
    action: str  # "like", "reply", "skip"
    dwell_seconds: Optional[float] = 12.0


class AttackInjectRequest(BaseModel):
    target_post_id: int
    attack_type: str = "bandwagon"  # "bandwagon" or "random"


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------

@router.get("/users")
def get_users():
    """Returns list of users with persona, department, followers, and interaction counts."""
    ensure_state_initialized()
    users: List[User] = state["users"]
    orchestrator: RecommendationOrchestrator = state["orchestrator"]

    results = []
    for u in users:
        inter_count = orchestrator.user_interaction_counts.get(u.user_id, 0)
        followers = orchestrator.hydrator.author_followers.get(u.user_id, 0)
        results.append({
            "user_id": u.user_id,
            "username": u.username,
            "department": u.department,
            "persona": u.persona,
            "preferred_tags": u.preferred_tags,
            "followers_count": followers,
            "following_count": len(u.followed_user_ids),
            "interaction_count": inter_count,
            "is_cold_start": inter_count < 3,
        })
    return {"users": results}


@router.get("/user/{user_id}")
def get_user_profile(user_id: int):
    """Returns user profile, follow stats, and recent activity."""
    ensure_state_initialized()
    user = state["user_map"].get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    orchestrator: RecommendationOrchestrator = state["orchestrator"]
    followers = orchestrator.hydrator.author_followers.get(user_id, 0)
    history = [it for it in state["interactions"] if it["user_id"] == user_id]

    recent_interactions = []
    for it in history[-10:]:
        post = state["post_map"].get(it["post_id"])
        if post:
            author = state["user_map"].get(post.author_id)
            recent_interactions.append({
                "post_id": post.post_id,
                "author_username": author.username if author else f"user_{post.author_id}",
                "content_snippet": post.content[:80] + "...",
                "liked": it.get("liked", False),
                "replied": it.get("replied", False),
                "dwell_seconds": it.get("dwell_seconds", 0.0),
                "timestamp": it.get("timestamp"),
            })

    return {
        "user_id": user.user_id,
        "username": user.username,
        "department": user.department,
        "persona": user.persona,
        "preferred_tags": user.preferred_tags,
        "followers_count": followers,
        "following_count": len(user.followed_user_ids),
        "total_interactions": len(history),
        "is_cold_start": len(history) < 3,
        "recent_interactions": recent_interactions,
    }


@router.get("/feed")
def get_feed(
    user_id: int = Query(0, description="Target User ID"),
    top_k: int = Query(FEED_TOP_K, ge=1, le=50),
    mmr_lambda: float = Query(0.7, ge=0.0, le=1.0),
    w_like: float = Query(0.6),
    w_reply: float = Query(1.2),
    w_skip: float = Query(0.4),
    lambda_age: float = Query(0.02),
    defense_active: bool = Query(False),
    retrieval_mode: str = Query("all", description="Retrieval tower: all, in_network, or out_network"),
    department_filter: Optional[str] = Query(None, description="Department constraint (L28)"),
):
    """Generates personalized feed with full explanation pills and funnel telemetry."""
    ensure_state_initialized()
    user = state["user_map"].get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    orchestrator: RecommendationOrchestrator = state["orchestrator"]

    feed, funnel_stats = orchestrator.generate_feed(
        user_id=user_id,
        top_k=top_k,
        mmr_lambda=mmr_lambda,
        w_like=w_like,
        w_reply=w_reply,
        w_skip=w_skip,
        lambda_age=lambda_age,
        defense_active=defense_active,
        retrieval_mode=retrieval_mode,
        department_filter=department_filter,
    )

    feed_items = []
    for sc in feed:
        post = sc.hydrated.candidate.post
        author = state["user_map"].get(post.author_id)
        author_name = author.username if author else f"user_{post.author_id}"
        author_dept = author.department if author else post.department

        feed_items.append({
            "post_id": post.post_id,
            "author_id": post.author_id,
            "author_username": author_name,
            "author_department": author_dept,
            "content": post.content,
            "department": post.department,
            "tags": post.tags,
            "created_at": post.created_at.isoformat(),
            "likes_count": post.likes_count,
            "replies_count": post.replies_count,
            "source": sc.hydrated.candidate.source,
            "p_like": sc.p_like,
            "p_reply": sc.p_reply,
            "p_skip": sc.p_skip,
            "final_score": sc.final_score,
            "explanation": sc.explanation,
        })

    return {
        "user_id": user_id,
        "feed": feed_items,
        "funnel_stats": funnel_stats,
        "config": {
            "mmr_lambda": mmr_lambda,
            "w_like": w_like,
            "w_reply": w_reply,
            "w_skip": w_skip,
            "lambda_age": lambda_age,
            "defense_active": defense_active,
            "retrieval_mode": retrieval_mode,
            "department_filter": department_filter,
        },
    }


@router.get("/search")
def search_posts(
    q: str = Query(..., min_length=1, description="Homework/project search query or topic"),
    user_id: int = Query(0, description="Requesting User ID"),
    top_k: int = Query(15, ge=1, le=50),
):
    """
    Case-Based Recommendation (Lecture L29):
    Searches the entire campus corpus for posts structurally and semantically similar
    to the student's problem query using TF-IDF cosine similarity.
    """
    ensure_state_initialized()
    user = state["user_map"].get(user_id) or state["users"][0]
    content_model: ContentBasedModel = state["content_model"]
    orchestrator: RecommendationOrchestrator = state["orchestrator"]

    results = content_model.case_based_query_search(q, top_n=top_k)
    feed_items = []

    for rank, (post_id, sim_score) in enumerate(results):
        post = state["post_map"].get(post_id)
        if not post:
            continue
        author = state["user_map"].get(post.author_id)
        author_name = author.username if author else f"user_{post.author_id}"
        author_dept = author.department if author else post.department

        matched_tags = [t for t in post.tags if t.lower() in q.lower() or any(w in t.lower() for w in q.lower().split())]

        feed_items.append({
            "post_id": post.post_id,
            "author_id": post.author_id,
            "author_username": author_name,
            "author_department": author_dept,
            "content": post.content,
            "department": post.department,
            "tags": post.tags,
            "created_at": post.created_at.isoformat(),
            "likes_count": post.likes_count,
            "replies_count": post.replies_count,
            "source": "case_based_search",
            "p_like": round(min(0.99, max(0.1, sim_score)), 4),
            "p_reply": round(min(0.50, max(0.05, sim_score * 0.4)), 4),
            "p_skip": round(max(0.01, 1.0 - sim_score), 4),
            "final_score": round(sim_score, 4),
            "explanation": {
                "primary_reason": f"Case-based match: structurally matches '{q[:30]}'",
                "badge_type": "topic",
                "badge_color": "emerald",
                "confidence_score": round(sim_score * 100.0, 1),
                "source": "case_based_search",
                "matched_tags": matched_tags,
                "feature_weights": {
                    "network_affinity": 10.0,
                    "topic_alignment": 75.0,
                    "peer_collaborative": 10.0,
                    "discussion_velocity": 5.0,
                },
                "probabilities": {
                    "p_like": round(min(99.0, sim_score * 100.0), 1),
                    "p_reply": round(min(50.0, sim_score * 40.0), 1),
                    "p_skip": round(max(1.0, (1.0 - sim_score) * 100.0), 1),
                },
            },
        })

    return {
        "query": q,
        "results_count": len(feed_items),
        "feed": feed_items,
    }


@router.post("/interact")
def record_interaction(req: InteractionRequest):
    """Logs real-time engagement and updates state."""
    ensure_state_initialized()
    if req.action not in ("like", "reply", "skip"):
        raise HTTPException(status_code=400, detail="Invalid action. Must be 'like', 'reply', or 'skip'")

    user = state["user_map"].get(req.user_id)
    post = state["post_map"].get(req.post_id)
    if not user or not post:
        raise HTTPException(status_code=404, detail="User or Post not found")

    orchestrator: RecommendationOrchestrator = state["orchestrator"]
    now = datetime.now(timezone.utc)

    liked = req.action == "like"
    replied = req.action == "reply"
    skipped = req.action == "skip"

    if liked:
        post.likes_count += 1
    if replied:
        post.replies_count += 1

    dwell = req.dwell_seconds or (25.0 if liked else (2.0 if skipped else 10.0))

    new_interaction = {
        "user_id": req.user_id,
        "post_id": req.post_id,
        "dwell_seconds": dwell,
        "liked": liked,
        "replied": replied,
        "skipped": skipped,
        "explicit_rating": 5.0 if liked else (1.0 if skipped else 3.0),
        "timestamp": now.isoformat(),
    }

    state["interactions"].append(new_interaction)
    orchestrator.user_seen_posts[req.user_id].add(req.post_id)
    orchestrator.user_interaction_counts[req.user_id] = orchestrator.user_interaction_counts.get(req.user_id, 0) + 1

    return {
        "status": "success",
        "action_recorded": req.action,
        "user_total_interactions": orchestrator.user_interaction_counts[req.user_id],
        "post_likes": post.likes_count,
        "post_replies": post.replies_count,
    }


@router.post("/attack/inject")
def inject_attack(req: AttackInjectRequest):
    """Simulates like-bombing shilling attack on a target post."""
    ensure_state_initialized()
    simulator: ShillingAttackSimulator = state["simulator"]
    detector: ShillingDefenseDetector = state["detector"]
    posts = state["posts"]

    result = simulator.inject_attack(
        target_post_id=req.target_post_id,
        attack_type=req.attack_type,
        posts=posts,
    )

    # Add bot interactions into detector analysis
    all_interactions = list(state["interactions"]) + simulator.bot_interactions
    detector.analyze_interactions(all_interactions)

    return result


@router.post("/attack/reset")
def reset_attack():
    """Removes all synthetic bots and restores organic metrics."""
    ensure_state_initialized()
    simulator: ShillingAttackSimulator = state["simulator"]
    detector: ShillingDefenseDetector = state["detector"]
    posts = state["posts"]

    result = simulator.reset_attack(posts=posts)
    detector.analyze_interactions(state["interactions"])

    return result


@router.get("/attack/status")
def attack_status(user_id: int = Query(0), target_post_id: Optional[int] = Query(None)):
    """Returns current attack status, target post ID, and its position in feed."""
    ensure_state_initialized()
    simulator: ShillingAttackSimulator = state["simulator"]
    detector: ShillingDefenseDetector = state["detector"]
    orchestrator: RecommendationOrchestrator = state["orchestrator"]

    target_id = simulator.target_post_id if simulator.target_post_id is not None else target_post_id
    attack_type = simulator.active_attack
    target_rank = None
    target_post = None

    if target_id is not None:
        target_post = state["post_map"].get(target_id)
        # Compute feed without defense to see rank
        feed, _ = orchestrator.generate_feed(user_id=user_id, top_k=50, defense_active=False)
        for idx, sc in enumerate(feed):
            if sc.hydrated.candidate.post.post_id == target_id:
                target_rank = idx + 1
                break

    is_anomalous, anomaly_score = (False, 0.0)
    if target_id is not None:
        is_anomalous, anomaly_score = detector.is_post_anomalous(target_id)

    return {
        "active_attack": attack_type,
        "target_post_id": target_id,
        "target_post_content": target_post.content if target_post else None,
        "target_post_likes": target_post.likes_count if target_post else None,
        "target_post_rank_unprotected": target_rank,
        "anomaly_score": anomaly_score,
        "is_flagged_as_botnet": is_anomalous,
        "num_bots": simulator.num_bots if simulator.active_attack else 0,
    }


@router.get("/benchmark/summary")
def get_benchmark_summary():
    """Returns precomputed offline evaluation tables and ablation results."""
    if not BENCHMARK_FILE.exists():
        raise HTTPException(status_code=404, detail="Benchmark summary not generated yet. Run evaluate.py first.")

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.get("/posts/popular")
def get_popular_posts():
    """Returns top 30 most popular posts for attack testing."""
    posts: List[Post] = state["posts"]
    sorted_posts = sorted(posts, key=lambda p: p.likes_count, reverse=True)[:30]
    return [
        {
            "post_id": p.post_id,
            "author_id": p.author_id,
            "likes_count": p.likes_count,
            "replies_count": p.replies_count,
            "department": p.department,
            "content_snippet": p.content[:90] + "...",
            "tags": p.tags,
        }
        for p in sorted_posts
    ]

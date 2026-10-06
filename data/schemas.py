"""
Pydantic v2 schemas and data models for the Mini-X recommendation pipeline.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class User(BaseModel):
    user_id: int
    username: str
    department: str
    persona: str
    preferred_tags: List[str]
    followed_user_ids: List[int] = Field(default_factory=list)
    blocked_user_ids: List[int] = Field(default_factory=list)


class Post(BaseModel):
    post_id: int
    author_id: int
    content: str
    department: str
    tags: List[str]
    created_at: datetime
    likes_count: int = 0
    replies_count: int = 0


class Interaction(BaseModel):
    user_id: int
    post_id: int
    dwell_seconds: float
    liked: bool
    replied: bool
    skipped: bool
    explicit_rating: Optional[float] = None
    timestamp: datetime


class CandidatePost(BaseModel):
    post: Post
    source: str  # "in_network", "als_latent", "item_knn", "content_tfidf", "cold_tag"
    retrieval_score: float


class HydratedCandidate(BaseModel):
    candidate: CandidatePost
    author_follower_count: int
    user_author_interaction_history: int
    post_age_hours: float
    author_reputation_score: float
    feature_vector: Optional[List[float]] = None


class ScoredCandidate(BaseModel):
    hydrated: HydratedCandidate
    p_like: float
    p_reply: float
    p_skip: float
    final_score: float
    explanation: Dict[str, Any]

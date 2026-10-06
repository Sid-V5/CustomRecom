"""
Shilling Attack Simulator (Section 3.5, 5.9).
Simulates adversarial profile injection attacks:
1. Random Attack: Bot accounts like target post + random catalog posts.
2. Bandwagon Attack: Bot accounts like target post + top-popular campus posts
   to artificially inflate target post into Top-3 feed.
"""

from datetime import datetime, timezone
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import SHILLING_FILLER_COUNT, SHILLING_NUM_BOTS
from data.schemas import Post, User


class ShillingAttackSimulator:
    """Injects and cleans synthetic adversarial bot accounts targeting a specific post."""

    def __init__(self, num_bots: int = SHILLING_NUM_BOTS, filler_count: int = SHILLING_FILLER_COUNT):
        self.num_bots = num_bots
        self.filler_count = filler_count
        self.active_attack: Optional[str] = None
        self.target_post_id: Optional[int] = None
        self.bot_user_ids: Set[int] = set()
        self.bot_interactions: List[Dict] = []
        self.original_post_likes: Optional[int] = None
        self.original_post_replies: Optional[int] = None

    def inject_attack(
        self,
        target_post_id: int,
        attack_type: str = "bandwagon",
        posts: Optional[List[Post]] = None,
        base_user_id: int = 1000,
    ) -> Dict:
        """
        Injects M bot accounts into simulated dataset targeting target_post_id.
        Attack types: 'bandwagon' or 'random'.
        """
        self.target_post_id = target_post_id
        self.active_attack = attack_type
        self.bot_user_ids.clear()
        self.bot_interactions.clear()

        if posts is None:
            raise ValueError("Posts list required for attack simulation.")

        post_map = {p.post_id: p for p in posts}
        target_post = post_map.get(target_post_id)
        if target_post is None:
            raise ValueError(f"Target post ID {target_post_id} not found.")

        self.original_post_likes = target_post.likes_count
        self.original_post_replies = target_post.replies_count

        # Identify filler posts
        if attack_type == "bandwagon":
            # The top 15 most popular campus posts to disguise as mainstream users
            popular_posts = sorted(posts, key=lambda p: p.likes_count, reverse=True)[:self.filler_count]
            filler_pool = [p.post_id for p in popular_posts if p.post_id != target_post_id]
        else:  # random attack
            # Coordinated botnet shares a template of random catalog posts
            rng = random.Random(42)
            all_ids = [p.post_id for p in posts if p.post_id != target_post_id]
            filler_pool = rng.sample(all_ids, k=self.filler_count)

        now = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

        # Generate bots
        for b in range(self.num_bots):
            bot_id = base_user_id + b
            self.bot_user_ids.add(bot_id)

            # 1. Mandatory like on target post
            self.bot_interactions.append({
                "user_id": bot_id,
                "post_id": target_post_id,
                "dwell_seconds": 45.0,
                "liked": True,
                "replied": True if b % 3 == 0 else False,
                "skipped": False,
                "explicit_rating": 5.0,
                "timestamp": now.isoformat(),
                "is_bot": True,
            })

            # 2. Filler likes
            chosen_fillers = filler_pool
            for f_id in chosen_fillers:
                self.bot_interactions.append({
                    "user_id": bot_id,
                    "post_id": f_id,
                    "dwell_seconds": 30.0,
                    "liked": True,
                    "replied": False,
                    "skipped": False,
                    "explicit_rating": 4.0,
                    "timestamp": now.isoformat(),
                    "is_bot": True,
                })

        # Artificially inflate target post metrics
        target_post.likes_count += self.num_bots
        target_post.replies_count += int(self.num_bots / 3)

        return {
            "status": "attack_injected",
            "attack_type": attack_type,
            "target_post_id": target_post_id,
            "num_bots": self.num_bots,
            "injected_interactions": len(self.bot_interactions),
            "new_target_likes": target_post.likes_count,
        }

    def reset_attack(self, posts: Optional[List[Post]] = None) -> Dict:
        """Removes all injected bot interactions and restores post likes."""
        if self.target_post_id is not None and posts is not None:
            post_map = {p.post_id: p for p in posts}
            target_post = post_map.get(self.target_post_id)
            if target_post is not None and self.original_post_likes is not None:
                target_post.likes_count = self.original_post_likes
                if self.original_post_replies is not None:
                    target_post.replies_count = self.original_post_replies
                else:
                    target_post.replies_count = max(0, target_post.replies_count - int(self.num_bots / 3))

        removed_bots = len(self.bot_user_ids)
        removed_interactions = len(self.bot_interactions)

        self.active_attack = None
        self.target_post_id = None
        self.bot_user_ids.clear()
        self.bot_interactions.clear()
        self.original_post_likes = None
        self.original_post_replies = None

        return {
            "status": "attack_cleared",
            "removed_bots": removed_bots,
            "removed_interactions": removed_interactions,
        }

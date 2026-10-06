"""
Synthetic Campus Social Dataset Generator.
Simulates a university social network with 400 users, 3,500 posts,
Barabási-Albert scale-free follow graph, multi-action implicit logs,
and a held-out explicit rating slice.
"""

import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np

from config import (
    BARABASI_M,
    EXPLICIT_HOLDOUT_FILE,
    FOLLOWS_FILE,
    INTERACTIONS_FILE,
    NUM_POSTS,
    NUM_USERS,
    POSTS_FILE,
    RANDOM_SEED,
    RAW_DATA_DIR,
    USERS_FILE,
)
from data.schemas import Interaction, Post, User

# Fixed Archetypes / Personas
PERSONAS = {
    "CS_Undergrad": {
        "dept": "Computer Science & AI",
        "tags": ["#cs", "#ai", "#hackathons", "#internships", "#coding", "#systems", "#webdev", "#python"],
        "ratio": 0.35,
    },
    "Bio_Researcher": {
        "dept": "Biology & Bioengineering",
        "tags": ["#biotech", "#research", "#lablife", "#academia", "#genomics", "#crispr", "#biochem"],
        "ratio": 0.22,
    },
    "Campus_Club": {
        "dept": "Student Life & Clubs",
        "tags": ["#campuslife", "#social", "#sports", "#music", "#fest", "#events", "#clubs"],
        "ratio": 0.10,
    },
    "Course_TA": {
        "dept": "Mathematics & Computing",
        "tags": ["#homework", "#exam", "#officehours", "#math", "#grading", "#cs", "#study"],
        "ratio": 0.13,
    },
    "Freshman": {
        "dept": "General Studies",
        "tags": ["#freshman", "#campuslife", "#study", "#clubs", "#housing", "#advice"],
        "ratio": 0.20,  # Cold-start population
    },
}

# Post templates for authentic campus content
POST_TEMPLATES = {
    "CS_Undergrad": [
        "Anyone looking for a teammate for HackMIT? Building an end-to-end agentic workflow with PyTorch and FastAPI! {tags}",
        "Finally got our distributed GPU training cluster working without CUDA out of memory errors. The feeling is unmatched! {tags}",
        "Summer SWE internship applications just opened for several tech firms. Updated my resume with latest ML projects. {tags}",
        "Hot take: writing custom CUDA kernels is actually fun once you understand warp synchronization. {tags}",
        "Just open-sourced our campus course planning tool! Feedback and PRs welcome on GitHub. {tags}",
        "Debugging a race condition that only reproduces at 3 AM. System programming strikes again. {tags}",
        "Attending the AI alignment guest lecture at Stata Center this evening. Who else is going? {tags}",
    ],
    "Bio_Researcher": [
        "Plate reader results came in: our CRISPR knockout cell line shows a 4x drop in target expression! {tags}",
        "Another 14-hour shift running confocal microscopy. Coffee is the primary solvent in this lab. {tags}",
        "Excited to present our computational genomics poster at next week's symposium! {tags}",
        "Pipetting 384-well plates on a Friday night is a rite of passage for every PhD student. {tags}",
        "New preprint out: identifying novel transcriptional regulators using single-cell RNA sequencing. {tags}",
        "Lab meeting discussion on ethical implications of gene editing in synthetic biology. {tags}",
    ],
    "Campus_Club": [
        "🎉 Battle of the Bands & Spring Festival is HAPPENING this Friday at 7 PM in the Central Quad! Free tacos & boba! {tags}",
        "⚽ Intramural soccer championship finals tomorrow afternoon! Come support your dorm team. {tags}",
        "Auditions open for the annual campus theater showcase! No prior experience needed, all majors welcome. {tags}",
        "Robotics Club demo day: see our autonomous rover navigate the engineering courtyard obstacles! {tags}",
        "Board Game Night at the Student Center lounge tonight starting 8 PM. Snacks provided! {tags}",
        "Campus Blood Drive next Tuesday in Hall B. Donate blood and receive a free club hoodie! {tags}",
    ],
    "Course_TA": [
        "Midterm review session scheduled for Thursday 6-8 PM in Hall 101. Topics: Eigenvalues, SVD, and PCA. {tags}",
        "Homework 3 autograder has been updated on Gradescope. Please verify your test submissions before 11:59 PM. {tags}",
        "Office hours today are moved to Room 32-044 due to the departmental seminar. {tags}",
        "Common mistake on Problem 4: remember that correlation is normalized covariance, bounded by [-1, 1]! {tags}",
        "Extra practice problems for final exam prep are now posted on the course Canvas page. {tags}",
        "Reminder: Late penalty policy applies after midnight unless you have an approved dean extension. {tags}",
    ],
    "Freshman": [
        "What is the best quiet floor in the central library for studying without distractions? {tags}",
        "Is the campus dining hall mac and cheese actually edible today or should I walk to the student center? {tags}",
        "Looking for freshman study partners for Intro to Data Structures! Let's connect. {tags}",
        "How do club recruitment interviews work? Any tips for first-year applicants? {tags}",
        "Lost my student ID card somewhere between the dining hall and quad. Has anyone turned it in? {tags}",
        "First week surviving morning 8:30 AM lectures... already running on 3 hours of sleep. {tags}",
    ],
}

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Casey", "Riley", "Avery", "Jamie",
    "Priya", "Rahul", "Chen", "Wei", "Mei", "Elena", "Mateo", "Lucas", "Sofia", "Hassan",
    "Amina", "Kavita", "Arjun", "Hiroshi", "Yuki", "Chloe", "Noah", "Liam", "Emma", "Maya",
    "David", "Sarah", "Daniel", "Zoe", "Leo", "Mia", "Ethan", "Hannah", "Oliver", "Ava"
]

LAST_NAMES = [
    "Patel", "Sharma", "Zhang", "Wang", "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Kim", "Lee", "Park", "Chen",
    "Gupta", "Rao", "Ito", "Tanaka", "Muller", "Schmidt", "Dubois", "Moreau", "Novak", "Rossi"
]


class CampusDatasetGenerator:
    """Generates synthetic campus social graph, posts, and multi-signal interactions."""

    def __init__(self, num_users: int = NUM_USERS, num_posts: int = NUM_POSTS, seed: int = RANDOM_SEED):
        self.num_users = num_users
        self.num_posts = num_posts
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.users: List[User] = []
        self.posts: List[Post] = []
        self.follows: List[Tuple[int, int]] = []
        self.interactions: List[Interaction] = []
        self.explicit_holdout: List[Dict] = []
        self.base_time = datetime(2026, 10, 6, 12, 0, 0, tzinfo=timezone.utc)

    def generate_users(self) -> List[User]:
        """Generate user archetypes, departments, and tag interests."""
        users = []
        persona_keys = list(PERSONAS.keys())
        ratios = [PERSONAS[p]["ratio"] for p in persona_keys]

        # Assign user personas proportionally
        persona_assignments = np.random.choice(persona_keys, size=self.num_users, p=ratios)

        for user_id in range(self.num_users):
            persona = str(persona_assignments[user_id])
            meta = PERSONAS[persona]
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            
            if persona == "Campus_Club":
                club_names = ["RoboticsClub", "HackersOrg", "BioSoc", "ChessClub", "ACM_StudentChapter", "CampusDaily"]
                username = f"{random.choice(club_names)}_{user_id}"
            elif persona == "Course_TA":
                username = f"ta_{first.lower()}_{user_id}"
            else:
                username = f"{first.lower()}_{last.lower()}{user_id % 100}"

            # Sample preferred tags (3 to 5 tags from archetype)
            k_tags = min(len(meta["tags"]), random.randint(3, 5))
            preferred_tags = random.sample(meta["tags"], k=k_tags)
            
            # Occasionally sample 1 cross-discipline tag
            if random.random() < 0.25:
                cross_tags = ["#campuslife", "#ai", "#events", "#research", "#sports"]
                extra_tag = random.choice(cross_tags)
                if extra_tag not in preferred_tags:
                    preferred_tags.append(extra_tag)

            user = User(
                user_id=user_id,
                username=username,
                department=meta["dept"],
                persona=persona,
                preferred_tags=preferred_tags,
                followed_user_ids=[],
                blocked_user_ids=[],
            )
            users.append(user)

        self.users = users
        return users

    def generate_follow_graph(self) -> List[Tuple[int, int]]:
        """
        Generate a scale-free directed follow graph using Barabási-Albert preferential attachment
        with intra-department homophily and persona-specific dynamics.
        """
        m = BARABASI_M
        in_degrees = {u.user_id: 1 for u in self.users}
        follows_set = set()

        # Seed initial core nodes (e.g. clubs and TAs)
        initial_nodes = [u.user_id for u in self.users[:m * 2]]
        for i in range(len(initial_nodes)):
            for j in range(len(initial_nodes)):
                if i != j and random.random() < 0.4:
                    follows_set.add((initial_nodes[i], initial_nodes[j]))
                    in_degrees[initial_nodes[j]] += 1

        # Preferential attachment for remaining nodes
        for u in self.users:
            u_id = u.user_id
            if u.persona == "Freshman":
                # Cold-start users follow very few accounts (1 to 3)
                target_count = random.randint(1, 3)
            elif u.persona == "Campus_Club":
                # Clubs follow 8-15 active organizers
                target_count = random.randint(8, 15)
            else:
                target_count = random.randint(m, m * 3)

            # Compute probability distribution based on in-degrees with persona homophily
            nodes = [v.user_id for v in self.users if v.user_id != u_id]
            weights = []
            for v_id in nodes:
                deg = in_degrees[v_id]
                target_user = self.users[v_id]
                homophily = 2.5 if target_user.department == u.department else 1.0
                # Clubs and TAs have higher structural visibility
                role_boost = 3.0 if target_user.persona in ("Campus_Club", "Course_TA") else 1.0
                weights.append(deg * homophily * role_boost)

            weights = np.array(weights, dtype=float)
            probs = weights / weights.sum()

            chosen_targets = np.random.choice(nodes, size=min(target_count, len(nodes)), replace=False, p=probs)
            for t_id in chosen_targets:
                follows_set.add((u_id, int(t_id)))
                in_degrees[int(t_id)] += 1

            # Sporadic random blocks (1-2 users blocked for ~5% of users to test BlockFilter)
            if random.random() < 0.08:
                blocked = random.choice([n for n in nodes if n != u_id])
                u.blocked_user_ids.append(int(blocked))

        # Assign follows back to User objects
        user_map = {u.user_id: u for u in self.users}
        for src, dst in follows_set:
            user_map[src].followed_user_ids.append(dst)

        self.follows = list(follows_set)
        return self.follows

    def generate_posts(self) -> List[Post]:
        """Generate 3,500 campus posts distributed across authors."""
        posts = []
        # Clubs and TAs author more posts; Freshmen author very few
        author_weights = []
        for u in self.users:
            if u.persona == "Campus_Club":
                author_weights.append(10.0)
            elif u.persona == "Course_TA":
                author_weights.append(6.0)
            elif u.persona == "CS_Undergrad":
                author_weights.append(3.0)
            elif u.persona == "Bio_Researcher":
                author_weights.append(2.5)
            else:  # Freshman
                author_weights.append(0.5)

        author_weights = np.array(author_weights)
        author_probs = author_weights / author_weights.sum()

        chosen_authors = np.random.choice([u.user_id for u in self.users], size=self.num_posts, p=author_probs)

        user_map = {u.user_id: u for u in self.users}

        for post_id in range(self.num_posts):
            author_id = int(chosen_authors[post_id])
            author = user_map[author_id]
            persona = author.persona

            templates = POST_TEMPLATES.get(persona, POST_TEMPLATES["CS_Undergrad"])
            template = random.choice(templates)

            # Choose 2-4 tags matching author's department and persona
            persona_tags = PERSONAS[persona]["tags"]
            k_tags = min(len(persona_tags), random.randint(2, 4))
            selected_tags = random.sample(persona_tags, k=k_tags)
            tags_str = " ".join(selected_tags)

            content = template.format(tags=tags_str)

            # Timestamp: past 14 days, with recent posts having higher density
            # Using exponential distribution for age
            age_days = random.weibullvariate(alpha=4.0, beta=1.2)  # clusters near 0-14 days
            age_days = min(14.0, max(0.01, age_days))
            created_at = self.base_time - timedelta(days=age_days, minutes=random.randint(0, 1440))

            post = Post(
                post_id=post_id,
                author_id=author_id,
                content=content,
                department=author.department,
                tags=selected_tags,
                created_at=created_at,
                likes_count=0,
                replies_count=0,
            )
            posts.append(post)

        self.posts = posts
        return posts

    def generate_interactions(self) -> List[Interaction]:
        """
        Generate multi-action implicit logs: Dwell time, Likes, Replies, Skips,
        and explicit holdout slice.
        Target matrix sparsity ~ 98.4%.
        """
        interactions = []
        user_map = {u.user_id: u for u in self.users}
        post_map = {p.post_id: p for p in self.posts}
        total_pairs = self.num_users * self.num_posts  # 1,400,000

        # Post popularity tracker for candidate sampling
        post_likes = {p.post_id: 0 for p in self.posts}
        post_replies = {p.post_id: 0 for p in self.posts}

        # Warm vs Cold interaction budgets
        for u in self.users:
            if u.persona == "Freshman":
                # Pure cold-start: 0, 1, or 2 interactions
                num_user_interactions = random.choice([0, 1, 1, 2])
            elif u.persona in ("Campus_Club", "Course_TA"):
                num_user_interactions = random.randint(50, 110)
            else:
                num_user_interactions = random.randint(40, 90)

            if num_user_interactions == 0:
                continue

            # Candidate sampling strategy for this user:
            # 45% in-network posts (followed authors)
            # 35% topic/tag matching posts
            # 15% popular posts
            # 5% random exploration
            in_network_posts = [p.post_id for p in self.posts if p.author_id in u.followed_user_ids]
            tag_matching_posts = [
                p.post_id for p in self.posts if any(t in u.preferred_tags for t in p.tags)
            ]
            all_post_ids = [p.post_id for p in self.posts]

            sampled_pool = set()
            
            # 1. In-network
            if in_network_posts:
                k_in = min(len(in_network_posts), int(num_user_interactions * 0.45))
                sampled_pool.update(random.sample(in_network_posts, k=k_in))

            # 2. Tag matches
            if tag_matching_posts:
                k_tag = min(len(tag_matching_posts), int(num_user_interactions * 0.35))
                sampled_pool.update(random.sample(tag_matching_posts, k=k_tag))

            # 3. Fill up to num_user_interactions with catalog samples
            needed = num_user_interactions - len(sampled_pool)
            if needed > 0:
                remaining = list(set(all_post_ids) - sampled_pool)
                if remaining:
                    sampled_pool.update(random.sample(remaining, k=min(needed, len(remaining))))

            # Now simulate realistic multi-action engagements for chosen posts
            for p_id in sampled_pool:
                post = post_map[p_id]
                
                # Cannot interact with own post
                if post.author_id == u.user_id:
                    continue

                # Compute topic affinity
                shared_tags = set(u.preferred_tags) & set(post.tags)
                tag_jaccard = len(shared_tags) / max(1, len(set(u.preferred_tags) | set(post.tags)))
                dept_match = 1.0 if u.department == post.department else 0.0
                is_followed = 1.0 if post.author_id in u.followed_user_ids else 0.0

                affinity = 0.5 * tag_jaccard + 0.3 * dept_match + 0.2 * is_followed

                # Sample dwell time from log-normal (mu=1.8, sigma=0.8) shifted by affinity
                mu = 1.8 + (affinity - 0.3) * 0.8
                dwell = float(np.random.lognormal(mean=mu, sigma=0.8))
                dwell = round(float(np.clip(dwell, 0.5, 120.0)), 2)

                # Action rules:
                # Skip if dwell < 3.0s or very low affinity
                skipped = False
                liked = False
                replied = False

                if dwell < 3.0 or (affinity < 0.15 and random.random() < 0.6):
                    skipped = True
                    liked = False
                    replied = False
                else:
                    # Liked probability increases with dwell and affinity
                    like_prob = 1.0 / (1.0 + math.exp(-3.5 * (affinity + math.log1p(dwell) / 4.0 - 1.0)))
                    if random.random() < like_prob or dwell > 15.0:
                        liked = True
                        # Reply probability is ~10-18% of liked posts
                        reply_prob = 0.14 * (affinity + 0.5)
                        if random.random() < reply_prob and dwell > 10.0:
                            replied = True

                if liked:
                    post_likes[p_id] += 1
                if replied:
                    post_replies[p_id] += 1

                # Timestamp between post creation and now
                post_created = post.created_at
                max_delta = max(60, int((self.base_time - post_created).total_seconds()))
                offset_seconds = random.randint(10, max_delta)
                interaction_time = post_created + timedelta(seconds=offset_seconds)

                # Holdout explicit rating (1 to 5 stars) for ~10% sample
                explicit_rating = None
                if random.random() < 0.10:
                    if liked and replied:
                        explicit_rating = round(random.choice([4.5, 5.0]), 1)
                    elif liked:
                        explicit_rating = round(random.choice([3.5, 4.0, 4.5, 5.0]), 1)
                    elif skipped:
                        explicit_rating = round(random.choice([1.0, 1.5, 2.0]), 1)
                    else:
                        explicit_rating = round(random.choice([2.5, 3.0, 3.5]), 1)

                    self.explicit_holdout.append({
                        "user_id": u.user_id,
                        "post_id": p_id,
                        "rating": explicit_rating,
                        "timestamp": interaction_time.isoformat(),
                    })

                interaction = Interaction(
                    user_id=u.user_id,
                    post_id=p_id,
                    dwell_seconds=dwell,
                    liked=liked,
                    replied=replied,
                    skipped=skipped,
                    explicit_rating=explicit_rating,
                    timestamp=interaction_time,
                )
                interactions.append(interaction)

        # Update post aggregate counters
        for post in self.posts:
            post.likes_count = post_likes.get(post.post_id, 0)
            post.replies_count = post_replies.get(post.post_id, 0)

        self.interactions = interactions
        return interactions

    def save(self):
        """Save generated synthetic dataset to disk in raw directory."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Users JSON
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump([u.model_dump() for u in self.users], f, indent=2)

        # 2. Posts JSON
        with open(POSTS_FILE, "w", encoding="utf-8") as f:
            posts_data = []
            for p in self.posts:
                d = p.model_dump()
                d["created_at"] = p.created_at.isoformat()
                posts_data.append(d)
            json.dump(posts_data, f, indent=2)

        # 3. Follows CSV
        with open(FOLLOWS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "followed_user_id"])
            for src, dst in self.follows:
                writer.writerow([src, dst])

        # 4. Interactions CSV
        with open(INTERACTIONS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "user_id", "post_id", "dwell_seconds", "liked", "replied",
                "skipped", "explicit_rating", "timestamp"
            ])
            for i in self.interactions:
                writer.writerow([
                    i.user_id,
                    i.post_id,
                    i.dwell_seconds,
                    int(i.liked),
                    int(i.replied),
                    int(i.skipped),
                    i.explicit_rating if i.explicit_rating is not None else "",
                    i.timestamp.isoformat(),
                ])

        # 5. Explicit Holdout CSV
        with open(EXPLICIT_HOLDOUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "post_id", "rating", "timestamp"])
            for row in self.explicit_holdout:
                writer.writerow([row["user_id"], row["post_id"], row["rating"], row["timestamp"]])

        # Summary statistics
        total_cells = self.num_users * self.num_posts
        num_interactions = len(self.interactions)
        sparsity = (1.0 - (num_interactions / total_cells)) * 100.0

        print(f"=== Mini-X Dataset Generated Successfully ===")
        print(f"Total Users: {len(self.users)}")
        print(f"Total Posts: {len(self.posts)}")
        print(f"Total Follow Edges: {len(self.follows)}")
        print(f"Total Interactions: {num_interactions}")
        print(f"Matrix Sparsity: {sparsity:.2f}% (Target: 98.2% - 98.8%)")
        print(f"Explicit Holdout Ratings: {len(self.explicit_holdout)}")
        print(f"Saved files to: {RAW_DATA_DIR}")


def run_generator(num_users: int = NUM_USERS, num_posts: int = NUM_POSTS, seed: int = RANDOM_SEED):
    gen = CampusDatasetGenerator(num_users=num_users, num_posts=num_posts, seed=seed)
    gen.generate_users()
    gen.generate_follow_graph()
    gen.generate_posts()
    gen.generate_interactions()
    gen.save()
    return gen

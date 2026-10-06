"""
PyTorch Two-Tower Neural Collaborative Filtering (Lecture L16).
Embeds User features (user_id, persona_id, tag profile) and Post features
(post_id, author_id, text reduced representation) into a shared 32-dimensional
representation space, optimizing cosine similarity with Binary Cross-Entropy loss.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import (
    ARTIFACTS_DIR,
    NEURAL_BATCH_SIZE,
    NEURAL_EMBED_DIM,
    NEURAL_EPOCHS,
    NEURAL_LR,
    NUM_POSTS,
    NUM_USERS,
)
from data.schemas import Post, User


class TwoTowerDataset(Dataset):
    """Dataset for training Two-Tower Neural CF."""

    def __init__(
        self,
        user_ids: np.ndarray,
        persona_ids: np.ndarray,
        user_tag_feats: np.ndarray,
        post_ids: np.ndarray,
        author_ids: np.ndarray,
        post_text_feats: np.ndarray,
        labels: np.ndarray,
    ):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.persona_ids = torch.tensor(persona_ids, dtype=torch.long)
        self.user_tag_feats = torch.tensor(user_tag_feats, dtype=torch.float32)
        self.post_ids = torch.tensor(post_ids, dtype=torch.long)
        self.author_ids = torch.tensor(author_ids, dtype=torch.long)
        self.post_text_feats = torch.tensor(post_text_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.persona_ids[idx],
            self.user_tag_feats[idx],
            self.post_ids[idx],
            self.author_ids[idx],
            self.post_text_feats[idx],
            self.labels[idx],
        )


class TwoTowerModel(nn.Module):
    """
    Two-Tower Neural Network Architecture:
    User Tower: Linear(32 + 8 + 16 = 56, 64) -> ReLU() -> Linear(64, 32)
    Post Tower: Linear(32 + 16 + 16 = 64, 64) -> ReLU() -> Linear(64, 32)
    """

    def __init__(
        self,
        num_users: int = NUM_USERS,
        num_posts: int = NUM_POSTS,
        num_personas: int = 5,
        embed_dim: int = NEURAL_EMBED_DIM,
        tag_dim: int = 16,
        text_dim: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # User Tower Embeddings & MLP
        self.user_embed = nn.Embedding(num_users, 32)
        self.persona_embed = nn.Embedding(num_personas, 8)
        self.user_mlp = nn.Sequential(
            nn.Linear(32 + 8 + tag_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

        # Post Tower Embeddings & MLP
        self.post_embed = nn.Embedding(num_posts, 32)
        self.author_embed = nn.Embedding(num_users, 16)
        self.post_mlp = nn.Sequential(
            nn.Linear(32 + 16 + text_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward_user(self, user_id, persona_id, user_tag_feat):
        u_emb = self.user_embed(user_id)
        p_emb = self.persona_embed(persona_id)
        user_input = torch.cat([u_emb, p_emb, user_tag_feat], dim=-1)
        out = self.user_mlp(user_input)
        return F.normalize(out, p=2, dim=-1)

    def forward_post(self, post_id, author_id, post_text_feat):
        pt_emb = self.post_embed(post_id)
        a_emb = self.author_embed(author_id)
        post_input = torch.cat([pt_emb, a_emb, post_text_feat], dim=-1)
        out = self.post_mlp(post_input)
        return F.normalize(out, p=2, dim=-1)

    def forward(self, user_id, persona_id, user_tag_feat, post_id, author_id, post_text_feat):
        u_vec = self.forward_user(user_id, persona_id, user_tag_feat)
        p_vec = self.forward_post(post_id, author_id, post_text_feat)
        # Cosine similarity bounded in [-1, 1], shifted to [0, 1] for BCE
        cosine_sim = (u_vec * p_vec).sum(dim=-1)
        prob = (cosine_sim + 1.0) / 2.0
        return prob


class TwoTowerRecommender:
    """Wrapper class managing training, embedding extraction, and inference."""

    def __init__(
        self,
        embed_dim: int = NEURAL_EMBED_DIM,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TwoTowerModel(embed_dim=embed_dim).to(self.device)
        self.persona_map = {
            "CS_Undergrad": 0,
            "Bio_Researcher": 1,
            "Campus_Club": 2,
            "Course_TA": 3,
            "Freshman": 4,
        }
        self.user_embs_cached: Optional[np.ndarray] = None
        self.post_embs_cached: Optional[np.ndarray] = None

    def prepare_data(
        self,
        users: List[User],
        posts: List[Post],
        interactions: List[Dict],
        content_model,
    ) -> DataLoader:
        """Create PyTorch DataLoader with positive interactions and negative samples."""
        from sklearn.decomposition import TruncatedSVD

        user_map = {u.user_id: u for u in users}
        post_map = {p.post_id: p for p in posts}

        # Reduce content vectors to 16-dim for post text feat
        svd_post = TruncatedSVD(n_components=16, random_state=42)
        if content_model.post_vectors is not None:
            post_text_reduced = svd_post.fit_transform(content_model.post_vectors)
        else:
            post_text_reduced = np.zeros((len(posts), 16), dtype=np.float32)

        # Reduce user tag vectors to 16-dim
        user_profiles_mat = np.array([content_model.get_user_profile(u).squeeze() for u in users])
        svd_user = TruncatedSVD(n_components=16, random_state=42)
        user_tag_reduced = svd_user.fit_transform(user_profiles_mat)

        u_ids = []
        p_ids = []
        u_tags = []
        p_texts = []
        auth_ids = []
        personas = []
        labels = []

        # Positive pairs
        positive_pairs = set()
        for inter in interactions:
            u_id = inter["user_id"]
            p_id = inter["post_id"]
            if u_id in user_map and p_id in post_map:
                liked = inter.get("liked", False)
                dwell = inter.get("dwell_seconds", 0.0)
                if liked or dwell >= 10.0:
                    positive_pairs.add((u_id, p_id))
                    u = user_map[u_id]
                    p = post_map[p_id]
                    u_ids.append(u_id)
                    personas.append(self.persona_map.get(u.persona, 0))
                    u_tags.append(user_tag_reduced[u_id])
                    p_ids.append(p_id)
                    auth_ids.append(p.author_id)
                    p_texts.append(post_text_reduced[content_model.post_id_to_idx[p_id]])
                    labels.append(1.0)

        # Negative sampling (1:1 ratio)
        num_neg = len(u_ids)
        np.random.seed(42)
        all_post_ids = list(post_map.keys())
        for _ in range(num_neg):
            rand_u = np.random.randint(0, len(users))
            rand_p = np.random.choice(all_post_ids)
            while (rand_u, rand_p) in positive_pairs:
                rand_p = np.random.choice(all_post_ids)

            u = user_map[rand_u]
            p = post_map[rand_p]
            u_ids.append(rand_u)
            personas.append(self.persona_map.get(u.persona, 0))
            u_tags.append(user_tag_reduced[rand_u])
            p_ids.append(rand_p)
            auth_ids.append(p.author_id)
            p_texts.append(post_text_reduced[content_model.post_id_to_idx[rand_p]])
            labels.append(0.0)

        dataset = TwoTowerDataset(
            user_ids=np.array(u_ids),
            persona_ids=np.array(personas),
            user_tag_feats=np.array(u_tags),
            post_ids=np.array(p_ids),
            author_ids=np.array(auth_ids),
            post_text_feats=np.array(p_texts),
            labels=np.array(labels),
        )
        return DataLoader(dataset, batch_size=NEURAL_BATCH_SIZE, shuffle=True)

    def train(self, dataloader: DataLoader, epochs: int = NEURAL_EPOCHS, lr: float = NEURAL_LR, verbose: bool = False):
        """Train Two-Tower network using BCE loss."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in dataloader:
                u_id, pers, u_tag, p_id, auth, p_text, target = [x.to(self.device) for x in batch]
                optimizer.zero_grad()
                pred = self.model(u_id, pers, u_tag, p_id, auth, p_text)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target)

            avg_loss = total_loss / len(dataloader.dataset)
            if verbose:
                print(f"Two-Tower Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        return self

    def precompute_embeddings(self, users: List[User], posts: List[Post], content_model):
        """Precompute and cache tower representations for fast dot-product inference."""
        self.model.eval()
        from sklearn.decomposition import TruncatedSVD

        svd_post = TruncatedSVD(n_components=16, random_state=42)
        if content_model.post_vectors is not None:
            post_text_reduced = svd_post.fit_transform(content_model.post_vectors)
        else:
            post_text_reduced = np.zeros((len(posts), 16), dtype=np.float32)

        user_profiles_mat = np.array([content_model.get_user_profile(u).squeeze() for u in users])
        svd_user = TruncatedSVD(n_components=16, random_state=42)
        user_tag_reduced = svd_user.fit_transform(user_profiles_mat)

        with torch.no_grad():
            # User embeddings
            u_ids = torch.tensor([u.user_id for u in users], dtype=torch.long, device=self.device)
            pers_ids = torch.tensor([self.persona_map.get(u.persona, 0) for u in users], dtype=torch.long, device=self.device)
            u_tags = torch.tensor(user_tag_reduced, dtype=torch.float32, device=self.device)
            self.user_embs_cached = self.model.forward_user(u_ids, pers_ids, u_tags).cpu().numpy()

            # Post embeddings
            p_ids = torch.tensor([p.post_id for p in posts], dtype=torch.long, device=self.device)
            auth_ids = torch.tensor([p.author_id for p in posts], dtype=torch.long, device=self.device)
            p_texts = torch.tensor(post_text_reduced, dtype=torch.float32, device=self.device)
            self.post_embs_cached = self.model.forward_post(p_ids, auth_ids, p_texts).cpu().numpy()

    def predict_score(self, user_id: int, post_id: int) -> float:
        """Fast prediction using cached embeddings."""
        if self.user_embs_cached is None or self.post_embs_cached is None:
            return 0.5
        if user_id >= len(self.user_embs_cached) or post_id >= len(self.post_embs_cached):
            return 0.5
        cosine = float(np.dot(self.user_embs_cached[user_id], self.post_embs_cached[post_id]))
        return (cosine + 1.0) / 2.0

    def recommend(self, user_id: int, top_n: int = 10, candidate_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        if self.user_embs_cached is None or self.post_embs_cached is None:
            return []
        u_emb = self.user_embs_cached[user_id]
        if candidate_ids is None:
            candidate_ids = list(range(len(self.post_embs_cached)))

        scores = [float(np.dot(u_emb, self.post_embs_cached[pid])) for pid in candidate_ids]
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [(candidate_ids[idx], (scores[idx] + 1.0) / 2.0) for idx in top_indices]

    def save(self, filepath: Optional[Path] = None):
        if filepath is None:
            filepath = ARTIFACTS_DIR / "two_tower_model.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "user_embs": self.user_embs_cached,
            "post_embs": self.post_embs_cached,
        }, filepath)

    def load(self, filepath: Optional[Path] = None):
        if filepath is None:
            filepath = ARTIFACTS_DIR / "two_tower_model.pt"
        data = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(data["model_state"])
        self.user_embs_cached = data.get("user_embs")
        self.post_embs_cached = data.get("post_embs")
        return self

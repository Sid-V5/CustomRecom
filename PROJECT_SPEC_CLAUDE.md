# TECHNICAL SPECIFICATION & IMPLEMENTATION BLUEPRINT
## Hybrid Collaborative-Content-Knowledge Recommender System ("Mini-X For You" Pipeline)

> **Document Type:** Master Implementation Plan & System Architecture Specification  
> **Domain:** Campus / Academic Social media Feed Recommender  
> **Inspired By:** Open-source X (Twitter) Recommendation Pipeline (`xai-org/x-algorithm`)  opensource

---

## 1. Executive Summary & Syllabus Alignment

This project designs and implements an end-to-end, multi-stage hybrid recommendation pipeline inspired directly by X's open-source architecture. Unlike standard academic toy projects (e.g., MovieLens user-item SVD), this system simulates a live social network feed with explicit social graphs, multi-action implicit engagements, candidate retrieval funnels, feature hydration, safety filters, calibrated multi-signal ranking, diversity selection (MMR), cold-start switching, explainability, and adversarial shilling attack defenses.
THE UI SHOULD BE LIKE A SOCIAL MEDIA APP POPULATED WITH YK DATA

### 1.1 Exhaustive Course Syllabus Mapping (AIM3143: All 36 Lectures)

Every single lecture from the approved course handout is systematically implemented and demonstrated in this project:

| Lecture No. | Course Handout Lecture Topic | Exact Implementation in Mini-X Project | Project File / Module Reference |
| :---: | :--- | :--- | :--- |
| **L1** | **Introduction to Recommender Systems** | Core problem formulation: personalized "For You" timeline distilling 3.5k campus posts to Top-15 feed. | `run_server.py`, `README.md` |
| **L2** | **Functions and Applications** | Multi-objective feed curation: discovery, social connectivity, academic announcements, and club events. | `data/schemas.py`, `data/generator.py` |
| **L3** | **Explicit vs. Implicit Feedback** | Implicit logging (dwell time proxy, likes, replies, skips) vs. held-out 1–5 explicit rating slice for validation. | `data/schemas.py`, `data/generator.py` |
| **L4** | **Challenges: Sparsity, Cold-Start, Scalability** | High sparsity matrix (~98.5%), dedicated cold-start user/item routing paths, and two-tower candidate reduction for $O(N)$ scalability. | `models/collaborative_knn.py`, `pipeline/orchestrator.py` |
| **L5** | **Matrix Representation & Multiplication** | Sparse user-item matrix $\mathbf{R} \in \mathbb{R}^{|U| \times |I|}$ using `scipy.sparse.csr_matrix` and inner-product projections. | `models/als_factorization.py`, `models/collaborative_knn.py` |
| **L6** | **Similarity Measures: Cosine & Pearson** | Vectorized Cosine Similarity and Mean-Centered Pearson Correlation with significance weighting ($\tau = 5$). | `models/collaborative_knn.py` |
| **L7** | **User-Based Collaborative Filtering** | User-neighborhood scoring predicting interaction scores based on top-k peer correlation vectors. | `models/collaborative_knn.py` |
| **L8** | **Item-Based Collaborative Filtering** | Item-neighborhood scoring based on co-engagement cosine similarity of historical liked posts. | `models/collaborative_knn.py`, `pipeline/candidate_retrieval.py` |
| **L9** | **Limitations of Memory-Based Methods** | Documented & benchmarked failure modes of kNN: memory footprint, $O(|U|^2)$ latency, and extreme cold-start brittleness. | `evaluation/evaluator.py`, report section |
| **L10** | **Model-Based Recommendation Systems** | Transition from memory-based lookups to parameterized latent representation models. | `models/als_factorization.py` |
| **L11** | **Matrix Factorization using SVD** | TruncatedSVD baseline decomposing user-item interaction matrix into $U \Sigma V^T$ latent components. | `models/als_factorization.py` |
| **L12** | **Alternating Least Squares (ALS)** | Alternating optimization fixing user factors $X$ to solve for item factors $Y$ and vice versa. | `models/als_factorization.py` |
| **L13** | **Latent Factor Models** | Low-rank embedding space ($d=32$) capturing implicit topical affinities without explicit tags. | `models/als_factorization.py` |
| **L14** | **Handling Sparsity and Scalability** | Low-rank factorization compressing $400 \times 3500$ matrix into compact factor matrices, scaling retrieval to $<10$ms. | `models/als_factorization.py` |
| **L15** | **Implicit Feedback Models** | Confidence-weighted iALS objective ($C_{ui} = 1 + \alpha r_{ui}$) where $\alpha = 40$ and $r_{ui} = \log(1 + \text{dwell})$. | `models/als_factorization.py` |
| **L16** | **Neural Collaborative Filtering** | PyTorch Two-Tower Neural Network embedding User and Post features into shared 32-dim space with non-linear MLP layers. | `models/neural_two_tower.py` |
| **L17** | **Data Normalization** | Z-score user rating normalization, min-max score scaling, and log1p transformation of dwell and counts. | `data/generator.py`, `pipeline/hydrator.py` |
| **L18** | **Missing Values & Dimensionality Reduction** | Imputation of unobserved pairs as negative samples in iALS; SVD/TF-IDF max-features truncation. | `data/generator.py`, `models/content_based.py` |
| **L19** | **Security Issues in RecSys** | Vulnerability analysis of collaborative systems to malicious profile manipulation and like-farming. | `security/shilling_simulator.py` |
| **L20** | **Shilling Attacks & Defense Mechanisms** | Random & Bandwagon attack injection + Interaction Velocity deviation detector and co-action Jaccard clustering defense. | `security/shilling_simulator.py`, `security/shilling_detector.py` |
| **L21** | **Evaluation of CF Models** | Offline split evaluating User-kNN, Item-kNN, SVD, and iALS on NDCG@10, Precision@10, and explicit RMSE. | `evaluation/evaluator.py` |
| **L22** | **Intro to Content-Based Recommendation** | Unsupervised text and metadata recommendation bypassing collaborative interaction history. | `models/content_based.py` |
| **L23** | **Architecture, Advantages & Drawbacks** | Over-specialization vs. zero-cold-start trade-off analysis in pipeline routing. | `pipeline/orchestrator.py` |
| **L24** | **Item Profiles and Feature Discovery** | Tokenization, TF-IDF vectorization, hashtag feature extraction, and author metadata extraction from post text. | `models/content_based.py`, `pipeline/hydrator.py` |
| **L25** | **Tags, User Profiles & Similarity Retrieval** | Dynamic user centroid profile vector $\vec{U}_u$ constructed from engagement-weighted post vectors. | `models/content_based.py` |
| **L26** | **Classification Algorithms for Content-Based** | Supervised multi-action Logistic Regression / LightGBM models predicting $P(\text{like})$, $P(\text{reply})$, $P(\text{skip})$. | `pipeline/heavy_ranker.py` |
| **L27** | **Knowledge-Based Recommendation Systems** | Explicit requirements and domain rule engine operating without interaction history. | `pipeline/filters.py`, `pipeline/orchestrator.py` |
| **L28** | **Constraint-Based Recommendation** | Hard constraints: Department matching filter, language check, max post age cutoff ($t \le 7$ days). | `pipeline/filters.py` |
| **L29** | **Case-Based Recommendation** | Problem-case similarity matching: student searches a specific homework/project query case and finds structurally similar past posts. | `models/content_based.py`, `pipeline/filters.py` |
| **L30** | **Hybrid Recommendation Systems** | Unifying collaborative, content, and knowledge paradigms into an integrated pipeline. | `pipeline/orchestrator.py` |
| **L31** | **Monolithic Hybridization (Feature Combination & Augmentation)** | Constructing unified 12-dim feature vectors combining CF latent dot product, TF-IDF cosine, and author features for the Heavy Ranker. | `pipeline/heavy_ranker.py`, `pipeline/hydrator.py` |
| **L32** | **Parallel Hybridization (Weighted, Switching & Mixed)** | **Weighted:** Linear combination of scoring heads.<br>**Switching:** Cold-start routing switch ($<3$ interactions).<br>**Mixed:** 50/50 mix of In-Network and Out-of-Network candidates. | `pipeline/orchestrator.py`, `mixer/diversity_selector.py` |
| **L33** | **Pipelined Hybridization (Cascade & Meta-Level)** | **Cascade:** High-recall retrieval funnels feed candidate pool to Heavy Ranker.<br>**Meta-Level:** iALS latent factors fed as dense features into Heavy Ranker. | `pipeline/candidate_retrieval.py`, `pipeline/heavy_ranker.py` |
| **L34** | **Limitations of Hybridization Strategies** | Weight sensitivity analysis, latency overheads, and error propagation across cascade stages documented in eval suite. | `evaluation/evaluator.py`, report section |
| **L35** | **Evaluation Metrics: Error, Decision-Support, User-Centered** | **Error:** RMSE, MAE.<br>**Decision-Support:** Precision@10, Recall@10, NDCG@10.<br>**User-Centered:** Intra-List Distance (ILD), Catalog Coverage (%), Novelty. | `evaluation/metrics.py`, `evaluation/evaluator.py` |
| **L36** | **Case Study and Course Review** | End-to-end deployment: live "Mini-X" FastAPI + Tailwind web client with explainable "Why this post?" UI. | `run_server.py`, `frontend/index.html` |

---

## 2. System Architecture: The "Mini-X" Recommendation Pipeline

The recommendation pipeline runs a multi-stage funnel transforming candidate posts from the entire corpus down to a curated, diverse Top-K feed in under 50ms.

```
                           [ Entire Post Corpus (~3k-5k posts) ]
                                            │
        ┌───────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                       ▼
 ┌──────────────────────────────┐                       ┌───────────────────────────────────────────────┐
 │   In-Network Candidates      │                       │           Out-of-Network Candidates           │
 │ (Earlybird / Thunder Proxy)  │                       │               (Phoenix Proxy)                 │
 │ - Follow Graph Posts (Recent)│                       │ - Implicit ALS Latent Dot Product             │
 │ - 2nd-degree Friend Engagements│                     │ - Item-kNN Latent Similarities                │
 └──────────────┬───────────────┘                       │ - TF-IDF Content Vector Search                │
                │                                       └───────────────────────┬───────────────────────┘
                │                                                               │
                └───────────────────────────────┬───────────────────────────────┘
                                                ▼
                             [ Candidate Union (~200-400 Candidates) ]
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │          Feature Hydration           │
                             │ - Author Reputation / Follower Stats │
                             │ - Post Age Decay & Interaction Counts│
                             │ - User-Author Historic Affinity      │
                             └──────────────────┬───────────────────┘
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │          Pre-Scoring Filters         │
                             │ - Deduplication & Self-Post Removal  │
                             │ - Blocked / Muted Authors            │
                             │ - Already-Seen / Interacted Filter   │
                             │ - Shilling Defense Anomaly Filter    │
                             └──────────────────┬───────────────────┘
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │             Heavy Ranker             │
                             │  Calibrated Multi-Task Engagement    │
                             │  P(Like), P(Reply), P(Skip) Head     │
                             │  Final Score = Weighted Combination  │
                             └──────────────────┬───────────────────┘
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │     Mixer & Diversity Selection      │
                             │ - Maximal Marginal Relevance (MMR)   │
                             │ - Max-Per-Author Constraints (<=2)   │
                             │ - In/Out Network Balance (50/50 Mix) │
                             └──────────────────┬───────────────────┘
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │         Explainability Layer         │
                             │  Generates "Why this post?" metadata │
                             └──────────────────┬───────────────────┘
                                                │
                                                ▼
                             [ Final "For You" Feed (Top 10-20 Posts) ]
```

### 2.1 Stage-by-Stage Functional Contracts

1. **Candidate Retrieval (Two Towers):**
   - **In-Network:** Gathers posts authored by users that the target user directly follows (`social_graph.follows`), sorted by recency and social velocity.
   - **Out-of-Network:** Gathers posts from authors outside the user's follow network using three concurrent retrieval engines:
     - `iALS Engine`: Latent dot product $\hat{r}_{ui} = p_u^T q_i$.
     - `Item-kNN Engine`: Nearest neighbors of posts the user has recently liked.
     - `Content-TFIDF Engine`: Cosine similarity between post text/tags and the user's aggregated topic vector.
2. **Hydration:**
   - Enrich candidate identifiers with real-time metadata: author metadata, follower count, total likes/replies, post creation timestamp, text length, and tag list.
3. **Filtering:**
   - Drop posts already seen or dismissed by the user.
   - Drop posts authored by blocked/muted users or the user themselves.
   - Drop posts identified as synthetic spam or flagged by the Shilling Defense module.
4. **Heavy Ranker (Scoring):**
   - If user interaction count $< 3$ (Cold-Start User), execute the **Hybrid Switch**: rank using Content TF-IDF + Department Tag Match + Recency Decay.
   - If user is Warm, pass hydrated feature vectors into a multi-action calibrated scoring model predicting probabilities $P(\text{like})$, $P(\text{reply})$, and $P(\text{skip})$.
   - Compute the aggregate score:
     $$\text{Score}(u, i) = \left[ w_{\text{like}} \cdot P(\text{like}) + w_{\text{reply}} \cdot P(\text{reply}) - w_{\text{skip}} \cdot P(\text{skip}) \right] \times \exp(-\lambda_{\text{age}} \cdot \Delta t)$$
5. **Mixer & Diversity Selection:**
   - Run greedy Maximal Marginal Relevance (MMR) with trade-off parameter $\lambda \in [0, 1]$.
   - Enforce author diversity constraint: no author may appear more than twice in the Top-10.
   - Interleave in-network and out-of-network candidates to achieve a target ratio (default 50% / 50%).
6. **Explainability Engine:**
   - Assign an explanatory badge and tooltip to each post in the final feed:
     - *"Followed Author"* (In-network match)
     - *"Because you engaged with #[Tag]"* (Content match)
     - *"Students with similar interests liked this"* (Collaborative latent match)
     - *"High engagement in [Department]"* (Popularity/Knowledge match)

---

## 3. Mathematical Formulations & Theoretical Rigor

### 3.1 Similarity Measures
For user-user and item-item collaborative baselines, calculate:

**Centered Cosine (Pearson Correlation Coefficient):**
$$\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$

With significance weighting threshold $\tau = 5$:
$$\text{sim}_{\text{adj}}(u, v) = \text{sim}(u, v) \times \frac{\min(|I_{uv}|, \tau)}{\tau}$$

### 3.2 Implicit Alternating Least Squares (iALS)
Given implicit binary preferences $p_{ui} \in \{0, 1\}$ derived from implicit actions (likes, replies, dwell $\ge 10\text{s}$):
$$p_{ui} = \begin{cases} 1 & \text{if } r_{ui} > 0 \\ 0 & \text{if } r_{ui} = 0 \end{cases}$$

Define the confidence matrix $C_{ui}$:
$$C_{ui} = 1 + \alpha r_{ui}$$
where $\alpha = 40$ and $r_{ui}$ is the log-normalized interaction frequency.

The loss function optimized alternatingly over user latent vectors $x_u \in \mathbb{R}^d$ and item latent vectors $y_i \in \mathbb{R}^d$:
$$\mathcal{L}_{ALS} = \sum_{u, i} C_{ui} (p_{ui} - x_u^T y_i)^2 + \lambda \left( \sum_u \|x_u\|_2^2 + \sum_i \|y_i\|_2^2 \right)$$

### 3.3 Content-Based Profiling
For post text and tag metadata, compute TF-IDF representations:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

Construct the dynamic User Interest Profile $\vec{U}_u$ as the dwell-weighted centroid of liked/replied posts:
$$\vec{U}_u = \frac{\sum_{i \in \text{Liked}(u)} w_i \cdot \vec{V}_i}{\|\sum_{i \in \text{Liked}(u)} w_i \cdot \vec{V}_i\|_2}$$

### 3.4 Maximal Marginal Relevance (MMR)
Given candidate set $C$, selected set $S$, and trade-off parameter $\lambda \in [0, 1]$:
$$\text{MMR}(u) = \arg\max_{i \in C \setminus S} \left[ \lambda \cdot \text{Score}_{\text{norm}}(u, i) - (1 - \lambda) \max_{j \in S} \text{Sim}_{\text{content}}(i, j) \right]$$

### 3.5 Shilling Attack & Defense Formulations
1. **Random Attack:** Bot accounts insert likes on target post $i^*$ plus a set of randomly sampled posts across the catalog.
2. **Bandwagon Attack:** Bot accounts insert likes on target post $i^*$ plus the top-5% most popular campus posts to disguise their profiles as typical active users.
3. **Detection Heuristics:**
   - **Interaction Velocity & Deviation:**
     $$\text{Dev}(u) = \frac{1}{|I_u|} \sum_{i \in I_u} |r_{ui} - \bar{r}_i|$$
   - **Degree Anomaly & Entropy:** Flag users whose engagement timestamps cluster within an abnormal delta ($\Delta t < 2\text{s}$) or whose co-action Jaccard similarity across identical subsets exceeds threshold $\theta_{\text{bot}} = 0.75$.

---

## 4. Complete Project Directory Structure

```
campus_recsys/
│
├── config.py                          # Global hyperparameters, weights, paths, seeds
├── requirements.txt                   # Production dependencies
├── generate_dataset.py                # Standalone synthetic campus social dataset generator
├── train.py                           # Retrains all models, saves artifacts to /artifacts
├── evaluate.py                        # Standalone evaluation & cold-start ablation runner
├── run_server.py                      # FastAPI Uvicorn entry point
│
├── data/
│   ├── __init__.py
│   ├── schemas.py                     # Pydantic v2 data models for User, Post, Interaction
│   ├── generator.py                   # Barabási-Albert social graph and persona engine
│   └── raw/                           # Output directory for generated CSV/JSON files
│       ├── users.json
│       ├── posts.json
│       ├── follows.csv
│       ├── interactions.csv
│       └── explicit_holdout.csv
│
├── models/
│   ├── __init__.py
│   ├── collaborative_knn.py           # User-User and Item-Item Pearson / Cosine CF
│   ├── als_factorization.py           # Implicit ALS matrix factorization engine
│   ├── content_based.py               # TF-IDF vectorizer + tag profile matching
│   └── neural_two_tower.py            # PyTorch 2-Tower User & Post embedding network
│
├── pipeline/
│   ├── __init__.py
│   ├── candidate_retrieval.py         # In-network + Out-of-network multi-engine retrieval
│   ├── hydrator.py                    # Metadata enrichment and feature vector assembly
│   ├── filters.py                     # Deduplication, safety, mute/block, already-seen filters
│   ├── heavy_ranker.py                # Calibrated Logistic/LightGBM multi-action ranker
│   ├── explainer.py                   # Rule & feature attribution for "Why this post?"
│   └── orchestrator.py                # End-to-end pipeline execution & cold-start switcher
│
├── mixer/
│   ├── __init__.py
│   └── diversity_selector.py          # MMR re-ranking, author cap, in/out ratio mixer
│
├── security/
│   ├── __init__.py
│   ├── shilling_simulator.py          # Random & Bandwagon bot attack injector
│   └── shilling_detector.py           # Anomaly deviation & co-action bot cluster detector
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     # NDCG@K, Precision@K, Recall@K, RMSE, ILD, Novelty
│   └── evaluator.py                   # Split manager (Warm vs Cold User/Item) & ablation tests
│
├── api/
│   ├── __init__.py
│   └── routes.py                      # FastAPI REST endpoints for feed, actions, attack, eval
│
└── frontend/                          # Decoupled web client
    ├── index.html                     # Semantic HTML5 layout with tabbed dashboard
    ├── app.js                         # Dynamic state management, feed rendering, sliders
    └── styles.css                     # Modern dark-mode styling (Tailwind CSS CDN)
```

---

## 5. Detailed Component Specifications

### 5.1 Synthetic Dataset Generator (`data/generator.py`)

The generator must operate with zero external API calls or downloads, using fixed seeds for reproducible generation:

- **Entity Volumes:**
  - **Users:** 400 total users.
  - **Posts:** 3,500 total posts.
  - **Sparsity:** Target interaction sparsity of 98.2% – 98.8%.
- **User Personas (Archetypes):**
  - `CS_Undergrad`: High interest in `#cs`, `#ai`, `#hackathons`, `#internships`.
  - `Bio_Researcher`: High interest in `#biotech`, `#research`, `#lablife`, `#academia`.
  - `Campus_Club`: High out-degree publishing, events, `#campuslife`, `#social`, `#sports`.
  - `Course_TA`: Focused on `#homework`, `#exam`, `#officehours`, `#math`.
  - `Freshman`: Cold-start candidate; few follows, general broad exploration.
- **Social Graph Topologies:**
  - Generate a scale-free directed follow graph using the **Barabási-Albert preferential attachment model** ($m = 6$ outgoing edges per node), ensuring power-law in-degree distribution (a few popular professors/clubs, many standard students).
- **Engagement Types & Implicit Dwell Log:**
  - `VIEW`: Base impression.
  - `DWELL_TIME`: Seconds spent viewing (sampled from log-normal distribution; $\mu = 1.8, \sigma = 0.8$).
  - `LIKE`: Binary flag triggered if dwell exceeds threshold and topic alignment is positive.
  - `REPLY`: Binary flag with higher friction (probability ~ 8-15% of likes).
  - `SKIP`: Negative signal (dwell $< 3$ seconds).
  - `BOOKMARK`: High-intent positive signal.
- **Explicit Holdout Slice:**
  - Generate a small synthetic 1–5 "Relevance Feedback" rating for ~10% of user-post interactions, held out strictly for computing collaborative filtering RMSE and MAE.

### 5.2 Pydantic Schemas (`data/schemas.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

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
    source: str  # "in_network", "als_latent", "item_knn", "content_tfidf"
    retrieval_score: float

class HydratedCandidate(BaseModel):
    candidate: CandidatePost
    author_follower_count: int
    user_author_interaction_history: int
    post_age_hours: float
    author_reputation_score: float

class ScoredCandidate(BaseModel):
    hydrated: HydratedCandidate
    p_like: float
    p_reply: float
    p_skip: float
    final_score: float
    explanation: Dict[str, str]
```

### 5.3 Candidate Retrieval Funnels (`pipeline/candidate_retrieval.py`)

The candidate retrieval stage pools together candidates from two towers:

1. **In-Network Tower:**
   - Query all posts authored by users in `user.followed_user_ids` published within the last 7 days.
   - Rank in-network candidates by:
     $$\text{Score}_{\text{in-net}}(i) = \text{likes}_i + 2 \cdot \text{replies}_i + \frac{10}{1 + \text{age\_hours}_i}$$
   - Return top 150 candidates.
2. **Out-of-Network Tower:**
   - **Engine A: Implicit ALS Latent Retrieval:**
     - Query user latent factor vector $x_u$.
     - Compute dot products $x_u^T y_i$ across all non-followed posts.
     - Return top 100 candidates.
   - **Engine B: Item-kNN Retrieval:**
     - Take the 3 most recently liked posts by user $u$.
     - Find top-5 nearest neighbors for each using precomputed Item Cosine Similarity matrix.
     - Return top 50 unique candidates.
   - **Engine C: Content TF-IDF Retrieval:**
     - Transform post contents with Scikit-learn `TfidfVectorizer`.
     - Calculate cosine similarity against the user's centroid vector.
     - Return top 50 candidates.
3. **Candidate Merger:**
   - Deduplicate candidates, tracking all origin sources in `CandidatePost.source`.

### 5.4 Pre-Scoring Filters (`pipeline/filters.py`)

Enforce strict safety and relevance rules in sequential order:
1. `SelfPostFilter`: Discard if `candidate.post.author_id == user.user_id`.
2. `BlockMuteFilter`: Discard if `candidate.post.author_id in user.blocked_user_ids`.
3. `AlreadySeenFilter`: Discard if the user has a recorded impression/interaction for this post in `history`.
4. `AuthorSpamFilter`: If an author has $> 5$ candidates in the pool, keep only the top 5 highest-scoring.
5. `ShillingDefenseFilter` *(Active when Defense toggle is ON)*:
   - Discard candidate if the post's recent engagement cluster has an anomaly score $> 0.85$ computed by `shilling_detector.py`.

### 5.5 Heavy Ranker & Cold-Start Routing (`pipeline/heavy_ranker.py`)

The Heavy Ranker acts as the scoring brain:

#### Cold-Start Switching Logic
```python
if user_interaction_count < 3:
    # Cold-Start User: Route to Knowledge + Content Scoring
    score = (
        0.50 * content_similarity(user.preferred_tags, post.tags) +
        0.30 * (1.0 if post.department == user.department else 0.0) +
        0.20 * math.exp(-0.05 * post_age_hours)
    )
    p_like, p_reply, p_skip = score, score * 0.2, 1.0 - score
```

#### Warm User Calibrated Multi-Action Ranker
For warm users, construct a 12-dimensional feature vector per hydrated post:
1. `als_latent_dot_product`
2. `content_tfidf_cosine`
3. `is_in_network` (0 or 1)
4. `author_reputation_score` (log-transformed follower count)
5. `user_author_historical_likes`
6. `post_total_likes` (log1p)
7. `post_total_replies` (log1p)
8. `post_age_hours`
9. `tag_overlap_jaccard`
10. `user_dwell_mean`
11. `item_popularity_percentile`
12. `department_match` (0 or 1)

Train 3 calibrated Logistic Regression models (or a Multi-Output LightGBM Classifier):
- Model 1 predicts $P(\text{like} = 1 \mid x)$
- Model 2 predicts $P(\text{reply} = 1 \mid x)$
- Model 3 predicts $P(\text{skip} = 1 \mid x)$

Calculate composite ranking score:
$$\text{RawScore} = w_{\text{like}} \cdot P(\text{like}) + w_{\text{reply}} \cdot P(\text{reply}) - w_{\text{skip}} \cdot P(\text{skip})$$
$$\text{FinalScore} = \text{RawScore} \times \exp(-\lambda_{\text{age}} \cdot \text{post\_age\_hours})$$
*(Default hyperparameter weights: $w_{\text{like}} = 0.6, w_{\text{reply}} = 1.2, w_{\text{skip}} = 0.4, \lambda_{\text{age}} = 0.02$)*

### 5.6 Deep Learning Baseline: Two-Tower Neural CF (`models/neural_two_tower.py`)
To fulfill the neural stretch goal and provide a deep learning ablation baseline:
- **User Tower:** Embeds `user_id` ($d=32$), `persona_id` ($d=8$), and dense historical tag preference vector ($d=16$). Passes through `Linear(56, 64) -> ReLU() -> Linear(64, 32)`.
- **Post Tower:** Embeds `post_id` ($d=32$), `author_id` ($d=16$), and dense TF-IDF reduced vector ($d=16$). Passes through `Linear(64, 64) -> ReLU() -> Linear(64, 32)`.
- **Interaction Head:** Computes cosine similarity of output embeddings, optimized with Binary Cross-Entropy loss on implicit positive/negative interactions.

### 5.7 Diversity & Selection Mixer (`mixer/diversity_selector.py`)

Applies post-processing to avoid recommendation echo chambers:
1. **Ratio Balancer:** Target 50% in-network and 50% out-of-network candidates.
2. **Max-per-Author Constraint:** Maintain an author frequency counter; skip any candidate whose author already occupies 2 slots in the current feed.
3. **Maximal Marginal Relevance (MMR):**
   - Initialize selected feed $S = []$.
   - Iteratively select post $i^*$ from remaining candidate pool $C$:
     $$i^* = \arg\max_{i \in C} \left[ \lambda \cdot \text{Score}_{\text{Heavy}}(i) - (1 - \lambda) \max_{j \in S} \text{Sim}_{\text{TF-IDF}}(i, j) \right]$$
   - Stop when target feed length $K = 15$ is satisfied.

### 5.8 Explainability Engine (`pipeline/explainer.py`)

Every post delivered to the client includes a structured explanation dictionary:
- `primary_reason`: Short string badge, e.g.:
  - `"From an account you follow"` (In-Network)
  - `"Matches your interest in #ai"` (High content alignment)
  - `"Popular with CS students"` (Department cluster match)
  - `"Discovered via your reading history"` (Latent ALS similarity)
- `confidence_score`: Normalized percent confidence.
- `feature_weights`: Dictionary breakdown of how like, reply, in-network, and topic signals contributed to the final score.

### 5.9 Security: Shilling Attack & Defense (`security/`)

#### 1. Attack Simulator (`security/shilling_simulator.py`)
- Injects $M = 25$ synthetic bot accounts into the database targeting a specific low-ranked post $i^*_{\text{target}}$ (e.g., ranked position #45):
  - **Random Attack:** Each bot issues a `LIKE` on $i^*_{\text{target}}$ and likes 15 uniformly random posts.
  - **Bandwagon Attack:** Each bot issues a `LIKE` on $i^*_{\text{target}}$ and likes the top 15 most popular campus posts to blend into mainstream clusters.
- Updates the interaction matrix and triggers a recalculation of the feed. Graders can verify that under the attack, the target post inflates from position #45 into the Top 3.

#### 2. Defense Detector (`security/shilling_detector.py`)
- Computes two anomaly signals across all users:
  1. **Interaction Velocity & Target Deviation:** Flag accounts with excessive like-to-view ratios and bursty timestamps.
  2. **Co-Action Correlation Matrix:** Compute pairwise Jaccard similarity across user interaction profiles:
     $$J(u, v) = \frac{|I_u \cap I_v|}{|I_u \cup I_v|}$$
     Dense subgraphs with high average Jaccard similarity ($J > 0.70$) indicate coordinated botnets.
- Identifies bot accounts and applies an attenuation penalty or total exclusion in the pre-scoring filters. When toggled on, the target post drops back down to its organic rank.

---

## 6. Evaluation Protocol & Offline Benchmarking Suite

The evaluation runner (`evaluate.py`) implements a rigorous, reproducible benchmark comparing all algorithms.

### 6.1 Evaluation Splits
1. **Temporal/Interaction Split:** For each warm user, hold out their last 20% of interactions for testing; use the initial 80% for training.
2. **Cold-Start Cohort:**
   - **Cold Users:** Users with $\le 2$ interactions in training data.
   - **Cold Items (New Posts):** Posts published within the final simulation interval with $\le 1$ interaction.
   - **Warm Users:** Users with $> 10$ interactions.

### 6.2 Benchmark Algorithms to Compare
1. `Random Baseline`: Uniform random selection from catalog.
2. `Popularity Baseline`: Top most-liked posts campus-wide.
3. `User-kNN CF`: Pearson-correlation neighborhood collaborative filtering.
4. `Item-kNN CF`: Item-item cosine similarity collaborative filtering.
5. `Matrix Factorization (iALS)`: Latent factor implicit ALS model.
6. `Content-Based`: TF-IDF vector space model.
7. `Two-Tower Neural CF`: PyTorch deep representation baseline.
8. **Full Mini-X Hybrid Pipeline (Our System)**: Sourcing + Hydration + Filter + Heavy Ranker + Diversity.

### 6.3 Evaluation Metrics
- **Ranking Accuracy:**
  - **Precision@10:**
    $$\text{P@10} = \frac{|\text{Recommended@10} \cap \text{Relevant}|}{10}$$
  - **Recall@10:**
    $$\text{R@10} = \frac{|\text{Recommended@10} \cap \text{Relevant}|}{|\text{Relevant}|}$$
  - **NDCG@10:** Normalized Discounted Cumulative Gain at rank 10:
    $$\text{DCG@10} = \sum_{k=1}^{10} \frac{2^{rel_k} - 1}{\log_2(k + 1)}, \quad \text{NDCG@10} = \frac{\text{DCG@10}}{\text{IDCG@10}}$$
- **Error Metric (Held-out Explicit Slice):**
  - **RMSE:** Root Mean Square Error on predicted vs actual 1–5 ratings.
- **Beyond-Accuracy Metrics:**
  - **Catalog Coverage (%):** Percentage of total unique posts recommended across all users in top-10 feeds:
    $$\text{Coverage} = \frac{|\bigcup_{u \in U} \text{Recs}_{10}(u)|}{|I|} \times 100\%$$
  - **Intra-List Distance (ILD / Diversity):** Average cosine distance between recommended items in a user's feed:
    $$\text{ILD}(L) = \frac{2}{|L|(|L|-1)} \sum_{i \in L} \sum_{j \in L, j \ne i} (1 - \text{Sim}_{\text{TF-IDF}}(i, j))$$

### 6.4 Ablation Experiment Matrix
The evaluation suite must automatically run and output an ablation table:
- *Ablation 1:* In-Network Retrieval Only (Follow graph only).
- *Ablation 2:* Out-of-Network Retrieval Only (No follow graph).
- *Ablation 3:* Full Pipeline without MMR Diversity ($\lambda = 1.0$).
- *Ablation 4:* Full Pipeline with MMR Diversity ($\lambda = 0.7$).
- *Ablation 5:* Under Shilling Attack (Unprotected).
- *Ablation 6:* Under Shilling Attack (With Defense Filter Active).

---

## 7. FastAPI Backend & Interactive Web UI Specification

### 7.1 REST API Specification (`api/routes.py`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/users` | Returns list of all simulated users with persona and department tags. |
| `GET` | `/api/user/{user_id}` | Returns user profile, follow list, and recent interaction history. |
| `GET` | `/api/feed?user_id=12&mmr_lambda=0.7&w_like=0.6&w_reply=1.2&defense_active=true` | Returns personalized Top-15 For You feed with full explanation badges and pipeline trace. |
| `POST` | `/api/interact` | Logs user engagement (Like, Reply, Skip) in real-time and updates state. |
| `POST` | `/api/attack/inject` | Triggers synthetic like-bomb attack on a target post (`attack_type: "bandwagon" \| "random"`). |
| `POST` | `/api/attack/reset` | Clears all synthetic bot accounts and restores organic state. |
| `GET` | `/api/benchmark/summary` | Returns precomputed evaluation metrics table and ablation comparisons as JSON. |

### 7.2 Frontend Web Interface (`frontend/`)

A single-page, responsive dashboard built with semantic HTML5, modern Tailwind CSS (via CDN), and vanilla ES6 JavaScript:

1. **Header & Persona Switcher:**
   - Dropdown to immediately impersonate different users:
     - `Alice (CS Undergrad - Warm)`
     - `Dr. Vance (Bio Faculty - Warm)`
     - `Charlie (Freshman - Pure Cold-Start)`
     - `Robotics Club (High Publisher)`
   - Displays User Avatar, Department, Follower Count, and Following Count.
2. **Main "For You" Feed Column:**
   - Post cards styled like modern social media feeds (X/Twitter aesthetic).
   - Author name, handle, department tag, and timestamp.
   - Post body text and clickable hashtag badges.
   - Interactive action buttons: `Like (Heart)`, `Reply (Chat)`, `Skip (Cross)`.
   - **Explainability Banner:** Every post card includes an expandable "Why this post?" pill:
     - Shows the source badge (e.g. `In-Network`, `ALS Latent`, `Content Match`).
     - Shows predicted probabilities: `P(Like): 78%`, `P(Reply): 24%`, `P(Skip): 5%`.
     - Visual horizontal progress bar breaking down the weighted scoring contribution.
3. **Interactive Control & Pipeline Inspector Sidebar:**
   - **Weight Sliders:** Live sliders to tweak $w_{\text{like}}$, $w_{\text{reply}}$, $w_{\text{skip}}$, and MMR $\lambda$ (Diversity vs Relevance) in real-time; updates the feed without restarting the server.
   - **Pipeline Stage Counter:** Shows real-time candidate funnel stats:
     `Corpus: 3,500` $\rightarrow$ `Retrieved: 350` $\rightarrow$ `Hydrated: 350` $\rightarrow$ `Filtered: 298` $\rightarrow$ `Ranked: 50` $\rightarrow$ `Top Feed: 15`.
   - **Security Sandbox Panel:**
     - Target Post Selector (e.g. Post #42: "Obscure Campus Notice").
     - Button: `🚀 Launch Like-Bomb Attack (Bandwagon)`
     - Live indicator showing Target Post position (e.g. Jumping from #42 to #2).
     - Toggle: `🛡️ Enable Shilling Defense Filter`. Target post immediately drops back to organic rank.
4. **Evaluation Benchmarks Tab:**
   - Formatted comparison tables for NDCG@10, Precision@10, RMSE, and Diversity across all baselines and cold-start splits.

---

## 8. Execution, Installation & Verification Guide

### 8.1 Dependencies (`requirements.txt`)
```txt
fastapi>=0.110.0
uvicorn>=0.28.0
pydantic>=2.6.0
numpy>=1.26.0
scipy>=1.12.0
scikit-learn>=1.4.0
torch>=2.2.0
pandas>=2.2.0
python-multipart>=0.0.9
jinja2>=3.1.3
```

### 8.2 End-to-End Run Sequence
1. **Generate Dataset:**
   ```bash
   python generate_dataset.py --seed 42 --users 400 --posts 3500
   ```
2. **Train Models & Precompute Latent Factors:**
   ```bash
   python train.py
   ```
3. **Run Evaluation Suite & Produce Academic Tables:**
   ```bash
   python evaluate.py
   ```
4. **Launch FastAPI Interactive Demo Server:**
   ```bash
   python run_server.py --port 8000
   ```
   *Access demo in browser at `http://localhost:8000`.*

---

## 9. Academic Project Grading Checklist

When presenting or submitting this project, the implementation satisfies the highest evaluation criteria:
- [x] **Non-generic domain:** Realistic campus social graph, not MovieLens or Spotify clone.
- [x] **Industry-faithful architecture:** Exact candidate sourcing $\rightarrow$ hydration $\rightarrow$ filters $\rightarrow$ heavy ranker $\rightarrow$ diversity pipeline mirroring X's production architecture.
- [x] **True hybrid design:** Switching (cold-start $\rightarrow$ content/knowledge), Cascade (retrieval $\rightarrow$ heavy ranker), and Weighted multi-signal scoring.
- [x] **Multi-signal implicit feedback:** Dwell time proxy, likes, replies, and skip signals with confidence-weighted ALS.
- [x] **Security & robustness:** Shilling botnet injection, correlation-clustering defense, and live rank-inflation visualization.
- [x] **Beyond-accuracy metrics:** Evaluates catalog coverage and intra-list distance (ILD) alongside NDCG@10 and precision.
- [x] **Live explainable UI:** Graders can interactively inspect "Why this post?" with real-time parameter tuning.

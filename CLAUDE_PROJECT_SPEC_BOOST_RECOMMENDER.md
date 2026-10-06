# MASTER SPECIFICATION & BUILD INSTRUCTION: THE /boost RECOMMENDER SYSTEM

> **Target:** Claude 3.5 Sonnet / Claude 3.7 Sonnet  
> **Project:** `/boost` — A Hybrid Collaborative Content Knowledge Recommender ("For You" Feed Inspired by the Open X Algorithm)  
> **Course Domain:** Recommender Systems, Machine Learning & Information Retrieval  

---

## I. SYSTEM OVERVIEW & ARCHITECTURAL INSPIRATION

`/boost` is an end-to-end, scaled-down social/academic timeline recommender designed for a university campus environment. It faithfully mirrors the real-world 5-stage recommendation pipeline open-sourced by X (Twitter) in `xai-org/x-algorithm` and `twitter/the-algorithm` (**Candidate Sourcing $\to$ Hydration $\to$ Visibility Filtering $\to$ Multi-Signal Heavy Ranking $\to$ Diversity Selection**).

Unlike standard toy recommenders (e.g., MovieLens user-item matrix factorizations), `/boost` solves the complex challenges of modern social feeds:
1. **Multi-Modal Implicit Signals**: Optimizing for dwell time, replies, likes, bookmarks, and skips rather than 1–5 star ratings.
2. **Cold-Start Routing**: Seamlessly switching between Content/Knowledge filters for new users/posts and Collaborative/Matrix Factorization for warm users.
3. **Multi-Action Heavy Ranking**: Training multiple engagement prediction heads ($\hat{P}(\text{like})$, $\hat{P}(\text{reply})$, $\hat{P}(\text{dwell})$, $\hat{P}(\text{skip})$) combined into an X-style composite utility function.
4. **Feed Diversity & Fatigue**: Implementing Maximal Marginal Relevance (MMR) with author fatigue penalties.
5. **Security & Shilling Robustness**: Injecting synthetic bot attacks (Random, Average, Bandwagon) and defending with statistical deviation metrics (RDMA and DegSim).
6. **Transparent Explainability**: "Why This Post?" badges derived via linear attribution.

```
                          [Raw Post Corpus: ~2,500 Posts]
                                         │
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 1. CANDIDATE SOURCING (Multi-Tower Mixed Retrieval: ~150 candidates)      │
   │   ├─ Tower A (In-Network / Thunder-style): Recent posts from followed     │
   │   ├─ Tower B (Collaborative Out-of-Network / Phoenix): Implicit ALS / kNN │
   │   ├─ Tower C (Content Out-of-Network): TF-IDF user-profile similarity     │
   │   └─ Tower D (Cold-Start / Exploration): Fresh posts matching user tags   │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 2. QUERY HYDRATION                                                        │
   │   Enrich post IDs with author stats, follow status, post velocity         │
   │   (likes/hr, replies/hr), post age, and TF-IDF vectors                    │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 3. VISIBILITY & HARD FILTERS                                              │
   │   - Drop seen posts (already viewed/interacted)                           │
   │   - Drop muted/blocked authors                                            │
   │   - Apply knowledge-based prerequisite & course-code constraints          │
   │   - Author fatigue pre-filter (max 3 candidate posts per author)          │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 4. MULTI-SIGNAL HEAVY RANKER (Feature-Augmented Scoring)                  │
   │   Predicts action probabilities: P(like), P(reply), P(dwell>15s), P(skip) │
   │   Utility: Score = 1.0·P(like) + 13.5·P(reply) + 3.0·P(dwell) - 2.0·P(skip)│
   │            + Recency Decay (β · exp(-γ · age_hours))                     │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 5. DIVERSITY SELECTION (MMR + Author Fatigue Caps)                        │
   │   Maximal Marginal Relevance balances utility against topic redundancy    │
   │   Hard constraint: Max 2 posts per author in Top-10                       │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ 6. EXPLAINABILITY ENGINE ("Why This Post?")                               │
   │   Linear attribution decomposition tags dominant positive signal:        │
   │   [In-Network], [Similar Tastes (CF)], [Topic Affinity], [Hot Discussion] │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         ▼
                   [Final "For You" Feed: Top 10-20 Posts]
```

---

## II. SYSTEM DATA MODELS & CONTRACTS (`pipeline/contracts.py`)

All pipeline components must strictly communicate via typed Python dataclasses:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class UserContext:
    user_id: str
    username: str
    department: str
    year: str
    interest_tags: List[str]
    followed_user_ids: List[str]
    blocked_user_ids: List[str]
    interaction_count: int
    is_cold_start: bool

@dataclass
class PostEntity:
    post_id: str
    author_id: str
    content: str
    tags: List[str]
    course_code: Optional[str]
    created_at: datetime
    age_hours: float
    ground_truth_relevance: float  # 1.0 to 5.0 for offline evaluation benchmark only

@dataclass
class CandidatePost:
    post: PostEntity
    source_tower: str  # 'in_network', 'cf_als', 'content_tfidf', 'cold_explore'
    cf_score: float = 0.0
    content_score: float = 0.0
    velocity_score: float = 0.0

@dataclass
class ScoredPost:
    candidate: CandidatePost
    p_like: float
    p_reply: float
    p_dwell: float
    p_skip: float
    utility_score: float
    recency_boost: float
    final_score: float
    explanation_badge: str
    attribution_weights: Dict[str, float] = field(default_factory=dict)
```

---

## III. SYNTHETIC DATASET GENERATION (`data/generate_dataset.py`)

Generate a realistic, controllable campus social feed using `numpy`, `pandas`, and `networkx`:

### 1. Scale & Distributions
- **Users**: 300 accounts.
  - Follow graph modeled via a **Barabási-Albert scale-free network** ($m=4$) to simulate real-world hub concentration (professors/clubs have 80–120 followers, regular students have 5–15).
- **Posts**: 2,500 posts generated across a 14-day timeline.
  - Topics: Academic courses (`#cs101`, `#math201`, `#stat300`), research (`#deep_learning`, `#nlp`, `#robotics`), campus life (`#hackathon`, `#career_fair`, `#study_group`).
- **Interactions**: ~35,000 implicit interaction events:
  - `view` (impression): 60%
  - `dwell` (dwell time > 15s): 20% (dwell duration sampled from Log-Normal($\mu=2.3, \sigma=0.8$))
  - `like` (weak explicit): 12%
  - `reply` (high intent): 5%
  - `bookmark`: 2%
  - `skip` (negative fast scroll): 1%
- **Ground Truth Relevance ($r_{ui}^* \in [1.0, 5.0]$)**:
  - Computed from: $\text{Jaccard}(Tags_u, Tags_i) \times 2.0 + \mathbb{I}(\text{Follows}) \times 1.5 + \mathcal{N}(1.5, 0.4)$.
  - Clipped to $[1.0, 5.0]$, reserved exclusively for offline RMSE/MAE regression benchmarking.

### 2. Strict Partitioning Protocol (Preventing Data Leakage)
- **Temporal Split**:
  - Events occurring at $t \le 11$ days $\to$ **Train Set** (~80% interactions).
  - Events occurring at $t > 11$ days $\to$ **Test Set** (~20% interactions).
- **Cold-Start Holdouts**:
  - **Cold Users**: 25 isolated accounts with 0 interactions in the training window (only interest tags & demographics known).
  - **Cold Posts**: 100 recent posts published within the last 24 hours with 0 historical interactions in train.

---

## IV. ALGORITHM IMPLEMENTATIONS (`models/`)

### 1. Memory-Based CF (`models/cf_knn.py`)
- **User-Based and Item-Based kNN**:
  - Implement centered cosine / Pearson correlation:
    $$r(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2} + \epsilon}$$
  - Neighbor rating aggregation:
    $$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u, v)(r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u, v)| + \epsilon}$$

### 2. Model-Based CF: Implicit ALS & Truncated SVD (`models/mf_als.py`)
- **Hu, Koren, Volinsky (2008) Implicit ALS**:
  - Continuous implicit feedback synthesis:
    $$r_{ui} = \ln(1 + \text{dwell}_{ui}) + 2.0 \cdot \mathbb{I}(\text{like}) + 4.0 \cdot \mathbb{I}(\text{reply}) + 3.0 \cdot \mathbb{I}(\text{bookmark}) - 2.0 \cdot \mathbb{I}(\text{skip})$$
    (Clipped at $r_{ui} \ge 0$).
  - Binary preference $p_{ui} = \mathbb{I}(r_{ui} > 0)$ and confidence $c_{ui} = 1 + \alpha r_{ui}$ ($\alpha = 40$).
  - Loss function optimized via alternating updates:
    $$\mathcal{L}_{\text{ALS}} = \sum_{u, i} c_{ui} (p_{ui} - x_u^T y_i)^2 + \lambda \left( \sum_u \|x_u\|_2^2 + \sum_i \|y_i\|_2^2 \right)$$
- **Baseline Truncated SVD**: Sparse matrix decomposition via `scipy.sparse.linalg.svds` ($k=20$).

### 3. Neural Collaborative Filtering Two-Tower (`models/neural_cf.py`)
- Compact PyTorch implementation:
  - **User Tower**: Embedding layer $E_u \in \mathbb{R}^{32}$.
  - **Candidate Item Tower**: Embedding layer $E_i \in \mathbb{R}^{32}$.
  - **Branches**:
    - GMF branch: $h_{\text{GMF}} = E_u \odot E_i$.
    - MLP branch: $h_{\text{MLP}} = \text{ReLU}(W_2 \text{ReLU}(W_1 [E_u, E_i]))$.
    - Fusion head: $\hat{y}_{ui} = \sigma(h^T [h_{\text{GMF}}, h_{\text{MLP}}])$.
  - Optimization: Binary Cross-Entropy (BCE) with 4 negative samples per positive interaction.

### 4. Content-Based Filtering (`models/content_tfidf.py`)
- TF-IDF Vectorizer fitted on post text, course codes, and hashtag tokens.
- **User Profile Vector**: Centroid of historic interactions weighted by engagement intensity:
  $$\vec{u}_u = \frac{\sum_{i \in \mathcal{H}_u} w_{ui} \vec{v}_i}{\sum_{i \in \mathcal{H}_u} w_{ui} + \epsilon}$$
- Score is the cosine similarity $S_{\text{CB}}(u, i) = \frac{\vec{u}_u \cdot \vec{v}_i}{\|\vec{u}_u\|_2 \|\vec{v}_i\|_2}$.

### 5. Knowledge-Based Constraint Engine (`models/knowledge_rules.py`)
- Hard Boolean filters: Course registration prerequisites, department filters, and muted/blocked account exclusions.
- Temporal freshness decay: $f(\Delta t) = e^{-\gamma \cdot \text{age\_hours}}$ ($\gamma = 0.05$).

### 6. Robin Burke's Hybrid Taxonomy (`models/hybrid.py`)
- **Mixed Retrieval**: Candidate sourcing unions candidate streams from In-Network, ALS, Content, and Fresh Explore.
- **Switching Strategy**:
  - If $|Interactions_u| < \tau_{\text{cold}}$ (e.g. $\tau = 5$): Route strictly to **Content + Knowledge + In-Network**.
  - Else: Route to **Collaborative / Neural CF + Content Heavy Ranker**.
- **Cascade & Feature Augmentation**:
  - Top 150 candidates from retrieval are hydrated with model scores and fed as tabular features into the Heavy Ranker.
- **Per-Query Min-Max Normalization**:
  $$\tilde{S}_k(i) = \frac{S_k(i) - \min_{j} S_k(j)}{\max_j S_k(j) - \min_j S_k(j) + \epsilon}$$

---

## V. MULTI-SIGNAL HEAVY RANKER (`pipeline/ranker.py`)

Mirrors X's Heavy Ranker with an interpretable, trainable multi-target tabular model:

1. **Feature Engineering**: For each user-candidate pair $(u, i)$:
   - `cf_score`: Min-max normalized ALS / Neural CF prediction.
   - `content_score`: Cosine similarity to user profile.
   - `is_following`: Binary indicator $\mathbb{I}(i.\text{author} \in u.\text{follows})$.
   - `author_followers`: $\log(1 + \text{author followers})$.
   - `post_reply_velocity`: $\frac{\text{reply count}}{\text{age hours} + 1.0}$.
   - `post_like_velocity`: $\frac{\text{like count}}{\text{age hours} + 1.0}$.
   - `user_activity_level`: $\log(1 + u.\text{interaction count})$.
   - `tag_jaccard`: Jaccard similarity between user interests and post tags.

2. **Action Heads**:
   - 4 independent Logistic Regression models (or LightGBM classifiers) trained on temporal train logs:
     - $\hat{P}(\text{like} \mid u, i)$
     - $\hat{P}(\text{reply} \mid u, i)$
     - $\hat{P}(\text{dwell} \mid u, i)$
     - $\hat{P}(\text{skip} \mid u, i)$

3. **Composite Utility Formula** (Inspired by X's production weights):
   $$\text{Utility}(u, i) = w_{\text{like}} \hat{P}(\text{like}) + w_{\text{reply}} \hat{P}(\text{reply}) + w_{\text{dwell}} \hat{P}(\text{dwell}) - w_{\text{skip}} \hat{P}(\text{skip}) + \beta \cdot e^{-\gamma \cdot \text{age\_hours}}$$
   - Defaults: $w_{\text{like}} = 1.0$, $w_{\text{reply}} = 13.5$, $w_{\text{dwell}} = 3.0$, $w_{\text{skip}} = 2.0$, $\beta = 1.5$, $\gamma = 0.05$.

---

## VI. DIVERSITY SELECTION & EXPLAINABILITY (`pipeline/`)

### 1. Maximal Marginal Relevance (MMR) with Author Fatigue (`pipeline/diversity_mmr.py`)
Greedy selection of Top-K items from candidate pool $C$:
$$\text{next\_item} = \arg\max_{d_i \in C \setminus S} \left[ \lambda \cdot \text{Score}(u, d_i) - (1 - \lambda) \max_{d_j \in S} \text{Sim}_{\text{content}}(d_i, d_j) - \mu \cdot \text{AuthorCount}(d_i.\text{author}, S) \right]$$
- Hard constraint: If $\text{AuthorCount}(d_i.\text{author}, S) \ge 2$, skip item for Top-10.
- $\lambda \in [0.0, 1.0]$ exposed in the UI (default $\lambda = 0.65$).

### 2. Explainability Engine via Linear Attribution (`pipeline/explainability.py`)
Decompose the final score into feature contributions:
$$\text{Contr}_k = \frac{w_k \phi_k(u, i)}{\sum_m |w_m \phi_m(u, i)| + \epsilon}$$
Assign the dominant positive badge:
- `In-Network`: "You follow @{author}" (if following signal $> 35\%$).
- `Topic-Match`: "Matches your interest in #{top_tag} ({score:.0%})" (if content signal dominates).
- `Collaborative`: "Students with similar tastes also engaged" (if CF signal dominates).
- `Discussion-Velocity`: "Trending campus discussion ({replies} replies)" (if reply velocity dominates).

---

## VII. SECURITY & SHILLING ATTACK TESTBED (`security/`)

### 1. Attack Generators (`security/shilling_generator.py`)
- **Random Shilling Attack**: Injects $M$ fake accounts that give maximum likes to target post $p^*$ and random likes to $F$ filler posts.
- **Average Shilling Attack**: Injects $M$ bots that give maximum likes to $p^*$ and assign likes to filler posts matching each item's average popularity.
- **Bandwagon Attack**: Injects $M$ bots giving maximum likes to $p^*$ AND maximum likes to the Top 5 most viral posts on campus (blending into the mainstream).

### 2. Defense Heuristics (`security/shilling_detector.py`)
- **Rating Deviation from Mean Agreement (RDMA)**:
  $$\text{RDMA}_u = \frac{1}{|I_u|} \sum_{i \in I_u} \frac{|r_{ui} - \bar{r}_i|}{N_i + 1}$$
  where $N_i$ is the total interaction count of item $i$.
- **Degree of Similarity with Top Neighbors (DegSim)**:
  $$\text{DegSim}_u = \frac{1}{k} \sum_{v \in \text{TopK}(u)} \cos(u, v)$$
  Detects bot networks exhibiting artificially high pairwise correlation.
- **Defense Protocol**: Compute RDMA and DegSim per account, prune the top outlier percentile prior to CF model fitting.
- **Security Metrics**:
  - Target Prediction Shift: $\Delta \hat{r} = \bar{r}_{\text{attacked}} - \bar{r}_{\text{clean}}$.
  - Target Rank Inflation: $\text{Rank}_{\text{clean}} \to \text{Rank}_{\text{attacked}} \to \text{Rank}_{\text{defended}}$.
  - Bot Detection: Precision, Recall, and False Alarm Rate (FAR).

---

## VIII. COMPREHENSIVE EVALUATION BENCHMARK (`eval/`)

### 1. Rating Prediction Error (Calibrated to 1–5 Relevance)
- Map utility scores to $[1.0, 5.0]$ using min-max scaling.
- Compute **RMSE** and **MAE** against held-out `ground_truth_relevance`.

### 2. Top-K Ranking Metrics ($K \in \{5, 10, 20\}$)
- **Precision@K**: $\frac{|\text{Rec}_K \cap \text{Relevant}|}{K}$.
- **Recall@K**: $\frac{|\text{Rec}_K \cap \text{Relevant}|}{|\text{Relevant}|}$.
- **MAP@K**: Mean Average Precision.
- **nDCG@K**: Normalized Discounted Cumulative Gain.

### 3. Beyond-Accuracy Metrics
- **Intra-List Diversity (ILD)**:
  $$\text{ILD} = \frac{2}{K(K-1)} \sum_{i \in R} \sum_{j \in R, j > i} (1 - \cos(\vec{v}_i, \vec{v}_j))$$
- **Catalog Coverage**: Percentage of distinct corpus items recommended across all users.
- **Novelty (Self-Information)**: $-\frac{1}{K} \sum_{i \in R} \log_2 P(i)$.

### 4. Ablation Study Grid (`eval/benchmark.py`)
Compare and tabulate:
1. `In-Network Only`
2. `Out-of-Network Only (Implicit ALS)`
3. `Content-Based Only (TF-IDF)`
4. `Full Hybrid (No MMR)`
5. `Full Hybrid + MMR Diversity`
6. `Full Hybrid under Bandwagon Attack`
7. `Full Hybrid under Attack + RDMA Defense Active`
8. `Cold-Start User Subset (Switching Hybrid Active)`

---

## IX. INTERACTIVE STREAMLIT APPLICATION (`app/streamlit_app.py`)

Build a clean, dark-themed Streamlit dashboard with 4 tabs:
- **Tab 1: "For You" Feed Explorer**: User picker (warm vs. cold), live timeline cards, explainability badges, interactive weight/MMR sliders.
- **Tab 2: Pipeline Waterfall Diagnostics**: Visual funnel displaying candidate attrition from 2,500 $\to$ 150 $\to$ 108 $\to$ Top 10, with sourcing tower breakdowns.
- **Tab 3: Shilling Attack & Defense Visualizer**: Target post selector, attack injection controls, clean vs. attacked vs. defended rank comparison, bot detection stats.
- **Tab 4: Benchmark & Evaluation Dashboard**: Benchmark comparison table across nDCG, Precision, Coverage, and Diversity with CSV export.

---

## X. PROJECT REPOSITORY STRUCTURE

```
boost_recommender/
├── data/
│   ├── generate_dataset.py       # Barabási-Albert graph & campus feed generator
│   └── dataset_loader.py         # Temporal train/test split & sparse matrix builder
├── pipeline/
│   ├── contracts.py              # Strict dataclass data models
│   ├── candidate_sources.py      # Multi-tower retrieval (In-Network + ALS + Content)
│   ├── hydration.py              # Feature hydration engine
│   ├── filters.py                # Visibility, seen, blocked, and safety filters
│   ├── ranker.py                 # Multi-task feature-augmented Heavy Ranker
│   ├── diversity_mmr.py          # MMR re-ranker with author fatigue caps
│   ├── explainability.py         # Linear attribution "Why this post" engine
│   └── home_mixer.py             # Master pipeline orchestrator
├── models/
│   ├── cf_knn.py                 # User-kNN and Item-kNN with Pearson/Cosine
│   ├── mf_als.py                 # Hu-Koren-Volinsky Implicit ALS & SVD
│   ├── neural_cf.py              # PyTorch Two-Tower / Neural CF model
│   ├── content_tfidf.py          # TF-IDF vectorizer & user profile centroid
│   ├── knowledge_rules.py        # Course & department constraint rules
│   └── hybrid.py                 # Robin Burke's Switching, Mixed, and Cascade hybrids
├── security/
│   ├── shilling_generator.py     # Random, Average, and Bandwagon attack injection
│   └── shilling_detector.py      # RDMA & DegSim bot detection and pruning
├── eval/
│   ├── metrics.py                # RMSE, P@K, R@K, nDCG@K, ILD, Coverage, Novelty
│   └── benchmark.py              # Automated ablation runner & table generator
├── app/
│   └── streamlit_app.py          # 4-tab interactive web interface
├── tests/
│   ├── test_pipeline.py          # End-to-end pipeline contract tests
│   └── test_metrics.py           # Verification tests for math & metrics
├── requirements.txt              # numpy, pandas, scipy, scikit-learn, torch, streamlit, networkx
└── README.md                     # Academic report, architecture diagrams & run instructions
```

---

## XI. CLAUDE EXECUTION GUIDELINES
- Implement all files completely with robust error handling. Do not omit functions or leave `# TODO` placeholders.
- Ensure all modules run cleanly with:
  ```bash
  python data/generate_dataset.py
  python eval/benchmark.py
  streamlit run app/streamlit_app.py
  ```

# Mini-X: Campus Social Feed Recommender Pipeline ("For You" Feed)

> **Course Handout:** AIM3143 Recommender Systems (All 36 Lectures Implemented)  
> **Architecture Inspiration:** Open-Source X (Twitter) Recommendation Pipeline (`xai-org/x-algorithm`)  
> **Domain:** University Campus Academic & Social Timeline

---

## 1. System Overview & Architecture

Mini-X implements an end-to-end, multi-stage hybrid recommendation funnel that mirrors production systems at scale:

```
                      [ Corpus: 3,500 Campus Posts ]
                                     │
      ┌──────────────────────────────┴──────────────────────────────┐
      ▼                                                             ▼
┌───────────────────────────┐                 ┌───────────────────────────────────────────┐
│ In-Network Candidates     │                 │ Out-of-Network Candidates                 │
│ (Follow Graph, Recent)    │                 │ - Implicit ALS Latent Dot Products        │
│                           │                 │ - Item-kNN Latent Similarities            │
│                           │                 │ - Content TF-IDF Centroid Search          │
└─────────────┬─────────────┘                 └─────────────────────┬─────────────────────┘
              │                                                     │
              └──────────────────────┬──────────────────────────────┘
                                     ▼
                  [ Candidate Union (~150-350 Posts) ]
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          Feature Hydration          │
                  │ - Author Reputation / Follower Stats│
                  │ - Social Velocity (Likes/Replies)   │
                  │ - 12-Dimensional Feature Vectors    │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │         Pre-Scoring Filters         │
                  │ - Self-Post & Block/Mute Filters    │
                  │ - Already-Seen / Interacted Filter  │
                  │ - Shilling Defense Anomaly Filter   │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │      Multi-Signal Heavy Ranker      │
                  │ - Cold-Start Switch (<3 interactions)│
                  │ - Calibrated P(like), P(reply),     │
                  │   P(skip) Logistic Classifiers      │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │     Mixer & Diversity Selection     │
                  │ - Maximal Marginal Relevance (MMR)  │
                  │ - Author Fatigue Cap (<= 2 per feed)│
                  │ - 50/50 In-Network / Out-of-Network │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │        Explainability Engine        │
                  │ - "Why this post?" Attribution      │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  [ Final Top-15 Personalized Feed ]
```

---

## 2. Exhaustive Course Syllabus Mapping (All 36 Lectures)

Every lecture from AIM3143 is systematically implemented and demonstrated in the codebase:

| Lecture | Topic | Project Implementation & File Reference |
|:---:|:---|:---|
| **L1** | Intro to RecSys | Top-15 personalized feed curation (`run_server.py`, `pipeline/orchestrator.py`) |
| **L2** | Functions & Applications | Multi-objective feed curation: discovery, social connectivity, academic events (`data/generator.py`) |
| **L3** | Explicit vs Implicit Feedback | Multi-action implicit logs (dwell time, likes, replies, skips) vs 10% explicit holdout (`data/generator.py`) |
| **L4** | RecSys Challenges | Sparsity (98.4%), cold-start routing, candidate indexing scalability (`pipeline/orchestrator.py`) |
| **L5** | Matrix Representation | Sparse interaction matrices and vector projections (`models/als_factorization.py`) |
| **L6** | Similarity: Cosine & Pearson | Centered Cosine / Pearson with significance weighting ($\tau = 5$) (`models/collaborative_knn.py`) |
| **L7** | User-Based Collaborative Filtering | User neighborhood scoring with top-$k$ peer correlation (`models/collaborative_knn.py`) |
| **L8** | Item-Based Collaborative Filtering | Co-engagement item cosine similarity and neighbor expansion (`models/collaborative_knn.py`) |
| **L9** | Limitations of Memory Methods | Documented $O(\|U\|^2)$ memory footprint and cold-start brittleness (`evaluation/evaluator.py`) |
| **L10** | Model-Based RecSys | Parameterized low-rank factor models (`models/als_factorization.py`) |
| **L11** | Matrix Factorization with SVD | `TruncatedSVDBaseline` decomposing user-item matrix into $U \Sigma V^T$ (`models/als_factorization.py`) |
| **L12** | Alternating Least Squares (ALS) | Alternating optimization fixing $X$ to solve $Y$ and vice versa (`models/als_factorization.py`) |
| **L13** | Latent Factor Models | Low-rank embedding space ($d=32$) capturing topical affinities (`models/als_factorization.py`) |
| **L14** | Handling Sparsity & Scalability | Factor projection scaling candidate retrieval to $<10$ms (`pipeline/candidate_retrieval.py`) |
| **L15** | Implicit Feedback Models | Confidence-weighted iALS: $C_{ui} = 1 + \alpha r_{ui}$ where $\alpha = 40$ (`models/als_factorization.py`) |
| **L16** | Neural Collaborative Filtering | PyTorch Two-Tower Neural Network embedding User & Post features (`models/neural_two_tower.py`) |
| **L17** | Data Normalization | Z-score user rating centering, min-max scaling, log1p transformation (`pipeline/hydrator.py`) |
| **L18** | Missing Values & Dim Reduction | Negative sampling in iALS; SVD and TF-IDF feature truncation (`models/content_based.py`) |
| **L19** | Security Issues in RecSys | Vulnerability analysis of collaborative systems to like-bombing (`security/shilling_simulator.py`) |
| **L20** | Shilling Attacks & Defenses | Random & Bandwagon attack injection + co-action Jaccard clustering defense (`security/`) |
| **L21** | Evaluation of CF Models | Offline evaluation on NDCG@10, Precision@10, and explicit RMSE (`evaluation/evaluator.py`) |
| **L22** | Intro to Content-Based RecSys | Unsupervised text and hashtag matching without interaction history (`models/content_based.py`) |
| **L23** | CB Advantages & Drawbacks | Zero-cold-start capability vs over-specialization routing (`pipeline/orchestrator.py`) |
| **L24** | Item Profiles & Feature Discovery | Tokenization, TF-IDF vectorization, hashtag feature extraction (`models/content_based.py`) |
| **L25** | Dynamic User Profiles | Dynamic centroid profile $\vec{U}_u$ constructed from engagement-weighted posts (`models/content_based.py`) |
| **L26** | Classification for Content-Based | Supervised calibrated multi-action rankers predicting $P(\text{like})$, $P(\text{reply})$, $P(\text{skip})$ (`pipeline/heavy_ranker.py`) |
| **L27** | Knowledge-Based RecSys | Explicit domain rules and constraints (`pipeline/filters.py`) |
| **L28** | Constraint-Based RecSys | Hard constraints: department matching, author spam cap, age cutoff (`pipeline/filters.py`) |
| **L29** | Case-Based RecSys | Problem-case similarity search: matches user queries to similar past posts (`models/content_based.py`) |
| **L30** | Hybrid Recommendation Systems | Unifying collaborative, content, and knowledge paradigms (`pipeline/orchestrator.py`) |
| **L31** | Monolithic Hybridization | Unified 12-dimensional feature vector combining CF dot product, TF-IDF cosine, and graph stats (`pipeline/hydrator.py`) |
| **L32** | Parallel Hybridization | **Weighted:** Multi-signal combination; **Switching:** Cold-start routing switch; **Mixed:** 50/50 In/Out network balance (`mixer/diversity_selector.py`) |
| **L33** | Pipelined Hybridization | **Cascade:** High-recall retrieval funnels feed Heavy Ranker; **Meta-Level:** iALS factors fed into ranker (`pipeline/`) |
| **L34** | Hybridization Trade-Offs | Latency overheads and error propagation benchmarked (`evaluation/evaluator.py`) |
| **L35** | RecSys Evaluation Metrics | Accuracy (NDCG@10, P@10, R@10), Error (RMSE), Diversity (ILD), Coverage (`evaluation/metrics.py`) |
| **L36** | Real-World Case Study | Complete deployment with live FastAPI server & responsive social UI (`run_server.py`, `frontend/`) |

---

## 3. Quickstart & Execution Sequence

### 3.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.2 Generate Synthetic Campus Social Dataset
```bash
python generate_dataset.py --seed 42 --users 400 --posts 3500
```
- Creates 400 users with personas (`CS_Undergrad`, `Bio_Researcher`, `Campus_Club`, `Course_TA`, `Freshman`).
- Generates Barabási-Albert scale-free follow graph ($m = 6$).
- Produces 3,500 realistic campus posts and 22,000+ multi-signal implicit interactions (98.4% matrix sparsity).
- Outputs saved in `data/raw/`.

### 3.3 Train All Models
```bash
python train.py
```
- Trains User-kNN, Item-kNN, iALS, Content-Based TF-IDF, PyTorch Two-Tower Neural CF, Heavy Ranker, and Shilling Detector.
- Saves artifacts in `artifacts/`.

### 3.4 Run Offline Evaluation Suite
```bash
python evaluate.py
```
- Benchmarks all 8 algorithms, Warm vs Cold cohorts, and the 6-stage Ablation Matrix.
- Outputs formatted comparison tables to the terminal and saves `artifacts/benchmark_summary.json`.

### 3.5 Launch the Interactive Web Dashboard
```bash
python run_server.py --port 8000
```
- Open browser at `http://localhost:8000`.

---

## 4. REST API Specification

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/users` | List all 400 simulated students with personas, departments, and stats. |
| `GET` | `/api/user/{user_id}` | User profile, follow relationships, and recent activity. |
| `GET` | `/api/feed` | Personalized Top-15 feed with explainability pills and funnel telemetry. |
| `POST` | `/api/interact` | Real-time logging of user engagements (Like, Reply, Skip). |
| `POST` | `/api/attack/inject` | Injects synthetic like-bomb attack on target post (Bandwagon or Random). |
| `POST` | `/api/attack/reset` | Excises bot accounts and restores organic baseline. |
| `GET` | `/api/attack/status` | Current attack status and target post rank in feed. |
| `GET` | `/api/benchmark/summary`| Precomputed academic benchmark comparison tables. |
| `GET` | `/api/posts/popular` | Top popular campus posts. |

---

## 5. Automated Test Suite
Run the full test suite verifying schemas, models, pipeline funnel, attack defense, and API endpoints:
```bash
python test_system.py
```
Result: `Ran 10 tests: OK`.

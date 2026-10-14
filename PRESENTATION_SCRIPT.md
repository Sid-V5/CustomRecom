# Varsity: How a Campus Feed Decides What You See
## Slide-by-Slide Presentation & Viva Defense Script

This document provides a simple, direct, spoken presentation script accompanying **`Varsity-how-a-campus-feed-decides-what-you-see (1).pptx`**. It explains how core Recommender System concepts are applied in a production-grade campus social feed.

---

### Slide 1: Feed Architecture · Case Study
**Title**: *Varsity: how a campus feed decides what you see — A multi-stage hybrid recommender inspired by X's open-source pipeline.*

#### 🎙️ Spoken Script
> *"Hello everyone. Today, I'm presenting **Varsity**, a campus social recommender system.*
>
> *Most textbook recommender projects simply take a MovieLens rating matrix and predict 1-to-5 star ratings. But real social media feeds don't work like that. Nobody sits on Twitter, Instagram, or LinkedIn giving 3 stars to a post. Real feeds have to deal with thousands of live posts, fast scrolling, implicit signals like likes and skips, and strict sub-50 millisecond latency.*
>
> *Inspired by the open-source recommendation architecture of X (formerly Twitter), we built a 4-stage hybrid recommender specifically for a university campus. In the next few minutes, I’ll show you step-by-step how our pipeline decides exactly what each student sees on their feed."*

* **RecSys Concept**: Industrial Cascaded Recommender Systems vs. Traditional Single-Model Rating Predictors.
* **Our Project Context**: Moving beyond simple matrix completion to a production-style retrieval → ranking → mixing pipeline.

---

### Slide 2: Dataset & Operating Scale
**Title**: *The setup: 400 students, 3,500 posts, 22,000+ interactions.*

#### 🎙️ Spoken Script
> *"Let's look at the campus environment we modeled.*
>
> *Our dataset contains **400 students** connected via a scale-free follow graph. This means popular clubs and professors have lots of followers, while average students have smaller circles. There are **3,500 active campus posts** covering club announcements, academic notices, and student chatter, resulting in over **22,000 logged interactions**.*
>
> *Our engineering constraint is real-time delivery: from the moment a student opens the app, the system must evaluate those 3,500 candidate posts, personalize the rankings, and render the top-10 feed in under **50 milliseconds**."*

* **RecSys Concept**: Power-law / scale-free network distributions (Barabási-Albert model) and strict real-time latency budgets.
* **Our Project Context**: 400 user nodes with distinct personas (e.g., CS undergrads, bio researchers, campus clubs) and 98.7% interaction sparsity.
* **Q&A Defense**: 
  - *Q: Why not evaluate all 3,500 posts with your best neural network?*
  - *A: Running a heavy deep learning model across 3,500 items on every scroll takes hundreds of milliseconds, violating web latency limits. That's why we need a funnel.*

---

### Slide 3: Core Architecture (The 4-Stage Funnel)
**Title**: *Every feed request runs a four-stage funnel (Pool 3,500 → Sourcing ~350 → Hydration 12 → Heavy Ranker 1 → MMR Mixer 10).*

#### 🎙️ Spoken Script
> *"This slide represents the core backbone of our system: the **Four-Stage Funnel**.*
>
> *Instead of asking one giant model to do everything, we break the problem into four focused stages:*
> 1. *First, **Candidate Retrieval**: We quickly scan the 3,500 posts and grab roughly 350 viable candidates in under 5 milliseconds.*
> 2. *Second, **Feature Hydration**: For each of those 350 candidates, we extract a 12-dimensional feature vector—such as author popularity, distance in the follow graph, and post age.*
> 3. *Third, **Heavy Ranker**: A calibrated machine learning model evaluates these 12 features and predicts how likely the student is to engage with the post.*
> 4. *Finally, **MMR Diversity Mixer**: We de-duplicate similar posts, apply author caps, and deliver the final balanced top-10 feed.*
>
> *This funnel gives us the best of both worlds: extreme speed at the top, and deep precision at the bottom."*

* **RecSys Concept**: Cascaded Hybrid Ranking (Burke, 2002; X Corp., 2023). High-recall retrieval followed by high-precision re-ranking.
* **Our Project Context**: Eliminates 90% of irrelevant items in $<5\text{ms}$, allowing the expensive Heavy Ranker to score only 350 items.

---

### Slide 4: Retrieval & Representation (Stages 1 & 2)
**Title**: *Stages 1 and 2: gather ~350 candidates, then describe each one.*

#### 🎙️ Spoken Script
> *"Now let's zoom in on Stages 1 and 2.*
>
> *How do we find those 350 candidates? We don't rely on just one algorithm; we pull from four parallel sources:*
> - *150 posts come from **In-Network Graph Walks** (what your campus friends are posting or interacting with).*
> - *100 posts come from **Implicit ALS Matrix Factorization** (finding latent taste matches).*
> - *100 posts come from a **Two-Tower Neural Collaborative Filtering** network (capturing non-linear interactions).*
> - *50 posts come from **Content TF-IDF** (matching department tags and hashtags like #hackathon or #midterm).*
>
> *Once merged, Stage 2 **hydrates** each candidate with 12 real-time features—including author follower count, network hop distance, topic overlap, and post age in hours. This 12-number vector becomes the input for our ranker."*

* **RecSys Concept**: Multi-Channel Sourcing and Feature Engineering. Combining collaborative graph signals with semantic content signals.
* **Our Project Context**: Ensures both "in-network" familiarity (friends) and "out-of-network" discovery (interesting campus events from other departments).

---

### Slide 5: Implicit Feedback Formulation
**Title**: *Learning from likes and skips: a like counts 41x more than silence ($C_{ui} = 1 + \alpha \cdot r_{ui}$).*

#### 🎙️ Spoken Script
> *"One of the biggest differences between theory and reality is how feedback is captured.*
>
> *In classroom examples, we assume students explicitly rate items from 1 to 5 stars. In a social app, over 99% of user signals are **implicit**—clicks, likes, replies, and dwell time.*
>
> *Crucially: **an unseen post is NOT a negative rating**. It simply means the student hasn't opened the app yet. However, if a student scrolls past a post in less than 1.5 seconds, that **is** an implicit negative signal—a skip.*
>
> *To handle this mathematically, we implemented **Hu, Koren, and Volinsky’s confidence formulation**: $C_{ui} = 1 + \alpha \cdot r_{ui}$. If a post was never seen, its confidence is 1. But if a student likes or replies to it, its confidence jumps up to 41! In other words, a positive action counts 41 times more than passive silence."*

* **RecSys Concept**: Implicit Feedback vs. Explicit Feedback; Confidence-weighted Matrix Factorization; Negative Dwell Sampling.
* **Our Project Context**: Dwell timer in frontend measures viewport seconds; $<1.5\text{s}$ generates a skip penalty ($-0.5$), whereas replies give a strong signal ($+2.0$).
* **Q&A Defense**:
  - *Q: Why use $\alpha = 40$?*
  - *A: In Hu et al.'s formulation, $\alpha$ controls how aggressively we separate observed interactions from unobserved ones. $\alpha=40$ is the standard empirical value that prevents the model from assuming missing items are dislikes.*

---

### Slide 6: Similarity Metrics (Cosine vs. Pearson vs. iALS)
**Title**: *Three ways to measure "similar", each used where it fits.*

#### 🎙️ Spoken Script
> *"A key topic in recommender systems is choosing the right similarity metric. There is no single 'best' metric—we use three different metrics, each where it fits best mathematically:*
>
> 1. *For **Post Content**, we use **Cosine Similarity**. Word frequencies and hashtags are always positive numbers ($\ge 0$). Cosine measures the angle between two topic vectors, meaning a short tweet and a long departmental announcement about #robotics match perfectly regardless of length.*
> 2. *For **Student-to-Student kNN**, we use **Pearson Correlation**. Some students are naturally generous and like almost every post, while others are very critical. Pearson subtracts each student’s average activity level, centering their behavior so we compare true taste rather than baseline activity levels.*
> 3. *For **Global Latent Tastes**, we use **Implicit ALS (iALS)**. It maps students and posts into a shared 32-dimensional embedding space, solving the fact that our matrix is 98.7% empty."*

* **RecSys Concept**: 
  - **Cosine**: Vector angle alignment, length invariant, ideal for TF-IDF.
  - **Pearson**: Mean-centering, de-biases optimistic vs. stingy raters, ideal for Memory-based kNN.
  - **Matrix Factorization (iALS)**: Dimensionality reduction, dense dot products, overcomes sparsity.
* **Q&A Defense**:
  - *Q: Why not use Pearson correlation on text vectors?*
  - *A: Word count vectors are non-negative and sparse. Mean-centering sparse text vectors turns zeros into negative numbers, destroying vector sparsity and computational efficiency.*

---

### Slide 7: Cold-Start Strategy (The Switching Hybrid)
**Title**: *New students take a different route (Decision Gate: $|H_u| < 3$? Warm Path vs. Cold Path).*

#### 🎙️ Spoken Script
> *"Every recommender system faces the **Cold-Start Problem**: what happens when a brand new freshman joins Varsity with zero likes or follows?*
>
> *Collaborative filtering models like iALS completely fail here because their user vector is all zeros. As shown on the slide, the cold NDCG@10 for pure collaborative filtering is essentially zero: **0.0109**.*
>
> *To fix this, we implemented a **Switching Hybrid architecture** based on Robin Burke's taxonomy:*
> - *When a student opens the feed, we check their history length $|H_u|$.*
> - *If they have fewer than 3 interactions, they take the **Cold Path**: we bypass matrix factorization and recommend posts using **Content-Based TF-IDF** matched to their department and chosen interest tags.*
> - *Once they accumulate 3 or more interactions, the gate dynamically switches them to the **Warm Path** using collaborative iALS.*
>
> *This simple decision rule boosts our cold-start ranking performance from **0.0109 to 0.6719**—a massive **61.6-times improvement**."*

* **RecSys Concept**: The Cold-Start Problem (User cold start); Switching Hybrid Recommender (Burke, 2002); Dynamic thresholding.
* **Our Project Context**: When a new persona (like "Charlie (Freshman)") logs in, the UI displays "Cold-Start Mode" and routes to semantic department/tag matching.

---

### Slide 8: Scoring & Ranking (Heavy Ranker & Age Decay)
**Title**: *The ranker predicts three behaviours, then scores the post ($P(\text{Like}), P(\text{Reply}), P(\text{Skip})$ + Recency Decay).*

#### 🎙️ Spoken Script
> *"Once candidates reach Stage 3, our **Heavy Ranker** scores them using calibrated multi-action probabilities.*
>
> *Rather than predicting a single generic score, we train classifiers to predict three separate real-world behaviors:*
> - *$P(\text{Like})$ — probability the student likes the post.*
> - *$P(\text{Reply})$ — probability the student leaves a comment.*
> - *$P(\text{Skip})$ — probability the student scrolls right past it.*
>
> *We combine these into an **Expected Utility Score**: we reward high-effort replies with a high weight (+1.2), reward likes (+0.6), and penalize predicted skips (-0.4).*
>
> *Finally, we apply an **exponential recency decay**: $e^{-\lambda \cdot \Delta t}$. For example, on the slide, a candidate post has an initial engagement score of 0.192. But because it is 15 hours old, the decay multiplier scales it down to **0.1367**. This prevents yesterday's viral posts from permanently clogging today's feed."*

* **RecSys Concept**: Multi-Task Ranking / Expected Utility Theory; Calibration; Exponential Time-Decay Half-Life.
* **Our Project Context**: Formula: $\text{Score} = [0.60 \cdot P(\text{Like}) + 1.20 \cdot P(\text{Reply}) - 0.40 \cdot P(\text{Skip})] \times e^{-0.02 \cdot \Delta t}$. The numbers on the slide match our prototype calculation directly.

---

### Slide 9: Diversity & Fairness (Maximal Marginal Relevance)
**Title**: *MMR keeps the feed from becoming ten posts about midterms.*

#### 🎙️ Spoken Script
> *"If a recommender only maximizes relevance, it creates a serious problem: the **Filter Bubble**.*
>
> *If a student likes one post about CS midterms, a greedy ranker will fill their entire top-10 feed with 10 nearly identical posts about midterms. That makes the app boring and unhelpful.*
>
> *To prevent this, our Stage 4 runs **Maximal Marginal Relevance (MMR)**, introduced by Carbonell and Goldstein:*
> 1. *We start with the highest-scoring candidate.*
> 2. *For each subsequent slot, we pick the post that has high relevance **minus** its similarity to the posts already chosen.*
> 3. *We also enforce a hard **Author Cap** of at most 2 posts per author so one active account doesn't dominate the feed.*
>
> *By tuning our diversity parameter to $\lambda = 0.70$, our Intra-List Diversity increases by **+5.1%** with **zero loss** in Top-10 precision. Students get high relevance, but with broad campus variety."*

* **RecSys Concept**: Filter Bubbles & Popularity Bias; Intra-List Diversity (ILD); Maximal Marginal Relevance (MMR); Fair Exposure Constraints.
* **Our Project Context**: Controlled via the live slider in our prototype (`MMR Diversity \lambda = 0.70`).

---

### Slide 10: Evaluation Benchmark & References
**Title**: *The hybrid beats every single model (Offline NDCG@10 comparison across 8 baselines).*

#### 🎙️ Spoken Script
> *"Finally, we evaluated our complete architecture against 8 standard industry baselines using an **80/20 time-based train/test split**.*
>
> *The results confirm our design choices:*
> - *Simple baselines like Random or Popularity offer poor personalization.*
> - *Single models like pure User-kNN or pure iALS perform reasonably on warm users, but collapse whenever cold-start users appear.*
> - *Our **Varsity Multi-Stage Hybrid** achieves the highest overall ranking accuracy (**NDCG@10 = 0.0186**), while delivering **0.9599 Intra-List Diversity** and **10% Catalog Coverage** across the entire campus.*
>
> *For our references, we didn't use generic textbook theory. We based the system directly on three applied industry blueprints:*
> 1. *First, **Twitter's 2023 Open-Source Algorithm** on GitHub, which gave us the blueprint for our 4-stage funnel and sub-50ms latency.*
> 2. *Second, **Sarwar et al. (2001)** from ACM WWW, which provides the foundation for our Item-kNN similarity and Pearson rating de-biasing.*
> 3. *Third, **Mobasher et al. (2007)** from ACM TIST, which gave us the math for our Bandwagon attack simulation and Jaccard botnet defense.*
>
> *Thank you! I’m happy to take any questions."*

* **RecSys Concept**: Offline Evaluation; Temporal Split vs. Random Split; Applied Industry Blueprints.
* **Our Project Context**: The evaluation suite runs automatically in our backend and is viewable live inside the prototype's settings drawer.

---

## 💡 Quick Cheat Sheet for Examiner Q&A

| If the Professor asks... | Say this 10-second answer: |
|---|---|
| **"Why did you use a temporal train/test split instead of standard k-fold cross-validation?"** | *"In social media feeds, time moves forward. If you randomly split interactions, the model trains on future likes to predict past likes, causing severe data leakage. An 80/20 time-based split tests true real-world forward prediction."* |
| **"Why didn't you just use Cosine similarity everywhere?"** | *"Cosine similarity does not account for user rating bias. Generous students who like everything would falsely appear dissimilar to conservative students. Pearson correlation mean-centers each user to fix this, while Cosine is reserved for non-negative word TF-IDF."* |
| **"What is the mathematical justification for your skip penalty?"** | *"In social feeds, unseen items are unobserved, not negative. But a post that enters the viewport and is scrolled past in under 1.5 seconds is an active skip. Penalizing it by $-0.5$ provides negative sampling without relying on explicit dislikes."* |
| **"How does the system prevent the cold-start problem from crashing matrix factorization?"** | *"We use Robin Burke's Switching Hybrid pattern: a routing gate checks if interaction count $|H_u| < 3$. If yes, it routes the student to a semantic TF-IDF tag matcher; once they reach 3 interactions, it seamlessly switches to collaborative iALS."* |

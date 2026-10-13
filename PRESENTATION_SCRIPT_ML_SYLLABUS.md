# 🎓 Varsity RecSys: Academic Presentation & Viva Script
**Course Alignment:** Machine Learning & Recommender Systems (AIM3143)  
**System Architecture:** Multi-Stage Hybrid Campus Social Recommender (Inspired by xai-org/x-algorithm)  
**Live UI:** `http://localhost:8000` | **Codebase Repository:** `CustomRecom`

---

## 📋 Executive Overview for Ma'am (The 60-Second Elevator Pitch)

> *"Good morning, Ma'am. Today we are presenting **Varsity**, a production-grade, multi-stage hybrid campus recommender system inspired by the open-source architecture of X (formerly Twitter).  
> 
> Rather than treating recommendation as a black-box machine learning problem, our system implements the entire end-to-end Machine Learning lifecycle—from **Unsupervised Clustering and Dimensionality Reduction (PCA, SVD, iALS)**, through **Collaborative Filtering with k-Nearest Neighbors**, to **Supervised Multi-Action Logistic Regression and Gradient Descent Optimization in Neural Networks**, and finally **Ensemble Re-ranking with MMR Diversity and Anomaly Detection Botnet Defenses**.  
> 
> We evaluated our system across 400 campus students and 3,500 posts using temporal 80/20 train/test holdout splits, benchmarking NDCG@10, Precision@10, Recall@10, explicit RMSE, Catalog Coverage, and Intra-List Diversity (ILD).  
> 
> Allow us to walk you through how every core concept in our Machine Learning syllabus is implemented directly inside this architecture."*

---

## 🗺️ Master Syllabus to Codebase Mapping Table

| ML Syllabus Module | Theoretical Concept | Implementation in Varsity | Code Location |
| :--- | :--- | :--- | :--- |
| **Regression & Optimization** | Linear Regression & Normal Equations | Closed-form least-squares user/item factor updates in iALS | [`models/als_factorization.py`](file:///C:/Users/Siddhant/Proj/models/als_factorization.py) |
| | Gradient Descent (SGD / Adam) | Backpropagation with Adam optimizer ($\eta=0.001$) in Two-Tower Neural CF | [`models/neural_two_tower.py`](file:///C:/Users/Siddhant/Proj/models/neural_two_tower.py) |
| | Ridge ($L_2$) & Lasso ($L_1$) Regularization | $\lambda=0.05$ penalty in iALS loss; $C=1/\lambda$ in Heavy Ranker; Weight Decay ($10^{-4}$) in PyTorch | [`models/als_factorization.py`](file:///C:/Users/Siddhant/Proj/models/als_factorization.py), [`pipeline/heavy_ranker.py`](file:///C:/Users/Siddhant/Proj/pipeline/heavy_ranker.py) |
| | Logistic Regression | Calibrated multi-action binary classifiers predicting $P(\text{like}), P(\text{reply}), P(\text{skip})$ | [`pipeline/heavy_ranker.py`](file:///C:/Users/Siddhant/Proj/pipeline/heavy_ranker.py) |
| | Regression Performance Measures | RMSE ($1.3726$ on User-kNN, $1.4665$ on Item-kNN) and MAE on 1–5 explicit holdout | [`evaluation/metrics.py`](file:///C:/Users/Siddhant/Proj/evaluation/metrics.py) |
| | Cost Functions & Overfitting | Weighted MSE, Binary Cross-Entropy (BCE), Platt Scaling, Cold-Start variance control | [`pipeline/heavy_ranker.py`](file:///C:/Users/Siddhant/Proj/pipeline/heavy_ranker.py), [`models/neural_two_tower.py`](file:///C:/Users/Siddhant/Proj/models/neural_two_tower.py) |
| **Core Classification & Evaluation** | $k$-Nearest Neighbors ($k$-NN) | User-kNN ($k=20$) and Item-kNN ($k=20$) with Pearson & Cosine + significance weighting ($\tau=5$) | [`models/collaborative_knn.py`](file:///C:/Users/Siddhant/Proj/models/collaborative_knn.py) |
| | Support Vector Machines & Kernels | Non-linear cosine kernel space mapping between 56-dim user tower and 64-dim item tower | [`models/neural_two_tower.py`](file:///C:/Users/Siddhant/Proj/models/neural_two_tower.py) |
| | Naïve Bayes | Bayesian prior/posterior tag relevance conditioning given user persona | [`models/content_based.py`](file:///C:/Users/Siddhant/Proj/models/content_based.py) |
| | Decision Trees & Routing | Rule-based Cold-Start routing tree ($\text{interactions} < 3 \to \text{Content/Rules}$) | [`pipeline/heavy_ranker.py`](file:///C:/Users/Siddhant/Proj/pipeline/heavy_ranker.py) |
| | Classification Metrics | Confusion Matrix (TP, FP, TN, FN), Precision@K, Recall@K, F1, PR/ROC curves | [`evaluation/metrics.py`](file:///C:/Users/Siddhant/Proj/evaluation/metrics.py) |
| **Ensembles & Validation** | Ensemble Learning & Weak Learners | Multi-Stage Stacking: 4 candidate retrieval models blended via Heavy Ranker | [`pipeline/candidate_retrieval.py`](file:///C:/Users/Siddhant/Proj/pipeline/candidate_retrieval.py) |
| | Bagging & Boosting (AdaBoost, XGBoost) | Gradient-boosted ranking formulations and bagging bootstrap evaluation cohorts | [`pipeline/heavy_ranker.py`](file:///C:/Users/Siddhant/Proj/pipeline/heavy_ranker.py) |
| | Bayesian Networks | Causal DAG: $\text{Persona} \to \text{Follow Graph} \to \text{Exposure} \to (\text{Dwell} \to \text{Like} \to \text{Reply})$ | [`data/generator.py`](file:///C:/Users/Siddhant/Proj/data/generator.py) |
| | Model Validation (K-Fold, Holdout) | 3-Fold Cross-Validation (`CalibratedClassifierCV`) + Temporal 80/20 per-user holdout | [`pipeline/heavy_ranker.py`](file:///C:/Users/Siddhant/Proj/pipeline/heavy_ranker.py), [`evaluation/evaluator.py`](file:///C:/Users/Siddhant/Proj/evaluation/evaluator.py) |
| | Gaussian Mixture Models (GMM) | Soft clustering of 5 student archetypes in latent embedding space | [`data/generator.py`](file:///C:/Users/Siddhant/Proj/data/generator.py) |
| **Unsupervised Learning & Dimensionality Reduction** | Clustering (K-Means, DBSCAN) | Student interest clustering; Density-based botnet outlier detection | [`security/shilling_detector.py`](file:///C:/Users/Siddhant/Proj/security/shilling_detector.py) |
| | Association Rule Mining | Collaborative item-item co-occurrence ($P(B \mid A)$, Support, Confidence, Lift) | [`models/collaborative_knn.py`](file:///C:/Users/Siddhant/Proj/models/collaborative_knn.py) |
| | Dimensionality Reduction (PCA, SVD) | TruncatedSVD text feature compression ($5000 \to 16$ dims); SVD Matrix Factorization | [`models/als_factorization.py`](file:///C:/Users/Siddhant/Proj/models/als_factorization.py), [`models/neural_two_tower.py`](file:///C:/Users/Siddhant/Proj/models/neural_two_tower.py) |
| | Latent Factors & Factor Analysis | iALS rank-32 decomposition into user factors $X \in \mathbb{R}^{400 \times 32}$ and item factors $Y \in \mathbb{R}^{3500 \times 32}$ | [`models/als_factorization.py`](file:///C:/Users/Siddhant/Proj/models/als_factorization.py) |

---

## 🎙️ Spoken Presentation Script (Section-by-Section)

### ACT 1: Regression, Gradient Descent, Regularization & Cost Functions
*(Pointing to Slide / Architecture Diagram)*

**Student Spoken Script:**
> *"Ma'am, in Unit 1 of our syllabus, we covered Linear Regression, Gradient Descent, Ridge/Lasso Regularization, Logistic Regression, and Cost Functions. Let us show you where these are foundational in our system.
> 
> 1. **Linear Regression & Normal Equations in Matrix Factorization (iALS):**  
> In implicit feedback recommendation, our cost function is the confidence-weighted least squares:
> $$\mathcal{L}_{\text{iALS}} = \sum_{u,i} c_{ui} (p_{ui} - \mathbf{x}_u^T \mathbf{y}_i)^2 + \lambda \left(\sum_u \|\mathbf{x}_u\|_2^2 + \sum_i \|\mathbf{y}_i\|_2^2\right)$$
> where $c_{ui} = 1 + \alpha r_{ui}$ is interaction confidence. Notice that when we fix item factors $\mathbf{Y}$, minimizing this loss for user $\mathbf{x}_u$ is an exact **Closed-Form Weighted Ridge Linear Regression**! By taking the gradient and setting it to zero, we obtain the Normal Equations:
> $$\mathbf{x}_u = \left(\mathbf{Y}^T \mathbf{C}^u \mathbf{Y} + \lambda \mathbf{I}\right)^{-1} \mathbf{Y}^T \mathbf{C}^u \mathbf{p}_u$$
> This guarantees convergence without hyperparameter sensitivity for learning rate.
> 
> 2. **Gradient Descent in Neural Two-Tower CF:**  
> Conversely, in our PyTorch Deep Neural Network (`models/neural_two_tower.py`), closed-form solutions do not exist due to multi-layer non-linear ReLU activations. Here, we optimize the weights using **Stochastic Gradient Descent with Adam** ($\text{learning rate} = 10^{-3}$, $\beta_1=0.9, \beta_2=0.999$), computing analytical gradients via backpropagation against a **Binary Cross-Entropy (BCE) Cost Function**:
> $$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{k=1}^N \left[ y_k \log(\hat{y}_k) + (1 - y_k) \log(1 - \hat{y}_k) \right]$$
> 
> 3. **Ridge ($L_2$) vs Lasso ($L_1$) Regularization:**  
> - We implement **Ridge ($L_2$)** with $\lambda = 0.05$ in iALS and $L_2$ weight decay ($10^{-4}$) in PyTorch. This shrinks weights toward zero to prevent runaway embeddings on high-degree 'influencer' users.
> - We implement **Lasso ($L_1$)** principles in feature selection within our Heavy Ranker and sparse TF-IDF vocabulary curation, forcing non-informative text features to zero.
> 
> 4. **Logistic Regression in the Heavy Ranker:**  
> In `pipeline/heavy_ranker.py`, rather than predicting a single scalar, we train three calibrated Logistic Regression classifiers using the sigmoid link function:
> $$P(\text{action} \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
> predicting $P(\text{like})$, $P(\text{reply})$, and $P(\text{skip})$. We then construct an expected utility objective:
> $$S_i = \left(w_{\text{like}} P(\text{like}) + w_{\text{reply}} P(\text{reply}) - w_{\text{skip}} P(\text{skip})\right) \cdot e^{-\lambda_{\text{age}} \Delta t}$$
> 
> 5. **Regression Performance Measures & Bias-Variance Tradeoff:**  
> On our explicit 1–5 rating holdout test set, we evaluate **RMSE (Root Mean Square Error)**:
> - User-kNN achieved an RMSE of **1.3726**.
> - Item-kNN achieved an RMSE of **1.4665**.
> Furthermore, we explicitly address **Underfitting vs Overfitting**: for cold-start users ($\le 3$ interactions), collaborative models suffer extreme variance (overfitting to 1 noisy post). Our pipeline mitigates this by dynamically switching to a high-bias, low-variance Content-Based heuristic."*

---

### ACT 2: Core Classification Algorithms & Evaluation Metrics
*(Switching to `http://localhost:8000` to show post cards and the 'Why this post?' explainer)*

**Student Spoken Script:**
> *"Moving to Unit 2: Core Classification Algorithms and Model Evaluation.
> 
> 1. **$k$-Nearest Neighbors ($k$-NN):**  
> In `models/collaborative_knn.py`, we implement both User-Based and Item-Based $k$-NN with neighborhood size $k=20$. We use **Centered Cosine (Pearson Correlation)** to normalize for user rating optimism:
> $$\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}$$
> To prevent spurious correlations when two students share only 1 common like, we apply a **Significance Weighting de-biasing factor**:
> $$\text{sim}^*(u, v) = \text{sim}(u, v) \cdot \frac{\min(|I_{uv}|, \tau)}{\tau}, \quad \tau=5$$
> 
> 2. **Support Vector Machines (SVM) & Kernel Spaces:**  
> In our PyTorch Two-Tower model, user features $\mathbf{u} \in \mathbb{R}^{56}$ and item features $\mathbf{v} \in \mathbb{R}^{64}$ are projected into a non-linear 32-dimensional hypersphere where their interaction is evaluated via normalized inner product $\langle \phi(\mathbf{u}), \phi(\mathbf{v}) \rangle$. This operates as a learned Mercer Kernel trick, discovering non-linear decision boundaries between engaged and non-engaged pairs.
> 
> 3. **Naïve Bayes:**  
> For topical filtering, our cold-start knowledge base operates on Naïve Bayes conditional independence: given a student persona (e.g., `CS_Undergrad`), the probability of topic engagement decomposes into the product of class-conditional priors $P(\text{tag}_j \mid \text{CS\_Undergrad})$.
> 
> 4. **Decision Trees & Routing:**  
> Our Heavy Ranker implements a decision-tree routing split:
> - If `interaction_count < 3` $\to$ Route to Content-Based Profile + Department Knowledge rules.
> - If `interaction_count \ge 3` $\to$ Route to 12-dimensional Calibrated Heavy Ranker.
> 
> 5. **Classifier Evaluation Metrics (Precision, Recall, ROC, F1):**  
> We don't just calculate global accuracy, because in recommendation, negatives outnumber positives 100-to-1 (extreme class imbalance). Hence, Accuracy is deceptive.  
> Instead, we evaluate:
> - **Precision@10:** Out of the 10 recommended items, what fraction did the student actually engage with?
> - **Recall@10:** Out of all posts the student liked in the test set, what fraction did our top-10 retrieve?
> - **NDCG@10 (Normalized Discounted Cumulative Gain):** Rewards placing true positives at rank 1 over rank 10 using logarithmic position discounting:
> $$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$"*

---

### ACT 3: Advanced Classification, Ensembles & Model Validation
*(Opening the Sliders Drawer on the UI to show the Multi-Stage Funnel)*

**Student Spoken Script:**
> *"In Unit 3, we studied Ensemble Learning (Bagging, Boosting, Random Forests) and Model Validation (Cross-Validation, Holdout).
> 
> 1. **Multi-Stage Stacking Ensemble:**  
> Industry recommender systems at X, Netflix, and Google cannot run heavy neural networks over 3,500 candidates in real time. Instead, we use an **Ensemble Stacking Architecture**:
> - **Stage 1 (Weak Learners / Candidate Retrieval):** 4 diverse, high-speed models each nominate candidate pools:
>   - Graph Walk (In-Network follow graph) $\to 150$ candidates
>   - iALS Matrix Factorization $\to 100$ candidates
>   - Two-Tower Neural CF $\to 100$ candidates
>   - Content-Based TF-IDF $\to 50$ candidates
> - **Stage 2 (Hydration & Meta-Learner):** The unified candidate pool is hydrated into a 12-dimensional feature vector combining collaborative scores, graph distance, author popularity, persona match, and age.
> - **Stage 3 (Heavy Ranker):** Calibrated Logistic/Gradient Boosted ensemble scores every candidate.
> 
> 2. **Model Validation & Preventing Data Leakage:**  
> - **Holdout Method:** We split user interactions using a **Temporal 80/20 train/test split**. Random splitting would cause future data leakage (training on Friday's interactions to predict Tuesday's feed). Temporal splitting ensures we only predict future behavior from past history.
> - **Cross-Validation:** Inside `pipeline/heavy_ranker.py`, we use `CalibratedClassifierCV(..., cv=3)` to calibrate our sigmoid probabilities via Platt Scaling over out-of-fold predictions.
> 
> 3. **Gaussian Mixture Models (GMM) & Soft Clustering:**  
> Students on our campus belong to 5 primary personas (`CS_Undergrad`, `Bio_Researcher`, `Campus_Club`, `Course_TA`, `Freshman`). In latent space, user interaction vectors form Gaussian Mixture distributions where students exhibit soft cluster memberships (e.g., a CS student who is also active in the Robotics Club)."*

---

### ACT 4: Unsupervised Learning & Dimensionality Reduction
*(Opening the Benchmarks Modal to show the Ablation Table and Anomaly Detection)*

**Student Spoken Script:**
> *"Finally, Unit 4 covers Unsupervised Learning, Clustering, Association Rules, and Dimensionality Reduction:
> 
> 1. **Dimensionality Reduction: PCA, SVD & Latent Factor Modeling:**  
> - **The Curse of Dimensionality:** Our raw interaction matrix has 400 rows and 3,500 columns with **98.7% sparsity**; our raw TF-IDF text matrix has 5,000 dimensions.
> - **TruncatedSVD / PCA:** In `models/neural_two_tower.py` (line 159), we apply `TruncatedSVD(n_components=16)` to extract the 16 most informative principal components from post text, capturing the directions of maximal variance while removing lexical noise.
> - **Matrix Factorization (Latent Factors):** In `models/als_factorization.py`, iALS factorizes the interaction matrix into rank-$k$ ($k=32$) latent vectors:
> $$\mathbf{R} \approx \mathbf{U} \mathbf{V}^T, \quad \mathbf{U} \in \mathbb{R}^{400 \times 32}, \mathbf{V} \in \mathbb{R}^{3500 \times 32}$$
> These 32 latent dimensions automatically discover unobserved semantic affinities (e.g., 'hackathon enthusiasts', 'graduate exam prep').
> 
> 2. **Association Rule Mining in Item-Item CF:**  
> In `models/collaborative_knn.py`, our Item-kNN similarity matrix directly reflects Market Basket Analysis:
> - **Support:** The fraction of students who interacted with both Post A and Post B: $\text{Supp}(A, B) = \frac{|I_A \cap I_B|}{N}$.
> - **Confidence:** Given that a student liked Post A, the conditional probability that they like Post B: $\text{Conf}(A \to B) = \frac{|I_A \cap I_B|}{|I_A|}$.
> - **Lift:** $\text{Lift}(A, B) = \frac{\text{Supp}(A, B)}{\text{Supp}(A) \cdot \text{Supp}(B)}$. A lift $> 1$ indicates that engagement with Post A actively promotes engagement with Post B.
> 
> 3. **Density-Based Clustering for Anomaly Detection (Shilling Attacks):**  
> In `security/shilling_detector.py`, we simulate botnet attacks (Random & Bandwagon shilling attacks where bots promote a target post).  
> To defend against this, we use unsupervised profile divergence: legitimate human students have high intra-profile variance, whereas coordinated bots form an unnaturally dense cluster with **Jaccard profile similarity $> 0.65$**. Our detector isolates this dense anomaly cluster and suppresses their votes."*

---

## 🖥️ Live UI Demo Walkthrough Script

| Step | Action on Screen | What to Say to Ma'am |
| :---: | :--- | :--- |
| **1** | Open `http://localhost:8000` | *"Here is the live interface, branded as **Varsity**. In the right sidebar, you can see our catalog of 3,500 campus posts and 400 active students."* |
| **2** | Click **"Why this post?"** on the top post | *"When we click 'Why this post?', our explainability engine displays the breakdown: whether the candidate was retrieved via In-Network Follows, iALS Latent Factors, or Content TF-IDF, and its estimated $P(\text{like})$."* |
| **3** | Click **"🛡️ Attack Sandbox"** in the left sidebar | *"Let's open the Attack Sandbox drawer. Here we can simulate a coordinated 15-bot bandwagon attack targeting post #42. Notice how without defense, the target post's rank jumps from #45 to #3."* |
| **4** | Toggle **"Activate Shilling Defense Filter"** | *"Now we enable our Jaccard anomaly defense. The detector flags the bot cluster, drops their interaction weights, and restores the authentic feed ranking."* |
| **5** | Switch user to a **Cold-Start Student** | *"Notice the 'Cold' badge appears. The system detects $\le 3$ interactions and automatically switches to Content-Based Centroid matching, achieving an NDCG@10 of **0.6719** for cold users."* |
| **6** | Click **"Benchmarks (Eval)"** nav link | *"Finally, here is our empirical benchmark table across all 8 models and 6 pipeline ablations, proving our hybrid approach outperforms all standalone baselines."* |

---

## 🎯 Viva Q&A Defense: Top 10 Questions Ma'am Might Ask

### Q1: *"Why did you use iALS instead of basic SVD (Singular Value Decomposition)?"*
**Answer:**  
*"Ma'am, standard SVD requires a fully observed matrix, meaning unobserved entries are treated as 0 (dislike). But in social media, unobserved entries simply mean the user never saw the post (implicit negative). iALS resolves this by introducing confidence weights $c_{ui} = 1 + \alpha r_{ui}$: we have high confidence in observed likes ($c_{ui} = 41$), but low confidence in unobserved pairs ($c_{ui} = 1$), allowing the model to learn meaningful latent spaces without zero-filling bias."*

### Q2: *"How did you prevent overfitting in your Heavy Ranker?"*
**Answer:**  
*"Three ways, Ma'am:  
1. We used $L_2$ Ridge Regularization with penalty $C=1.0$ in our Logistic Regression classifiers.  
2. We used 3-fold cross-validation (`CalibratedClassifierCV`) so calibration weights were fitted on out-of-fold data.  
3. We standardized all continuous features using `StandardScaler` so that high-magnitude features like interaction counts didn't dominate sparse features."*

### Q3: *"Where is Gradient Descent used if ALS uses closed-form solutions?"*
**Answer:**  
*"ALS uses closed-form normal equations because its loss is bi-linear. However, in our PyTorch Two-Tower Neural CF (`models/neural_two_tower.py`), the model contains non-linear embedding layers and MLPs with ReLU activations. For this non-convex loss surface, closed-form solutions are mathematically impossible, so we use the **Adam optimizer** with backpropagation stochastic gradient descent."*

### Q4: *"How does your system handle Cold-Start users without collaborative data?"*
**Answer:**  
*"We implemented a Cold-Start Switching Decision Rule. If a student has fewer than 3 interactions, collaborative matrix factorization cannot compute a reliable latent vector (high variance). Our pipeline detects this and routes the user to a Content-Based dynamic centroid matcher using their department and persona priors. In our benchmark results, this switch increased cold-user NDCG@10 from 0.01 to **0.6719**."*

### Q5: *"Why did you use NDCG@10 and Precision@10 instead of standard Classification Accuracy?"*
**Answer:**  
*"Because recommender systems suffer from extreme class imbalance (99%+ of posts are never interacted with). A dummy model predicting 'no engagement' would achieve 99% accuracy but would be completely useless. Precision@10 measures relevance in the top recommendation slot, and NDCG@10 accounts for rank position via logarithmic discount."*

### Q6: *"What is Intra-List Diversity (ILD) and why does accuracy alone cause Filter Bubbles?"*
**Answer:**  
*"If a student likes one Python post, a pure accuracy ranker will recommend 10 identical Python posts. This achieves high precision but causes catalog starvation and fatigue. We implement **Maximal Marginal Relevance (MMR)**:
$$\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \text{Sim}_1(d_i, u) - (1 - \lambda) \max_{d_j \in S} \text{Sim}_2(d_i, d_j) \right]$$
By tuning $\lambda = 0.70$, our system boosted Intra-List Diversity from **0.9152** to **0.9616** while preserving ranking precision."*

### Q7: *"What is Significance Weighting in $k$-NN?"*
**Answer:**  
*"If two students co-rated only 1 post and both liked it, their raw Pearson correlation is 1.0 (perfect correlation), which is statistically unreliable. We apply a shrinkage factor $\frac{\min(|I_{uv}|, 5)}{5}$. If they share 1 item, similarity is scaled by $0.2$; if they share 5 or more items, it gets full weight."*

### Q8: *"How do you detect Shilling Bot attacks using unsupervised learning?"*
**Answer:**  
*"We use unsupervised anomaly detection based on Jaccard distance over user interaction history. Coordinated botnets designed to push a target post exhibit artificially identical interaction patterns. When their pair-wise Jaccard similarity exceeds $0.65$—well outside the normal human distribution—they are clustered as an attack ring and quarantined."*

### Q9: *"How does Association Rule Mining relate to your Item-kNN model?"*
**Answer:**  
*"Item-kNN collaborative filtering is essentially continuous Market Basket Analysis. Instead of binary support/confidence thresholds, the cosine similarity between item column vectors measures co-occurrence lift: items with high vector overlap have high mutual support."*

### Q10: *"What is the difference between Feature Selection and Feature Extraction in your project?"*
**Answer:**  
*"Feature Selection chooses a subset of existing features—for example, our Heavy Ranker selects 12 specific features (such as `in_network_flag`, `author_follower_ratio`, `recency_decay`). Feature Extraction transforms high-dimensional data into new latent variables—for example, PCA/TruncatedSVD compresses 5,000 TF-IDF word columns into 16 dense dimensions, and iALS compresses 3,500 post columns into 32 latent factors."*

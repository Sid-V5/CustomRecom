"""
Global Configuration and Hyperparameters for Mini-X Recommendation System.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

USERS_FILE = RAW_DATA_DIR / "users.json"
POSTS_FILE = RAW_DATA_DIR / "posts.json"
FOLLOWS_FILE = RAW_DATA_DIR / "follows.csv"
INTERACTIONS_FILE = RAW_DATA_DIR / "interactions.csv"
EXPLICIT_HOLDOUT_FILE = RAW_DATA_DIR / "explicit_holdout.csv"
BENCHMARK_FILE = ARTIFACTS_DIR / "benchmark_summary.json"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Dataset Volumes
NUM_USERS = 400
NUM_POSTS = 3500
BARABASI_M = 6  # Outgoing edges per node for preferential attachment follow graph

# Cold-Start Definition
COLD_USER_THRESHOLD = 3  # Interactions < 3 is cold-start
COLD_ITEM_THRESHOLD = 2  # Interactions <= 1 is cold item

# iALS Hyperparameters
ALS_FACTORS = 32
ALS_ALPHA = 40.0
ALS_REGULARIZATION = 0.1
ALS_ITERATIONS = 15

# kNN Hyperparameters
KNN_K = 20
SIGNIFICANCE_THRESHOLD_TAU = 5.0

# Content-Based TF-IDF Hyperparameters
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 2)

# Two-Tower Neural CF
NEURAL_EMBED_DIM = 32
NEURAL_BATCH_SIZE = 64
NEURAL_LR = 0.001
NEURAL_EPOCHS = 10

# Heavy Ranker Default Weights
DEFAULT_WEIGHT_LIKE = 0.6
DEFAULT_WEIGHT_REPLY = 1.2
DEFAULT_WEIGHT_SKIP = 0.4
DEFAULT_LAMBDA_AGE = 0.02

# Mixer & Diversity Defaults
DEFAULT_MMR_LAMBDA = 0.7  # 1.0 = purely relevance, 0.0 = purely diversity
FEED_TOP_K = 15
MAX_POSTS_PER_AUTHOR = 2
TARGET_IN_NETWORK_RATIO = 0.50

# Candidate Sourcing Limits
LIMIT_IN_NETWORK = 150
LIMIT_ALS_LATENT = 100
LIMIT_ITEM_KNN = 50
LIMIT_CONTENT_TFIDF = 50
LIMIT_TRENDING = 50

# Security / Shilling Attack Simulation
SHILLING_NUM_BOTS = 25
SHILLING_FILLER_COUNT = 15
SHILLING_ANOMALY_THRESHOLD = 0.85

"""
CLI Runner for synthetic campus dataset generation.
Usage:
    python generate_dataset.py --seed 42 --users 400 --posts 3500
"""

import argparse
from config import NUM_POSTS, NUM_USERS, RANDOM_SEED
from data.generator import run_generator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic campus social dataset.")
    parser.add_argument("--users", type=int, default=NUM_USERS, help="Total number of users (default: 400)")
    parser.add_argument("--posts", type=int, default=NUM_POSTS, help="Total number of posts (default: 3500)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()
    print(f"Starting dataset generation: users={args.users}, posts={args.posts}, seed={args.seed}")
    run_generator(num_users=args.users, num_posts=args.posts, seed=args.seed)


if __name__ == "__main__":
    main()

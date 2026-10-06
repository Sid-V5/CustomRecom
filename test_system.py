"""
Comprehensive End-to-End Verification Test Suite.
Tests:
- Schema validation
- Model loading & predictions (kNN, iALS, Content, Two-Tower, Heavy Ranker)
- Pipeline orchestrator funnel execution (Warm and Cold users)
- Shilling attack injection & defense anomaly filtering
- FastAPI REST endpoints with TestClient
"""

import unittest
from fastapi.testclient import TestClient

from config import ARTIFACTS_DIR, RAW_DATA_DIR, BENCHMARK_FILE
from data.schemas import User, Post, Interaction
from run_server import app


class TestMiniXSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_raw_data_files_exist(self):
        """Verify generated dataset files exist on disk."""
        users_file = RAW_DATA_DIR / "users.json"
        posts_file = RAW_DATA_DIR / "posts.json"
        follows_file = RAW_DATA_DIR / "follows.csv"
        interactions_file = RAW_DATA_DIR / "interactions.csv"
        holdout_file = RAW_DATA_DIR / "explicit_holdout.csv"

        self.assertTrue(users_file.exists(), "users.json missing")
        self.assertTrue(posts_file.exists(), "posts.json missing")
        self.assertTrue(follows_file.exists(), "follows.csv missing")
        self.assertTrue(interactions_file.exists(), "interactions.csv missing")
        self.assertTrue(holdout_file.exists(), "explicit_holdout.csv missing")

    def test_02_model_artifacts_exist(self):
        """Verify trained model artifacts exist on disk."""
        self.assertTrue((ARTIFACTS_DIR / "user_knn.pkl").exists(), "user_knn.pkl missing")
        self.assertTrue((ARTIFACTS_DIR / "item_knn.pkl").exists(), "item_knn.pkl missing")
        self.assertTrue((ARTIFACTS_DIR / "ials_model.pkl").exists(), "ials_model.pkl missing")
        self.assertTrue((ARTIFACTS_DIR / "content_model.pkl").exists(), "content_model.pkl missing")
        self.assertTrue((ARTIFACTS_DIR / "two_tower_model.pt").exists(), "two_tower_model.pt missing")
        self.assertTrue((ARTIFACTS_DIR / "heavy_ranker.pkl").exists(), "heavy_ranker.pkl missing")
        self.assertTrue((ARTIFACTS_DIR / "detector.pkl").exists(), "detector.pkl missing")
        self.assertTrue(BENCHMARK_FILE.exists(), "benchmark_summary.json missing")

    def test_03_get_users_endpoint(self):
        """Test GET /api/users endpoint."""
        res = self.client.get("/api/users")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("users", data)
        self.assertGreaterEqual(len(data["users"]), 400)
        # Check first user fields
        u0 = data["users"][0]
        self.assertIn("user_id", u0)
        self.assertIn("username", u0)
        self.assertIn("persona", u0)
        self.assertIn("is_cold_start", u0)

    def test_04_get_user_profile(self):
        """Test GET /api/user/{user_id}."""
        res = self.client.get("/api/user/0")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["user_id"], 0)
        self.assertIn("department", data)
        self.assertIn("recent_interactions", data)

    def test_05_feed_warm_user(self):
        """Test GET /api/feed for a warm user."""
        res = self.client.get("/api/feed?user_id=0&top_k=15")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["user_id"], 0)
        feed = data["feed"]
        self.assertGreater(len(feed), 0)
        self.assertLessEqual(len(feed), 15)

        # Check post structure
        p0 = feed[0]
        self.assertIn("post_id", p0)
        self.assertIn("content", p0)
        self.assertIn("p_like", p0)
        self.assertIn("final_score", p0)
        self.assertIn("explanation", p0)
        self.assertIn("primary_reason", p0["explanation"])

        # Check funnel telemetry
        funnel = data["funnel_stats"]
        self.assertEqual(funnel["corpus_count"], 3500)
        self.assertGreater(funnel["retrieved_count"], 0)
        self.assertGreater(funnel["hydrated_count"], 0)
        self.assertGreater(funnel["filtered_count"], 0)
        self.assertGreater(funnel["latency_ms"], 0)

    def test_06_feed_cold_start_user(self):
        """Test GET /api/feed for a cold-start user routes to cold logic."""
        # Find a cold user
        users_res = self.client.get("/api/users").json()["users"]
        cold_user = next((u for u in users_res if u["is_cold_start"]), None)
        self.assertIsNotNone(cold_user)

        res = self.client.get(f"/api/feed?user_id={cold_user['user_id']}&top_k=15")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["funnel_stats"]["is_cold_start"])
        self.assertGreater(len(data["feed"]), 0)

    def test_07_post_interaction(self):
        """Test logging like engagement."""
        res = self.client.post("/api/interact", json={
            "user_id": 0,
            "post_id": 10,
            "action": "like",
            "dwell_seconds": 25.0,
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["action_recorded"], "like")

    def test_08_shilling_attack_and_defense(self):
        """Test attack injection, rank inflation, and defense filtering (Lecture L20, L33)."""
        target_post_id = 42

        # 1. Inject bandwagon attack
        inject_res = self.client.post("/api/attack/inject", json={
            "target_post_id": target_post_id,
            "attack_type": "bandwagon",
        })
        self.assertEqual(inject_res.status_code, 200)
        inject_data = inject_res.json()
        self.assertEqual(inject_data["status"], "attack_injected")

        # 2. Check attack status and rank inflation
        status_res = self.client.get(f"/api/attack/status?user_id=0")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertEqual(status_data["target_post_id"], target_post_id)
        self.assertTrue(status_data["is_flagged_as_botnet"])
        self.assertGreater(status_data["anomaly_score"], 0.80)
        self.assertIsNotNone(status_data["target_post_rank_unprotected"], "Target post rank should be computed")
        self.assertLessEqual(status_data["target_post_rank_unprotected"], 5, "Attacked post should inflate to Top 5")

        # Verify target post is present in unprotected feed
        feed_unprot = self.client.get("/api/feed?user_id=0&top_k=20&defense_active=false")
        self.assertEqual(feed_unprot.status_code, 200)
        unprot_ids = [p["post_id"] for p in feed_unprot.json()["feed"]]
        self.assertIn(target_post_id, unprot_ids, "Attacked post should appear in unprotected feed")

        # 3. Test defense active: target post must be filtered out
        feed_def_res = self.client.get("/api/feed?user_id=0&top_k=20&defense_active=true")
        self.assertEqual(feed_def_res.status_code, 200)
        feed_posts = [p["post_id"] for p in feed_def_res.json()["feed"]]
        self.assertNotIn(target_post_id, feed_posts, "Anomalous target post should be filtered by defense")

        # 4. Reset attack
        reset_res = self.client.post("/api/attack/reset")
        self.assertEqual(reset_res.status_code, 200)
        self.assertEqual(reset_res.json()["status"], "attack_cleared")

    def test_09_benchmark_summary_endpoint(self):
        """Test GET /api/benchmark/summary."""
        res = self.client.get("/api/benchmark/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("main_benchmarks", data)
        self.assertIn("cold_start_benchmarks", data)
        self.assertIn("ablations", data)
        self.assertGreaterEqual(len(data["main_benchmarks"]), 8)

    def test_10_serve_frontend(self):
        """Test GET / serves the HTML frontend."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Mini-X | Campus Social Feed Recommender", res.text)

    def test_11_case_based_search(self):
        """Test GET /api/search implements Case-Based retrieval (Lecture L29)."""
        res = self.client.get("/api/search?q=machine+learning+homework&user_id=0&top_k=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["query"], "machine learning homework")
        self.assertGreater(data["results_count"], 0)
        feed = data["feed"]
        self.assertGreater(len(feed), 0)
        first_item = feed[0]
        self.assertEqual(first_item["source"], "case_based_search")
        self.assertIn("explanation", first_item)
        self.assertIn("primary_reason", first_item["explanation"])

    def test_12_feed_retrieval_modes_and_filters(self):
        """Test retrieval_mode (in_network / out_network) and department_filter (L28)."""
        # 1. In-network only
        res_in = self.client.get("/api/feed?user_id=0&retrieval_mode=in_network&top_k=10")
        self.assertEqual(res_in.status_code, 200)
        feed_in = res_in.json()["feed"]
        for p in feed_in:
            self.assertEqual(p["source"], "in_network")

        # 2. Department filter constraint
        dept = "Computer Science & AI"
        res_dept = self.client.get(f"/api/feed?user_id=0&department_filter={dept}&top_k=10")
        self.assertEqual(res_dept.status_code, 200)
        feed_dept = res_dept.json()["feed"]
        for p in feed_dept:
            self.assertEqual(p["department"], dept)

    def test_13_all_six_ablations_present(self):
        """Verify all 6 required ablations from Section 6.4 exist in benchmark report."""
        res = self.client.get("/api/benchmark/summary")
        self.assertEqual(res.status_code, 200)
        ablations = res.json()["ablations"]
        self.assertEqual(len(ablations), 6, "Expected exactly 6 ablations per Section 6.4")
        ablation_names = [a["model"] for a in ablations]
        self.assertTrue(any("In-Network Only" in name for name in ablation_names))
        self.assertTrue(any("Out-of-Network Only" in name for name in ablation_names))
        self.assertTrue(any("No MMR" in name for name in ablation_names))
        self.assertTrue(any("MMR Diversity" in name for name in ablation_names))
        self.assertTrue(any("Unprotected" in name for name in ablation_names))
        self.assertTrue(any("Defense Active" in name for name in ablation_names))


if __name__ == "__main__":
    unittest.main()


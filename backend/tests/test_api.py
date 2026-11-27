"""
ASTRA — Test Suite (Unit and API Integration Tests)
Runs without external dependencies using Python's built-in unittest framework.
"""

import os
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


class TestPredictionEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import predict
        predict.load_model()
        cls.predict = predict

    def test_risk_classification_tiers(self):
        # High: > -12.0
        # Medium: -20.0 < x <= -12.0
        # Low: <= -20.0
        p = self.predict.predict_data({"time_to_tca": 1.0, "miss_distance": 50.0})
        self.assertIn(p["prediction"], ["Low", "Medium", "High"])
        self.assertIsInstance(p["log10_probability_of_collision"], float)
        self.assertTrue(0.0 <= p["probability_of_collision"] <= 1.0)
        self.assertTrue(0.0 <= p["calibrated_collision_probability"] <= 1.0)
        self.assertTrue(0 <= p["risk_score"] <= 100)

    def test_uncertainty_quantification(self):
        p = self.predict.predict_data({"time_to_tca": 2.0, "miss_distance": 150.0})
        self.assertIn("uncertainty", p)
        unc = p["uncertainty"]
        self.assertIn("standard_error", unc)
        self.assertIn("confidence_interval_95", unc)
        self.assertEqual(len(unc["confidence_interval_95"]), 2)
        ci_low, ci_high = unc["confidence_interval_95"]
        self.assertLessEqual(ci_low, ci_high)

    def test_shap_explanation(self):
        p = self.predict.predict_data({"time_to_tca": 1.5, "miss_distance": 100.0, "mahalanobis_distance": 2.5})
        self.assertIn("shap_explanation", p)
        shap_info = p["shap_explanation"]
        self.assertIsNotNone(shap_info["primary_driver"])
        self.assertTrue(len(shap_info["top_features"]) > 0)
        first_feat = shap_info["top_features"][0]
        self.assertIn("feature", first_feat)
        self.assertIn("attribution", first_feat)
        self.assertIn("impact", first_feat)

    def test_batch_prediction(self):
        records = [
            {"time_to_tca": 1.0, "miss_distance": 50.0},
            {"time_to_tca": 3.0, "miss_distance": 300.0},
            {"time_to_tca": 6.0, "miss_distance": 1200.0},
        ]
        results = self.predict.predict_batch(records)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn(r["prediction"], ["Low", "Medium", "High"])


class TestFastAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import predict
        predict.load_model()
        import main
        from fastapi.testclient import TestClient
        cls.client = TestClient(main.app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["project"], "ASTRA SDC-II")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["features_loaded"], 102)

    def test_auth_login_and_me(self):
        # Invalid
        inv = self.client.post("/auth/login", json={"username": "admin", "password": "wrongpassword"})
        self.assertEqual(inv.status_code, 401)

        # Valid
        val = self.client.post("/auth/login", json={"username": "admin", "password": "astra2024"})
        self.assertEqual(val.status_code, 200)
        token = val.json()["access_token"]

        # Profile
        me = self.client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["username"], "admin")

    def test_single_predict_json(self):
        resp = self.client.post("/predict/json", json={"features": {"time_to_tca": 2.5, "miss_distance": 250.0}})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("prediction", data)
        self.assertIn("calibrated_collision_probability", data)
        self.assertIn("uncertainty", data)
        self.assertIn("shap_explanation", data)

    def test_batch_predict_json(self):
        items = [{"time_to_tca": 1.0, "miss_distance": 90.0}, {"time_to_tca": 4.0, "miss_distance": 400.0}]
        resp = self.client.post("/predict/batch", json={"items": items})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["total_processed"], 2)

    def test_tabular_file_upload(self):
        csv_content = b"time_to_tca,miss_distance,relative_speed\n2.0,150.0,12000.0\n3.5,450.0,11500.0\n"
        resp = self.client.post(
            "/predict/tabular",
            content=csv_content,
            headers={"Content-Type": "text/csv", "X-Filename": "test_cdm_batch.csv"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("batch_mode"))
        self.assertEqual(data["total_cdms_processed"], 2)

    def test_model_info(self):
        resp = self.client.get("/model/info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_type"], "CatBoostRegressor")
        self.assertTrue(data["leakage_free"])
        self.assertEqual(data["feature_count"], 102)


if __name__ == "__main__":
    unittest.main()

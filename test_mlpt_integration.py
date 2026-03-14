"""
Integration tests for MLPT API using the real trained model.
Tests are skipped if no model checkpoint is available.

Run with: python3 -m pytest test_mlpt_integration.py -v
"""

import os
import pytest
from app import app
from mlpt.config import MODEL_SAVE_DIR

# Check whether any model checkpoint exists
_checkpoint_exists = (
    os.path.exists(os.path.join(MODEL_SAVE_DIR, "mlpt_80_20.pt"))
    or os.path.exists(os.path.join(MODEL_SAVE_DIR, "mlpt_70_30.pt"))
)

skip_no_model = pytest.mark.skipif(
    not _checkpoint_exists,
    reason="No MLPT model checkpoint found in mlpt/saved_models/",
)


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@skip_no_model
class TestRealPredictions:
    """Integration tests that use the real trained model."""

    def test_real_prediction_single(self, client):
        """Submit a real peptide and verify valid prediction with all fields."""
        resp = client.post("/api/mlpt/predict", json={
            "sequences": ["FIASNGVKLV"],
        })

        assert resp.status_code == 200
        data = resp.get_json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1

        pred = data["predictions"][0]
        assert "predicted_class" in pred
        assert "confidence" in pred
        assert "probabilities" in pred
        assert "kt_scores" in pred
        assert "antigenicity_score" in pred
        assert "antigenic_regions" in pred

        # Verify confidence is a valid probability
        assert 0.0 <= pred["confidence"] <= 1.0

        # Verify probabilities sum to ~1.0
        prob_sum = sum(pred["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01

        # Verify predicted_class matches highest probability
        max_class = max(pred["probabilities"], key=pred["probabilities"].get)
        assert pred["predicted_class"] == max_class

    def test_real_prediction_batch(self, client):
        """Submit 4 example sequences and verify all get predictions."""
        sequences = [
            "FIASNGVKLV",
            "MVLSPADKTN",
            "ACDEFGHIKL",
            "YWQRSTVACD",
        ]
        resp = client.post("/api/mlpt/predict", json={"sequences": sequences})

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["predictions"]) == 4

        for pred in data["predictions"]:
            # Each prediction should either be a valid result or have an error
            assert "predicted_class" in pred or "error" in pred

    def test_real_prediction_short_sequence(self, client):
        """Submit 'AA' which is too short for MLPT (needs >= 3 AAs), verify error in result."""
        resp = client.post("/api/mlpt/predict", json={"sequences": ["AA"]})

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["predictions"]) == 1
        pred = data["predictions"][0]
        # The predictor should return an error dict for too-short sequences
        assert "error" in pred
        assert "too short" in pred["error"].lower() or "short" in pred["error"].lower()

    def test_real_prediction_long_sequence(self, client):
        """Submit a 60-residue peptide and verify it works."""
        long_peptide = "FIASNGVKLV" * 6  # 60 residues
        resp = client.post("/api/mlpt/predict", json={"sequences": [long_peptide]})

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["predictions"]) == 1
        pred = data["predictions"][0]
        assert "predicted_class" in pred
        assert "confidence" in pred
        assert 0.0 <= pred["confidence"] <= 1.0

    def test_real_prediction_response_format(self, client):
        """Verify all expected fields exist in a real prediction response."""
        resp = client.post("/api/mlpt/predict", json={"sequences": ["FIASNGVKLV"]})

        assert resp.status_code == 200
        data = resp.get_json()

        # Top-level keys
        assert "predictions" in data
        assert "model_info" in data

        pred = data["predictions"][0]
        expected_fields = [
            "predicted_class",
            "confidence",
            "probabilities",
            "kt_scores",
            "antigenicity_score",
            "antigenic_regions",
            "sequence",
            "sequence_length",
        ]
        for field in expected_fields:
            assert field in pred, f"Missing field: {field}"

        # Validate types
        assert isinstance(pred["predicted_class"], str)
        assert isinstance(pred["confidence"], float)
        assert isinstance(pred["probabilities"], dict)
        assert isinstance(pred["kt_scores"], list)
        assert isinstance(pred["antigenicity_score"], float)
        assert isinstance(pred["antigenic_regions"], list)
        assert isinstance(pred["sequence_length"], int)

        # Verify probabilities have all 6 classes
        assert len(pred["probabilities"]) == 6

        # Verify model_info structure
        model_info = data["model_info"]
        assert "loaded" in model_info
        assert model_info["loaded"] is True
        assert "num_classes" in model_info
        assert model_info["num_classes"] == 6


@skip_no_model
class TestRealModelHealth:
    """Integration tests for health endpoint with a real loaded model."""

    def test_real_model_health(self, client):
        """GET /api/mlpt/health when model is loaded should show model_loaded=true."""
        # First trigger a prediction to ensure the model is lazily loaded
        client.post("/api/mlpt/predict", json={"sequences": ["FIASNGVKLV"]})

        resp = client.get("/api/mlpt/health")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "service" in data

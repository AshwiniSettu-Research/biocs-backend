"""
Tests for MLPT API endpoints with mocked predictor.
Run with: python3 -m pytest test_mlpt_endpoints.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _make_mock_prediction(sequence):
    """Create a realistic mock prediction result for a single sequence."""
    return {
        "sequence": sequence,
        "sequence_length": len(sequence),
        "predicted_class": "Cancer Antigenic Peptides",
        "confidence": 0.8523,
        "probabilities": {
            "Cancer Antigenic Peptides": 0.8523,
            "Inactive Peptides-Lung Breast": 0.0412,
            "Moderately Active-Lung Breast": 0.0301,
            "Natural Peptide": 0.0254,
            "Non-Natural Peptide": 0.0318,
            "Very Active-Lung Breast": 0.0192,
        },
        "kt_scores": [1.064, 1.152, 1.064, 1.012, 0.874, 1.015, 1.383, 0.930, 1.250, 1.383],
        "antigenicity_score": 1.0627,
        "antigenic_regions": [(2, 5), (7, 9)],
    }


def _make_mock_model_info():
    """Create a realistic mock model info dict."""
    return {
        "loaded": True,
        "num_classes": 6,
        "class_names": [
            "Cancer Antigenic Peptides",
            "Inactive Peptides-Lung Breast",
            "Moderately Active-Lung Breast",
            "Natural Peptide",
            "Non-Natural Peptide",
            "Very Active-Lung Breast",
        ],
        "device": "cpu",
        "accuracy": 0.9123,
        "macro_f1": 0.8876,
    }


class TestPredictEndpoint:
    """Tests for POST /api/mlpt/predict."""

    @patch("mlpt.routes._get_predictor")
    def test_predict_valid_single_sequence(self, mock_get_pred, client):
        """POST with one sequence should return 200 with correct structure."""
        predictor = MagicMock()
        predictor.loaded = True
        predictor.predict.return_value = [_make_mock_prediction("FIASNGVKLV")]
        predictor.get_model_info.return_value = _make_mock_model_info()
        mock_get_pred.return_value = predictor

        resp = client.post("/api/mlpt/predict", json={"sequences": ["FIASNGVKLV"]})

        assert resp.status_code == 200
        data = resp.get_json()
        assert "predictions" in data
        assert "model_info" in data
        assert len(data["predictions"]) == 1

        pred = data["predictions"][0]
        assert "predicted_class" in pred
        assert "confidence" in pred
        assert "probabilities" in pred
        assert "kt_scores" in pred
        assert "antigenicity_score" in pred
        assert "antigenic_regions" in pred
        assert pred["predicted_class"] == "Cancer Antigenic Peptides"
        assert pred["confidence"] == 0.8523

    @patch("mlpt.routes._get_predictor")
    def test_predict_valid_batch(self, mock_get_pred, client):
        """POST with 3 sequences should return 3 predictions."""
        sequences = ["FIASNGVKLV", "MVLSPADKTN", "ACDEFGHIKL"]
        predictor = MagicMock()
        predictor.loaded = True
        predictor.predict.return_value = [_make_mock_prediction(s) for s in sequences]
        predictor.get_model_info.return_value = _make_mock_model_info()
        mock_get_pred.return_value = predictor

        resp = client.post("/api/mlpt/predict", json={"sequences": sequences})

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["predictions"]) == 3
        # Verify predictor was called with the correct sequences
        predictor.predict.assert_called_once_with(sequences)

    @patch("mlpt.routes._get_predictor")
    def test_predict_empty_sequences_list(self, mock_get_pred, client):
        """POST with empty sequences list should return 400."""
        predictor = MagicMock()
        predictor.loaded = True
        mock_get_pred.return_value = predictor

        resp = client.post("/api/mlpt/predict", json={"sequences": []})

        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "No sequences" in data["error"]

    @patch("mlpt.routes._get_predictor")
    def test_predict_missing_sequences_key(self, mock_get_pred, client):
        """POST with no 'sequences' key should return 400."""
        predictor = MagicMock()
        predictor.loaded = True
        mock_get_pred.return_value = predictor

        resp = client.post("/api/mlpt/predict", json={"peptides": ["FIASNGVKLV"]})

        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "No sequences" in data["error"]

    @patch("mlpt.routes._get_predictor")
    def test_predict_exceeds_max_sequences(self, mock_get_pred, client):
        """POST with 101 sequences should return 400."""
        predictor = MagicMock()
        predictor.loaded = True
        mock_get_pred.return_value = predictor

        sequences = [f"FIASNGVKLV" for _ in range(101)]
        resp = client.post("/api/mlpt/predict", json={"sequences": sequences})

        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
        assert "100" in data["error"]

    @patch("mlpt.routes._get_predictor")
    def test_predict_string_input_converted_to_list(self, mock_get_pred, client):
        """POST with sequences as a single string (not list) should work."""
        predictor = MagicMock()
        predictor.loaded = True
        predictor.predict.return_value = [_make_mock_prediction("FIASNGVKLV")]
        predictor.get_model_info.return_value = _make_mock_model_info()
        mock_get_pred.return_value = predictor

        resp = client.post("/api/mlpt/predict", json={"sequences": "FIASNGVKLV"})

        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["predictions"]) == 1
        # Verify the string was converted to a list before calling predict
        predictor.predict.assert_called_once_with(["FIASNGVKLV"])

    @patch("mlpt.routes._get_predictor")
    def test_predict_model_not_loaded(self, mock_get_pred, client):
        """When predictor.loaded is False, should return 503."""
        predictor = MagicMock()
        predictor.loaded = False
        mock_get_pred.return_value = predictor

        resp = client.post("/api/mlpt/predict", json={"sequences": ["FIASNGVKLV"]})

        assert resp.status_code == 503
        data = resp.get_json()
        assert "error" in data
        assert "not loaded" in data["error"].lower() or "Model" in data["error"]

    @patch("mlpt.routes._get_predictor")
    def test_predict_server_error_returns_generic_message(self, mock_get_pred, client):
        """When predictor raises Exception, return 500 with generic message (no details)."""
        predictor = MagicMock()
        predictor.loaded = True
        predictor.predict.side_effect = RuntimeError("CUDA out of memory: secret internal detail")
        mock_get_pred.return_value = predictor

        resp = client.post("/api/mlpt/predict", json={"sequences": ["FIASNGVKLV"]})

        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data
        # Ensure the internal error message is NOT leaked to the client
        assert "CUDA" not in data["error"]
        assert "secret" not in data["error"]
        assert "internal" in data["error"].lower() or "error" in data["error"].lower()


class TestHealthEndpoint:
    """Tests for GET /api/mlpt/health."""

    @patch("mlpt.routes._get_predictor")
    def test_health_endpoint(self, mock_get_pred, client):
        """GET /api/mlpt/health should return status, model_loaded, and service."""
        predictor = MagicMock()
        predictor.loaded = True
        mock_get_pred.return_value = predictor

        resp = client.get("/api/mlpt/health")

        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert data["model_loaded"] is True
        assert "service" in data


class TestClassesEndpoint:
    """Tests for GET /api/mlpt/classes."""

    def test_classes_endpoint(self, client):
        """GET /api/mlpt/classes should return 6 classes with name and description."""
        resp = client.get("/api/mlpt/classes")

        assert resp.status_code == 200
        data = resp.get_json()
        assert "num_classes" in data
        assert data["num_classes"] == 6
        assert "classes" in data
        assert len(data["classes"]) == 6

        for cls in data["classes"]:
            assert "index" in cls
            assert "name" in cls
            assert "description" in cls
            assert isinstance(cls["name"], str)
            assert isinstance(cls["description"], str)
            assert len(cls["name"]) > 0
            assert len(cls["description"]) > 0

        # Verify the expected class names are all present
        names = {c["name"] for c in data["classes"]}
        expected_names = {
            "Cancer Antigenic Peptides",
            "Inactive Peptides-Lung Breast",
            "Moderately Active-Lung Breast",
            "Natural Peptide",
            "Non-Natural Peptide",
            "Very Active-Lung Breast",
        }
        assert names == expected_names

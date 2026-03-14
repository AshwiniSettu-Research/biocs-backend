"""
Flask Blueprint for the MLPT Antigenic Peptide Predictor API.

Endpoints:
  POST /api/mlpt/predict  - Predict class for peptide sequences
  GET  /api/mlpt/health   - Health check
  GET  /api/mlpt/classes  - List the 6 prediction classes
"""

import logging
import threading
from flask import Blueprint, request, jsonify
from .predict import MLPTPredictor
from .config import CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)

mlpt_bp = Blueprint("mlpt", __name__)

# Lazy-loaded predictor (initialized on first request, thread-safe)
_predictor = None
_predictor_lock = threading.Lock()


def _get_predictor():
    """Get or initialize the MLPT predictor singleton (thread-safe)."""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            # Double-checked locking
            if _predictor is None:
                _predictor = MLPTPredictor()
    return _predictor


@mlpt_bp.route("/api/mlpt/predict", methods=["POST"])
def predict():
    """Predict antigenic peptide class for one or more sequences."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON data provided."}), 400

        sequences = data.get("sequences", [])
        if isinstance(sequences, str):
            sequences = [sequences]

        if not sequences:
            return jsonify({"error": "No sequences provided."}), 400

        if len(sequences) > 100:
            return jsonify({"error": "Maximum 100 sequences per request."}), 400

        predictor = _get_predictor()
        if not predictor.loaded:
            return jsonify({
                "error": "Model not loaded. Please train the model first.",
            }), 503

        predictions = predictor.predict(sequences)
        model_info = predictor.get_model_info()

        return jsonify({
            "predictions": predictions,
            "model_info": model_info,
        })

    except Exception as e:
        logger.error(f"MLPT prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "An internal error occurred during prediction. Please try again."}), 500


@mlpt_bp.route("/api/mlpt/health", methods=["GET"])
def health():
    """Health check for the MLPT service."""
    return jsonify({
        "status": "ok",
        "model_loaded": _predictor is not None and _predictor.loaded,
        "service": "MLPT Antigenic Peptide Predictor",
    })


@mlpt_bp.route("/api/mlpt/classes", methods=["GET"])
def classes():
    """Return the prediction class labels."""
    class_descriptions = {
        "Cancer Antigenic Peptides": "Peptides identified as cancer-specific T-cell epitopes",
        "Inactive Peptides-Lung Breast": "Peptides inactive against lung and breast cancer cell lines",
        "Moderately Active-Lung Breast": "Peptides with moderate activity against lung/breast cancer",
        "Natural Peptide": "Naturally occurring peptides from biological sources",
        "Non-Natural Peptide": "Synthetically designed or modified peptides",
        "Very Active-Lung Breast": "Peptides highly active against lung/breast cancer cell lines",
    }
    return jsonify({
        "num_classes": NUM_CLASSES,
        "classes": [
            {"index": i, "name": name, "description": class_descriptions.get(name, "")}
            for i, name in enumerate(CLASS_NAMES)
        ],
    })

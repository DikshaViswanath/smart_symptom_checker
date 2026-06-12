"""
Flask REST API for the Smart Symptom Checker backend.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS

from predictor import ModelLoadError, get_predictor
from severity_checker import get_severity
from specialist_mapper import get_specialist

app = Flask(__name__)
# Allow Live Server and other local dev origins (localhost vs 127.0.0.1, varying ports).
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=["Content-Type"],
)


def _load_model_on_startup() -> None:
    try:
        get_predictor()
    except ModelLoadError:
        # Endpoints will return a clear error until train_model.py is run.
        pass


@app.route("/", methods=["GET"])
def health():
    try:
        predictor = get_predictor()
        model_ready = predictor.is_loaded
    except ModelLoadError:
        model_ready = False

    return jsonify(
        {
            "status": "ok",
            "service": "Smart Symptom Checker API",
            "model_loaded": model_ready,
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 400

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON payload."}), 400

    if "symptoms" not in payload:
        return jsonify({"error": "Missing required field: 'symptoms'."}), 400

    symptoms = payload.get("symptoms")
    if not isinstance(symptoms, list):
        return jsonify({"error": "'symptoms' must be a list of strings."}), 400

    if len(symptoms) == 0:
        return jsonify({"error": "Symptoms list cannot be empty."}), 400

    if not all(isinstance(s, str) for s in symptoms):
        return jsonify({"error": "Each symptom must be a string."}), 400

    if not any(isinstance(s, str) and s.strip() for s in symptoms):
        return jsonify({"error": "At least one non-empty symptom is required."}), 400

    try:
        predictor = get_predictor()
        normalized = predictor.normalize_symptoms(symptoms)
        if not normalized:
            return jsonify({"error": "At least one valid symptom is required."}), 400

        predicted_disease = predictor.predict(normalized)
        specialist = get_specialist(predicted_disease)
        severity = get_severity(len(normalized))

        return jsonify(
            {
                "predicted_disease": str(predicted_disease),
                "specialist": str(specialist),
                "severity": str(severity),
            }
        ), 200
    except ModelLoadError as exc:
        return jsonify(
            {
                "error": str(exc),
                "hint": "Run 'python train_model.py' to generate symptom_model.pkl.",
            }
        ), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "An unexpected error occurred during prediction."}), 500


_load_model_on_startup()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

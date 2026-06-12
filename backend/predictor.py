"""Load the trained model and predict disease from symptom lists."""

import pickle
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "symptom_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"

SYMPTOM_CANONICAL = {
    "high fever": "fever",
    "mild fever": "fever",
    "high temperature": "fever",
    "temperature": "fever",
    "chronic cough": "cough",
    "breathing difficulty": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "breathlessness": "shortness of breath",
    "body pain": "body ache",
    "muscle pain": "body ache",
    "aches": "body ache",
    "head ache": "headache",
    "head pain": "headache",
    "vertigo": "dizziness",
    "lightheaded": "dizziness",
    "chest pressure": "chest discomfort",
}


class ModelLoadError(Exception):
    """Raised when model or metadata files cannot be loaded."""


class Predictor:
    def __init__(self) -> None:
        self._model = None
        self._feature_columns: list[str] = []
        self._loaded = False

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise ModelLoadError(
                f"Model file not found at {MODEL_PATH}. Run: python train_model.py"
            )
        if not METADATA_PATH.exists():
            raise ModelLoadError(
                f"Metadata file not found at {METADATA_PATH}. Run: python train_model.py"
            )
        try:
            with open(MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            with open(METADATA_PATH, "rb") as f:
                metadata = pickle.load(f)
            self._feature_columns = metadata.get("feature_columns", [])
            if not self._feature_columns:
                raise ModelLoadError("Model metadata is missing feature_columns.")
            self._loaded = True
        except (pickle.UnpicklingError, EOFError, KeyError, TypeError) as exc:
            raise ModelLoadError(f"Failed to load model files: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @staticmethod
    def normalize_symptoms(symptoms: list) -> list[str]:
        normalized = []
        seen = set()
        for item in symptoms:
            if not isinstance(item, str):
                continue
            value = SYMPTOM_CANONICAL.get(item.strip().lower(), item.strip().lower())
            if value and value not in seen:
                seen.add(value)
                normalized.append(value)
        return normalized

    def _symptoms_to_vector(self, symptoms: list[str]) -> list[int]:
        symptom_set = set(symptoms)
        return [1 if col in symptom_set else 0 for col in self._feature_columns]

    def predict(self, symptoms: list) -> str:
        if not self._loaded or self._model is None:
            raise ModelLoadError("Predictor is not loaded. Call load() first.")

        normalized = self.normalize_symptoms(symptoms)
        if not normalized:
            raise ValueError("At least one valid symptom is required.")

        vector = self._symptoms_to_vector(normalized)
        features = pd.DataFrame([vector], columns=self._feature_columns)
        prediction = self._model.predict(features)[0]
        return str(prediction)


_predictor_instance: Predictor | None = None
_model_mtime: float | None = None


def get_predictor() -> Predictor:
    global _predictor_instance, _model_mtime
    current_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    if (
        _predictor_instance is None
        or _model_mtime is None
        or current_mtime != _model_mtime
    ):
        _predictor_instance = Predictor()
        _predictor_instance.load()
        _model_mtime = current_mtime
    return _predictor_instance

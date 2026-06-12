"""
Train a DecisionTreeClassifier on disease-symptom data and save symptom_model.pkl.
Generates data/Training.csv from disease_symptom_map.json if the CSV is missing.
"""

import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MAP_PATH = DATA_DIR / "disease_symptom_map.json"
TRAINING_CSV = DATA_DIR / "Training.csv"
MODEL_PATH = BASE_DIR / "symptom_model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.pkl"
RANDOM_SEED = 42

# Map variant symptom labels to canonical names used in the feature matrix.
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


def _normalize_symptom(symptom: str) -> str:
    value = symptom.strip().lower()
    return SYMPTOM_CANONICAL.get(value, value)


def load_disease_map() -> dict[str, list[str]]:
    with open(MAP_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        disease: [_normalize_symptom(s) for s in symptoms]
        for disease, symptoms in raw.items()
    }


def build_training_csv(disease_map: dict[str, list[str]], rows_per_disease: int = 80) -> pd.DataFrame:
    """Build binary symptom matrix rows for each disease."""
    all_symptoms = sorted({s for symptoms in disease_map.values() for s in symptoms})
    records: list[dict] = []

    rng = random.Random(RANDOM_SEED)
    for disease, symptoms in disease_map.items():
        for _ in range(rows_per_disease // 2):
            row = {symptom: 0 for symptom in all_symptoms}
            # Full symptom profile (typical clinical presentation)
            for symptom in symptoms[: max(2, len(symptoms) // 2)]:
                row[symptom] = 1
            optional = symptoms[max(2, len(symptoms) // 2) :]
            for symptom in optional:
                if rng.random() < 0.65:
                    row[symptom] = 1
            noise_pool = [s for s in all_symptoms if s not in symptoms]
            if noise_pool and rng.random() < 0.1:
                row[rng.choice(noise_pool)] = 1
            row["prognosis"] = disease
            records.append(row)

        # Sparse rows (2-4 symptoms) — matches typical frontend/API input
        for _ in range(rows_per_disease // 2):
            row = {symptom: 0 for symptom in all_symptoms}
            count = rng.randint(2, min(4, len(symptoms)))
            chosen = rng.sample(symptoms, k=count)
            for symptom in chosen:
                row[symptom] = 1
            row["prognosis"] = disease
            records.append(row)

    # Reinforce frequent frontend symptom pairs
    signature_rows = [
        ("Flu", ["fever", "cough"]),
        ("Flu", ["fever", "cough", "fatigue"]),
        ("Common Cold", ["cough", "runny nose"]),
        ("Pneumonia", ["fever", "cough", "chest pain"]),
    ]
    for disease, combo in signature_rows:
        row = {symptom: 0 for symptom in all_symptoms}
        for symptom in combo:
            if symptom in row:
                row[symptom] = 1
        row["prognosis"] = disease
        for _ in range(30):
            records.append(row.copy())

    return pd.DataFrame(records)


def ensure_training_csv() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TRAINING_CSV.exists():
        return pd.read_csv(TRAINING_CSV)
    disease_map = load_disease_map()
    df = build_training_csv(disease_map)
    df.to_csv(TRAINING_CSV, index=False)
    print(f"Created {TRAINING_CSV} with {len(df)} rows.")
    return df


def train_and_save() -> None:
    df = ensure_training_csv()
    if "prognosis" not in df.columns:
        raise ValueError("Training.csv must contain a 'prognosis' column.")

    feature_columns = [col for col in df.columns if col != "prognosis"]
    X = df[feature_columns].astype(int)
    y = df["prognosis"]

    model = DecisionTreeClassifier(
        random_state=RANDOM_SEED,
        max_depth=12,
        min_samples_leaf=2,
    )
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "feature_columns": feature_columns,
        "diseases": sorted(y.unique().tolist()),
    }
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    train_score = model.score(X, y)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")
    print(f"Features: {len(feature_columns)} | Diseases: {len(metadata['diseases'])}")
    print(f"Training accuracy: {train_score:.4f}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    train_and_save()

"""Prediction helpers shared by the API and Streamlit app."""

from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/fraud_model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


def predict_transaction(transaction: dict):
    model = load_model()
    frame = pd.DataFrame([transaction])
    probability = float(model.predict_proba(frame)[0, 1])

    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "fraud_probability": probability,
        "risk": risk,
    }

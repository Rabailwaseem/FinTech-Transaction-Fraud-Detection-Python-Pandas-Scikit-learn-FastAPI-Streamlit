"""Train and evaluate fraud detection models."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/transactions.csv")
MODEL_PATH = Path("models/fraud_model.joblib")
METRICS_PATH = Path("reports/metrics.json")

TARGET = "is_fraud"
NUMERIC_FEATURES = [
    "amount",
    "hour",
    "country_mismatch",
    "device_trust",
    "transactions_24h",
    "card_age_days",
    "account_age_days",
    "previous_chargebacks",
]
CATEGORICAL_FEATURES = ["merchant_category"]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        ),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/transactions.csv not found. Run `python -m src.generate_data` first."
        )

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = pipeline

        print(f"\n{name}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1:        {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        print(metrics["classification_report"])

    selected_name = max(results, key=lambda name: results[name]["pr_auc"])
    selected_model = fitted_models[selected_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH = METRICS_PATH
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected_model, MODEL_PATH)

    serializable_results = {
        name: {k: v for k, v in metrics.items() if k != "classification_report"}
        for name, metrics in results.items()
    }
    report = {
        "selected_model": selected_name,
        "test_size": 0.20,
        "random_state": 42,
        "results": serializable_results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"\nSelected model: {selected_name}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {REPORT_PATH}")


if __name__ == "__main__":
    main()

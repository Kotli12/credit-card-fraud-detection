"""
predict.py — Run inference on new transaction data using saved models.

Usage:
    python src/predict.py --input data/new_transactions.csv --threshold 0.3
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fraud_detection import engineer_features


def load_models(model_dir: str = "models"):
    model = joblib.load(f"{model_dir}/xgboost_fraud.pkl")
    iso = joblib.load(f"{model_dir}/isolation_forest.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    return model, iso, scaler


def predict(input_path: str, threshold: float = 0.5, model_dir: str = "models"):
    model, iso_model, scaler = load_models(model_dir)

    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df)} transactions.")

    df = engineer_features(df)
    feature_cols = [c for c in df.columns if c not in ["Class", "Time"]]
    X = df[feature_cols].copy()

    # Scale
    for col in ["Amount", "log_amount", "amount_zscore", "pca_magnitude"]:
        if col in X.columns:
            X[col] = scaler.transform(X[[col]])

    # Isolation Forest
    X["iso_score"] = iso_model.decision_function(X)
    X["iso_flag"] = (iso_model.predict(X.drop("iso_score", axis=1)) == -1).astype(int)

    # Predict
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    df["fraud_probability"] = probs
    df["fraud_prediction"] = preds

    flagged = df[df["fraud_prediction"] == 1]
    print(f"[INFO] Transactions flagged as fraud: {len(flagged)} / {len(df)}")

    output_path = "outputs/predictions.csv"
    df[["fraud_probability", "fraud_prediction"]].to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to transaction CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()

    predict(args.input, args.threshold, args.model_dir)

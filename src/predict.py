"""
predict.py — Batch inference on new transaction data.

Loads the three serialised artefacts produced by fraud_detection.py
(XGBoost model, Isolation Forest, StandardScaler), applies the same
feature engineering and scaling pipeline used at training time, and
writes a CSV with fraud probability scores and binary predictions.

Usage:
    python src/predict.py --input data/new_transactions.csv --threshold 0.3
    python src/predict.py --input data/new_transactions.csv --threshold 0.3 \\
        --model_dir models/ --output outputs/predictions.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fraud_detection import engineer_features

SCALE_COLS = ["Amount", "log_amount", "amount_zscore", "pca_magnitude"]


def load_models(model_dir: str = "models") -> tuple:
    """Load the three serialised model artefacts from disk.

    Args:
        model_dir: Directory containing xgboost_fraud.pkl,
                   isolation_forest.pkl, and scaler.pkl.

    Returns:
        Tuple of (xgb_model, iso_model, scaler).

    Raises:
        FileNotFoundError: If any artefact is missing from model_dir.
    """
    model  = joblib.load(os.path.join(model_dir, "xgboost_fraud.pkl"))
    iso    = joblib.load(os.path.join(model_dir, "isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    print(f"[INFO] Loaded model artefacts from {model_dir}/")
    return model, iso, scaler


def predict(
    input_path: str,
    threshold: float = 0.5,
    model_dir: str = "models",
    output_path: str = "outputs/predictions.csv",
) -> pd.DataFrame:
    """Run fraud detection inference on a batch of transactions.

    Applies the same preprocessing steps used at training time, in the
    same order:
        1. Feature engineering (log_amount, hour_of_day, pca_magnitude, etc.)
        2. StandardScaler on the four numeric features
        3. Isolation Forest anomaly scores (iso_score, iso_flag)
        4. XGBoost probability estimation
        5. Binary prediction at the specified threshold

    Args:
        input_path:  Path to CSV file with transaction data. Must contain
                     the same columns as creditcard.csv (Time, Amount, V1–V28).
        threshold:   Decision threshold for binary classification.
                     Lower values increase recall at the cost of precision.
                     Use the threshold printed at the end of training.
        model_dir:   Directory containing the serialised model artefacts.
        output_path: Path to write the results CSV.

    Returns:
        Input DataFrame with two appended columns:
            fraud_probability — model's estimated probability of fraud [0, 1]
            fraud_prediction  — 1 if fraud_probability >= threshold, else 0
    """
    model, iso_model, scaler = load_models(model_dir)

    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df):,} transactions from {input_path}")

    df_eng = engineer_features(df)
    feature_cols = [c for c in df_eng.columns if c not in ["Class", "Time"]]
    X = df_eng[feature_cols].copy()

    X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])

    X["iso_score"] = iso_model.decision_function(X)
    X["iso_flag"]  = (iso_model.predict(X.drop("iso_score", axis=1)) == -1).astype(int)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    df["fraud_probability"] = probs
    df["fraud_prediction"]  = preds

    n_flagged = int(preds.sum())
    print(f"[INFO] Flagged {n_flagged:,} / {len(df):,} transactions as fraud "
          f"({n_flagged / len(df) * 100:.2f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df[["fraud_probability", "fraud_prediction"]].to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved to {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fraud detection inference on a batch of transactions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to transaction CSV (same column format as creditcard.csv)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold. Use the value printed at the end of training.",
    )
    parser.add_argument(
        "--model_dir", type=str, default="models",
        help="Directory containing serialised model artefacts",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/predictions.csv",
        help="Path to write the predictions CSV",
    )
    args = parser.parse_args()
    predict(args.input, args.threshold, args.model_dir, args.output)

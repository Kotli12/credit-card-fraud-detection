"""
Credit Card Fraud Detection Pipeline
Combines Isolation Forest (anomaly detection) + XGBoost (supervised classification)
with SMOTE oversampling to handle severe class imbalance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load the Kaggle Credit Card Fraud dataset."""
    print(f"[INFO] Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Fraud rate: {df['Class'].mean() * 100:.4f}%")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom feature engineering to help discriminate fraud from legitimate transactions.
    """
    df = df.copy()

    # Transaction amount features
    df["log_amount"] = np.log1p(df["Amount"])
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    # Time-based features
    df["hour_of_day"] = (df["Time"] % 86400) // 3600
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)

    # High-value transaction flag (top 5%)
    threshold = df["Amount"].quantile(0.95)
    df["high_value_txn"] = (df["Amount"] > threshold).astype(int)

    # Interaction: high value at night
    df["high_value_night"] = df["high_value_txn"] * df["is_night"]

    # PCA component magnitude (proxy for deviation from normal)
    pca_cols = [c for c in df.columns if c.startswith("V")]
    df["pca_magnitude"] = np.sqrt((df[pca_cols] ** 2).sum(axis=1))

    print(f"[INFO] Features after engineering: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────
# 3. ISOLATION FOREST ANOMALY SCORES
# ─────────────────────────────────────────────

def add_isolation_forest_scores(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    contamination: float = 0.002
) -> tuple:
    """
    Fit Isolation Forest on training data and add anomaly scores as a feature.
    A lower (more negative) score = more anomalous.
    """
    print("[INFO] Fitting Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["iso_score"] = iso.decision_function(X_train)
    X_test["iso_score"] = iso.decision_function(X_test)

    # Binary anomaly flag
    X_train["iso_flag"] = (iso.predict(X_train.drop("iso_score", axis=1)) == -1).astype(int)
    X_test["iso_flag"] = (iso.predict(X_test.drop("iso_score", axis=1)) == -1).astype(int)

    print(f"[INFO] Isolation Forest flagged {X_train['iso_flag'].sum()} anomalies in training set.")
    return X_train, X_test, iso


# ─────────────────────────────────────────────
# 4. SMOTE + XGBOOST TRAINING
# ─────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> XGBClassifier:
    """
    Apply SMOTE then train XGBoost classifier.
    """
    print(f"[INFO] Class distribution before SMOTE: {dict(y_train.value_counts())}")

    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"[INFO] Class distribution after SMOTE:  {dict(pd.Series(y_resampled).value_counts())}")

    scale_pos_weight = (y_resampled == 0).sum() / (y_resampled == 1).sum()

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1,
        early_stopping_rounds=30
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.1, stratify=y_resampled, random_state=42
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    print(f"[INFO] Best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────
# 5. THRESHOLD OPTIMISATION (maximise recall)
# ─────────────────────────────────────────────

def find_optimal_threshold(model, X_test, y_test, min_precision=0.10):
    """
    Find probability threshold that maximises recall while keeping
    precision above a minimum acceptable level.
    """
    probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    best_threshold = 0.5
    best_recall = 0.0

    for p, r, t in zip(precisions, recalls, thresholds):
        if p >= min_precision and r > best_recall:
            best_recall = r
            best_threshold = t

    print(f"[INFO] Optimal threshold: {best_threshold:.4f} → Recall: {best_recall:.4f}")
    return best_threshold, probs


# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test, threshold=0.5):
    """Full evaluation suite."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_test, preds, target_names=["Legitimate", "Fraud"]))

    roc_auc = roc_auc_score(y_test, probs)
    avg_prec = average_precision_score(y_test, probs)
    print(f"ROC-AUC Score:            {roc_auc:.4f}")
    print(f"Average Precision (AUPR): {avg_prec:.4f}")
    print("=" * 55 + "\n")

    return probs, preds


# ─────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_test, preds, save_path="outputs/confusion_matrix.png"):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_roc_pr_curves(y_test, probs, save_path="outputs/roc_pr_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    axes[0].plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    # Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    axes[1].plot(recalls, precisions, color="#16A34A", lw=2, label=f"AP = {ap:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_feature_importance(model, feature_names, save_path="outputs/feature_importance.png", top_n=20):
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_names).nlargest(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    fi.sort_values().plot(kind="barh", color="#7C3AED", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_fraud_patterns(df, save_path="outputs/fraud_patterns.png"):
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(n=min(5000, len(df[df["Class"] == 0])), random_state=42)
    combined = pd.concat([legit, fraud])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Amount distribution
    axes[0].hist(np.log1p(legit["Amount"]), bins=60, alpha=0.6, label="Legitimate", color="#2563EB")
    axes[0].hist(np.log1p(fraud["Amount"]), bins=60, alpha=0.7, label="Fraud", color="#DC2626")
    axes[0].set_xlabel("log(Amount + 1)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Transaction Amount Distribution")
    axes[0].legend()

    # Hour of day
    combined["hour_of_day"] = (combined["Time"] % 86400) // 3600
    fraud_hours = (fraud["Time"] % 86400) // 3600
    legit_hours = (legit["Time"] % 86400) // 3600

    axes[1].hist(legit_hours, bins=24, alpha=0.6, label="Legitimate", color="#2563EB", density=True)
    axes[1].hist(fraud_hours, bins=24, alpha=0.7, label="Fraud", color="#DC2626", density=True)
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Fraud by Hour of Day")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(data_path: str = "data/creditcard.csv"):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load & engineer features
    df = load_data(data_path)
    df = engineer_features(df)
    plot_fraud_patterns(df)

    # Split
    feature_cols = [c for c in df.columns if c not in ["Class", "Time"]]
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale Amount
    scaler = StandardScaler()
    for col in ["Amount", "log_amount", "amount_zscore", "pca_magnitude"]:
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col] = scaler.transform(X_test[[col]])

    # Isolation Forest scores
    X_train, X_test, iso_model = add_isolation_forest_scores(X_train, X_test)

    # Train XGBoost
    model = train_model(X_train, y_train)

    # Optimise threshold for high recall
    threshold, probs = find_optimal_threshold(model, X_test, y_test, min_precision=0.10)

    # Evaluate
    probs, preds = evaluate(model, X_test, y_test, threshold)

    # Plots
    plot_confusion_matrix(y_test, preds)
    plot_roc_pr_curves(y_test, probs)
    plot_feature_importance(model, X_train.columns.tolist())

    # Save models
    joblib.dump(model, "models/xgboost_fraud.pkl")
    joblib.dump(iso_model, "models/isolation_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("[INFO] Models saved to /models/")

    return model, iso_model, scaler, threshold


if __name__ == "__main__":
    run_pipeline()

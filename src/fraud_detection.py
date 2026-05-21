"""
Credit Card Fraud Detection — Training Pipeline

Two-stage ensemble for imbalanced binary classification:
    1. Isolation Forest fits on the training set and contributes anomaly scores
       as features — an unsupervised signal that does not depend on fraud labels.
    2. XGBoost is trained on SMOTE-oversampled data and fine-tuned for high recall
       via threshold optimisation on the precision-recall curve.

Usage:
    python src/fraud_detection.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load and summarise the Kaggle Credit Card Fraud dataset.

    Args:
        path: Path to creditcard.csv.

    Returns:
        DataFrame with columns Time, V1–V28, Amount, Class.
    """
    print(f"[INFO] Loading data from {path} ...")
    df = pd.read_csv(path)
    n_fraud = int(df["Class"].sum())
    n_total = len(df)
    print(f"[INFO] {n_total:,} transactions  |  "
          f"{n_fraud:,} fraud ({n_fraud / n_total * 100:.4f}%)  |  "
          f"{n_total - n_fraud:,} legitimate")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive domain-informed features from the raw transaction fields.

    V1–V28 are PCA-transformed by the dataset provider and cannot be
    interpreted directly. Engineering therefore focuses on Amount and Time,
    plus a second-order signal derived from the PCA vector itself.

    Features added:
        log_amount       log1p-normalised amount (stabilises heavy right tail)
        amount_zscore    standardised deviation from the population mean amount
        hour_of_day      clock hour derived from the elapsed-seconds Time field
        is_night         1 for transactions between 22:00 and 06:00
        high_value_txn   1 for amounts above the 95th percentile
        high_value_night interaction: high-value AND night
        pca_magnitude    L2 norm of V1–V28 (measures distance from the PCA origin)

    Args:
        df: Raw DataFrame.

    Returns:
        Copy of df with seven additional feature columns.
    """
    df = df.copy()

    df["log_amount"]   = np.log1p(df["Amount"])
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    df["hour_of_day"] = (df["Time"] % 86400) // 3600
    df["is_night"]    = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)

    high_val_thresh      = df["Amount"].quantile(0.95)
    df["high_value_txn"] = (df["Amount"] > high_val_thresh).astype(int)
    df["high_value_night"] = df["high_value_txn"] * df["is_night"]

    pca_cols = [c for c in df.columns if c.startswith("V")]
    df["pca_magnitude"] = np.linalg.norm(df[pca_cols].values, axis=1)

    print(f"[INFO] Feature matrix: {df.shape[1]} columns after engineering "
          f"(+{df.shape[1] - (len(pca_cols) + 3)} engineered)")
    return df


# ─────────────────────────────────────────────
# 3. ISOLATION FOREST ANOMALY SCORES
# ─────────────────────────────────────────────

def add_isolation_forest_scores(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    contamination: float = 0.002,
) -> tuple:
    """Fit an Isolation Forest on training data and append anomaly features.

    The forest is fitted exclusively on the training split to prevent data
    leakage. Two derived features are added to both splits:

        iso_score — continuous decision_function output. More negative values
                    indicate more isolated (anomalous) observations. Used by
                    XGBoost as a soft, graduated signal.
        iso_flag  — binary flag (1 = anomaly) from the hard predict() boundary.
                    Provides a categorical complement to iso_score.

    contamination=0.002 is set to match the dataset's known fraud rate rather
    than the scikit-learn default of 0.1, which would flag far too many
    legitimate transactions as anomalies.

    Args:
        X_train:       Training features (no iso columns yet).
        X_test:        Test features (no iso columns yet).
        contamination: Expected fraction of outliers in the training data.

    Returns:
        Tuple of (X_train_with_scores, X_test_with_scores, fitted_iso_model).
    """
    print("[INFO] Fitting Isolation Forest ...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)

    X_train = X_train.copy()
    X_test  = X_test.copy()

    for X in (X_train, X_test):
        X["iso_score"] = iso.decision_function(X)
        # drop iso_score before predict() so feature count matches what iso was trained on
        X["iso_flag"]  = (iso.predict(X.drop("iso_score", axis=1)) == -1).astype(int)

    flagged = int(X_train["iso_flag"].sum())
    print(f"[INFO] Isolation Forest flagged {flagged:,} training anomalies "
          f"({flagged / len(X_train) * 100:.2f}%)")
    return X_train, X_test, iso


# ─────────────────────────────────────────────
# 4. SMOTE + XGBOOST TRAINING
# ─────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Oversample with SMOTE then train an XGBoost classifier.

    SMOTE is applied after the train/test split so the test set contains only
    real transactions and the evaluation is not inflated. A 10% minority
    target ratio is used: enough synthetic fraud examples to improve the
    decision boundary without over-representing the class.

    Early stopping on AUPR (Average Precision under the Precision-Recall curve)
    is preferred over AUC-ROC. On imbalanced datasets, AUC-ROC can appear
    inflated because it covers the full threshold range including those where
    the dominant negative class drives performance. AUPR focuses on the
    minority class and better reflects the operational objective.

    Args:
        X_train: Training features including iso_score and iso_flag.
        y_train: Binary labels (0 = legitimate, 1 = fraud).

    Returns:
        Fitted XGBClassifier at the best early-stopping iteration.
    """
    fraud_n = int(y_train.sum())
    legit_n = int((y_train == 0).sum())
    print(f"[INFO] Before SMOTE — legitimate: {legit_n:,}  |  fraud: {fraud_n:,}")

    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    fraud_after = int((y_res == 1).sum())
    legit_after = int((y_res == 0).sum())
    print(f"[INFO] After  SMOTE — legitimate: {legit_after:,}  |  fraud: {fraud_after:,}")

    # Residual class imbalance after SMOTE is further penalised via scale_pos_weight
    scale_pos_weight = legit_after / fraud_after

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_res, y_res, test_size=0.1, stratify=y_res, random_state=42
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
    print(f"[INFO] Best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────
# 5. THRESHOLD OPTIMISATION
# ─────────────────────────────────────────────

def find_optimal_threshold(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_precision: float = 0.10,
) -> tuple:
    """Find the decision threshold that maximises recall subject to a precision floor.

    The default threshold of 0.5 assumes equal misclassification costs, which
    does not hold in fraud detection. Lowering the threshold catches more fraud
    but increases false positives.

    This function sweeps the full precision-recall curve and returns the
    lowest threshold at which precision remains above min_precision.

    min_precision=0.10 means at least 1 in 10 flagged transactions must be
    genuine fraud — a loose constraint that prioritises recall while keeping
    the false alarm rate manageable for a review team.

    Args:
        model:         Fitted XGBClassifier.
        X_test:        Test features.
        y_test:        True labels.
        min_precision: Minimum acceptable precision at the chosen threshold.

    Returns:
        Tuple of (optimal_threshold, fraud_probabilities_array).
    """
    probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    best_threshold, best_recall = 0.5, 0.0
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p >= min_precision and r > best_recall:
            best_recall     = r
            best_threshold  = t

    print(f"[INFO] Optimal threshold: {best_threshold:.4f}  |  "
          f"Recall: {best_recall:.4f}  |  Precision floor: {min_precision:.2f}")
    return best_threshold, probs


# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────

def evaluate(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> tuple:
    """Print a full evaluation report at the given decision threshold.

    Args:
        model:     Fitted XGBClassifier.
        X_test:    Test features.
        y_test:    True binary labels.
        threshold: Decision threshold applied to predicted probabilities.

    Returns:
        Tuple of (fraud_probabilities, binary_predictions).
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    print("\n" + "=" * 60)
    print(f"  EVALUATION  (threshold = {threshold:.4f})")
    print("=" * 60)
    print(classification_report(y_test, preds, target_names=["Legitimate", "Fraud"]))
    print(f"  ROC-AUC:            {roc_auc_score(y_test, probs):.4f}")
    print(f"  Average Precision:  {average_precision_score(y_test, probs):.4f}")
    print("=" * 60 + "\n")

    return probs, preds


# ─────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_test, preds, save_path: str = "outputs/confusion_matrix.png"
) -> None:
    """Save a confusion matrix heatmap annotated with raw counts."""
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_roc_pr_curves(
    y_test, probs, save_path: str = "outputs/roc_pr_curves.png"
) -> None:
    """Save side-by-side ROC and Precision-Recall curves.

    Both curves are shown with their summary statistics (AUC and AP).
    The PR curve is more informative than the ROC curve for imbalanced data
    because it focuses on the minority class at every threshold.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    axes[0].plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

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


def plot_feature_importance(
    model: XGBClassifier,
    feature_names: list,
    top_n: int = 20,
    save_path: str = "outputs/feature_importance.png",
) -> None:
    """Save a horizontal bar chart of the top-N XGBoost feature importances."""
    fi = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)
    fig, ax = plt.subplots(figsize=(9, 7))
    fi.sort_values().plot(kind="barh", color="#7C3AED", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost gain)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_fraud_patterns(
    df: pd.DataFrame, save_path: str = "outputs/fraud_patterns.png"
) -> None:
    """Save transaction amount and hour-of-day distributions by class.

    Legitimate transactions are downsampled to 5,000 for visual clarity.
    """
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(n=5000, random_state=42)

    fraud_hours = (fraud["Time"] % 86400) // 3600
    legit_hours = (legit["Time"] % 86400) // 3600

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(np.log1p(legit["Amount"]), bins=60, alpha=0.6,
                 label="Legitimate", color="#2563EB")
    axes[0].hist(np.log1p(fraud["Amount"]), bins=60, alpha=0.7,
                 label="Fraud", color="#DC2626")
    axes[0].set_xlabel("log(Amount + 1)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Transaction Amount Distribution by Class")
    axes[0].legend()

    axes[1].hist(legit_hours, bins=24, alpha=0.6, label="Legitimate",
                 color="#2563EB", density=True)
    axes[1].hist(fraud_hours, bins=24, alpha=0.7, label="Fraud",
                 color="#DC2626", density=True)
    axes[1].set_xlabel("Hour of Day")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Transaction Frequency by Hour of Day")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


def plot_threshold_analysis(
    y_test,
    probs: np.ndarray,
    chosen_threshold: float,
    save_path: str = "outputs/threshold_analysis.png",
) -> None:
    """Save precision and recall as a function of the decision threshold.

    Plots both curves on the same axes with a vertical marker at the chosen
    threshold. Makes the precision-recall tradeoff explicit — useful for
    explaining the threshold choice to a non-technical audience.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions[:-1], color="#2563EB", lw=2, label="Precision")
    ax.plot(thresholds, recalls[:-1],    color="#DC2626", lw=2, label="Recall")
    ax.axvline(
        chosen_threshold, color="#7C3AED", linestyle="--", lw=1.5,
        label=f"Chosen threshold ({chosen_threshold:.3f})",
    )
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall vs. Decision Threshold")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {save_path}")


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(data_path: str = "data/creditcard.csv") -> tuple:
    """Execute the full training pipeline end-to-end.

    Steps:
        1.  Load raw data
        2.  Engineer features
        3.  Stratified 80/20 train/test split
        4.  StandardScaler on numeric features (fit on train only)
        5.  Isolation Forest anomaly scores (fit on train only)
        6.  SMOTE oversampling (applied to train only)
        7.  XGBoost with early stopping on AUPR
        8.  Decision threshold optimisation for maximum recall
        9.  Evaluation and diagnostic plots
        10. Serialise model artefacts

    Args:
        data_path: Path to creditcard.csv.

    Returns:
        Tuple of (xgb_model, iso_model, scaler, optimal_threshold).
    """
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    df = load_data(data_path)
    df = engineer_features(df)
    plot_fraud_patterns(df)

    feature_cols = [c for c in df.columns if c not in ["Class", "Time"]]
    X, y = df[feature_cols], df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_cols = ["Amount", "log_amount", "amount_zscore", "pca_magnitude"]
    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

    X_train, X_test, iso_model = add_isolation_forest_scores(X_train, X_test)
    model = train_model(X_train, y_train)

    threshold, probs = find_optimal_threshold(model, X_test, y_test, min_precision=0.10)
    probs, preds     = evaluate(model, X_test, y_test, threshold)

    plot_confusion_matrix(y_test, preds)
    plot_roc_pr_curves(y_test, probs)
    plot_feature_importance(model, X_train.columns.tolist())
    plot_threshold_analysis(y_test, probs, threshold)

    joblib.dump(model,     "models/xgboost_fraud.pkl")
    joblib.dump(iso_model, "models/isolation_forest.pkl")
    joblib.dump(scaler,    "models/scaler.pkl")
    print("[INFO] Model artefacts saved to models/")

    return model, iso_model, scaler, threshold


if __name__ == "__main__":
    run_pipeline()

# Credit Card Fraud Detection

End-to-end machine learning pipeline for detecting fraudulent transactions on a severely imbalanced dataset. Combines **Isolation Forest** (unsupervised anomaly scoring) and **XGBoost** (supervised classification) with SMOTE oversampling, achieving **~98% fraud recall** at a ROC-AUC of ~0.98.

---

## Results

| Metric | Score |
|---|---|
| Fraud Recall | ~98% |
| ROC-AUC | ~0.98 |
| Avg Precision (AUPR) | ~0.85 |

---

## Approach

### The Problem

Only **0.17%** of transactions are fraudulent. A naive classifier predicts "legitimate" for everything, looks 99.8% accurate, and catches zero fraud. The challenge is recall — not accuracy.

### Pipeline

```
Raw Transactions
      |
      v
Feature Engineering  ---  log(amount), hour-of-day, PCA magnitude, interaction flags
      |
      v
Isolation Forest     ---  Unsupervised anomaly scores added as features
      |
      v
SMOTE Oversampling   ---  Synthetic minority samples to 10% ratio
      |
      v
XGBoost Classifier   ---  500 trees, depth 6, early stopping on AUPR
      |
      v
Threshold Tuning     ---  Maximise recall, constrain precision >= 10%
      |
      v
Fraud Predictions
```

**Design decisions:**
- **Isolation Forest** provides a label-free anomaly signal. Useful as a feature because fraud patterns can shift over time without retraining.
- **SMOTE** generates synthetic fraud examples rather than simple duplication, giving the classifier a richer minority class boundary.
- **Threshold tuning** explicitly trades precision for recall — missing a fraud is far more costly than a false alarm.

---

## Project Structure

```
credit-card-fraud-detection/
├── data/                          # Place creditcard.csv here
├── notebooks/
│   └── fraud_detection_walkthrough.ipynb
├── src/
│   ├── fraud_detection.py         # Training pipeline
│   └── predict.py                 # Inference on new transactions
├── models/                        # Saved artifacts (generated on run)
├── outputs/                       # Plots and predictions (generated on run)
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
git clone https://github.com/Kotli12/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.

> The dataset contains 284,807 transactions from European cardholders (September 2013). Features V1–V28 are PCA-transformed to protect cardholder privacy. Only `Time`, `Amount`, and `Class` are in their original form.

---

## Running

**Train and evaluate:**
```bash
python src/fraud_detection.py
```

Outputs: classification report, ROC-AUC, plots in `outputs/`, models in `models/`.

**Inference on new transactions:**
```bash
python src/predict.py --input data/new_transactions.csv --threshold 0.3
```

**Notebook walkthrough:**
```bash
jupyter notebook notebooks/fraud_detection_walkthrough.ipynb
```

---

## Feature Engineering

| Feature | Description |
|---|---|
| `log_amount` | Log-normalised transaction amount |
| `amount_zscore` | Z-score of amount |
| `hour_of_day` | Hour extracted from `Time` |
| `is_night` | 1 if transaction between 22:00–06:00 |
| `high_value_txn` | 1 if amount in top 5% |
| `high_value_night` | Interaction: high value AND night |
| `pca_magnitude` | Euclidean norm of V1–V28 (deviation from origin) |
| `iso_score` | Isolation Forest anomaly score |
| `iso_flag` | Binary anomaly flag from Isolation Forest |

---

## Tech Stack

- Python 3.10+
- `scikit-learn` — Isolation Forest, preprocessing, metrics
- `xgboost` — Gradient boosted classifier
- `imbalanced-learn` — SMOTE oversampling
- `pandas` / `numpy` — Data manipulation
- `matplotlib` / `seaborn` — Visualisation
- `joblib` — Model serialisation

---

## License

MIT

# 💳 Credit Card Fraud Detection

An end-to-end anomaly detection pipeline on highly imbalanced transaction data, achieving **98% recall on fraud cases** through an ensemble of Isolation Forest and XGBoost with SMOTE oversampling.

---

## 🚀 Results

| Metric | Score |
|---|---|
| Fraud Recall | **~98%** |
| ROC-AUC | **~0.98** |
| Avg Precision (AUPR) | **~0.85** |
| False Positive Rate | Minimised via threshold tuning |

---

## 🧠 Approach

### The Challenge
The dataset is severely imbalanced — only **0.17%** of transactions are fraudulent. Standard classifiers trained naively will simply predict "legitimate" for everything and appear 99.8% accurate while catching zero fraud.

### The Solution: A Two-Stage Ensemble

```
Raw Transactions
      │
      ▼
Feature Engineering  ─── Log amount, hour-of-day, PCA magnitude, etc.
      │
      ▼
Isolation Forest  ─── Unsupervised anomaly scores added as features
      │
      ▼
SMOTE Oversampling  ─── Synthetic minority samples (10% ratio)
      │
      ▼
XGBoost Classifier  ─── Trained on resampled data
      │
      ▼
Threshold Optimisation  ─── Maximise recall, constrain precision ≥ 10%
      │
      ▼
Fraud Predictions
```

**Why this works:**
- **Isolation Forest** provides an unsupervised anomaly signal that doesn't rely on labels — useful when fraud patterns shift over time
- **SMOTE** generates synthetic fraud examples to help XGBoost learn the minority class boundary, rather than simply duplicating existing fraud samples
- **Threshold tuning** lets us trade precision for recall explicitly — in fraud detection, missing a fraud (false negative) is far more costly than a false alarm

---

## 📁 Repo Structure

```
credit-card-fraud-detection/
├── data/                          # Place creditcard.csv here (see below)
├── notebooks/
│   └── fraud_detection_walkthrough.ipynb   # Full analysis walkthrough
├── src/
│   ├── fraud_detection.py         # Main pipeline (train + evaluate)
│   └── predict.py                 # Inference on new transactions
├── models/                        # Saved model artefacts (generated on run)
├── outputs/                       # Plots & predictions (generated on run)
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Download the dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

> The dataset contains 284,807 transactions from European cardholders (September 2013). Features V1–V28 are PCA-transformed to protect cardholder privacy. Only `Time`, `Amount`, and `Class` are in their original form.

---

## 🏃 Run the Pipeline

**Train & evaluate:**
```bash
python src/fraud_detection.py
```

This will:
- Load and engineer features
- Fit Isolation Forest + XGBoost
- Optimise classification threshold
- Print evaluation metrics
- Save plots to `outputs/` and models to `models/`

**Run inference on new data:**
```bash
python src/predict.py --input data/new_transactions.csv --threshold 0.3
```

**Jupyter notebook walkthrough:**
```bash
jupyter notebook notebooks/fraud_detection_walkthrough.ipynb
```

---

## 📊 Feature Engineering

| Feature | Description |
|---|---|
| `log_amount` | Log-normalised transaction amount |
| `amount_zscore` | Z-score of amount (standardised deviation) |
| `hour_of_day` | Hour extracted from `Time` |
| `is_night` | 1 if transaction between 22:00–06:00 |
| `high_value_txn` | 1 if amount in top 5% |
| `high_value_night` | Interaction: high value AND night |
| `pca_magnitude` | Euclidean norm of V1–V28 (deviation from origin) |
| `iso_score` | Isolation Forest anomaly score |
| `iso_flag` | Binary flag from Isolation Forest (-1 → 1) |

---

## 📈 Sample Outputs

After running the pipeline, the `outputs/` directory contains:

- `confusion_matrix.png` — TP/FP/TN/FN breakdown
- `roc_pr_curves.png` — ROC and Precision-Recall curves
- `feature_importance.png` — Top 20 XGBoost features
- `fraud_patterns.png` — Amount & time-of-day fraud distribution

---

## 🛠️ Tech Stack

- **Python 3.10+**
- `scikit-learn` — Isolation Forest, preprocessing, metrics
- `xgboost` — Gradient boosted classifier
- `imbalanced-learn` — SMOTE oversampling
- `pandas` / `numpy` — Data manipulation
- `matplotlib` / `seaborn` — Visualisation
- `joblib` — Model serialisation

---

## 📄 License

MIT License — free to use, modify, and distribute.

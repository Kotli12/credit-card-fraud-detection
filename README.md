# Credit Card Fraud Detection

End-to-end machine learning pipeline for detecting fraudulent credit card transactions on a severely imbalanced dataset (0.17% fraud rate). The system combines an unsupervised anomaly detector (Isolation Forest) with a supervised gradient-boosted classifier (XGBoost), bridged by SMOTE oversampling, and achieves approximately **98% fraud recall** at a ROC-AUC of ~0.98.

---

## Problem Context

Credit card fraud costs the global economy over $30 billion per year. The core modelling challenge is not predictive accuracy in the classical sense — it is **class imbalance** and **asymmetric misclassification costs**.

A dataset that is 99.83% legitimate transactions yields 99.83% accuracy by predicting "legitimate" for every row. That model catches zero fraud. The useful metric is **recall on the fraud class**: of all actual fraud cases, what fraction did the model detect?

A false negative (missed fraud) is also far more costly to a business than a false positive (legitimate transaction flagged for review). This pipeline is therefore optimised for **maximum recall subject to a precision floor**, not for accuracy or balanced F1.

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Source | European cardholders, September 2013 |
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.17%) |
| Features | 30 (Time, Amount, V1–V28) |
| Class label | 0 = legitimate, 1 = fraud |

V1–V28 are PCA-transformed by the dataset provider to protect cardholder privacy. Only `Time` (seconds elapsed since the first transaction), `Amount`, and `Class` are in their original form. This limits domain-specific feature engineering to those two interpretable fields.

---

## Architecture

```
Raw Transactions (284,807 × 30)
         |
         v
Feature Engineering
    log_amount, amount_zscore         stabilise the heavy-tailed amount distribution
    hour_of_day, is_night             time-of-day signals for fraud pattern detection
    high_value_txn, high_value_night  interaction flags for high-value overnight transactions
    pca_magnitude                     L2 norm of V1–V28; measures deviation from the PCA origin
         |
         v
Stratified 80/20 Train/Test Split
         |
         v
StandardScaler  (fit on train only — prevents leakage)
    Scales Amount, log_amount, amount_zscore, pca_magnitude
         |
         v
Isolation Forest  (fit on train only)
    Adds iso_score  — continuous anomaly score from decision_function()
    Adds iso_flag   — binary anomaly indicator from predict()
    contamination=0.002 matches the dataset's known fraud rate
         |
         v
SMOTE Oversampling  (applied to train only)
    sampling_strategy=0.10  generates synthetic fraud samples to 10% ratio
    k_neighbors=5
         |
         v
XGBoost Classifier
    n_estimators=500, max_depth=6, learning_rate=0.05
    eval_metric=aucpr, early_stopping_rounds=30
    scale_pos_weight set to post-SMOTE class ratio
         |
         v
Threshold Optimisation
    Sweeps the full precision-recall curve
    Selects the threshold that maximises recall with precision >= 10%
         |
         v
Artefacts: xgboost_fraud.pkl, isolation_forest.pkl, scaler.pkl
Plots:     confusion_matrix, roc_pr_curves, feature_importance,
           fraud_patterns, threshold_analysis
```

---

## Design Decisions

### Isolation Forest as a feature, not a standalone classifier

Isolation Forest learns what "normal" looks like without fraud labels. Its `decision_function` score is added as a feature to XGBoost rather than used directly for classification for two reasons:

1. The unsupervised anomaly signal degrades more gracefully than a purely supervised model under **distribution shift** — fraud patterns evolve over time, but a model of normality remains useful even as the fraud landscape changes.
2. XGBoost can learn when `iso_score` is informative and when it is not. Feeding it as a feature is more robust than hard-thresholding the Isolation Forest output directly.

### SMOTE instead of class weights or simple duplication

Simple oversampling repeats existing fraud examples and risks overfitting to those specific transactions. SMOTE generates **synthetic** fraud observations by interpolating between nearest neighbours in feature space, giving the classifier a broader view of the minority class boundary.

SMOTE is applied after the train/test split so the test set contains only real transactions and the held-out evaluation is unbiased.

### AUPR for early stopping, not AUC-ROC

On imbalanced datasets, AUC-ROC can appear inflated because it evaluates performance across all thresholds, including those where the dominant negative class drives the result. **Average Precision (AUPR)** focuses on precision-recall space, where the minority class is the subject of interest, and is more sensitive to meaningful improvements in fraud detection.

### Threshold tuning rather than accepting 0.5

The default threshold of 0.5 assumes symmetric misclassification costs. The threshold sweep finds the operating point that **maximises recall** while keeping precision above 10% (at least 1 in 10 flagged transactions is genuine fraud). This constraint limits false alarm volume to a level a review team can manage, while catching as many fraud cases as possible.

---

## Feature Engineering

| Feature | Derivation | Rationale |
|---|---|---|
| `log_amount` | `log(Amount + 1)` | Transaction amounts are heavily right-skewed. The log transform compresses the range and improves gradient stability. |
| `amount_zscore` | `(Amount − μ) / σ` | Normalised deviation from the mean; makes unusually large or small transactions directly comparable across the feature space. |
| `hour_of_day` | `(Time % 86400) // 3600` | The raw `Time` field is seconds elapsed since the first transaction, not a clock time. This converts it to a 0–23 hour signal. |
| `is_night` | 1 if hour ∈ [22, 6) | Fraud is disproportionately concentrated in late-night hours when account monitoring is reduced. |
| `high_value_txn` | 1 if Amount > 95th percentile | High-value transactions are a known fraud indicator, flagging the top 5% of amounts. |
| `high_value_night` | `high_value_txn × is_night` | Interaction term: high-value transactions at night are especially rare and suspicious. |
| `pca_magnitude` | `‖[V1,...,V28]‖₂` | Measures how far the PCA vector is from the origin. Fraud transactions tend to cluster further from the dense legitimate region in PCA space. |
| `iso_score` | Isolation Forest `decision_function` | Continuous anomaly score; more negative values indicate more isolated (anomalous) observations. |
| `iso_flag` | 1 if Isolation Forest `predict == −1` | Binary hard-boundary anomaly indicator; categorical complement to `iso_score`. |

---

## Results

| Metric | Score |
|---|---|
| Fraud Recall | ~98% |
| ROC-AUC | ~0.98 |
| Average Precision (AUPR) | ~0.85 |

At ~98% recall, approximately 98 of every 100 fraudulent transactions are caught. The residual 2% false negative rate is a deliberate operating choice — pushing recall to 100% would require a threshold so low that precision collapses and the review team is overwhelmed with false alarms.

---

## Project Structure

```
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv             # Download from Kaggle (not tracked in git)
├── notebooks/
│   └── fraud_detection_walkthrough.ipynb
├── src/
│   ├── fraud_detection.py         # Training pipeline
│   └── predict.py                 # Batch inference
├── models/                        # Generated on first run (not tracked in git)
│   ├── xgboost_fraud.pkl
│   ├── isolation_forest.pkl
│   └── scaler.pkl
├── outputs/                       # Generated on first run
│   ├── confusion_matrix.png
│   ├── roc_pr_curves.png
│   ├── feature_importance.png
│   ├── fraud_patterns.png
│   └── threshold_analysis.png
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

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`.

---

## Running

**Train and evaluate:**
```bash
python src/fraud_detection.py
```

Runs the full pipeline and writes:
- Classification report, ROC-AUC, and Average Precision to stdout
- Five diagnostic plots to `outputs/`
- Three serialised model artefacts to `models/`

**Batch inference on new transactions:**
```bash
# Use the threshold printed at the end of training
python src/predict.py --input data/new_transactions.csv --threshold 0.3

# All options
python src/predict.py \
    --input data/new_transactions.csv \
    --threshold 0.3 \
    --model_dir models/ \
    --output outputs/predictions.csv
```

Writes a CSV with `fraud_probability` and `fraud_prediction` appended to the original columns.

**Interactive notebook walkthrough:**
```bash
jupyter notebook notebooks/fraud_detection_walkthrough.ipynb
```

---

## Limitations

**Opaque PCA features.** V1–V28 are already transformed, so domain-specific features (merchant category, geographic location, card type, transaction velocity) cannot be engineered. In a real deployment these would be available and the feature space would be significantly richer.

**Random train/test split.** The 80/20 split is stratified by class but not by time. A rigorous evaluation would use a temporal split — training on earlier transactions and testing on later ones — to simulate production conditions and measure resilience to temporal distribution shift.

**Static threshold.** The threshold is calibrated once on the test set and serialised with the model. In production it would be recalibrated periodically as fraud patterns and business costs evolve.

**No concept drift monitoring.** The pipeline does not detect shifts between training data and live traffic. In production, the `iso_score` distribution over live transactions would be monitored as an early warning signal, since Isolation Forest degrades gracefully and its scores reflect structural changes in the input data even without retraining.

---

## License

MIT

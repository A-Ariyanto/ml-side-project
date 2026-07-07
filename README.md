N# Vehicle Insurance Risk Modelling

An end-to-end machine learning pipeline that predicts two targets from vehicle insurance policy data:

- **Safety rating** (regression) — a continuous vehicle safety score
- **Claim lodgement** (classification) — whether a policyholder will lodge a claim, with a severe ~94/6 class imbalance

Built with pandas, scikit-learn, LightGBM and XGBoost. The full pipeline (feature engineering → training seven gradient-boosted models → prediction export) runs in about a minute on a laptop.

## Repository structure

| Path | Description |
|---|---|
| [`analysis.ipynb`](analysis.ipynb) | EDA, baseline model comparison, and hyperparameter tuning — the full development story |
| [`predict.py`](predict.py) | Final production pipeline: feature engineering + model ensembles, end to end |
| [`data/train.csv`](data/train.csv) | 40,194 labelled policies, 33 raw columns |
| [`data/test.csv`](data/test.csv) | 4,689 held-out policies to predict |
| [`requirements.txt`](requirements.txt) | Pinned dependencies |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 predict.py data/train.csv data/test.csv
```

This trains everything from scratch and writes two files: `predictions_regression.csv` (`policy_id, safety_rating`) and `predictions_classification.csv` (`policy_id, claim`).

To explore the analysis, open the notebook:

```bash
jupyter notebook analysis.ipynb
```

## Approach

### Feature engineering (no target leakage)

All transformations live in a single `preprocess()` function applied independently to train and test, so no target information ever crosses over:

- **String parsing** — `torque` (`"91Nm@4250rpm"`) and `power` (`"67.06bhp@5500rpm"`) are split into numeric magnitude + RPM columns; `car_age` (`"4 years and 6 months"`) becomes months.
- **Feature-list expansion** — the `features` column (a stringified list of vehicle features) is expanded into 17 binary safety-equipment flags plus a feature count.
- **Domain ratios** — power-to-weight, torque-to-weight, displacement-per-cylinder, estimated total mileage, vehicle volume/footprint, and more.
- **Log transforms** of skewed columns (power, weight, displacement, mileage, population density).
- **Robust handling of unseen data** — median imputation uses training-set medians only, and the `OrdinalEncoder` maps categories never seen in training to `-1` instead of crashing.

### Models

- **Regression** (`safety_rating`): a weighted ensemble of four gradient-boosted regressors — two LightGBM configurations with deliberately different tree structures, XGBoost, and scikit-learn's HistGradientBoosting.
- **Classification** (`claim`): the predicted safety rating is fed in as an extra feature, then three classifiers (LightGBM, XGBoost, HistGradientBoosting) vote by averaged probability. Class imbalance is handled at the algorithm level via `class_weight='balanced'` / `scale_pos_weight`, avoiding the leakage risks of oversampling.

### Development process

The notebook documents how the final design was reached on an 80/20 validation split:

| Stage | Regression (RMSE ↓) | Classification (F1 macro ↑) |
|---|---|---|
| Baseline (default params) | 3.181 (LightGBM) | 0.484 (Random Forest) |
| After `RandomizedSearchCV` tuning | 3.176 | 0.510 |

Tuning alone bought little, which motivated the final pipeline's heavier investment in feature engineering and multi-model ensembling instead.

## Author

Abdullah Ariyanto

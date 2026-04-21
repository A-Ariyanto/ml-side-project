"""
Improved regression + classification pipeline for Assignment 3.
Key differences from original z5543164.py:
  - Deep feature engineering from string columns (torque, power, car_age, features list)
  - Binary indicators for individual safety features (critical for safety_rating prediction)
  - Physics-based interaction features
  - LightGBM + CatBoost ensemble for regression
  - Better hyperparameter configuration

Usage (command line):
    python3 z5543164_improved.py train.csv test.csv

Usage (Google Colab):
    Set COLAB_MODE = True below and adjust paths.
"""

import sys
import time
import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import root_mean_squared_error, f1_score

# ============================================================
# COLAB MODE: Set to True to run interactively in Colab
# ============================================================
COLAB_MODE = False
COLAB_TRAIN_PATH = "train.csv"
COLAB_TEST_PATH = "test.csv"
STUDENT_ID = "z5543164"

# Key safety-related features found in the 'features' column
SAFETY_FEATURES = [
    'esc', 'tpms', 'brake_assist', 'parking_camera',
    'parking_sensors', 'front_fog_lights', 'adjustable_steering',
    'rear_defogger', 'power_door_locks', 'central_locking',
    'power_steering', 'driver_seat_adjustable', 'day_night_mirror',
    'ecw', 'speed_alert', 'rear_wiper', 'rear_washer'
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering applied independently to train and test.
    NO target information is used here, so no data leakage.
    """
    df_clean = df.copy()

    # ── 1. Parse torque: "91Nm@4250rpm" → torque_nm=91, torque_rpm=4250 ──
    if 'torque' in df_clean.columns:
        torque_str = df_clean['torque'].astype(str)
        df_clean['torque_nm'] = pd.to_numeric(
            torque_str.str.extract(r'([\d.]+)\s*Nm', flags=re.IGNORECASE)[0],
            errors='coerce'
        )
        df_clean['torque_rpm'] = pd.to_numeric(
            torque_str.str.extract(r'@\s*([\d.]+)\s*rpm', flags=re.IGNORECASE)[0],
            errors='coerce'
        )
        df_clean.drop(columns=['torque'], inplace=True)

    # ── 2. Parse power: "67.06bhp@5500rpm" → power_bhp=67.06, power_rpm=5500 ──
    if 'power' in df_clean.columns:
        power_str = df_clean['power'].astype(str)
        df_clean['power_bhp'] = pd.to_numeric(
            power_str.str.extract(r'([\d.]+)\s*bhp', flags=re.IGNORECASE)[0],
            errors='coerce'
        )
        df_clean['power_rpm'] = pd.to_numeric(
            power_str.str.extract(r'@\s*([\d.]+)\s*rpm', flags=re.IGNORECASE)[0],
            errors='coerce'
        )
        df_clean.drop(columns=['power'], inplace=True)

    # ── 3. Parse car_age: "4 years and 6 months" → 54 months ──
    if 'car_age' in df_clean.columns:
        age_str = df_clean['car_age'].astype(str)
        years = pd.to_numeric(
            age_str.str.extract(r'(\d+)\s*year')[0], errors='coerce'
        ).fillna(0)
        months = pd.to_numeric(
            age_str.str.extract(r'(\d+)\s*month')[0], errors='coerce'
        ).fillna(0)
        df_clean['car_age_months'] = (years * 12 + months).astype(int)
        df_clean.drop(columns=['car_age'], inplace=True)

    # ── 4. Parse features list → count + binary indicators ──
    #    This is the MOST CRITICAL step for safety_rating prediction.
    #    The features column lists safety equipment that directly determines the rating.
    if 'features' in df_clean.columns:
        feat_str = df_clean['features'].astype(str)

        # Count total features (each feature name is in single quotes)
        df_clean['n_features'] = feat_str.str.count("'") // 2

        # Binary indicator for each key safety feature
        for feat in SAFETY_FEATURES:
            df_clean[f'has_{feat}'] = feat_str.str.contains(
                feat, case=False, na=False
            ).astype(int)

        df_clean.drop(columns=['features'], inplace=True)

    # ── 5. Physics-based interaction features ──
    # Power-to-weight ratio (acceleration proxy)
    if 'power_bhp' in df_clean.columns and 'gross_weight' in df_clean.columns:
        df_clean['power_to_weight'] = (
            df_clean['power_bhp'] / df_clean['gross_weight'].replace(0, np.nan)
        )

    # Torque-to-weight ratio (low-speed pulling power)
    if 'torque_nm' in df_clean.columns and 'gross_weight' in df_clean.columns:
        df_clean['torque_to_weight'] = (
            df_clean['torque_nm'] / df_clean['gross_weight'].replace(0, np.nan)
        )

    # Displacement per cylinder (engine efficiency)
    if 'displacement' in df_clean.columns and 'cylinder' in df_clean.columns:
        df_clean['disp_per_cylinder'] = (
            df_clean['displacement'] / df_clean['cylinder'].replace(0, np.nan)
        )

    # Vehicle volume proxy
    if all(c in df_clean.columns for c in ['length', 'width', 'height']):
        df_clean['volume'] = df_clean['length'] * df_clean['width'] * df_clean['height']

    # Safety feature density = airbags × number of features
    if 'airbags' in df_clean.columns and 'n_features' in df_clean.columns:
        df_clean['safety_density'] = df_clean['airbags'] * df_clean['n_features']

    # Estimated total mileage
    if 'car_age_months' in df_clean.columns and 'annual_mileage_km' in df_clean.columns:
        df_clean['est_total_mileage'] = (
            df_clean['car_age_months'] * df_clean['annual_mileage_km'] / 12.0
        )

    return df_clean


def main():
    start_time = time.time()

    # ── Get paths ──
    if COLAB_MODE:
        train_path = COLAB_TRAIN_PATH
        test_path = COLAB_TEST_PATH
    else:
        if len(sys.argv) != 3:
            print(f"Usage: python3 {STUDENT_ID}.py <train_file> <test_file>")
            sys.exit(1)
        train_path = sys.argv[1]
        test_path = sys.argv[2]

    # ── Load data ──
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    # ── Feature engineering (applied independently — no leakage) ──
    train_clean = engineer_features(train_df)
    test_clean = engineer_features(test_df)
    print(f"After feature engineering — train: {train_clean.shape}, test: {test_clean.shape}")

    # ── Missing value imputation (use training statistics only) ──
    exclude_targets = ['safety_rating', 'claim', 'policy_id']
    num_cols = [
        c for c in train_clean.select_dtypes(include=['number']).columns
        if c not in exclude_targets
    ]
    medians = train_clean[num_cols].median()
    train_clean[num_cols] = train_clean[num_cols].fillna(medians)
    test_clean[num_cols] = test_clean[num_cols].fillna(medians)

    cat_cols = [
        c for c in train_clean.select_dtypes(exclude=['number']).columns
        if c != 'policy_id'
    ]
    if cat_cols:
        train_clean[cat_cols] = train_clean[cat_cols].fillna('Missing')
        test_clean[cat_cols] = test_clean[cat_cols].fillna('Missing')

    # ── Categorical encoding ──
    if cat_cols:
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1
        )
        train_clean[cat_cols] = encoder.fit_transform(train_clean[cat_cols])
        test_clean[cat_cols] = encoder.transform(test_clean[cat_cols])

    # ── Define feature columns ──
    base_features = [
        c for c in train_clean.columns
        if c not in ['policy_id', 'safety_rating', 'claim']
    ]
    print(f"Number of features: {len(base_features)}")

    # =============================================================
    # PART II: REGRESSION — Predict safety_rating
    # =============================================================
    X_train_reg = train_clean[base_features]
    y_train_reg = train_clean['safety_rating']
    X_test_reg = test_clean[base_features]

    # ── LightGBM Regressor ──
    lgbm_reg = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=-1,           # unlimited depth
        num_leaves=127,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_reg.fit(X_train_reg, y_train_reg)
    pred_lgbm = lgbm_reg.predict(X_test_reg)
    print(f"LightGBM regression done — {time.time() - start_time:.1f}s elapsed")

    # ── CatBoost Regressor (ensemble partner) ──
    cat_reg = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        verbose=0,
        random_seed=42,
        thread_count=-1
    )
    cat_reg.fit(X_train_reg, y_train_reg)
    pred_cat = cat_reg.predict(X_test_reg)
    print(f"CatBoost regression done — {time.time() - start_time:.1f}s elapsed")

    # ── Ensemble (simple average) ──
    test_clean['safety_rating'] = 0.5 * pred_lgbm + 0.5 * pred_cat

    # =============================================================
    # PART III: CLASSIFICATION — Predict claim
    # =============================================================
    clf_features = base_features + ['safety_rating']
    X_train_clf = train_clean[clf_features]
    y_train_clf = train_clean['claim']
    X_test_clf = test_clean[clf_features]

    clf_model = LGBMClassifier(
        class_weight='balanced',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf_model.fit(X_train_clf, y_train_clf)

    # Threshold tuning — try a range and pick the best on train as proxy
    test_pred_probs = clf_model.predict_proba(X_test_clf)[:, 1]
    test_clean['claim'] = (test_pred_probs >= 0.5).astype(int)
    print(f"Classification done — {time.time() - start_time:.1f}s elapsed")

    # =============================================================
    # VALIDATION (only when test set has ground truth, e.g., local eval)
    # =============================================================
    if 'safety_rating' in test_df.columns and test_df['safety_rating'].notna().all():
        rmse = root_mean_squared_error(
            test_df['safety_rating'], test_clean['safety_rating']
        )
        print(f"\n[EVAL] Regression RMSE: {rmse:.4f}")
        if rmse <= 1.0:
            print(f"[EVAL] ✅ Full marks (5/5)")
        elif rmse < 10.0:
            score = (1 - (rmse - 1) / 9) * 5
            print(f"[EVAL] Regression score: {score:.2f}/5")

    if 'claim' in test_df.columns and test_df['claim'].notna().all():
        f1 = f1_score(test_df['claim'], test_clean['claim'], average='macro')
        print(f"[EVAL] Classification F1 Macro: {f1:.4f}")

    # =============================================================
    # EXPORT
    # =============================================================
    reg_out = f"{STUDENT_ID}_regression.csv"
    clf_out = f"{STUDENT_ID}_classification.csv"
    test_clean[['policy_id', 'safety_rating']].to_csv(reg_out, index=False)
    test_clean[['policy_id', 'claim']].to_csv(clf_out, index=False)

    total_time = time.time() - start_time
    print(f"\nPredictions saved to {reg_out} and {clf_out}")
    print(f"Total pipeline runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()

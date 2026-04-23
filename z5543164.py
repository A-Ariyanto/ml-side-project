import sys
import time
import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

STUDENT_ID = "z5543164"

SAFETY_FEATURES = [
    'esc', 'tpms', 'brake_assist', 'parking_camera',
    'parking_sensors', 'front_fog_lights', 'adjustable_steering',
    'rear_defogger', 'power_door_locks', 'central_locking',
    'power_steering', 'driver_seat_adjustable', 'day_night_mirror',
    'ecw', 'speed_alert', 'rear_wiper', 'rear_washer'
]


def preprocess(df) -> pd.DataFrame:
    """
    Rich feature engineering applied independently to train and test.
    NO target information is used which means there is no data leakage.
    """
    df_clean = df.copy()

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

    if 'features' in df_clean.columns:
        feat_str = df_clean['features'].astype(str)
        df_clean['n_features'] = feat_str.str.count("'") // 2
        for feat in SAFETY_FEATURES:
            df_clean[f'has_{feat}'] = feat_str.str.contains(
                feat, case=False, na=False
            ).astype(int)
        df_clean.drop(columns=['features'], inplace=True)

    if 'gross_weight' in df_clean.columns:
        df_clean['gross_weight'] = pd.to_numeric(
            df_clean['gross_weight'].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        )

    if 'power_bhp' in df_clean.columns and 'gross_weight' in df_clean.columns:
        gw_safe = df_clean['gross_weight'].replace(0, np.nan)
        df_clean['power_to_weight'] = df_clean['power_bhp'] / gw_safe

    if 'torque_nm' in df_clean.columns and 'gross_weight' in df_clean.columns:
        gw_safe = df_clean['gross_weight'].replace(0, np.nan)
        df_clean['torque_to_weight'] = df_clean['torque_nm'] / gw_safe

    if 'displacement' in df_clean.columns and 'cylinder' in df_clean.columns:
        cyl_safe = df_clean['cylinder'].replace(0, np.nan)
        df_clean['disp_per_cylinder'] = df_clean['displacement'] / cyl_safe

    if all(c in df_clean.columns for c in ['length', 'width', 'height']):
        df_clean['volume'] = df_clean['length'] * df_clean['width'] * df_clean['height']

    if 'airbags' in df_clean.columns and 'n_features' in df_clean.columns:
        df_clean['safety_density'] = df_clean['airbags'] * df_clean['n_features']

    if 'car_age_months' in df_clean.columns and 'annual_mileage_km' in df_clean.columns:
        df_clean['est_total_mileage'] = (
            df_clean['car_age_months'] * df_clean['annual_mileage_km'] / 12.0
        )

    if 'policyholder_age' in df_clean.columns and 'car_age_months' in df_clean.columns:
        df_clean['driver_car_age_ratio'] = (
            df_clean['policyholder_age'] / df_clean['car_age_months'].replace(0, np.nan)
        )

    if 'policy_age_months' in df_clean.columns and 'car_age_months' in df_clean.columns:
        df_clean['policy_vs_car_age'] = (
            df_clean['policy_age_months'] / df_clean['car_age_months'].replace(0, np.nan)
        )

    if 'power_bhp' in df_clean.columns and 'torque_nm' in df_clean.columns:
        df_clean['power_torque_ratio'] = (
            df_clean['power_bhp'] / df_clean['torque_nm'].replace(0, np.nan)
        )

    if all(c in df_clean.columns for c in ['length', 'width']):
        df_clean['footprint'] = df_clean['length'] * df_clean['width']

    if 'turning_radius' in df_clean.columns and 'length' in df_clean.columns:
        df_clean['turn_length_ratio'] = (
            df_clean['turning_radius'] / df_clean['length'].replace(0, np.nan)
        )

    has_cols = [c for c in df_clean.columns if c.startswith('has_')]
    if has_cols:
        df_clean['total_safety_score'] = df_clean[has_cols].sum(axis=1)

    if 'airbags' in df_clean.columns:
        df_clean['airbags_sq'] = df_clean['airbags'] ** 2

    if 'n_features' in df_clean.columns:
        df_clean['n_features_sq'] = df_clean['n_features'] ** 2

    if 'power_bhp' in df_clean.columns:
        df_clean['log_power'] = np.log1p(df_clean['power_bhp'].clip(lower=0))

    if 'gross_weight' in df_clean.columns:
        df_clean['log_weight'] = np.log1p(df_clean['gross_weight'].clip(lower=0))

    if 'displacement' in df_clean.columns:
        df_clean['log_displacement'] = np.log1p(df_clean['displacement'].clip(lower=0))

    if 'annual_mileage_km' in df_clean.columns:
        df_clean['log_mileage'] = np.log1p(df_clean['annual_mileage_km'].clip(lower=0))

    if 'population_density' in df_clean.columns:
        df_clean['log_pop_density'] = np.log1p(df_clean['population_density'].clip(lower=0))

    return df_clean


def main():
    start_time = time.time()

    if len(sys.argv) != 3:
        print(f"Usage: python3 {STUDENT_ID}.py <train_file> <test_file>")
        sys.exit(1)
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Feature engineering
    train_clean = preprocess(train_df)
    test_clean = preprocess(test_df)

    # Missing value imputation (use training medians only)
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

    # Categorical encoding
    if cat_cols:
        encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1
        )
        train_clean[cat_cols] = encoder.fit_transform(train_clean[cat_cols])
        test_clean[cat_cols] = encoder.transform(test_clean[cat_cols])

    base_features = [
        c for c in train_clean.columns
        if c not in ['policy_id', 'safety_rating', 'claim']
    ]

    # PART II: REGRESSION — 4-model ensemble for safety_rating
    X_train_reg = train_clean[base_features]
    y_train_reg = train_clean['safety_rating']
    X_test_reg = test_clean[base_features]

    # Model 1: LightGBM (deep, many leaves)
    lgbm_reg_1 = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=127,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    lgbm_reg_1.fit(X_train_reg, y_train_reg)
    pred_lgbm_1 = lgbm_reg_1.predict(X_test_reg)
    print(f"  LGBM reg v1 done   — {time.time() - start_time:.1f}s")

    # Model 2: LightGBM (wider trees, different structure for diversity)
    lgbm_reg_2 = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=255,
        min_child_samples=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=123,
        verbose=-1
    )
    lgbm_reg_2.fit(X_train_reg, y_train_reg)
    pred_lgbm_2 = lgbm_reg_2.predict(X_test_reg)
    print(f"  LGBM reg v2 done   — {time.time() - start_time:.1f}s")

    # Model 3: XGBoost
    xgb_reg = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        random_state=42,
        verbosity=0
    )
    xgb_reg.fit(X_train_reg, y_train_reg)
    pred_xgb_reg = xgb_reg.predict(X_test_reg)
    print(f"  XGBoost reg done   — {time.time() - start_time:.1f}s")

    # Model 4: sklearn HistGradientBoosting
    hgb_reg = HistGradientBoostingRegressor(
        max_iter=1500,
        learning_rate=0.03,
        max_depth=8,
        max_leaf_nodes=127,
        min_samples_leaf=10,
        l2_regularization=0.1,
        random_state=42
    )
    hgb_reg.fit(X_train_reg, y_train_reg)
    pred_hgb_reg = hgb_reg.predict(X_test_reg)
    print(f"  HistGBR reg done   — {time.time() - start_time:.1f}s")

    # Ensemble: weighted average
    test_clean['safety_rating'] = (
        0.30 * pred_lgbm_1 +
        0.25 * pred_lgbm_2 +
        0.25 * pred_xgb_reg +
        0.20 * pred_hgb_reg
    )

    # PART III: CLASSIFICATION — 3-model ensemble for claim
    clf_features = base_features + ['safety_rating']
    X_train_clf = train_clean[clf_features]
    y_train_clf = train_clean['claim']
    X_test_clf = test_clean[clf_features]

    # Model 1: LightGBM Classifier
    lgbm_clf = LGBMClassifier(
        class_weight='balanced',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgbm_clf.fit(X_train_clf, y_train_clf)
    prob_lgbm_clf = lgbm_clf.predict_proba(X_test_clf)[:, 1]
    print(f"  LightGBM clf done  — {time.time() - start_time:.1f}s")

    # Model 2: XGBoost Classifier
    n_neg = (y_train_clf == 0).sum()
    n_pos = (y_train_clf == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)

    xgb_clf = XGBClassifier(
        scale_pos_weight=scale_pos,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42,
        verbosity=0
    )
    xgb_clf.fit(X_train_clf, y_train_clf)
    prob_xgb_clf = xgb_clf.predict_proba(X_test_clf)[:, 1]
    print(f"  XGBoost clf done   — {time.time() - start_time:.1f}s")

    # Model 3: sklearn HistGradientBoosting Classifier
    hgb_clf = HistGradientBoostingClassifier(
        class_weight='balanced',
        max_iter=500,
        learning_rate=0.05,
        max_depth=7,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42
    )
    hgb_clf.fit(X_train_clf, y_train_clf)
    prob_hgb_clf = hgb_clf.predict_proba(X_test_clf)[:, 1]
    print(f"  HistGBC clf done   — {time.time() - start_time:.1f}s")

    # Ensemble: average probabilities, then threshold
    avg_prob = (prob_lgbm_clf + prob_xgb_clf + prob_hgb_clf) / 3.0
    test_clean['claim'] = (avg_prob >= 0.5).astype(int)

    # EXPORT PREDICTIONS
    reg_out = f"{STUDENT_ID}_regression.csv"
    clf_out = f"{STUDENT_ID}_classification.csv"
    test_clean[['policy_id', 'safety_rating']].to_csv(reg_out, index=False)
    test_clean[['policy_id', 'claim']].to_csv(clf_out, index=False)

    total_time = time.time() - start_time
    print(f"\nPredictions saved to {reg_out} and {clf_out}")
    print(f"Total pipeline runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()

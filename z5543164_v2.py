import sys
import time
import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder

STUDENT_ID = "z5543164"

# Key safety-related features found in the 'features' column
SAFETY_FEATURES = [
    'esc', 'tpms', 'brake_assist', 'parking_camera',
    'parking_sensors', 'front_fog_lights', 'adjustable_steering',
    'rear_defogger', 'power_door_locks', 'central_locking',
    'power_steering', 'driver_seat_adjustable', 'day_night_mirror',
    'ecw', 'speed_alert', 'rear_wiper', 'rear_washer'
]


def preprocess(df) -> pd.DataFrame:
    """
    Feature engineering applied independently to train and test.
    NO target information is used here, so no data leakage.
    """
    df_clean = df.copy()

    # 1. Parse torque: "91Nm@4250rpm" → torque_nm=91, torque_rpm=4250
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

    # 2. Parse power: "67.06bhp@5500rpm" → power_bhp=67.06, power_rpm=5500
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

    # 3. Parse car_age: "4 years and 6 months" → 54 months
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

    # 4. Parse features list → count + binary indicators
    if 'features' in df_clean.columns:
        feat_str = df_clean['features'].astype(str)
        df_clean['n_features'] = feat_str.str.count("'") // 2
        for feat in SAFETY_FEATURES:
            df_clean[f'has_{feat}'] = feat_str.str.contains(
                feat, case=False, na=False
            ).astype(int)
        df_clean.drop(columns=['features'], inplace=True)

    # 5. Parse gross_weight to numeric
    if 'gross_weight' in df_clean.columns:
        df_clean['gross_weight'] = pd.to_numeric(
            df_clean['gross_weight'].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        )

    # 6. Interaction features
    if 'power_bhp' in df_clean.columns and 'gross_weight' in df_clean.columns:
        df_clean['power_to_weight'] = (
            df_clean['power_bhp'] / df_clean['gross_weight'].replace(0, np.nan)
        )
    if 'torque_nm' in df_clean.columns and 'gross_weight' in df_clean.columns:
        df_clean['torque_to_weight'] = (
            df_clean['torque_nm'] / df_clean['gross_weight'].replace(0, np.nan)
        )
    if 'displacement' in df_clean.columns and 'cylinder' in df_clean.columns:
        df_clean['disp_per_cylinder'] = (
            df_clean['displacement'] / df_clean['cylinder'].replace(0, np.nan)
        )
    if all(c in df_clean.columns for c in ['length', 'width', 'height']):
        df_clean['volume'] = df_clean['length'] * df_clean['width'] * df_clean['height']
    if 'airbags' in df_clean.columns and 'n_features' in df_clean.columns:
        df_clean['safety_density'] = df_clean['airbags'] * df_clean['n_features']
    if 'car_age_months' in df_clean.columns and 'annual_mileage_km' in df_clean.columns:
        df_clean['est_total_mileage'] = (
            df_clean['car_age_months'] * df_clean['annual_mileage_km'] / 12.0
        )

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

    # Feature engineering (applied independently — no leakage)
    train_clean = preprocess(train_df)
    test_clean = preprocess(test_df)

    # Missing value imputation (use training statistics only)
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

    # Define feature columns
    base_features = [
        c for c in train_clean.columns
        if c not in ['policy_id', 'safety_rating', 'claim']
    ]

    # ---------------------------------------------------------
    # PART II: REGRESSION — Predict safety_rating
    # ---------------------------------------------------------
    X_train_reg = train_clean[base_features]
    y_train_reg = train_clean['safety_rating']
    X_test_reg = test_clean[base_features]

    # Optimized for speed: fewer trees + higher learning rate
    # Still accurate thanks to rich feature engineering
    reg_model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    reg_model.fit(X_train_reg, y_train_reg)
    test_clean['safety_rating'] = reg_model.predict(X_test_reg)

    # ---------------------------------------------------------
    # PART III: CLASSIFICATION — Predict claim
    # ---------------------------------------------------------
    clf_features = base_features + ['safety_rating']
    X_train_clf = train_clean[clf_features]
    y_train_clf = train_clean['claim']
    X_test_clf = test_clean[clf_features]

    clf_model = LGBMClassifier(
        class_weight='balanced',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    clf_model.fit(X_train_clf, y_train_clf)

    test_pred_probs = clf_model.predict_proba(X_test_clf)[:, 1]
    test_clean['claim'] = (test_pred_probs >= 0.5).astype(int)

    # ---------------------------------------------------------
    # EXPORT PREDICTIONS
    # ---------------------------------------------------------
    reg_out = f"{STUDENT_ID}_regression.csv"
    clf_out = f"{STUDENT_ID}_classification.csv"
    test_clean[['policy_id', 'safety_rating']].to_csv(reg_out, index=False)
    test_clean[['policy_id', 'claim']].to_csv(clf_out, index=False)

    print("Predictions generated successfully.")
    print(f"Total pipeline runtime: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
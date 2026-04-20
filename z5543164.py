import sys
import time
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder

def preprocess(df) -> pd.DataFrame:
    """
    Vectorized preprocessing. NO LOOPS. Prevents memory fragmentation and 
    disk-swapping on heavily restricted university servers.
    """
    df_clean = df.copy()
    
    # 1. Fast regex extraction (avoids string expansion memory issues)
    if 'power' in df_clean.columns:
        df_clean['power'] = pd.to_numeric(df_clean['power'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    if 'gross_weight' in df_clean.columns:
        df_clean['gross_weight'] = pd.to_numeric(df_clean['gross_weight'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        
    # 2. Feature Engineering
    if 'power' in df_clean.columns and 'gross_weight' in df_clean.columns:
        df_clean['power_to_weight'] = df_clean['power'] / df_clean['gross_weight']
        
    if 'car_age' in df_clean.columns and 'annual_mileage_km' in df_clean.columns:
        df_clean['est_total_mileage'] = df_clean['car_age'] * df_clean['annual_mileage_km']

    # 3. Vectorized Missing Value Handling (Done all at once, no loops)
    # Fill numbers with median
    num_cols = df_clean.select_dtypes(include=['number']).columns.drop(['safety_rating', 'claim'], errors='ignore')
    if len(num_cols) > 0:
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
            
    # Fill categoricals with 'Missing'
    cat_cols = df_clean.select_dtypes(exclude=['number']).columns
    if len(cat_cols) > 0:
        df_clean[cat_cols] = df_clean[cat_cols].fillna('Missing')
        
    return df_clean

def main():
    # START THE TIMER
    start_time = time.time()
    
    if len(sys.argv) != 3:
        print("Usage: python3 z5543164.py <train_file> <test_file>")
        sys.exit(1)
        
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    
    # Load Datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Preprocess
    train_clean = preprocess(train_df)
    test_clean = preprocess(test_df)
    
    # Categorical Encoding
    cat_cols = train_clean.select_dtypes(exclude=['number']).columns.tolist()
    if 'policy_id' in cat_cols:
        cat_cols.remove('policy_id')
        
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_clean[cat_cols] = encoder.fit_transform(train_clean[cat_cols])
    test_clean[cat_cols] = encoder.transform(test_clean[cat_cols])
    
    base_features = [col for col in train_clean.columns if col not in ['policy_id', 'safety_rating', 'claim']]
    
    # --- EMERGENCY SPEED LIMITER: Train on random 20% ---
    train_fast = train_clean.sample(frac=0.20, random_state=42)
    
    # ---------------------------------------------------------
    # PART II: REGRESSION 
    # ---------------------------------------------------------
    X_train_reg = train_fast[base_features]
    y_train_reg = train_fast['safety_rating']
    X_test_reg = test_clean[base_features]
    
    # Double-enforced single threading (n_jobs=1 AND num_threads=1)
    reg_model = LGBMRegressor(
        random_state=42, 
        n_jobs=1,
        num_threads=1, 
        n_estimators=40,          
        learning_rate=0.15,       
        max_depth=7,              
        num_leaves=20
    )
    
    reg_model.fit(X_train_reg, y_train_reg)
    test_clean['safety_rating'] = reg_model.predict(X_test_reg)
    
    # ---------------------------------------------------------
    # PART III: CLASSIFICATION 
    # ---------------------------------------------------------
    clf_features = base_features + ['safety_rating']
    X_train_clf = train_fast[clf_features]
    y_train_clf = train_fast['claim']
    X_test_clf = test_clean[clf_features]
    
    clf_model = LGBMClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=1,                 
        num_threads=1,
        n_estimators=40,          
        learning_rate=0.15,       
        max_depth=7,
        num_leaves=20
    )
    
    clf_model.fit(X_train_clf, y_train_clf)
    
    # Apply Threshold
    test_pred_probs = clf_model.predict_proba(X_test_clf)[:, 1]
    test_clean['claim'] = (test_pred_probs >= 0.62).astype(int)
    
    # ---------------------------------------------------------
    # EXPORT PREDICTIONS
    # ---------------------------------------------------------
    test_clean[['policy_id', 'safety_rating']].to_csv('z5543164_regression.csv', index=False)
    test_clean[['policy_id', 'claim']].to_csv('z5543164_classification.csv', index=False)
    
    print("Predictions generated successfully.")
    print(f"Total pipeline runtime: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
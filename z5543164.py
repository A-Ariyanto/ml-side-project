import sys
import time
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder

def preprocess(df) -> pd.DataFrame:
    """
    Cleans data, engineers features, and aggressively downcasts data types to save RAM.
    """
    df_clean = df.copy()
    
    # Extract numerical digits and convert to float
    if 'power' in df_clean.columns:
        df_clean['power'] = df_clean['power'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    if 'gross_weight' in df_clean.columns:
        df_clean['gross_weight'] = df_clean['gross_weight'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        
    # Feature Engineering
    if 'power' in df_clean.columns and 'gross_weight' in df_clean.columns:
        df_clean['power_to_weight'] = df_clean['power'] / df_clean['gross_weight']
        
    if 'car_age' in df_clean.columns and 'annual_mileage_km' in df_clean.columns:
        df_clean['est_total_mileage'] = df_clean['car_age'] * df_clean['annual_mileage_km']

    # Handle Missing Values and Downcast to 32-bit
    num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if col not in ['safety_rating', 'claim']: # Prevent overwriting targets if present
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
        # Downcasting
        if df_clean[col].dtype == 'float64':
            df_clean[col] = df_clean[col].astype('float32')
        elif df_clean[col].dtype == 'int64':
            df_clean[col] = df_clean[col].astype('int32')
            
    cat_cols = df_clean.select_dtypes(exclude=['int32', 'float32']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna('Missing')
        
    return df_clean

def main():
    # START THE TIMER
    start_time = time.time()
    
    # 1. Argument Parsing
    if len(sys.argv) != 3:
        print("Usage: python3 z5543164.py <train_file> <test_file>")
        sys.exit(1)
        
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    
    # 2. Load Datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 3. Preprocess Separately (No concatenation)
    train_clean = preprocess(train_df)
    test_clean = preprocess(test_df)
    
    # 4. Robust Categorical Encoding
    cat_cols = train_clean.select_dtypes(exclude=['int32', 'float32']).columns.tolist()
    if 'policy_id' in cat_cols:
        cat_cols.remove('policy_id')
        
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_clean[cat_cols] = encoder.fit_transform(train_clean[cat_cols])
    test_clean[cat_cols] = encoder.transform(test_clean[cat_cols])
    
    # 5. Define Base Features (excluding targets and IDs)
    base_features = [col for col in train_clean.columns if col not in ['policy_id', 'safety_rating', 'claim']]
    
    # ---------------------------------------------------------
    # PART II: REGRESSION (Predicting safety_rating)
    # ---------------------------------------------------------
    X_train_reg = train_clean[base_features]
    y_train_reg = train_clean['safety_rating']
    X_test_reg = test_clean[base_features]
    
    # Initialize with tuned hyperparameters
    reg_model = LGBMRegressor(
        random_state=42, 
        n_jobs=-1,
        n_estimators=287,
        learning_rate=0.0309,
        max_depth=16,
        num_leaves=34,
        subsample=0.7824,
        colsample_bytree=0.8447
    )
    
    reg_model.fit(X_train_reg, y_train_reg)
    test_clean['safety_rating'] = reg_model.predict(X_test_reg)
    
    # ---------------------------------------------------------
    # PART III: CLASSIFICATION (Predicting claim)
    # ---------------------------------------------------------
    # Utilizing safety_rating as an additional feature
    clf_features = base_features + ['safety_rating']
    X_train_clf = train_clean[clf_features]
    y_train_clf = train_clean['claim']
    X_test_clf = test_clean[clf_features]
    
    # Initialize with tuned hyperparameters
    clf_model = LGBMClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        learning_rate=0.0371,
        max_depth=9,
        min_child_samples=33,
        n_estimators=316,
        num_leaves=79
    )
    
    clf_model.fit(X_train_clf, y_train_clf)
    test_clean['claim'] = clf_model.predict(X_test_clf)
    
    # ---------------------------------------------------------
    # EXPORT PREDICTIONS
    # ---------------------------------------------------------
    reg_output = test_clean[['policy_id', 'safety_rating']]
    reg_output.to_csv('z5543164_regression.csv', index=False)
    
    clf_output = test_clean[['policy_id', 'claim']]
    clf_output.to_csv('z5543164_classification.csv', index=False)
    
    # STOP THE TIMER AND PRINT
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("Predictions generated successfully.")
    print(f"Total pipeline runtime: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

def preprocess(df, encoder=None, imputer=None, is_train=True):
    """
    Cleans and prepares the data. 
    Strictly avoids merging train and test.
    """
    # 1. Drop the target columns if they exist and we are preprocessing for features
    # Note: Keep policy_id aside for the final output
    policy_ids = df['policy_id']
    
    # Identify feature columns (everything except targets and ID)
    features = df.drop(columns=['policy_id', 'safety_rating', 'claim'], errors='ignore')
    
    # 2. Handle Categorical Data
    # Using OrdinalEncoder with 'handle_unknown' to manage unseen values in test
    cat_cols = features.select_dtypes(include=['object']).columns
    
    if is_train:
        imputer = SimpleImputer(strategy='most_frequent')
        features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
        
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        features[cat_cols] = encoder.fit_transform(features[cat_cols])
        return features, policy_ids, encoder, imputer
    else:
        # Use the objects fitted on training data
        features = pd.DataFrame(imputer.transform(features), columns=features.columns)
        features[cat_cols] = encoder.transform(features[cat_cols])
        return features, policy_ids

def main():
    # 1. Check Command Line Arguments
    if len(sys.argv) != 3:
        print("Usage: python3 z5543164.py <train_file> <test_file>")
        sys.exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Preprocessing
    X_train, _, encoder, imputer = preprocess(train_df, is_train=True)
    y_reg = train_df['safety_rating']
    y_clf = train_df['claim']
    
    X_test, test_ids = preprocess(test_df, encoder, imputer, is_train=False)
    
    # 3. Part II: Regression (Safety Rating)
    # Using a fast model to stay under 2 mins
    reg_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    reg_model.fit(X_train, y_reg)
    reg_preds = reg_model.predict(X_test)
    
    # Create Regression Output
    reg_output = pd.DataFrame({'policy_id': test_ids, 'safety_rating': reg_preds})
    reg_output.to_csv('z5555555_regression.csv', index=False)
    
    # 4. Part III: Classification (Claim)
    # Target leakage rule: You CAN use safety_rating as a feature for claim
    # To keep it simple, we'll just train on the original features here.
    clf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    clf_model.fit(X_train, y_clf)
    clf_preds = clf_model.predict(X_test)
    
    # Create Classification Output
    clf_output = pd.DataFrame({'policy_id': test_ids, 'claim': clf_preds})
    clf_output.to_csv('z5555555_classification.csv', index=False)
    
    print("Predictions generated successfully.")

if __name__ == "__main__":
    main()
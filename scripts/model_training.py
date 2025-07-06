import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, classification_report)
import time
import joblib
import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data():
    """Load and preprocess the loan data."""
    print("Loading and preprocessing data...")
    df = pd.read_csv('data/processed/processed_loan_data_encoded.csv')
    
    # Check if target exists
    if 'default' not in df.columns:
        raise ValueError("Target column 'default' not found in the dataset")
    
    # Separate features and target
    X = df.drop('default', axis=1)
    y = df['default']
    
    # Convert all columns to numeric, coercing errors
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values - use median for numeric columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # For any remaining missing values, fill with 0
    X = X.fillna(0)
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    
    # Ensure no NaN values remain
    if X.isna().any().any():
        raise ValueError("NaN values still present in the data after preprocessing")
    
    return X, y, feature_names

def create_models():
    """Create model instances with initial parameters."""
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.5,  # Handle class imbalance
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    return models

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'inference_time': time.time() - start_time
    }
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return metrics

def tune_hyperparameters(model, X, y):
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    if isinstance(model, XGBClassifier):
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5),
            'min_child_weight': randint(1, 10)
        }
    elif isinstance(model, LGBMClassifier):
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        }
    else:  # Random Forest
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 11),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=30,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print(f"\nTuning {model.__class__.__name__}...")
    search.fit(X, y)
    
    return search.best_estimator_

def main():
    """
    Main function to run the model training pipeline.
    This includes data loading, preprocessing, model comparison,
    hyperparameter tuning, and saving the final artifacts.
    """
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    print("Applying SMOTE for class balancing...")
    print(f"Class distribution before SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        print(f"Class distribution after SMOTE: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
    except Exception as e:
        print(f"SMOTE failed: {e}. Falling back to RandomOverSampler.")
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)

    # --- Model Comparison ---
    models = create_models()
    results = []
    for name, model in models.items():
        print(f"\n{'='*50}\nTraining {name}...\n{'='*50}")
        start_time = time.time()
        model.fit(X_train_resampled, y_train_resampled)
        training_time = time.time() - start_time
        
        metrics = evaluate_model(model, X_test_scaled, y_test)
        metrics['model'] = name
        metrics['training_time'] = training_time
        results.append(metrics)

    # --- Hyperparameter Tuning ---
    results_df = pd.DataFrame(results)
    best_model_name = results_df.sort_values('roc_auc', ascending=False).iloc[0]['model']
    best_model = models[best_model_name]
    
    print(f"\n{'='*50}\nTuning Best Model: {best_model_name}\n{'='*50}")
    tuned_model = tune_hyperparameters(best_model, X_train_resampled, y_train_resampled)

    # --- Final Evaluation ---
    print("\nEvaluating tuned model...")
    tuned_metrics = evaluate_model(tuned_model, X_test_scaled, y_test)
    tuned_metrics['model'] = f'Tuned {best_model_name}'
    
    # --- Save Results ---
    tuned_df = pd.DataFrame([tuned_metrics])
    final_results_df = pd.concat([results_df, tuned_df], ignore_index=True)
    final_results_df.to_csv('reports/model_training_results.csv', index=False)
    
    print("\n--- Training Complete ---")
    print("\nFinal Results:")
    print(final_results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].round(4))
    print("\nResults saved to 'reports/model_training_results.csv'")

    # --- Save Final Artifacts ---
    print("\nSaving final model, scaler, and feature list...")
    
    # Save final tuned model
    joblib.dump(tuned_model, 'models/final_model.joblib')
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save the feature list
    with open('models/final_model_features.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
            
    # Save feature importances of the final model
    if hasattr(tuned_model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': tuned_model.feature_importances_
        }).sort_values('importance', ascending=False)
        importances.to_csv('reports/final_model_feature_importances.csv', index=False)
        print("Feature importances of the final model saved.")

    print("\nFinal artifacts saved successfully to 'models/' directory.")

if __name__ == "__main__":
    main()

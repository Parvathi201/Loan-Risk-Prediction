import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_data():
    """Load and preprocess the data"""
    print("Loading and preprocessing data...")
    
    try:
        # Load the processed and encoded data
        df = pd.read_csv('data/processed/processed_loan_data_encoded.csv')
        
        # Check if target column exists
        if 'default' not in df.columns:
            raise ValueError("Target column 'default' not found in the dataset")
        
        # Convert date column to datetime and extract useful features
        if 'application_date' in df.columns:
            df['application_date'] = pd.to_datetime(df['application_date'])
            df['app_year'] = df['application_date'].dt.year
            df['app_month'] = df['application_date'].dt.month
            df['app_day'] = df['application_date'].dt.day
            df = df.drop('application_date', axis=1)
        
        # Convert all columns to numeric, coercing errors
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values that might have been created
        df = df.dropna()
        
        # Separate features and target
        X = df.drop('default', axis=1)
        y = df['default']
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Default rate: {y.mean():.2%}")
        
        return X, y, df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def correlation_analysis(X, y, threshold=0.9):
    """
    Perform correlation analysis and remove highly correlated features.
    Returns filtered feature set and correlation matrix.
    """
    print("\nPerforming correlation analysis...")
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Create mask for heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot correlation heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Feature Correlation Heatmap")
    
    # Save the plot
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/feature_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # Identify highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"\nFeatures to drop due to high correlation (> {threshold}):")
    print(to_drop)
    
    # Drop highly correlated features
    X_filtered = X.drop(columns=to_drop)
    
    return X_filtered, corr_matrix, to_drop

def recursive_feature_elimination(X, y, n_features_to_select=20):
    """
    Perform Recursive Feature Elimination (RFE) to select top features.
    Returns selected features and their rankings.
    """
    print("\nPerforming Recursive Feature Elimination (RFE)...")
    
    # Use Random Forest as the estimator
    estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    
    # Create RFE object
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=0.1  # Remove 10% of features at each step
    )
    
    # Fit RFE
    rfe.fit(X, y)
    
    # Get selected features and rankings
    selected_features = X.columns[rfe.support_].tolist()
    feature_rankings = pd.DataFrame({
        'Feature': X.columns,
        'Ranking': rfe.ranking_,
        'Selected': rfe.support_
    }).sort_values('Ranking')
    
    print("\nTop features selected by RFE:")
    print(selected_features)
    
    # Plot feature rankings
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='Ranking', 
        y='Feature', 
        data=feature_rankings[feature_rankings['Ranking'] <= 30],  # Show top 30
        palette='viridis'
    )
    plt.title('Feature Rankings from RFE')
    plt.tight_layout()
    plt.savefig('reports/figures/rfe_feature_rankings.png', bbox_inches='tight')
    plt.close()
    
    return selected_features, feature_rankings

def shap_analysis(X, y, model_type='xgb'):
    """
    Perform SHAP analysis to understand feature importance.
    Returns SHAP values and plots.
    """
    print("\nPerforming SHAP analysis...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Train a model
    if model_type == 'xgb':
        model = XGBClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=RANDOM_STATE, 
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:  # Default to Random Forest
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )
    
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy for SHAP analysis: {accuracy:.4f}")
    
    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train)
    
    # Calculate SHAP values (using a subset for faster computation)
    shap_values = explainer(X_train[:100])  # Using first 100 samples for speed
    
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar)")
    plt.tight_layout()
    plt.savefig('reports/figures/shap_feature_importance_bar.png', bbox_inches='tight')
    plt.close()
    
    # Detailed summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title("SHAP Feature Importance (Detailed)")
    plt.tight_layout()
    plt.savefig('reports/figures/shap_feature_importance_detailed.png', bbox_inches='tight')
    plt.close()
    
    # Return SHAP values and feature importance
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': np.abs(shap_values.values).mean(0)
    }).sort_values('shap_importance', ascending=False)
    
    print("\nTop 10 features by SHAP importance:")
    print(shap_df.head(10).to_string(index=False))
    
    return shap_df

def main():
    # Load data
    X, y, df = load_data()
    
    # 1. Correlation Analysis
    X_filtered, corr_matrix, dropped_features = correlation_analysis(X, y)
    
    # 2. Recursive Feature Elimination
    selected_features, feature_rankings = recursive_feature_elimination(
        X_filtered, y, n_features_to_select=min(20, X_filtered.shape[1])
    )
    
    # 3. SHAP Analysis
    try:
        shap_df = shap_analysis(X_filtered, y)
        
        # Combine feature importance from different methods
        feature_importance = pd.DataFrame({
            'feature': X_filtered.columns,
            'rfe_ranking': feature_rankings.set_index('Feature')['Ranking'],
            'shap_importance': shap_df.set_index('feature')['shap_importance']
        }).fillna(0)
        
        # Save results
        os.makedirs('reports', exist_ok=True)
        feature_importance.to_csv('reports/feature_importance_analysis.csv', index=False)
        
        print("\nFeature selection completed successfully!")
        print(f"Results saved to 'reports/feature_importance_analysis.csv'")
        
    except Exception as e:
        print(f"\nError during SHAP analysis: {str(e)}")
        print("Continuing with other results...")
        
        # Save RFE results even if SHAP fails
        os.makedirs('reports', exist_ok=True)
        feature_rankings.to_csv('reports/rfe_feature_rankings.csv', index=False)
        print("RFE results saved to 'reports/rfe_feature_rankings.csv'")

if __name__ == "__main__":
    main()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy import stats
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
        
        return X, y
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def detect_outliers_zscore(data, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(data))
    outliers = (z_scores > threshold).any(axis=1)
    return outliers

def detect_outliers_isolation_forest(data, contamination=0.05):
    """Detect outliers using Isolation Forest"""
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    outliers = iso_forest.fit_predict(data) == -1
    return outliers

def detect_outliers_oneclass_svm(data, nu=0.05):
    """Detect outliers using One-Class SVM"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    oc_svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    outliers = oc_svm.fit_predict(X_scaled) == -1
    return outliers

def analyze_outliers(X, y, outliers, method_name):
    """Analyze and visualize the impact of outliers"""
    print(f"\n{'='*50}")
    print(f"Outlier Analysis - {method_name}")
    print("="*50)
    
    # Calculate basic statistics
    n_outliers = sum(outliers)
    outlier_percentage = n_outliers / len(X) * 100
    
    print(f"Number of outliers detected: {n_outliers} ({outlier_percentage:.2f}%)")
    
    # Analyze target distribution in outliers vs inliers
    if y is not None:
        outlier_targets = y[outliers]
        inlier_targets = y[~outliers]
        
        print("\nDefault rate comparison:")
        print(f"- In default rows: {sum(y) / len(y):.2%}")
        print(f"- In outliers: {sum(outlier_targets) / len(outlier_targets):.2%}" if len(outlier_targets) > 0 else "- No outliers detected")
        print(f"- In inliers: {sum(inlier_targets) / len(inlier_targets):.2%}" if len(inlier_targets) > 0 else "- No inliers detected")
    
    return n_outliers, outlier_percentage

def plot_outliers(X, y, outliers, method_name, n_features=5):
    """Plot outliers for the most important numerical features"""
    # Select numerical features with highest variance
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > n_features:
        # Select top n_features with highest variance
        top_variance_cols = X[numerical_cols].var().nlargest(n_features).index
    else:
        top_variance_cols = numerical_cols
    
    # Create subplots
    n_cols = min(2, len(top_variance_cols))
    n_rows = (len(top_variance_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.ravel()
    
    for i, col in enumerate(top_variance_cols):
        ax = axes[i]
        
        # Plot inliers
        inliers = X.loc[~outliers, col]
        sns.histplot(inliers, kde=True, color='blue', label='Inliers', ax=ax)
        
        # Plot outliers
        if sum(outliers) > 0:
            outlier_vals = X.loc[outliers, col]
            sns.histplot(outlier_vals, kde=True, color='red', label='Outliers', ax=ax)
        
        ax.set_title(f'Distribution of {col}')
        ax.legend()
    
    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f'Outlier Detection - {method_name}', y=1.02)
    plt.tight_layout()
    
    # Save the plot in the reports/outlier_detection directory
    os.makedirs('reports/outlier_detection', exist_ok=True)
    plt.savefig(f'reports/outlier_detection/outlier_analysis_{method_name.lower().replace(" ", "_")}.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load data
    X, y = load_data()
    
    # Standardize the data for outlier detection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Dictionary to store results
    results = []
    
    # 1. Z-score method
    print("\n" + "="*50)
    print("Detecting outliers using Z-score method...")
    outliers_zscore = detect_outliers_zscore(X_scaled)
    n_outliers, pct_outliers = analyze_outliers(X, y, outliers_zscore, "Z-score")
    plot_outliers(X, y, outliers_zscore, "Z-score")
    results.append({
        'Method': 'Z-score',
        'Outliers Detected': n_outliers,
        'Percentage': f"{pct_outliers:.2f}%"
    })
    
    # 2. Isolation Forest
    print("\n" + "="*50)
    print("Detecting outliers using Isolation Forest...")
    outliers_iso = detect_outliers_isolation_forest(X_scaled)
    n_outliers, pct_outliers = analyze_outliers(X, y, outliers_iso, "Isolation Forest")
    plot_outliers(X, y, outliers_iso, "Isolation Forest")
    results.append({
        'Method': 'Isolation Forest',
        'Outliers Detected': n_outliers,
        'Percentage': f"{pct_outliers:.2f}%"
    })
    
    # 3. One-Class SVM
    print("\n" + "="*50)
    print("Detecting outliers using One-Class SVM...")
    outliers_svm = detect_outliers_oneclass_svm(X_scaled)
    n_outliers, pct_outliers = analyze_outliers(X, y, outliers_svm, "One-Class SVM")
    plot_outliers(X, y, outliers_svm, "One-Class SVM")
    results.append({
        'Method': 'One-Class SVM',
        'Outliers Detected': n_outliers,
        'Percentage': f"{pct_outliers:.2f}%"
    })
    
    # Print summary of results
    print("\n" + "="*50)
    print("Outlier Detection Summary:")
    print("="*50)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save results to CSV in the reports/outlier_detection directory
    os.makedirs('reports/outlier_detection', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv('reports/outlier_detection/summary.csv', index=False)
    print(f"\nResults saved to 'reports/outlier_detection/summary.csv'")

if __name__ == "__main__":
    main()

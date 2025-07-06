import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory for plots
os.makedirs('reports/figures', exist_ok=True)

def load_data():
    """Load the dataset"""
    print("Loading dataset...")
    df = pd.read_csv('data/raw/loan_risk_dataset.csv')
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def basic_info(df):
    """Display basic information about the dataset"""
    print("\n" + "="*50)
    print("BASIC DATASET INFORMATION")
    print("="*50)
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    
    # Dataset shape
    print(f"\nDataset shape: {df.shape}")
    
    # Data types
    print("\nData types:")
    print(df.dtypes)
    
    # Basic statistics for numerical columns
    print("\nDescriptive statistics for numerical columns:")
    print(df.describe(include=[np.number]).to_string())
    
    # Basic statistics for categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        print("\nDescriptive statistics for categorical columns:")
        print(df[cat_cols].describe().to_string())

def check_missing_values(df):
    """Check for missing values"""
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS")
    print("="*50)
    
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    }).sort_values('Missing Values', ascending=False)
    
    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing Values'] > 0]
    
    if missing_df.empty:
        print("\nNo missing values found in the dataset!")
    else:
        print(f"\nFound {len(missing_df)} columns with missing values:")
        print(missing_df.to_string())
        
        # Plot missing values
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_df.index, y='Percentage', data=missing_df)
        plt.title('Percentage of Missing Values by Column')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('reports/figures/missing_values.png')
        plt.show()

def analyze_numerical_features(df):
    """Analyze numerical features"""
    print("\n" + "="*50)
    print("NUMERICAL FEATURES ANALYSIS")
    print("="*50)
    
    # Select numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target variable if it exists
    if 'default' in num_cols:
        num_cols.remove('default')
    
    # Plot distributions
    print("\nDistributions of numerical features:")
    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.ravel()
    
    for i, col in enumerate(num_cols):
        if i < len(axes):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel('')
    
    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('reports/figures/numerical_distributions.png')
    plt.show()
    
    # Boxplots to identify outliers
    print("\nBoxplots for numerical features (checking for outliers):")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.ravel()
    
    for i, col in enumerate(num_cols):
        if i < len(axes):
            sns.boxplot(data=df, y=col, ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')
            axes[i].set_ylabel('')
    
    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('reports/figures/boxplots.png')
    plt.show()

def analyze_categorical_features(df):
    """Analyze categorical features"""
    print("\n" + "="*50)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*50)
    
    # Select categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not cat_cols:
        print("\nNo categorical features found in the dataset.")
        return
    
    # Plot value counts for each categorical feature
    n_cols = 2
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.ravel()
    
    for i, col in enumerate(cat_cols):
        if i < len(axes):
            value_counts = df[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i])
            axes[i].set_title(f'Value Counts for {col}')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide any remaining empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('reports/figures/categorical_distributions.png')
    plt.show()

def analyze_target_variable(df):
    """Analyze the target variable"""
    if 'default' not in df.columns:
        print("\nNo 'default' column found in the dataset.")
        return
    
    print("\n" + "="*50)
    print("TARGET VARIABLE ANALYSIS")
    print("="*50)
    
    # Class distribution
    target_dist = df['default'].value_counts(normalize=True) * 100
    
    print("\nClass distribution (percentage):")
    print(target_dist)
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='default')
    plt.title('Distribution of Loan Defaults')
    plt.xlabel('Default (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.savefig('reports/figures/target_distribution.png')
    plt.show()
    
    # Analyze target vs numerical features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'default' in num_cols:
        num_cols.remove('default')
    
    if num_cols:
        print("\nCorrelation with target variable:")
        corr_with_target = df[num_cols + ['default']].corr()['default'].sort_values(ascending=False)
        print(corr_with_target.to_string())
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[num_cols + ['default']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('reports/figures/correlation_heatmap.png')
        plt.show()

def analyze_date_features(df):
    """Analyze date features"""
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if not date_cols:
        # Try to infer date columns
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    continue
    
    if not date_cols:
        print("\nNo date features found in the dataset.")
        return
    
    print("\n" + "="*50)
    print("DATE FEATURES ANALYSIS")
    print("="*50)
    
    for col in date_cols:
        print(f"\nAnalyzing date column: {col}")
        
        # Extract date components
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        
        # Plot time series
        plt.figure(figsize=(14, 5))
        
        # Plot 1: Count by date
        plt.subplot(1, 2, 1)
        df[col].value_counts().sort_index().plot()
        plt.title(f'Count by {col}')
        plt.xlabel('Date')
        plt.ylabel('Count')
        
        # Plot 2: Default rate by month
        plt.subplot(1, 2, 2)
        if 'default' in df.columns:
            monthly_default = df.groupby(df[col].dt.to_period('M'))['default'].mean()
            monthly_default.plot(kind='bar')
            plt.title(f'Default Rate by Month ({col})')
            plt.xlabel('Month')
            plt.ylabel('Default Rate')
        
        plt.tight_layout()
        plt.savefig(f'reports/figures/{col}_analysis.png')
        plt.show()

def plot_default_rates(df):
    """Plot default rates across different demographics"""
    print("\nGenerating demographic default rate visualizations...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Demographic Default Rate Analysis', fontsize=16)
    
    # 1. Default rate by age group
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], 
                               bins=[18, 25, 35, 45, 55, 65, 100],
                               labels=['18-25', '26-35', '36-45', '46-55', '56-65', '66+'])
        age_default = df.groupby('age_group', observed=False)['default'].mean().reset_index()
        
        sns.barplot(x='age_group', y='default', data=age_default, ax=axes[0,0])
        axes[0,0].set_title('Default Rate by Age Group', fontsize=12)
        axes[0,0].set_xlabel('Age Group')
        axes[0,0].set_ylabel('Default Rate')
    
    # 2. Default rate by income quartile
    if 'income_annual' in df.columns:
        try:
            df['income_quartile'] = pd.qcut(df['income_annual'], q=4, 
                                          labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'],
                                          duplicates='drop')
            income_default = df.groupby('income_quartile', observed=False)['default'].mean().reset_index()
            
            sns.barplot(x='income_quartile', y='default', data=income_default, ax=axes[0,1])
            axes[0,1].set_title('Default Rate by Income Quartile', fontsize=12)
            axes[0,1].set_xlabel('Income Quartile')
            axes[0,1].set_ylabel('Default Rate')
        except ValueError as e:
            print(f"Could not plot income quartiles: {e}")

    # 3. Default rate by employment length
    if 'employment_length' in df.columns:
        df['employment_group'] = pd.cut(df['employment_length'],
                                     bins=[-1, 1, 5, 10, 50],
                                     labels=['<1 year', '1-5 years', '6-10 years', '>10 years'])
        emp_default = df.groupby('employment_group', observed=False)['default'].mean().reset_index()
        
        sns.barplot(x='employment_group', y='default', data=emp_default, ax=axes[1,0])
        axes[1,0].set_title('Default Rate by Employment Length', fontsize=12)
        axes[1,0].set_xlabel('Employment Length')
        axes[1,0].set_ylabel('Default Rate')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Default rate by home ownership
    if 'home_ownership' in df.columns:
        home_default = df.groupby('home_ownership')['default'].mean().reset_index()
        
        sns.barplot(x='home_ownership', y='default', data=home_default, ax=axes[1,1])
        axes[1,1].set_title('Default Rate by Home Ownership', fontsize=12)
        axes[1,1].set_xlabel('Home Ownership')
        axes[1,1].set_ylabel('Default Rate')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('reports/figures/demographic_default_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved demographic default rate visualizations to 'reports/figures/demographic_default_rates.png'")


def analyze_risk_profiles(df):
    """Analyze risk profiles based on key features"""
    print("\nAnalyzing risk profiles...")
    
    try:
        def calculate_risk_score(row):
            score = 0
            if 'credit_score' in row and not pd.isna(row['credit_score']):
                if row['credit_score'] > 750: score += 30
                elif row['credit_score'] > 650: score += 15
            
            if 'debt_to_income_ratio' in row and not pd.isna(row['debt_to_income_ratio']):
                if row['debt_to_income_ratio'] < 0.2: score += 20
                elif row['debt_to_income_ratio'] < 0.4: score += 10
            
            if 'income_annual' in row and not pd.isna(row['income_annual']):
                if row['income_annual'] > 100000: score += 15
            
            if 'employment_length' in row and not pd.isna(row['employment_length']):
                if row['employment_length'] > 10: score += 10
            
            return score
        
        required_cols = ['credit_score', 'debt_to_income_ratio', 'income_annual', 'employment_length', 'loan_amount']
        if all(col in df.columns for col in required_cols):
            df['risk_score'] = df.apply(calculate_risk_score, axis=1)
            
            df['risk_segment'] = pd.cut(
                df['risk_score'],
                bins=[-1, 20, 40, 60, 80, 101],
                labels=['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk']
            )
            
            plt.figure(figsize=(12, 6))
            risk_default = df.groupby('risk_segment', observed=False)['default'].mean().sort_index()
            sns.barplot(x=risk_default.index, y=risk_default.values)
            plt.title('Default Rate by Risk Segment', fontsize=14)
            plt.xlabel('Risk Segment')
            plt.ylabel('Default Rate')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('reports/figures/risk_segments.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=df.sample(min(1000, len(df))),
                x='income_annual',
                y='loan_amount',
                hue='default',
                alpha=0.6,
                palette={0: 'green', 1: 'red'}
            )
            plt.title('Income vs Loan Amount by Default Status', fontsize=14)
            plt.xlabel('Annual Income')
            plt.ylabel('Loan Amount')
            plt.tight_layout()
            plt.savefig('reports/figures/income_vs_loan.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Saved risk profile visualizations.")
            
    except Exception as e:
        print(f"Error analyzing risk profiles: {str(e)}")

def save_eda_report(df):
    """Save EDA report"""
    report = []
    
    # Basic info
    report.append("# Exploratory Data Analysis Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n## Dataset Overview")
    report.append(f"- Number of observations: {df.shape[0]}")
    report.append(f"- Number of features: {df.shape[1]}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    }).sort_values('Missing Values', ascending=False)
    missing_df = missing_df[missing_df['Missing Values'] > 0]
    
    report.append("\n## Missing Values")
    if missing_df.empty:
        report.append("No missing values found in the dataset.")
    else:
        report.append("The following columns contain missing values:")
        report.append(missing_df.to_markdown())
    
    # Target variable analysis
    if 'default' in df.columns:
        target_dist = df['default'].value_counts(normalize=True) * 100
        report.append("\n## Target Variable Analysis")
        report.append("### Class Distribution")
        report.append(target_dist.to_markdown())
        
        # Correlation with target
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'default' in num_cols:
            num_cols.remove('default')
        if num_cols:
            corr_with_target = df[num_cols + ['default']].corr()['default'].sort_values(ascending=False)
            report.append("\n### Correlation with Target")
            report.append("Top 10 features most correlated with the target variable:")
            report.append(corr_with_target.head(10).to_markdown())
    
    # Save report
    with open('reports/eda_report.md', 'w') as f:
        f.write('\n'.join(report))
    
    print("\nEDA report saved to 'reports/eda_report.md'")

def main():
    # Load data
    df = load_data()
    
    # Basic info
    basic_info(df)
    
    # Check for missing values
    check_missing_values(df)
    
    # Analyze numerical features
    analyze_numerical_features(df)
    
    # Analyze categorical features
    analyze_categorical_features(df)
    
    # Analyze target variable
    analyze_target_variable(df)
    
    # Analyze date features
    analyze_date_features(df)

    # Analyze demographics and risk profiles
    plot_default_rates(df)
    analyze_risk_profiles(df)
    
    # Save EDA report
    save_eda_report(df)
    
    print("\n" + "="*50)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nCheck the 'reports' directory for visualizations and the EDA report.")

if __name__ == "__main__":
    main()

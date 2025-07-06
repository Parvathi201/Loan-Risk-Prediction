"""
Feature engineering pipeline for the loan risk prediction project.
Transforms raw loan data into features suitable for modeling.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import data utilities
from .data_utils import (
    DataValidationError, validate_features, load_raw_data,
    save_processed_data, PROJECT_ROOT, logger
)

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the input DataFrame against expected schema.
    
    Args:
        df: Input DataFrame to validate
        
    Returns:
        Validated DataFrame with correct data types
        
    Raises:
        DataValidationError: If validation fails
    """
    logger.info("Validating input data schema...")
    
    required_columns = {
        'age': 'int64',
        'income_annual': 'float64',
        'loan_amount': 'float64',
        'risk_score': 'float64',
        'debt_to_income_ratio': 'float64',
        'credit_history': 'float64',
        'loan_grade': 'object',
        'default': 'int64'
    }
    
    # Check for missing columns
    missing_columns = set(required_columns.keys()) - set(df.columns)
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")
    
    # Convert data types
    for col, dtype in required_columns.items():
        if col in df.columns:
            try:
                if dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
    
    logger.info("Data validation completed successfully")
    return df

def load_and_validate_data() -> pd.DataFrame:
    """
    Load and validate the raw dataset.
    
    Returns:
        Validated DataFrame with raw data
        
    Raises:
        DataValidationError: If data loading or validation fails
    """
    try:
        # Load raw data using data_utils
        df = load_raw_data()
        
        # Validate the dataframe
        df = validate_dataframe(df)
        
        # Log data quality metrics
        missing_values = df.isnull().sum().sum()
        dup_rows = df.duplicated().sum()
        
        logger.info(f"Data Quality Check:")
        logger.info(f"- Missing values: {missing_values}")
        logger.info(f"- Duplicate rows: {dup_rows}")
        
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in the dataset")
        if dup_rows > 0:
            logger.warning(f"Found {dup_rows} duplicate rows in the dataset")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading and validating data: {str(e)}")
        raise

def create_output_dirs() -> None:
    """Create necessary output directories."""
    dirs = [
        PROJECT_ROOT / 'data' / 'processed',
        PROJECT_ROOT / 'reports' / 'figures'
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")

def handle_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from date columns with error handling.
    
    Args:
        df: Input DataFrame with 'application_date' column
        
    Returns:
        DataFrame with added date features
    """
    logger.info("Extracting date features...")
    
    try:
        # Convert to datetime with error handling
        df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
        
        # Handle any invalid dates by filling with the most recent date
        if df['application_date'].isnull().any():
            most_recent = df['application_date'].max()
            df['application_date'].fillna(most_recent, inplace=True)
            logger.warning(
                f"{df['application_date'].isnull().sum()} invalid dates "
                f"filled with {most_recent.strftime('%Y-%m-%d')}"
            )
        
        # Calculate reference date (1 year before most recent application)
        reference_date = df['application_date'].max() - pd.DateOffset(years=1)
        
        # Extract date components
        df['app_year'] = df['application_date'].dt.year
        df['app_month'] = df['application_date'].dt.month
        df['app_quarter'] = df['application_date'].dt.quarter
        df['app_dayofweek'] = df['application_date'].dt.dayofweek
        df['app_weekend'] = df['app_dayofweek'].isin([5, 6]).astype(int)
        
        # Time-based features relative to reference date
        df['days_since_reference'] = (df['application_date'] - reference_date).dt.days
        
        logger.info(f"Created {len([c for c in df.columns if c.startswith('app_')])} date features")
        return df
        
    except Exception as e:
        logger.error(f"Error processing dates: {str(e)}")
        raise

def create_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new numerical features with validation.
    
    Args:
        df: Input DataFrame with raw numerical features
        
    Returns:
        DataFrame with added numerical features
    """
    logger.info("Creating numerical features...")
    
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Loan amount relative to income with validation
        df['loan_to_income_ratio'] = np.where(
            df['income_annual'] > 0,
            df['loan_amount'] / df['income_annual'],
            np.nan
        )
        
        # Monthly payment estimation with validation
        df['monthly_payment'] = np.where(
            df['loan_amount_term'] > 0,
            df['loan_amount'] / df['loan_amount_term'],
            np.nan
        )
        
        # Debt-to-income ratio with small epsilon to avoid division by zero
        df['debt_to_income'] = df['monthly_payment'] / (df['income_annual'] / 12 + 1e-6)
        
        # Credit score to loan amount ratio with small epsilon to avoid division by zero
        df['credit_to_loan_ratio'] = df['risk_score'] / (df['loan_amount'] + 1e-6)
        
        # Income per dependent with small epsilon to avoid division by zero
        df['income_per_dependent'] = df['income_annual'] / (df.get('dependents', 0) + 1)
        
        # Payment to income ratio
        df['payment_to_income_ratio'] = (df['monthly_payment'] * 12) / (df['income_annual'] + 1e-6)
        
        # Fill any infinite or NA values
        numerical_cols = [
            'loan_to_income_ratio', 'monthly_payment', 'debt_to_income',
            'credit_to_loan_ratio', 'income_per_dependent', 'payment_to_income_ratio'
        ]
        
        for col in numerical_cols:
            if col in df.columns:
                # Replace inf with NaN first
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                # Fill NA with median of the column
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        logger.info(f"Created {len(numerical_cols)} numerical features")
        return df
        
    except Exception as e:
        logger.error(f"Error creating numerical features: {str(e)}")
        raise

def create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new categorical features through binning and discretization.
    
    Args:
        df: Input DataFrame with numerical features
        
    Returns:
        DataFrame with added categorical features
    """
    logger.info("Creating categorical features...")
    
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Age groups
        age_bins = [18, 25, 35, 45, 55, 65, 100]
        age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        df['age_group'] = pd.cut(
            df['age'],
            bins=age_bins,
            labels=age_labels,
            right=False
        )
        
        # Income groups (in USD)
        income_bins = [0, 30_000, 60_000, 100_000, 200_000, float('inf')]
        income_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        df['income_group'] = pd.cut(
            df['income_annual'],
            bins=income_bins,
            labels=income_labels
        )
        
        # Loan amount groups (in USD)
        loan_bins = [0, 10_000, 25_000, 50_000, 100_000, float('inf')]
        loan_labels = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
        df['loan_amount_group'] = pd.cut(
            df['loan_amount'],
            bins=loan_bins,
            labels=loan_labels
        )
        
        # Debt burden categories
        if 'debt_to_income_ratio' in df.columns:
            df['debt_burden'] = pd.cut(
                df['debt_to_income_ratio'],
                bins=[0, 0.1, 0.2, 0.3, 0.4, float('inf')],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        # Risk score buckets (if risk_score exists)
        if 'risk_score' in df.columns:
            df['risk_score_bucket'] = pd.cut(
                df['risk_score'],
                bins=[0, 400, 500, 600, 700, 850, 1000],
                labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
            )
        
        logger.info(f"Created {len([c for c in df.columns if c.endswith('_group') or c.endswith('_bucket') or c == 'debt_burden'])} categorical features")
        return df
        
    except Exception as e:
        logger.error(f"Error creating categorical features: {str(e)}")
        raise

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using one-hot encoding.
    
    Args:
        df: Input DataFrame with categorical features
        
    Returns:
        DataFrame with one-hot encoded categorical features
    """
    logger.info("Encoding categorical features...")
    
    try:
        # Define all possible categorical columns
        categorical_cols = [
            'gender', 'marital_status', 'education', 'self_employed',
            'property_area', 'loan_grade', 'risk_score_bucket',
            'age_group', 'income_group', 'loan_amount_group', 'debt_burden'
        ]
        
        # Only encode columns that exist in the dataframe
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        if not categorical_cols:
            logger.warning("No categorical columns found to encode")
            return df
            
        # Get dtypes before encoding
        original_dtypes = df[categorical_cols].dtypes
        
        # Convert all categorical columns to string type to avoid mixed-type issues
        for col in categorical_cols:
            df[col] = df[col].astype(str)
        
        # Create dummy variables
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Log encoding results
        num_original_cols = len(df.columns)
        num_new_cols = len(df_encoded.columns) - num_original_cols + len(categorical_cols)
        
        logger.info(
            f"Encoded {len(categorical_cols)} categorical features into "
            f"{num_new_cols} binary columns"
        )
        
        return df_encoded
        
    except Exception as e:
        logger.error(f"Error encoding categorical features: {str(e)}")
        raise

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers in numerical features using IQR method with percentile-based bounds.
    
    Args:
        df: Input DataFrame with numerical features
        
    Returns:
        DataFrame with capped outliers
    """
    logger.info("Handling outliers in numerical features...")
    
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target variable if it exists
        if 'default' in numerical_cols:
            numerical_cols.remove('default')
        
        if not numerical_cols:
            logger.warning("No numerical columns found for outlier handling")
            return df
        
        # Track number of outliers per column
        outliers_info = {}
        
        # Cap outliers using IQR method with 5th and 95th percentiles
        for col in numerical_cols:
            # Calculate percentiles
            Q1 = df[col].quantile(0.05)
            Q3 = df[col].quantile(0.95)
            IQR = Q3 - Q1
            
            # Calculate bounds with 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
            num_outliers = is_outlier.sum()
            
            if num_outliers > 0:
                # Cap the outliers
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                
                # Store outlier info
                pct_outliers = (num_outliers / len(df)) * 100
                outliers_info[col] = {
                    'count': num_outliers,
                    'pct': round(pct_outliers, 2)
                }
        
        # Log outlier information
        if outliers_info:
            logger.info(f"Capped outliers in {len(outliers_info)} columns:")
            for col, info in outliers_info.items():
                logger.info(f"  - {col}: {info['count']} outliers ({info['pct']}%)")
        else:
            logger.info("No significant outliers found in the data")
            
        return df
        
    except Exception as e:
        logger.error(f"Error handling outliers: {str(e)}")
        raise

def save_processed_data(df, filename='processed_loan_data.csv'):
    """Save the processed dataset"""
    output_path = f'data/processed/{filename}'
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    print(f"Final shape: {df.shape}")

def main() -> None:
    """
    Main function to run the feature engineering pipeline.
    """
    try:
        logger.info("="*60)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("="*60)
        
        # 0. Create output directories
        create_output_dirs()
        
        # 1. Load and validate data
        logger.info("\n[1/6] Loading and validating data...")
        df = load_and_validate_data()
        
        # 2. Handle date features
        logger.info("\n[2/6] Processing date features...")
        df = handle_dates(df)
        
        # 3. Create new numerical features
        logger.info("\n[3/6] Creating numerical features...")
        df = create_numerical_features(df)
        
        # 4. Create categorical features
        logger.info("\n[4/6] Creating categorical features...")
        df = create_categorical_features(df)
        
        # 5. Handle outliers
        logger.info("\n[5/6] Handling outliers...")
        df_clean = handle_outliers(df)
        
        # 6. Encode categorical features
        logger.info("\n[6/6] Encoding categorical features...")
        df_encoded = encode_categorical_features(df_clean)
        
        # 7. Save processed data
        logger.info("\nSaving processed data...")
        save_processed_data(df_clean, 'processed_loan_data.csv')
        save_processed_data(df_encoded, 'processed_loan_data_encoded.csv')
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        # Calculate and log feature statistics
        original_cols = set(load_raw_data().columns)
        new_cols = set(df_encoded.columns) - original_cols
        
        # Categorize new features
        numerical_features = [
            col for col in new_cols 
            if df_encoded[col].dtype in ['int64', 'float64']
        ]
        
        categorical_features = [
            col for col in new_cols 
            if col not in numerical_features
        ]
        
        logger.info(f"Original number of features: {len(original_cols)}")
        logger.info(f"Total features after engineering: {len(df_encoded.columns)}")
        logger.info(f"New features created: {len(new_cols)}")
        logger.info(f"  - Numerical features: {len(numerical_features)}")
        logger.info(f"  - Categorical/dummy features: {len(categorical_features)}")
        
        # Log memory usage
        logger.info("\nMemory usage:")
        logger.info(f"- Raw data: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        logger.info(f"- Processed data: {df_encoded.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("\nFeature engineering pipeline completed")

if __name__ == "__main__":
    main()

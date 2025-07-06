"""
Data loading and preprocessing utilities for the loan risk prediction project.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DATA_PATH = DATA_DIR / 'processed' / 'processed_loan_data_encoded.csv'

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

def validate_features(X: pd.DataFrame, y: pd.Series = None) -> None:
    """
    Validate features and target variable.
    
    Args:
        X: Features DataFrame
        y: Target Series (optional)
        
    Raises:
        DataValidationError: If validation fails
    """
    # Check for missing values
    if X.isnull().any().any():
        missing = X.isnull().sum()
        missing = missing[missing > 0]
        raise DataValidationError(f"Missing values found in features: {missing.to_dict()}")
    
    # Check for infinite values
    if np.isinf(X).any().any():
        inf_cols = X.columns[np.isinf(X).any()].tolist()
        raise DataValidationError(f"Infinite values found in columns: {inf_cols}")
    
    if y is not None:
        # Check target variable
        if len(np.unique(y)) < 2:
            raise DataValidationError("Target variable has less than 2 classes")
        
        # Check for class imbalance
        class_ratio = y.value_counts(normalize=True)
        if class_ratio.min() < 0.1:  # Less than 10% in minority class
            logger.warning(
                f"Severe class imbalance detected. Class distribution:\n{class_ratio}"
            )

def load_raw_data() -> pd.DataFrame:
    """
    Load raw loan risk dataset.
    
    Returns:
        DataFrame containing the raw data
    """
    raw_path = DATA_DIR / 'raw' / 'loan_risk_dataset.csv'
    logger.info(f"Loading raw data from {raw_path}")
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
    
    try:
        df = pd.read_csv(raw_path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}")
        raise

def load_processed_data(validate: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and validate processed loan data.
    
    Args:
        validate: Whether to perform validation checks
        
    Returns:
        Tuple of (features, target)
    """
    logger.info(f"Loading processed data from {PROCESSED_DATA_PATH}")
    
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}. "
            "Please run feature_engineering.py first."
        )
    
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Check if target exists
        if 'default' not in df.columns:
            raise DataValidationError("Target column 'default' not found in the dataset")
        
        # Separate features and target
        X = df.drop('default', axis=1)
        y = df['default'].astype(int)
        
        if validate:
            validate_features(X, y)
            
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        raise

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(
        f"Split data into {len(X_train)} training and {len(X_test)} test samples\n"
        f"Class distribution - Train: {dict(y_train.value_counts())}, "
        f"Test: {dict(y_test.value_counts())}"
    )
    
    return X_train, X_test, y_train, y_test

def get_feature_names(X: pd.DataFrame) -> list:
    """Get list of feature names from a DataFrame."""
    return X.columns.tolist()

def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save processed data to the processed directory.
    
    Args:
        df: DataFrame to save
        filename: Output filename (will be saved in data/processed/)
    """
    output_path = DATA_DIR / 'processed' / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise

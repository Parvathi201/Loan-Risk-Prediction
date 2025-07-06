# Phase 1: Data Preprocessing

This phase is the foundation of our project. Here, we take the raw loan data and transform it into a clean, structured format that our machine learning models can understand. The entire preprocessing pipeline has been consolidated into a single, efficient script.

## Key Files and Scripts

-   `scripts/feature_engineering.py`: This is the primary script for the data preprocessing phase. It orchestrates the entire workflow, including:
    -   **Loading and Validation:** Loading the raw dataset from `data/raw/` and validating its schema to ensure data quality.
    -   **Feature Creation:** Engineering new features from existing data, such as date-based features (e.g., loan age) and numerical ratios (e.g., loan-to-income ratio).
    -   **Outlier Handling:** Detecting and treating outliers in numerical columns to prevent them from skewing the model.
    -   **Categorical Encoding:** Converting categorical variables into a numerical format (one-hot encoding) that can be used by machine learning models.
    -   **Saving Processed Data:** Saving two versions of the processed data to `data/processed/`: one with cleaned features (`processed_loan_data.csv`) and another with all features encoded for modeling (`processed_loan_data_encoded.csv`).

-   `scripts/data_utils.py`: This script provides helper functions that are used by `feature_engineering.py` and other scripts in the project. It contains reusable code for tasks like loading data, saving files, and logging.

-   `data/raw/`: This directory contains the original, untouched dataset.
-   `data/processed/`: After running the feature engineering script, the cleaned and encoded data is saved here, ready for the next phase.

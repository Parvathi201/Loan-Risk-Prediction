import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of samples
n_samples = 5000

# Generate synthetic data
print("Generating synthetic loan risk dataset...")

# Personal Information
data = {
    'age': np.random.normal(35, 10, n_samples).astype(int).clip(18, 80),
    'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
    'marital_status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples, p=[0.3, 0.5, 0.15, 0.05]),
    'dependents': np.random.poisson(0.8, n_samples).clip(0, 5),
    'education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.7, 0.3]),
    'self_employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
    
    # Financial Information
    'income_annual': np.random.lognormal(10.5, 0.5, n_samples).astype(int),
    'loan_amount': 0,  # Will be calculated based on income
    'loan_amount_term': np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
    'credit_history': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
    'property_area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.4, 0.3, 0.3]),
    
    # Loan characteristics
    'loan_grade': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1]),
    'debt_to_income_ratio': np.random.beta(2, 5, n_samples) * 0.5,  # Between 0 and 0.5
    
    # Historical data
    'delinquencies_2yrs': np.random.poisson(0.5, n_samples),
    'open_accounts': np.random.poisson(5, n_samples).clip(1, 15),
    'total_accounts': np.random.poisson(8, n_samples).clip(1, 30),
    
    # Target variables
    'default': 0,  # Will be set based on risk factors
    'risk_score': 0  # Will be calculated
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate loan amount (1-5x annual income, but not more than 1,000,000)
df['loan_amount'] = (df['income_annual'] * np.random.uniform(0.2, 1.5, n_samples)).astype(int).clip(1000, 1000000)

# Calculate risk score (0-1000) based on various factors
df['risk_score'] = (
    (df['age'].clip(25, 65) - 25) * 5 +  # 0-200 points for age (25-65)
    (df['income_annual'] / 1000).clip(0, 100) * 2 +  # 0-200 points for income
    (df['credit_history'] * 300) +  # 0 or 300 points for credit history
    (df['education'] == 'Graduate') * 100 +  # 100 points for education
    (df['property_area'] == 'Urban') * 50 +  # 50 points for urban property
    (df['marital_status'].isin(['Married', 'Widowed'])) * 50  # 50 points for stable marital status
).clip(300, 1000).astype(int)

# Add some noise to risk score
df['risk_score'] = (df['risk_score'] * np.random.normal(1, 0.05, n_samples)).clip(300, 1000).astype(int)

# Calculate default probability based on risk score (sigmoid function)
default_prob = 1 / (1 + np.exp((df['risk_score'] - 600) / 100))

# Add some noise to default probability
noise = np.random.normal(0, 0.05, n_samples)
default_prob = np.clip(default_prob + noise, 0, 1)

# Set default status based on probability
df['default'] = np.random.binomial(1, default_prob, n_samples)

# Add some realistic dates
start_date = datetime(2023, 1, 1)
df['application_date'] = [start_date + timedelta(days=int(x)) for x in np.random.uniform(0, 365, n_samples)]

# Reorder columns
cols = ['age', 'gender', 'marital_status', 'dependents', 'education', 'self_employed',
        'income_annual', 'loan_amount', 'loan_amount_term', 'credit_history',
        'property_area', 'loan_grade', 'debt_to_income_ratio', 'delinquencies_2yrs',
        'open_accounts', 'total_accounts', 'risk_score', 'default', 'application_date']
df = df[cols]

# Save to CSV
output_path = 'data/raw/loan_risk_dataset.csv'
df.to_csv(output_path, index=False)

print(f"\nSynthetic loan risk dataset generated with {len(df)} records")
print(f"Dataset saved to: {output_path}")
print("\nDataset Overview:")
print(f"- Default rate: {df['default'].mean():.2%}")
print(f"- Average age: {df['age'].mean():.1f}")
print(f"- Average income: ${df['income_annual'].mean():,.0f}")
print(f"- Average loan amount: ${df['loan_amount'].mean():,.0f}")
print("\nFirst few rows of the dataset:")
print(df.head().to_string())

# Create a smaller sample for quick testing
sample_df = df.sample(min(100, len(df)), random_state=42)
sample_path = 'data/raw/loan_risk_dataset_sample.csv'
sample_df.to_csv(sample_path, index=False)
print(f"\nSample dataset (100 records) saved to: {sample_path}")

import os
import pandas as pd
from sklearn.datasets import fetch_openml

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("Downloading UCI Credit Card Default dataset...")

# Download the dataset
X, y = fetch_openml('credit-g', as_frame=True, return_X_y=True, parser='auto')

# Combine features and target
data = pd.concat([X, y], axis=1)

# Save to CSV
file_path = 'data/raw/credit_card_default.csv'
data.to_csv(file_path, index=False)

print(f"Dataset successfully downloaded and saved to {file_path}")
print(f"Shape of the dataset: {data.shape}")
print("\nFirst few rows of the dataset:")
print(data.head())

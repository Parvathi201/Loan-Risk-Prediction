# Phase 3: Model Training and Evaluation

This is where the predictive power of our project comes to life. In this phase, we use the preprocessed data to train, tune, and evaluate several machine learning models to find the one that most accurately predicts loan defaults.

## Key Files and Scripts

-   `scripts/model_training.py`: This is the core script for the modeling phase. It automates the entire training and evaluation pipeline:
    -   **Data Loading:** It loads the fully processed and encoded data from `data/processed/processed_loan_data_encoded.csv`.
    -   **Data Splitting and Scaling:** The data is split into training and testing sets, and a `StandardScaler` is fitted to the training data.
    -   **Handling Class Imbalance:** It uses the Synthetic Minority Over-sampling Technique (SMOTE) to address the class imbalance in the training data, creating a more balanced dataset for training.
    -   **Model Training:** It trains three different classification models: `XGBoost`, `LightGBM`, and `Random Forest`.
    -   **Hyperparameter Tuning:** After evaluating the baseline models, it performs hyperparameter tuning on the best-performing model (Random Forest) using `GridSearchCV`.
    -   **Evaluation and Reporting:** It evaluates all models on the test set and saves the results—including accuracy, precision, recall, F1-score, and ROC-AUC—to `reports/model_training_results.csv`.

-   `models/`: The final, tuned model and its associated artifacts are saved in this directory:
    -   `final_model.joblib`: The trained and tuned Random Forest model.
    -   `final_model_features.txt`: A text file containing the list of features the model was trained on.
    -   `scaler.joblib`: The scaler object used to standardize the features.

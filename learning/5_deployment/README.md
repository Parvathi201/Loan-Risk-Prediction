# Phase 5: Deployment

In the final phase, we deploy the trained loan risk prediction model as an interactive web application using Streamlit. This allows users to input loan applicant data and receive a real-time risk assessment.

## Key Files and Scripts

-   `app.py`: This is the main script for the Streamlit application. It creates the user interface and handles the prediction logic:
    -   **Model Loading:** It loads the final trained model (`final_model.joblib`), the feature list (`final_model_features.txt`), and the scaler (`scaler.joblib`) from the `models/` directory.
    -   **User Interface:** It creates a user-friendly form where users can input applicant information, such as income, loan amount, and credit history.
    -   **Prediction Logic:** When the user submits the form, the script takes the input data, preprocesses it using the saved scaler, and feeds it to the model to get a prediction.
    -   **Displaying Results:** It displays the prediction result (e.g., "Low Risk" or "High Risk") to the user, along with the predicted probability of default.

## How to Run the Application

To run the Streamlit application, navigate to the project's root directory in your terminal and run the following command:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your browser.

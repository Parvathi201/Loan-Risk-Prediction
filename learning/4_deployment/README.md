# Phase 4: Deployment

In the final phase, we take our best-performing model and make it accessible to end-users through a web application. This allows stakeholders to use the model to get real-time loan risk predictions.

## Key Files and Scripts

-   `app.py`: This is the main script for the web application. It's built using **Streamlit**, a Python framework for creating interactive data apps. The script is responsible for:
    -   **Loading the model:** It loads the saved `.joblib` model file from the `models/` directory.
    -   **Creating the user interface (UI):** It builds a user-friendly form where users can input borrower information (e.g., age, income, loan amount).
    -   **Making predictions:** When the user submits the form, the app preprocesses the input data, feeds it to the model, and gets a prediction.
    -   **Displaying results:** It presents the prediction (e.g., "Approve" or "Decline") and the probability of default in an intuitive way, often using charts and metrics.

-   `requirements.txt`: This file lists all the Python libraries and packages required to run the project, including `streamlit`, `pandas`, `scikit-learn`, etc. This makes it easy for others to set up the correct environment.

### How to Run the App

To run the deployed application, you would typically run the following command in your terminal from the project's root directory:

```bash
streamlit run app.py
```

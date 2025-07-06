import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Set page config
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and data
@st.cache_resource
def load_model():
    """Load the trained model, scaler, and feature list."""
    try:
        model = joblib.load('models/final_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        with open('models/final_model_features.txt', 'r') as f:
            features = [line.strip() for line in f]
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return None, None, []

# Load sample data for reference
def load_sample_data():
    """Load sample data for reference ranges"""
    try:
        df = pd.read_csv('data/processed/processed_loan_data_encoded.csv')
        return df
    except Exception as e:
        st.warning(f"Could not load sample data: {str(e)}")
        return None

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Main app
def main():
    st.title("💰 Loan Default Predictor")
    st.write("Assess the risk of loan default based on borrower information")
    
    # Load model and sample data
    model, scaler, model_features = load_model()
    sample_data = load_sample_data()
    
    if model is None or scaler is None or not model_features:
        st.error("Failed to load the prediction model. Please check the model files.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Make Prediction", "View Risk Analysis"])
    
    if app_mode == "Make Prediction":
        render_prediction_form(model, scaler, model_features, sample_data)
    else:
        render_risk_analysis()

def render_prediction_form(model, scaler, model_features, sample_data):
    """Render the prediction form"""
    st.header("Borrower Information")
    
    # Create form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        # Personal Information
        with col1:
            st.subheader("Personal Details")
            age = st.slider("Age", 18, 100, 35)
            income_annual = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
            employment_length = st.slider("Employment Length (years)", 0, 50, 5)
            
        # Loan Information
        with col2:
            st.subheader("Loan Details")
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=10000, step=500)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
            interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 10.0, step=0.5)
        
        # Credit Information
        st.subheader("Credit Information")
        col3, col4 = st.columns(2)
        with col3:
            credit_score = st.slider("Credit Score", 300, 850, 700)
            credit_history = st.selectbox("Credit History (1=Good, 0=Bad)", [1, 0])
        with col4:
            debt_to_income_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, 0.01)
            credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.3, 0.01)
        
        # Submit button
        submitted = st.form_submit_button("Assess Default Risk")
    
    # Handle form submission
    if submitted:
        # Prepare input data
        input_data = {
            'age': age,
            'income_annual': income_annual,
            'employment_length': employment_length,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'interest_rate': interest_rate,
            'credit_score': credit_score,
            'credit_history': credit_history,
            'debt_to_income_ratio': debt_to_income_ratio,
            'credit_utilization': credit_utilization,
            'risk_score': calculate_risk_score(credit_score, debt_to_income_ratio, credit_utilization)
        }
        
        # Create DataFrame with all model features
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for feature in model_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match model's training order
        input_df = input_df[model_features]

        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        try:
            probability = model.predict_proba(input_scaled)[:, 1][0]
            prediction = model.predict(input_scaled)[0]
            
            st.session_state.prediction_made = True

            # Show results
            show_prediction_results(probability, prediction, input_data)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Show previous prediction if exists
    elif st.session_state.prediction_made:
        pred = st.session_state.prediction
        show_prediction_results(pred['probability'], pred['prediction'], pred['input_data'])

def calculate_risk_score(credit_score, dti_ratio, credit_utilization):
    """Calculate a simple risk score based on credit factors"""
    # Normalize credit score (300-850 to 0-1)
    score_norm = (credit_score - 300) / (850 - 300)
    
    # Invert DTI ratio (lower is better)
    dti_score = 1 - min(dti_ratio, 1.0)
    
    # Invert credit utilization (lower is better)
    util_score = 1 - min(credit_utilization, 1.0)
    
    # Weighted average
    risk_score = (score_norm * 0.5) + (dti_score * 0.3) + (util_score * 0.2)
    
    # Scale to 300-850 range
    return int(300 + (risk_score * 550))

def show_prediction_results(probability, prediction, input_data):
    """Display prediction results"""
    st.markdown("---")
    st.header("📊 Risk Assessment Results")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Default Probability", f"{probability:.1%}")
    
    with col2:
        risk_level = "High" if probability >= 0.5 else "Low"
        st.metric("Risk Level", risk_level, 
                 delta_color="inverse",
                 delta=f"{'⚠️ ' if risk_level == 'High' else '✅ '}{risk_level} Risk")
    
    with col3:
        recommendation = "Decline" if prediction else "Approve"
        st.metric("Recommendation", recommendation)
    
    # Show risk gauge
    st.markdown("### Risk Assessment")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Default Risk Score"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show key factors
    st.markdown("### Key Risk Factors")
    
    risk_factors = [
        ("Credit Score", input_data['credit_score'], 700, 850, "Higher is better"),
        ("Debt-to-Income Ratio", input_data['debt_to_income_ratio'], 0, 0.36, "Lower is better"),
        ("Credit Utilization", input_data['credit_utilization'], 0, 0.3, "Lower is better"),
        ("Income", f"${input_data['income_annual']:,.0f}", 50000, 100000, "Higher is better"),
        ("Loan-to-Income Ratio", input_data['loan_amount'] / max(1, input_data['income_annual']), 0, 0.36, "Lower is better")
    ]
    
    for i, (factor, value, good, bad, desc) in enumerate(risk_factors):
        cols = st.columns([1, 2])
        with cols[0]:
            st.metric(factor, str(value))
        with cols[1]:
            if isinstance(value, (int, float)):
                if factor in ["Debt-to-Income Ratio", "Credit Utilization", "Loan-to-Income Ratio"]:
                    good, bad = bad, good  # For metrics where lower is better
                
                if (value >= good and good > bad) or (value <= good and good < bad):
                    st.progress(0.8, f"✅ {desc}")
                else:
                    st.progress(0.4, f"⚠️ {desc}")
            else:
                st.caption(desc)

def render_risk_analysis():
    """Render the risk analysis dashboard"""
    st.header("📈 Loan Portfolio Risk Analysis")
    
    # Sample data for demonstration
    risk_data = pd.DataFrame({
        'Risk Segment': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        'Default Rate': [0.01, 0.05, 0.15, 0.35, 0.65],
        'Count': [1200, 1800, 2500, 1500, 500],
        'Avg. Loan Amount': [25000, 35000, 28000, 20000, 15000]
    })
    
    # Portfolio distribution
    st.subheader("Portfolio Distribution by Risk Segment")
    fig1 = px.pie(risk_data, values='Count', names='Risk Segment',
                 color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Default rate by segment
    st.subheader("Default Rate by Risk Segment")
    fig2 = px.bar(risk_data, x='Risk Segment', y='Default Rate',
                  color='Risk Segment',
                  color_discrete_sequence=px.colors.sequential.Viridis_r)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Loan amount distribution
    st.subheader("Average Loan Amount by Risk Segment")
    fig3 = px.bar(risk_data, x='Risk Segment', y='Avg. Loan Amount',
                  color='Risk Segment',
                  color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()

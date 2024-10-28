import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Title and description
st.title("Credit Card Fraud Detection")
st.markdown("""
This application predicts whether a credit card transaction is fraudulent or legitimate.
""")

def load_model():
    try:
        model = joblib.load('random_forest_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    return prediction[0], probability[0]

# Load the model
model = load_model()

# Create input form
st.subheader("Transaction Details")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

# Initialize a dictionary to store inputs
inputs = {}

# Create input fields for V1-V28
for i in range(1, 29):
    with col1 if i <= 9 else col2 if i <= 18 else col3:
        inputs[f'V{i}'] = st.number_input(f'V{i}', value=0.0, format='%f')

# Amount input
with col1:
    inputs['Amount'] = st.number_input('Transaction Amount ($)', min_value=0.0, format='%f')

if st.button('Predict'):
    if model is not None:
        # Create DataFrame from inputs
        input_df = pd.DataFrame([inputs])
        
        # Make prediction
        prediction, probability = make_prediction(model, input_df)
        
        # Show results
        st.subheader("Prediction Results")
        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
            st.write(f"Confidence: {probability[1]:.2%}")
        else:
            st.success("✅ Legitimate Transaction")
            st.write(f"Confidence: {probability[0]:.2%}")
        
        # Display probability gauge
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability[1] * 100,
            title = {'text': "Fraud Probability"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "lightred"}
                ]
            }
        ))
        
        st.plotly_chart(fig)

# Add additional information
st.sidebar.header("About")
st.sidebar.info("""
This application uses a Random Forest model trained on credit card transaction data 
to detect fraudulent transactions. The model takes various transaction features 
(V1-V28) which are PCA transformed features for privacy, along with the transaction amount.
""")

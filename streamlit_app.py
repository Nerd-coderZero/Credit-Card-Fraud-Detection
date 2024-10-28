import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Credit Card Fraud Detection Demo",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .fraud-high { color: #ff4b4b; font-weight: bold; }
    .fraud-medium { color: #ffa500; font-weight: bold; }
    .fraud-low { color: #00cc00; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

def load_model():
    try:
        model_path = "Models/best_random_forest_model.pkl"
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def generate_synthetic_features(transaction_data):
    """
    Generate synthetic features similar to the original dataset's PCA components
    based on basic transaction information
    """
    # Create a normalized version of the amount
    amount = transaction_data['amount']
    hour = transaction_data['hour']
    
    # Generate synthetic V1-V28 features based on the input data
    # This is a simplified example - in real world you'd have proper feature engineering
    synthetic_data = {
        'V1': np.sin(hour/24 * 2 * np.pi),  # Time-based feature
        'V2': amount / 1000,  # Normalized amount
        'V3': np.cos(hour/24 * 2 * np.pi),  # Another time-based feature
    }
    
    # Generate remaining features with some random noise
    # In real world, these would be actual transaction features
    for i in range(4, 29):
        synthetic_data[f'V{i}'] = np.random.normal(0, 0.1)
    
    synthetic_data['Amount'] = amount
    
    return pd.DataFrame([synthetic_data])

def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    return prediction[0], probability[0]

def get_risk_level(probability):
    if probability >= 0.7:
        return "High Risk", "fraud-high"
    elif probability >= 0.4:
        return "Medium Risk", "fraud-medium"
    else:
        return "Low Risk", "fraud-low"

def main():
    model = load_model()
    
    st.title("Credit Card Fraud Detection Demo")
    st.markdown("""
    ### 🚀 Demo Version
    This is a demonstration of a credit card fraud detection system. Enter transaction details below to see how the system analyzes potential fraud.
    
    ⚠️ Note: This is a simplified demo using synthetic data transformation. In a real system, many more transaction details would be considered.
    """)

    # Create a more user-friendly input form
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                max_value=25000.0,
                value=100.0,
                help="Enter the transaction amount"
            )
            
            hour = st.slider(
                "Hour of Day",
                0, 23, 12,
                help="Select the hour when the transaction occurred"
            )

        with col2:
            st.markdown("#### Transaction Type")
            transaction_type = st.selectbox(
                "Select the type of transaction",
                ["Online Purchase", "In-store Purchase", "ATM Withdrawal", "Money Transfer"]
            )
            
            merchant_category = st.selectbox(
                "Merchant Category",
                ["Retail", "Travel", "Entertainment", "Groceries", "Electronics", "Other"]
            )

        submitted = st.form_submit_button("Analyze Transaction")

    if submitted and model is not None:
        # Create transaction data dictionary
        transaction_data = {
            'amount': amount,
            'hour': hour,
            'type': transaction_type,
            'category': merchant_category
        }
        
        # Generate features for the model
        input_df = generate_synthetic_features(transaction_data)
        
        # Make prediction
        prediction, probability = make_prediction(model, input_df)
        
        # Display results
        st.header("Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ Potential Fraud Detected")
            else:
                st.success("✅ Transaction Appears Legitimate")
        
        with col2:
            risk_level, risk_class = get_risk_level(probability[1])
            st.markdown(f"Risk Level: <span class='{risk_class}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
        
        with col3:
            st.metric("Fraud Probability", f"{probability[1]:.1%}")

        # Display probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={'text': "Risk Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "orange"},
                    {'range': [66, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        st.plotly_chart(fig)
        
        # Add explanation
        st.markdown("""
        ### How to interpret these results
        
        This demo shows how a fraud detection system might analyze a transaction. The analysis considers:
        - Transaction amount
        - Time of day
        - Transaction type
        - Merchant category
        
        🔍 **Risk Levels Explained:**
        - **Low Risk** (0-40%): Transaction appears normal
        - **Medium Risk** (40-70%): Some unusual patterns detected
        - **High Risk** (70-100%): Multiple fraud indicators present
        
        ⚠️ **Demo Limitations:**
        This is a simplified demonstration. Real fraud detection systems use hundreds of features and more sophisticated analysis methods.
        """)

    # Add sidebar with additional information
    st.sidebar.title("About This Demo")
    st.sidebar.markdown("""
    This is a demonstration version of a credit card fraud detection system. 
    
    #### Features:
    - Real-time transaction analysis
    - Risk level assessment
    - Visual risk indicators
    
    #### Important Note:
    This demo uses synthetic data transformation for demonstration purposes. In a real production system, many more factors would be considered for fraud detection.
    """)

if __name__ == "__main__":
    main()

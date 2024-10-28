import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# Configure the page
st.set_page_config(
    page_title="Credit Card Fraud Detection Model Demo",
    page_icon="💳",
    layout="wide"
)

def load_model():
    try:
        model_path = "Models/best_random_forest_model.pkl"
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def make_prediction(model, input_df):
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    model = load_model()
    
    st.title("Credit Card Fraud Detection Model Demo")
    
    # Educational explanation
    st.markdown("""
    ### About This Model
    This fraud detection model was trained on a dataset of credit card transactions where:
    - Features V1-V28 are PCA (Principal Component Analysis) transformed features
    - These transformations were done to protect sensitive transaction details
    - Amount represents the transaction amount in dollars
    
    #### How to Use This Demo:
    1. You can input values for V1-V28 and Amount
    2. The model will analyze these inputs just like it would a real transaction
    3. You'll see the fraud probability and prediction
    
    #### Sample Transaction Values:
    - For a typical legitimate transaction, V1-V28 values usually range from -3 to +3
    - Amount typically ranges from $0 to $25,000
    """)

    # Create expandable section for sample values
    with st.expander("See sample values from training data"):
        st.markdown("""
        Here are some example values from legitimate transactions:
        ```
        V1: -1.3598, V2: -0.0728, V3: 2.5363
        Amount: $149.62
        ```
        And from fraudulent transactions:
        ```
        V1: -3.7578, V2: -3.1635, V3: -7.3674
        Amount: $378.66
        ```
        """)

    # Input form with better organization
    with st.form("transaction_form"):
        st.subheader("Enter Transaction Details")
        
        # Amount input
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=25000.0,
            value=149.62,
            help="Enter the transaction amount in dollars"
        )
        
        # Create 4 columns for V1-V28 inputs to make it more compact
        st.markdown("#### PCA Transformed Features (V1-V28)")
        cols = st.columns(4)
        v_inputs = []
        
        # Sample values from your training data
        sample_values = [-1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 
                        0.2396, 0.0987, 0.3638, 0.0908, -0.5516, -0.6178, 
                        -0.9914, -0.3112, 1.4682, -0.4704, 0.2080, 0.0258, 
                        0.4040, 0.2514, -0.0183, 0.2778, -0.1105, 0.0669,
                        0.1285, -0.1891, 0.1336, -0.0211]

        for i in range(28):
            with cols[i % 4]:
                v_inputs.append(
                    st.number_input(
                        f"V{i+1}",
                        value=sample_values[i],
                        format="%.4f",
                        help=f"PCA transformed feature {i+1}"
                    )
                )
        
        submitted = st.form_submit_button("Analyze Transaction")

    if submitted and model is not None:
        # Create input DataFrame
        input_data = {f'V{i+1}': v_inputs[i] for i in range(28)}
        input_data['Amount'] = amount
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction, probability = make_prediction(model, input_df)
        
        if prediction is not None:
            # Display results
            st.header("Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ Model predicts this transaction as potentially fraudulent")
                else:
                    st.success("✅ Model predicts this transaction as legitimate")
                
                st.metric("Fraud Probability", f"{probability[1]:.2%}")
            
            with col2:
                # Display probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1] * 100,
                    title={'text': "Fraud Probability"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "orange"},
                            {'range': [66, 100], 'color': "salmon"}
                        ]
                    }
                ))
                st.plotly_chart(fig)

    # Add information about model performance
    st.sidebar.title("Model Performance")
    st.sidebar.markdown("""
    This model was trained on a dataset of credit card transactions and achieved:
    - Accuracy: 99.95%
    - Precision: 0.97
    - Recall: 0.94
    - F1 Score: 0.95
    
    ⚠️ Note: This is a demo version using the actual trained model, but with PCA-transformed features that protect sensitive transaction details.
    """)

if __name__ == "__main__":
    main()

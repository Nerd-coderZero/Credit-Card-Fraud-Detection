import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("Models/best_random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details below to predict if it’s fraudulent:")

# Input fields for user data (assuming 29 features based on your description)
features = []
for i in range(1, 30):
    value = st.number_input(f"Feature {i}", step=0.01)
    features.append(value)

# Predict and display the result
if st.button("Predict"):
    prediction = model.predict([features])
    result = "Fraudulent" if prediction[0] == 1 else "Non-Fraudulent"
    st.write(f"Prediction: {result}")

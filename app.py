import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it is Fraud or Legitimate")

# Input fields (simplified example)
time = st.number_input("Time", 0.0)
amount = st.number_input("Amount", 0.0)

# For demo — V1 to V28 inputs
features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", 0.0)
    features.append(val)

input_data = np.array([time] + features + [amount]).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

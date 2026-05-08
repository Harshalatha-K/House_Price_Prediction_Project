import streamlit as st
import numpy as np
import joblib

# Load files
model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

feature_names = joblib.load("features.pkl")

# Title
st.title("House Price Prediction")

st.write("Enter Feature Values")

input_data = []

# Dynamic inputs
for feature in feature_names:

    value = st.number_input(
        f"{feature}",
        value=0.0
    )

    input_data.append(value)

# Predict
if st.button("Predict Price"):

    features = np.array([input_data])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    st.success(
        f"Predicted Price: ₹ {prediction[0]:,.2f}"
    )

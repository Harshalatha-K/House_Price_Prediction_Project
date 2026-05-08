
import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("House Price Prediction App")

st.write("Enter house details below:")

# Inputs
square_footage = st.number_input("Square Footage", min_value=0.0)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)

# Predict Button
if st.button("Predict Price"):

    features = np.array([[square_footage,
                          bedrooms,
                          bathrooms]])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

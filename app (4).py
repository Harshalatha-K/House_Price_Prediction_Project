
import streamlit as st
import numpy as np
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title
st.title("House Price Prediction")

st.write("Enter House Details")

# Inputs
area = st.number_input("Area", min_value=500)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

stories = st.number_input("Stories", min_value=1)

parking = st.number_input("Parking", min_value=0)

# Predict
if st.button("Predict Price"):

    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        parking
    ]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted House Price: ₹ {prediction[0]:,.2f}"
    )

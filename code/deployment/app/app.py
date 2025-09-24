import streamlit as st
import requests

st.title("House Price Predictor")

overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
gr_liv_area = st.number_input("Above ground living area (sq ft)", 500, 5000, 1500)
garage_cars = st.slider("Garage capacity (cars)", 0, 4, 1)
year_built = st.number_input("Year Built", 1870, 2020, 1990)
neighborhood = st.selectbox("Neighborhood", ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"])

if st.button("Predict Price"):
    payload = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "YearBuilt": year_built,
        "Neighborhood": neighborhood
    }
    response = requests.post("http://api:8000/predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Price: ${result['prediction']}")
    else:
        st.error("API request failed!")

import streamlit as st
import pandas as pd
import joblib

# Load pipeline
model = joblib.load("model.pkl")

st.title("Ship CO2 Emission Predictor")

# Sliders based on dataset distribution

engine_efficiency = st.slider(
    "Engine Efficiency (%)",
    min_value=70.0,
    max_value=95.0,
    value=82.5,
    step=0.1
)

fuel_consumption = st.slider(
    "Fuel Consumption",
    min_value=200.0,
    max_value=25000.0,
    value=3000.0,
    step=100.0
)

distance = st.slider(
    "Distance",
    min_value=20.0,
    max_value=500.0,
    value=120.0,
    step=1.0
)

df = pd.read_csv("data\ship_fuel_efficiency.csv")

ship_type = st.selectbox("Ship Type", sorted(df["ship_type"].unique()))
fuel_type = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))

# Convert to DataFrame
input_df = pd.DataFrame([{
    "engine_efficiency": engine_efficiency,
    "fuel_consumption": fuel_consumption,
    "distance": distance,
    "ship_type": ship_type,
    "fuel_type": fuel_type
}])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted CO2 Emission: {prediction[0]:.2f}")
    st.caption("Lower is better for environment 🌱")
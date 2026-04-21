import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Load saved files
# -----------------------
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
encoders = joblib.load("encoders.pkl")

# -----------------------
# App UI
# -----------------------
st.title("🚢 Ship CO2 Emission Predictor")

# Simple inputs
fuel = st.slider("⛽ Fuel Consumption", 0.0, 25000.0, 3000.0)
eff = st.slider("⚙️ Engine Efficiency", 70.0, 100.0, 80.0)
dist = st.slider("📏 Distance", 0.0, 500.0, 100.0)

input_df = pd.DataFrame([{
    "fuel_consumption": fuel,
    "engine_efficiency": eff,
    "distance": dist
}])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.metric("🌍 CO2 Emission", f"{prediction:.2f}")
    st.caption("Lower is better for environment 🌱")
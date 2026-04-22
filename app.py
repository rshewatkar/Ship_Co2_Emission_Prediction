from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "ship_fuel_efficiency.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import load_data, save_model, train_model


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            header = file.read(64)
    except OSError:
        return False

    return header.startswith(b"version https://git-lfs.github.com/spec/")


@st.cache_resource
def load_or_train_model():
    """
    Load a saved model when available, or rebuild it from the CSV dataset.
    This avoids app crashes in environments where Git LFS only provides a
    pointer file instead of the real model artifact.
    """
    if MODEL_PATH.exists() and not _is_git_lfs_pointer(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    df = load_data(DATA_PATH)
    pipeline = train_model(df)
    save_model(pipeline, MODEL_PATH)
    return pipeline


model = load_or_train_model()
df = pd.read_csv(DATA_PATH)

st.title("Ship CO2 Emission Predictor")

engine_efficiency = st.slider(
    "Engine Efficiency (%)",
    min_value=70.0,
    max_value=95.0,
    value=82.5,
    step=0.1,
)

fuel_consumption = st.slider(
    "Fuel Consumption",
    min_value=200.0,
    max_value=25000.0,
    value=3000.0,
    step=100.0,
)

distance = st.slider(
    "Distance",
    min_value=20.0,
    max_value=500.0,
    value=120.0,
    step=1.0,
)

ship_type = st.selectbox("Ship Type", sorted(df["ship_type"].unique()))
fuel_type = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))

input_df = pd.DataFrame(
    [
        {
            "engine_efficiency": engine_efficiency,
            "fuel_consumption": fuel_consumption,
            "distance": distance,
            "ship_type": ship_type,
            "fuel_type": fuel_type,
        }
    ]
)

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted CO2 Emission: {prediction[0]:.2f}")
    st.caption("Lower is better for the environment.")

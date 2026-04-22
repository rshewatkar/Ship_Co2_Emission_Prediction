# src/train.py

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from preprocess import get_preprocessor

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "ship_fuel_efficiency.csv")

def load_data(path):
    """
    Load dataset
    """
    df = pd.read_csv(path)
    return df


def train_model(df):
    """
    Train ML pipeline
    """

    # 👉 UPDATE target column
    target = "CO2_emissions"

    X = df.drop(columns=[target])
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessor
    preprocessor = get_preprocessor()

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # FULL PIPELINE (IMPORTANT)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    print("✅ Model trained successfully")

    return pipeline


def save_model(pipeline, path="model.pkl"):
    """
    Save trained pipeline
    """
    joblib.dump(pipeline, path)
    print(f"✅ Model saved at {path}")


if __name__ == "__main__":
    data_path = "data\ship_fuel_efficiency.csv"  # 👉 update path if needed

    df = load_data(data_path)

    pipeline = train_model(df)

    save_model(pipeline)
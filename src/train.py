# src/train.py

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import MODEL_FEATURES, get_preprocessor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ship_fuel_efficiency.csv"


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
    target = "CO2_emissions"

    X = df[MODEL_FEATURES].copy()
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = get_preprocessor()

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("Model trained successfully")

    return pipeline


def save_model(pipeline, path="model.pkl"):
    """
    Save trained pipeline
    """
    joblib.dump(pipeline, path)
    print(f"Model saved at {path}")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    pipeline = train_model(df)
    save_model(pipeline)

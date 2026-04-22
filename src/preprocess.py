# src/preprocess.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_FEATURES = [
    "engine_efficiency",
    "fuel_consumption",
    "distance",
    "ship_type",
    "fuel_type",
]


def get_preprocessor():
    """
    Returns a ColumnTransformer for preprocessing
    """
    numeric_features = [
        "engine_efficiency",
        "fuel_consumption",
        "distance",
    ]

    categorical_features = [
        "ship_type",
        "fuel_type",
    ]

    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features),
        ]
    )

    return preprocessor

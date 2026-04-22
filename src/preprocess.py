# src/preprocess.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_preprocessor():
    """
    Returns a ColumnTransformer for preprocessing
    """

    # 👉 UPDATE THESE BASED ON YOUR DATASET
    numeric_features = [
        "engine_efficiency",
        "fuel_consumption",
        "distance"
    ]

    categorical_features = [
        "ship_type",
        "fuel_type"
    ]

    # Numerical pipeline
    num_transformer = StandardScaler()

    # Categorical pipeline
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features)
        ]
    )

    return preprocessor
"""Shared feature definitions and preprocessing for the diabetes model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# In the Pima dataset a value of 0 in these columns is physiologically
# impossible and actually means "missing". Treating them as real zeros is the
# most common bug in diabetes-prediction projects.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Plausible input ranges, used for validating user input in the CLI / app.
VALID_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (0, 80),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (1, 120),
}


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN in columns where 0 means 'not measured'."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=FEATURES).copy()
        cols = self.columns if self.columns is not None else ZERO_AS_MISSING
        X[cols] = X[cols].replace(0, np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES)


def load_data(path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df.drop_duplicates()
    return df[FEATURES], df[TARGET].astype(int)


def validate_input(record: dict) -> dict:
    """Check a single patient record; return it as floats or raise ValueError."""
    clean = {}
    for name in FEATURES:
        if name not in record:
            raise ValueError(f"Missing value for '{name}'")
        try:
            value = float(record[name])
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number, got {record[name]!r}") from None
        lo, hi = VALID_RANGES[name]
        if not lo <= value <= hi:
            raise ValueError(f"'{name}'={value} is outside the valid range {lo}-{hi}")
        clean[name] = value
    return clean

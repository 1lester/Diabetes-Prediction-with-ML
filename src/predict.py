"""Predict diabetes risk for one patient.

Usage:
    python -m src.predict --pregnancies 2 --glucose 138 --blood-pressure 62 \
        --skin-thickness 35 --insulin 0 --bmi 33.6 --dpf 0.127 --age 47
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURES, validate_input

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "diabetes_model.joblib"


@lru_cache(maxsize=2)
def load_model(path: str | Path = MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict(record: dict, model=None, threshold: float = 0.5) -> dict:
    """Return {'probability': float, 'prediction': 0|1, 'label': str}."""
    clean = validate_input(record)
    model = model or load_model()
    X = pd.DataFrame([clean], columns=FEATURES)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= threshold)
    return {
        "probability": round(proba, 4),
        "prediction": pred,
        "label": "Diabetic" if pred else "Not diabetic",
    }


def main():
    p = argparse.ArgumentParser(description="Predict diabetes for one patient")
    p.add_argument("--pregnancies", type=float, required=True)
    p.add_argument("--glucose", type=float, required=True)
    p.add_argument("--blood-pressure", type=float, required=True)
    p.add_argument("--skin-thickness", type=float, required=True)
    p.add_argument("--insulin", type=float, required=True)
    p.add_argument("--bmi", type=float, required=True)
    p.add_argument("--dpf", type=float, required=True, help="Diabetes pedigree function")
    p.add_argument("--age", type=float, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()
    record = dict(zip(FEATURES, [a.pregnancies, a.glucose, a.blood_pressure, a.skin_thickness,
                                 a.insulin, a.bmi, a.dpf, a.age]))
    try:
        r = predict(record, threshold=a.threshold)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
    print(f"{r['label']}  (probability of diabetes: {r['probability']:.1%})")


if __name__ == "__main__":
    main()

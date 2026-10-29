import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, ZeroToNaN, load_data, validate_input
from src.predict import predict
from src.train import train

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "diabetes.csv"

HIGH_RISK = dict(zip(FEATURES, [8, 196, 76, 29, 280, 37.5, 0.605, 57]))
LOW_RISK = dict(zip(FEATURES, [1, 85, 66, 29, 0, 26.6, 0.351, 31]))


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    metrics = train(DATA, out)
    import joblib
    return joblib.load(out / "diabetes_model.joblib"), metrics, out


def test_dataset_loads():
    X, y = load_data(DATA)
    assert list(X.columns) == FEATURES
    assert len(X) == 768
    assert set(y.unique()) == {0, 1}


def test_zero_to_nan():
    df = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0.5, 30]], columns=FEATURES)
    out = ZeroToNaN().transform(df)
    assert out["Pregnancies"].iloc[0] == 0          # 0 pregnancies is valid
    assert np.isnan(out["Glucose"].iloc[0])          # 0 glucose means missing


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        validate_input({**LOW_RISK, "Glucose": -5})
    with pytest.raises(ValueError):
        validate_input({**LOW_RISK, "Age": "abc"})
    bad = dict(LOW_RISK)
    bad.pop("BMI")
    with pytest.raises(ValueError):
        validate_input(bad)


def test_model_quality(trained):
    _, metrics, out = trained
    assert metrics["test_metrics"]["roc_auc"] >= 0.80
    assert metrics["test_metrics"]["accuracy"] >= 0.70
    assert metrics["test_metrics"]["recall"] >= 0.65
    assert (out / "evaluation.png").exists()
    assert json.loads((out / "metrics.json").read_text())["best_model"] == metrics["best_model"]


def test_predictions_are_sensible(trained):
    model, _, _ = trained
    hi = predict(HIGH_RISK, model=model)
    lo = predict(LOW_RISK, model=model)
    assert 0 <= lo["probability"] < hi["probability"] <= 1
    assert hi["prediction"] == 1 and lo["prediction"] == 0


def test_missing_values_handled(trained):
    model, _, _ = trained
    r = predict({**LOW_RISK, "Insulin": 0, "SkinThickness": 0}, model=model)
    assert 0 <= r["probability"] <= 1

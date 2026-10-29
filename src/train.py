"""Train, compare and save the diabetes prediction model.

Usage:
    python -m src.train                 # uses data/diabetes.csv
    python -m src.train --data my.csv --out models/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from src.features import FEATURES, ZeroToNaN, load_data  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("zero_to_nan", ZeroToNaN()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


CANDIDATES = {
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "RandomForest": (
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        {"model__n_estimators": [200, 400], "model__max_depth": [4, 6, None], "model__min_samples_leaf": [1, 3, 5]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
    ),
    "SVM": (
        # SVC(probability=True) is deprecated in scikit-learn 1.9+; calibrate explicitly instead.
        CalibratedClassifierCV(SVC(class_weight="balanced", random_state=SEED), ensemble=False, cv=5),
        {"model__estimator__C": [0.1, 1, 10], "model__estimator__gamma": ["scale", 0.01, 0.1]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"model__n_neighbors": [5, 11, 15, 21], "model__weights": ["uniform", "distance"]},
    ),
}


def train(data_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = {}
    best_name, best_est, best_score = None, None, -np.inf
    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows\n")
    for name, (model, grid) in CANDIDATES.items():
        search = GridSearchCV(make_pipeline(model), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        scores = cross_validate(
            search.best_estimator_, X_train, y_train, cv=cv,
            scoring=["roc_auc", "accuracy", "f1", "recall"],
        )
        summary = {k.replace("test_", "cv_"): round(float(v.mean()), 4) for k, v in scores.items() if k.startswith("test_")}
        summary["best_params"] = {k.replace("model__estimator__", "").replace("model__", ""): v for k, v in search.best_params_.items()}
        results[name] = summary
        print(f"{name:20s} CV ROC-AUC={summary['cv_roc_auc']:.3f}  acc={summary['cv_accuracy']:.3f}  "
              f"recall={summary['cv_recall']:.3f}  f1={summary['cv_f1']:.3f}")
        if summary["cv_f1"] > best_score:  # F1 balances catching diabetics (recall) and precision
            best_name, best_est, best_score = name, search.best_estimator_, summary["cv_f1"]

    # Final evaluation on the untouched test set
    proba = best_est.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    print(f"\nBest model: {best_name}")
    print("Test-set metrics:", test_metrics)
    print(classification_report(y_test, pred, target_names=["No diabetes", "Diabetes"]))

    # Refit the winner on all data for deployment
    final = best_est.fit(X, y)
    joblib.dump(final, out_dir / "diabetes_model.joblib")

    metrics = {
        "best_model": best_name,
        "test_metrics": test_metrics,
        "cv_results": results,
        "features": FEATURES,
        "n_rows": int(len(X)),
        "sklearn_version": sklearn.__version__,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No", "Yes"], cmap="Blues", ax=axes[0], colorbar=False
    )
    axes[0].set_title(f"{best_name} – confusion matrix (test)")
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1], name=best_name)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("ROC curve (test)")
    fig.tight_layout()
    fig.savefig(out_dir / "evaluation.png", dpi=120)
    plt.close(fig)

    print(f"Saved model, metrics.json and evaluation.png to {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the diabetes prediction model")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "diabetes.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    train(args.data, args.out)


if __name__ == "__main__":
    main()

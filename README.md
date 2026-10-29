# Diabetes Prediction with ML

Predicts whether a patient has diabetes from 8 diagnostic measurements, using the
[Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
(768 patients, 268 diabetic). Includes a training pipeline that compares five models,
a command-line predictor, a Streamlit web app, and tests.

> Educational project — not a medical diagnostic tool.

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m src.train               # trains + saves models/diabetes_model.joblib (~30 s)
streamlit run app.py              # opens the web app at http://localhost:8501
```

Predict from the command line:

```bash
python -m src.predict --pregnancies 8 --glucose 196 --blood-pressure 76 \
  --skin-thickness 29 --insulin 280 --bmi 37.5 --dpf 0.605 --age 57
# Diabetic  (probability of diabetes: 84.7%)
```

Run the tests:

```bash
python -m pytest -q
```

## How it works

1. **Cleaning** – In this dataset, `0` for Glucose, BloodPressure, SkinThickness, Insulin
   and BMI means *not measured* (nearly half the Insulin values are 0). These are converted
   to missing and filled with the training-set median. `Pregnancies = 0` is kept as a real value.
2. **Pipeline** – `ZeroToNaN → median imputer → StandardScaler → model`, all inside one
   scikit-learn `Pipeline`, so there is no data leakage and the saved model accepts raw input.
3. **Model selection** – Logistic Regression, Random Forest, Gradient Boosting, SVM and KNN
   are each tuned with `GridSearchCV` (5-fold stratified CV) on 80% of the data. The model
   with the best cross-validated **F1** wins, because for screening, missing diabetic patients
   (recall) matters as much as precision.
4. **Evaluation** – The winner is scored once on the held-out 20% test set, then refit on all
   data and saved. Results go to `models/metrics.json` and `models/evaluation.png`.

## Results (held-out test set, 154 patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (selected) | 0.734 | 0.592 | 0.778 | 0.672 | 0.815 |

Cross-validation scores for all five models (ROC-AUC 0.83–0.85) are in `models/metrics.json`.
Around 75–78% accuracy is the realistic ceiling for this dataset; reports of 95%+ usually come
from data leakage or evaluating on the training data.

## Project structure

```
data/diabetes.csv        dataset
src/features.py          feature list, zero→missing transformer, input validation
src/train.py             model comparison, tuning, evaluation, saving
src/predict.py           load model + predict one patient (CLI)
app.py                   Streamlit web app
tests/test_model.py      data, validation, model-quality and prediction tests
models/                  trained model, metrics.json, evaluation.png (created by training)
```

## Input features

| Feature | Meaning |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose, 2 h oral glucose tolerance test (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) — 0 if not measured |
| Insulin | 2-hour serum insulin (µU/mL) — 0 if not measured |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Family-history score |
| Age | Years |

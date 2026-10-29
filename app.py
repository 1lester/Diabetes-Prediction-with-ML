"""Streamlit web app for diabetes prediction.

Run with:  streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st

from src.features import FEATURES
from src.predict import MODEL_PATH, load_model, predict

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction")
st.caption("Pima Indians Diabetes dataset · educational use only, not medical advice")

if not Path(MODEL_PATH).exists():
    st.error("No trained model found. Run `python -m src.train` first, then reload this page.")
    st.stop()

model = load_model()
metrics_path = Path(MODEL_PATH).parent / "metrics.json"
if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    with st.expander("Model details"):
        st.write(f"**Model:** {m['best_model']}  ·  trained on {m['n_rows']} patients")
        st.json(m["test_metrics"])

with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1, step=1)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
        bp = st.number_input("Blood pressure (mm Hg)", 0, 200, 70)
        skin = st.number_input("Skin thickness (mm)", 0, 100, 20, help="Enter 0 if not measured")
    with c2:
        insulin = st.number_input("Insulin (µU/mL)", 0, 900, 0, help="Enter 0 if not measured")
        bmi = st.number_input("BMI", 0.0, 80.0, 28.0, step=0.1)
        dpf = st.number_input("Diabetes pedigree function", 0.0, 3.0, 0.40, step=0.01, format="%.3f")
        age = st.number_input("Age", 1, 120, 33, step=1)
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    record = dict(zip(FEATURES, [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]))
    try:
        result = predict(record, model=model)
    except ValueError as e:
        st.error(str(e))
    else:
        p = result["probability"]
        if result["prediction"]:
            st.error(f"**High risk of diabetes** — probability {p:.1%}")
        else:
            st.success(f"**Low risk of diabetes** — probability {p:.1%}")
        st.progress(p)

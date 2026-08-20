"""Streamlit app — interactive churn prediction from a saved pipeline.

Load the pickled model with ``load_model`` and let the user fill in
customer attributes via sidebar widgets. Displays churn probability
and a plain-language label.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import pathlib

import pandas as pd
import streamlit as st

from src.predict import load_model, predict_proba

MODEL_PATH = pathlib.Path("models/churn_pipeline.joblib")

st.set_page_config(page_title="Churn predictor", layout="centered")
st.title("Churn predictor")
st.caption("Telco customer churn — sklearn baseline")


@st.cache_resource
def _load() -> object:
    """Load and cache the pipeline from disk."""
    return load_model(MODEL_PATH)


def _build_input_row() -> pd.DataFrame:
    """Collect customer features from sidebar widgets.

    Returns a single-row DataFrame matching the training feature schema.
    """
    with st.sidebar:
        st.header("Customer attributes")

        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total = st.number_input("Total charges ($)", 0.0, 10000.0, float(tenure * monthly), step=1.0)

        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior citizen", ["0", "1"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
        security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
        device = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
        tech = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])

    return pd.DataFrame(
        [
            {
                "tenure": tenure,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phone,
                "MultipleLines": multi,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": device,
                "TechSupport": tech,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
            }
        ]
    )


def main() -> None:
    """Render the prediction UI."""
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at `{MODEL_PATH}`. "
            "Train the model first: `python -m src.train`"
        )
        st.stop()

    pipeline = _load()
    row = _build_input_row()

    prob = float(predict_proba(pipeline, row)[0])
    label = "Churn" if prob >= 0.5 else "Retained"

    col1, col2 = st.columns(2)
    col1.metric("Churn probability", f"{prob:.1%}")
    col2.metric("Prediction", label)

    st.progress(prob)

    with st.expander("Input features"):
        st.dataframe(row.T, use_container_width=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Streamlit Dashboard for Churn Classifier.
Provides interactive EDA, model performance, and live predictions.
"""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = MODELS_DIR / "reports"

API_URL = "http://localhost:8000"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Classifier Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Churn Classifier")
st.sidebar.markdown("**Navigation**")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "🔍 EDA", "📈 Model Performance", "🎯 Predict"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard provides interactive insights into customer churn "
    "prediction using an ensemble of ML models."
)


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load()


@st.cache_data
def get_metrics():
    path = REPORTS_DIR / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def get_results():
    path = MODELS_DIR / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.title("Customer Churn Prediction Platform")
    st.markdown(
        """
        Welcome to the **Churn Classifier** — a production-grade machine learning
        platform that predicts whether a telecom customer will churn.

        ### What's inside?
        - 🤖 **4 ML models** (Logistic Regression, Random Forest, XGBoost, LightGBM)
        - 🔬 **Optuna hyperparameter tuning** for each model
        - ⚖️ **SMOTE** oversampling to handle class imbalance
        - 📊 **SHAP** explainability for model transparency
        - 🚀 **REST API** for real-time predictions
        """
    )

    df = get_data()
    churn_counts = df["Churn"].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Churned", int(churn_counts.get("Yes", 0)))
    col3.metric(
        "Churn Rate",
        f"{100 * churn_counts.get('Yes', 0) / len(df):.1f}%",
    )

    results = get_results()
    if results:
        st.markdown("### Model Leaderboard")
        df_results = pd.DataFrame(results)[["model", "f1", "roc_auc", "train_time_sec"]]
        df_results = df_results.sort_values("f1", ascending=False)
        df_results.columns = ["Model", "F1 Score", "ROC-AUC", "Train Time (s)"]
        st.dataframe(df_results.style.highlight_max(["F1 Score", "ROC-AUC"], color="lightgreen"))

elif page == "🔍 EDA":
    st.title("Exploratory Data Analysis")
    df = get_data()

    tab1, tab2, tab3 = st.tabs(["Overview", "Churn Distributions", "Correlations"])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(50), use_container_width=True)
        st.markdown("**Shape:**", help=f"{df.shape}")
        col1, col2 = st.columns(2)
        col1.write(df.dtypes.to_frame("dtype"))
        col2.write(df.describe())

    with tab2:
        st.subheader("Churn Distribution")
        fig = px.pie(
            df, names="Churn", title="Churn vs Retained",
            color_discrete_sequence=["#636EFA", "#EF553B"],
        )
        st.plotly_chart(fig, use_container_width=True)

        cat_col = st.selectbox(
            "Select a categorical column to see churn breakdown:",
            ["Contract", "InternetService", "PaymentMethod", "gender", "Partner"],
        )
        grouped = (
            df.groupby([cat_col, "Churn"])
            .size()
            .reset_index(name="Count")
        )
        fig2 = px.bar(grouped, x=cat_col, y="Count", color="Churn", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Numeric Feature Distributions by Churn")
        num_col = st.selectbox(
            "Select numeric feature:", ["tenure", "MonthlyCharges", "TotalCharges"]
        )
        fig3 = px.histogram(
            df, x=num_col, color="Churn", marginal="box",
            barmode="overlay", opacity=0.65,
        )
        st.plotly_chart(fig3, use_container_width=True)

elif page == "📈 Model Performance":
    st.title("Model Performance")
    metrics = get_metrics()
    results = get_results()

    if metrics is None:
        st.warning("No metrics found. Run `python scripts/train_pipeline.py` first.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Best Model F1 Score", f"{metrics['f1']:.4f}", delta="Target: 0.75")
        col2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Confusion Matrix", "ROC Curve", "PR Curve", "SHAP"]
        )
        image_map = {
            tab1: REPORTS_DIR / "confusion_matrix.png",
            tab2: REPORTS_DIR / "roc_curve.png",
            tab3: REPORTS_DIR / "pr_curve.png",
            tab4: REPORTS_DIR / "shap_summary.png",
        }
        for tab, img_path in image_map.items():
            with tab:
                if img_path.exists():
                    st.image(str(img_path), use_column_width=True)
                else:
                    st.info("Train the model to generate this plot.")

    if results:
        st.subheader("All Models")
        df_r = pd.DataFrame(results)
        fig = px.bar(
            df_r.sort_values("f1"),
            x="f1", y="model", orientation="h",
            color="f1", color_continuous_scale="Viridis",
            title="F1 Score by Model",
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Predict":
    st.title("Live Churn Prediction")
    st.markdown("Fill in the customer details below and click **Predict**.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )

        with col2:
            internet_service = st.selectbox(
                "Internet Service", ["Fiber optic", "DSL", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )

        with col3:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0
            )
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, value=780.0
            )

        submitted = st.form_submit_button("🔮 Predict Churn")

    if submitted:
        payload = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            prob = result["churn_probability"]
            will_churn = result["will_churn"]
            confidence = result["confidence"]
            color = "🔴" if will_churn else "🟢"
            st.markdown(f"## {color} {'Will Churn' if will_churn else 'Will NOT Churn'}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Churn Probability", f"{prob:.1%}")
            col2.metric("Decision", "Churn" if will_churn else "Retain")
            col3.metric("Confidence", confidence.capitalize())

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "red" if will_churn else "green"},
                    "steps": [
                        {"range": [0, 40], "color": "lightgreen"},
                        {"range": [40, 60], "color": "yellow"},
                        {"range": [60, 100], "color": "salmon"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error(
                "⚠️ API is not running. Start it with: `uvicorn api.main:app --port 8000`"
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

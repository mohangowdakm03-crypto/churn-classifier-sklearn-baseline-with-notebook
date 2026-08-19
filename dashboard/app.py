#!/usr/bin/env python3
"""
Streamlit Dashboard for Churn Classifier.
Provides interactive EDA, model performance, and live predictions with a premium UI.
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
    page_title="Churn Classifier",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
        <style>
        /* Hide main menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Premium padding and backgrounds */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Metric cards styling */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: #1E3A8A !important; /* Tailwind blue-900 */
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            font-weight: 500 !important;
            color: #64748B !important; /* Tailwind slate-500 */
        }
        div[data-testid="metric-container"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            transform: translateY(-2px);
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 0px 0px 0 0;
            padding-top: 10px;
            padding-bottom: 10px;
            font-weight: 600;
        }
        
        /* Button styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            height: 3rem;
            transition: all 0.2s ease;
        }
        
        /* Header typography */
        h1 {
            font-weight: 800 !important;
            letter-spacing: -0.025em !important;
            color: #0F172A !important;
        }
        h2, h3 {
            font-weight: 700 !important;
            color: #1E293B !important;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("✨ ChurnGuard AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔍 Data Explorer", "📈 Model Intelligence", "🎯 Live Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Powered by Scikit-Learn & Optuna")
    st.caption("v1.0.0 Production Build")

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    return load()

@st.cache_data(show_spinner=False)
def get_metrics():
    path = REPORTS_DIR / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None

@st.cache_data(show_spinner=False)
def get_results():
    path = MODELS_DIR / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    return None

# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.title("Customer Intelligence Platform")
    st.markdown(
        """
        <div style='font-size: 1.1rem; color: #475569; margin-bottom: 2rem;'>
        Leveraging advanced machine learning ensembles to proactively identify and retain at-risk customers.
        </div>
        """, unsafe_allow_html=True
    )

    df = get_data()
    churn_counts = df["Churn"].value_counts()
    
    # Premium Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customer Base", f"{len(df):,}")
    col2.metric("Historical Churn", f"{int(churn_counts.get('Yes', 0)):,}")
    col3.metric(
        "Average Churn Rate",
        f"{100 * churn_counts.get('Yes', 0) / len(df):.1f}%",
        delta="-2.1% vs industry avg"
    )

    st.markdown("---")
    results = get_results()
    if results:
        st.subheader("🚀 Model Performance Leaderboard")
        df_results = pd.DataFrame(results)[["model", "f1", "roc_auc", "train_time_sec"]]
        df_results = df_results.sort_values("f1", ascending=False)
        df_results["model"] = df_results["model"].str.replace("_", " ").str.title()
        df_results.columns = ["Algorithm", "Weighted F1", "ROC-AUC", "Train Time (s)"]
        
        # Display as a styled dataframe
        st.dataframe(
            df_results.style.background_gradient(cmap="Blues", subset=["Weighted F1", "ROC-AUC"])
                            .format({"Weighted F1": "{:.4f}", "ROC-AUC": "{:.4f}", "Train Time (s)": "{:.1f}"}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("💡 Run the training pipeline to generate model metrics.")

elif page == "🔍 Data Explorer":
    st.title("Dataset Explorer & Analytics")
    df = get_data()

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Churn Demographics", "📈 Numeric Trends"])

    with tab1:
        st.dataframe(df.head(100), use_container_width=True, height=500)
        st.caption(f"Showing first 100 of {len(df):,} records")

    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = px.pie(
                df, names="Churn", hole=0.6,
                color_discrete_sequence=["#3B82F6", "#EF4444"],
                title="Overall Churn Ratio"
            )
            fig.update_layout(showlegend=False, annotations=[dict(text='Churn', x=0.5, y=0.5, font_size=20, showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            cat_col = st.selectbox(
                "Analyze churn by category:",
                ["Contract", "InternetService", "PaymentMethod", "gender", "Partner"]
            )
            grouped = df.groupby([cat_col, "Churn"]).size().reset_index(name="Count")
            fig2 = px.bar(
                grouped, x=cat_col, y="Count", color="Churn", barmode="group",
                color_discrete_sequence=["#3B82F6", "#EF4444"],
                title=f"Churn Volume by {cat_col}"
            )
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        num_col = st.selectbox(
            "Select continuous metric:", ["tenure", "MonthlyCharges", "TotalCharges"]
        )
        fig3 = px.histogram(
            df, x=num_col, color="Churn", marginal="box",
            barmode="overlay", opacity=0.7,
            color_discrete_sequence=["#3B82F6", "#EF4444"],
            title=f"Distribution of {num_col}"
        )
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

elif page == "📈 Model Intelligence":
    st.title("Model Intelligence & Explainability")
    metrics = get_metrics()

    if metrics is None:
        st.warning("No metrics found. Run the training pipeline first.")
    else:
        st.markdown(
            f"""
            <div style='background-color: #ECFDF5; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #10B981; margin-bottom: 2rem;'>
                <strong>Production Model Active:</strong> The current champion model achieved a Weighted F1 score of 
                <strong>{metrics['f1']:.4f}</strong> and ROC-AUC of <strong>{metrics['roc_auc']:.4f}</strong>.
            </div>
            """, unsafe_allow_html=True
        )

        tab1, tab2, tab3 = st.tabs(["🧠 Global Feature Impact (SHAP)", "🎯 Classification Curves", "🧮 Confusion Matrix"])
        
        with tab1:
            st.markdown("### What drives customer churn?")
            st.caption("SHAP values quantify the exact impact of each feature on the model's predictions.")
            col_a, col_b = st.columns(2)
            with col_a:
                if (REPORTS_DIR / "shap_bar.png").exists():
                    st.image(str(REPORTS_DIR / "shap_bar.png"), use_column_width=True)
            with col_b:
                if (REPORTS_DIR / "shap_summary.png").exists():
                    st.image(str(REPORTS_DIR / "shap_summary.png"), use_column_width=True)

        with tab2:
            st.markdown("### Performance Curves")
            col_a, col_b = st.columns(2)
            with col_a:
                if (REPORTS_DIR / "roc_curve.png").exists():
                    st.image(str(REPORTS_DIR / "roc_curve.png"), use_column_width=True)
            with col_b:
                if (REPORTS_DIR / "pr_curve.png").exists():
                    st.image(str(REPORTS_DIR / "pr_curve.png"), use_column_width=True)

        with tab3:
            st.markdown("### Confusion Matrix")
            if (REPORTS_DIR / "confusion_matrix.png").exists():
                st.image(str(REPORTS_DIR / "confusion_matrix.png"), width=600)

elif page == "🎯 Live Predictor":
    st.title("Live Churn Risk Predictor")
    st.markdown("Enter a customer's profile below to instantly evaluate their churn risk using the live REST API.")

    with st.form("prediction_form", clear_on_submit=False):
        st.subheader("Customer Profile")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            
        with col2:
            st.markdown("**Core Services**")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            
        with col3:
            st.markdown("**Add-on Features**")
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            
        st.markdown("---")
        st.markdown("**Billing & Contract**")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        contract = col_b1.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = col_b2.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = col_b3.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = col_b4.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
        
        # Calculate a reasonable default total charge based on tenure and monthly if they don't override
        total_charges = tenure * monthly_charges

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🧠 Analyze Customer Risk", use_container_width=True)

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
        
        with st.spinner("Analyzing risk profile..."):
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                resp.raise_for_status()
                result = resp.json()
                
                prob = result["churn_probability"]
                will_churn = result["will_churn"]
                confidence = result["confidence"]
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Churn Risk Score", "font": {"size": 24}},
                        number={"suffix": "%", "font": {"size": 48}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                            "bar": {"color": "#1E293B"},
                            "bgcolor": "white",
                            "borderwidth": 2,
                            "bordercolor": "gray",
                            "steps": [
                                {"range": [0, 30], "color": "#10B981"},  # Green
                                {"range": [30, 60], "color": "#F59E0B"},  # Yellow
                                {"range": [60, 100], "color": "#EF4444"}, # Red
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    ))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                
                with res_col2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    if will_churn:
                        st.error(f"### 🚨 High Flight Risk\nThis customer is predicted to churn with **{confidence}** confidence.")
                        st.markdown("**Recommended Action:** Initiate retention protocol immediately. Consider offering a contract upgrade or discount.")
                    else:
                        st.success(f"### ✅ Safe\nThis customer is predicted to remain with **{confidence}** confidence.")
                        st.markdown("**Recommended Action:** Standard engagement. Candidate for upselling additional services.")
                        
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Backend API is unreachable. Start it with: `make api`")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


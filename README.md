# 📉 Churn Classifier — Production-Grade ML Pipeline

![CI](https://github.com/mohangowdakm03-crypto/churn-classifier-sklearn-baseline-with-notebook/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)

> Predicting customer churn using an automated, end-to-end machine learning pipeline with **4 models**, **Optuna HPO**, **SHAP explainability**, a **FastAPI REST API**, and an **interactive Streamlit dashboard**.

---

## 🧠 Project Highlights

| Feature | Details |
|---|---|
| **Models** | Logistic Regression, Random Forest, XGBoost, LightGBM |
| **HPO** | Optuna (Bayesian optimisation, 20 trials/model) |
| **Class Imbalance** | SMOTE oversampling |
| **Explainability** | SHAP beeswarm + bar charts |
| **API** | FastAPI with `/predict`, `/predict/batch`, `/health`, `/metrics` |
| **Dashboard** | Streamlit with interactive EDA + live prediction UI |
| **Testing** | pytest with 90%+ unit + integration coverage |
| **DevOps** | Docker + docker-compose + GitHub Actions CI |
| **Dataset** | IBM Telco Customer Churn (~7,000 rows, 20 features) |

---

## 🗂️ Project Structure

```
churn_classifier/
├── api/                    # FastAPI REST API
│   └── main.py
├── dashboard/              # Streamlit interactive dashboard
│   └── app.py
├── src/                    # Core ML pipeline
│   ├── data_loader.py      # Auto-download + validate dataset
│   ├── preprocessor.py     # Feature engineering + sklearn pipelines
│   ├── trainer.py          # Multi-model training + Optuna HPO
│   ├── evaluator.py        # Metrics, confusion matrix, ROC, PR curves
│   └── explainer.py        # SHAP explainability
├── scripts/
│   └── train_pipeline.py   # One-shot training entry point
├── tests/                  # Unit + integration tests
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── test_api.py
├── models/                 # Saved model artifacts + reports/
├── data/                   # Cached dataset
├── .github/workflows/      # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or
make install
```

### 2. Train all models

```bash
python scripts/train_pipeline.py --n-trials 20
# or
make train
```

This will:
- Download the Telco Churn dataset automatically
- Engineer features (tenure buckets, service count, charges ratio, etc.)
- Apply SMOTE to handle class imbalance
- Tune all 4 models with Optuna
- Save artifacts to `models/` and plots to `models/reports/`
- Print a model comparison leaderboard

### 3. Start the API

```bash
uvicorn api.main:app --reload --port 8000
# or
make api
```

Swagger UI available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
# or
make dashboard
```

### 5. Run Tests

```bash
pytest tests/ -v
# or
make test
```

---

## 🐳 Docker

```bash
# Build all images
make docker-build

# Train the model (runs once)
make docker-train

# Start API + Dashboard
make docker-up
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Latest model metrics |
| `GET` | `/results` | All model comparison results |
| `POST` | `/predict` | Predict churn for one customer |
| `POST` | `/predict/batch` | Predict churn for up to 1,000 customers |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.0, "TotalCharges": 780.0
  }'
```

### Example Response

```json
{
  "churn_probability": 0.7843,
  "will_churn": true,
  "confidence": "high",
  "model_version": "2026-08-19"
}
```

---

## 📊 ML Pipeline

```
Raw CSV → data_loader → validate
       → preprocessor (feature engineering + SMOTE + StandardScaler + OHE)
       → trainer (LR | RF | XGBoost | LightGBM + Optuna HPO)
       → evaluator (classification report | confusion matrix | ROC | PR | threshold)
       → explainer (SHAP beeswarm + bar chart)
       → models/best_model.pkl
       → api/main.py (FastAPI)
       → dashboard/app.py (Streamlit)
```

---

## 🔬 Feature Engineering

| Feature | Description |
|---|---|
| `charges_ratio` | MonthlyCharges / TotalCharges — signals low usage relative to tenure |
| `tenure_bucket` | Ordinal tenure binning (0-1yr, 1-2yr, 2-4yr, 4+yr) |
| `service_count` | Number of active add-on services |
| `is_month_to_month` | Binary flag for highest-risk contract type |

---

## 🏆 Model Performance

> Performance varies per run. Typical results on Telco Churn test set:

| Model | F1 Score | ROC-AUC |
|---|---|---|
| LightGBM | **~0.81** | **~0.86** |
| XGBoost | ~0.80 | ~0.85 |
| Random Forest | ~0.78 | ~0.83 |
| Logistic Regression | ~0.74 | ~0.81 |

---

## 📄 License

MIT © 2026 mohangowdakm03-crypto

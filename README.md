# churn-classifier

Telco customer churn prediction using scikit-learn. Trains and cross-validates
LogisticRegression, RandomForest, and GradientBoosting on the
[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset, then serves predictions through a Streamlit app.

## What's here

```
src/
  preprocess.py   ColumnTransformer pipeline (numeric + categorical)
  train.py        cross-validated model comparison and serialisation
  predict.py      load a saved pipeline and run inference
notebooks/
  01_eda.ipynb    class balance, distributions, correlation heatmap
app/
  app.py          Streamlit prediction UI
tests/            pytest suite (added in later commits)
data/raw/         place the raw CSV here (not committed)
```

## Dataset

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle and place it at
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the notebook

```bash
jupyter lab notebooks/01_eda.ipynb
```

## Train

```bash
python -m src.train
```

Trains all three models via 5-fold cross-validation, prints the comparison
table, and saves the best pipeline to `models/churn_pipeline.joblib`.

## Streamlit app

```bash
streamlit run app/app.py
```

Loads the saved pipeline. Adjust customer attributes in the sidebar and
the churn probability updates live.

## Test

```bash
pytest
```

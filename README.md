# FinTech Fraud Detection — End-to-End Machine Learning Project

An end-to-end machine learning project that detects potentially fraudulent card transactions.

This project is designed as a portfolio project for a Software Engineer transitioning into AI/ML. It demonstrates:

- Data generation and exploratory analysis
- Feature engineering
- Supervised learning for binary classification
- Train/test splitting and stratification
- Preprocessing with scikit-learn pipelines
- Logistic Regression and Random Forest model comparison
- Precision, Recall, F1, ROC-AUC and PR-AUC
- Model persistence with Joblib
- A FastAPI prediction API
- A Streamlit web interface
- Basic unit tests


## Project Structure

```text
fintech-fraud-ai/
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_pipeline.py
├── app.py
├── api.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Generate the dataset

```bash
python -m src.generate_data
```

This creates:

```text
data/transactions.csv
```

The generated dataset contains transaction features such as:

- transaction amount
- transaction hour
- merchant category
- country mismatch
- device trust score
- transactions in the previous 24 hours
- card age
- account age
- previous chargebacks
- and a fraud label

## 3. Train the models

```bash
python -m src.train
```

This:

1. Loads the synthetic transaction data
2. Splits it into train/test sets
3. Builds preprocessing pipelines
4. Trains Logistic Regression and Random Forest models
5. Evaluates both models
6. Selects the model using PR-AUC
7. Saves the selected pipeline to `models/fraud_model.joblib`
8. Writes evaluation metrics to `reports/metrics.json`

The script prints the metrics, so the README does not contain invented performance numbers.

## 4. Run the Streamlit application

First train the model, then:

```bash
streamlit run app.py
```

The browser interface allows a user to enter transaction information and receive:

- fraud probability
- risk classification
- the model threshold used by the application

## 5. Run the FastAPI service

```bash
uvicorn api:app --reload
```

Open the automatically generated API documentation at:

```text
http://127.0.0.1:8000/docs
```

The API endpoint is:

```text
POST /predict
```

Example JSON:

```json
{
  "amount": 850.0,
  "hour": 2,
  "merchant_category": "electronics",
  "country_mismatch": 1,
  "device_trust": 0.15,
  "transactions_24h": 9,
  "card_age_days": 120,
  "account_age_days": 180,
  "previous_chargebacks": 1
}
```

## 6. Run tests

```bash
pytest
```

## Machine Learning Approach

The target is binary classification:

```text
0 = legitimate
1 = potentially fraudulent
```

Categorical features are one-hot encoded and numerical features are standardized inside a scikit-learn `ColumnTransformer`.

The project compares:

- Logistic Regression — interpretable baseline
- Random Forest — nonlinear ensemble model

Because fraud detection is an imbalanced classification problem, the project emphasizes **Precision, Recall, F1 and PR-AUC**, rather than relying only on accuracy.

## Why PR-AUC matters

In fraud detection, the positive class can be much smaller than the legitimate class. A model can achieve high accuracy while still missing many fraudulent transactions.

Therefore, this project evaluates:

- Precision — how many flagged transactions are actually fraud
- Recall — how much of the fraud the model catches
- F1 — balance between precision and recall
- PR-AUC — performance across classification thresholds
- ROC-AUC — overall ranking performance




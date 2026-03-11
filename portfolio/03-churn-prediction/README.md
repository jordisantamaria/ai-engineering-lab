# Customer Churn Prediction

## Business Problem

Customer retention is one of the most critical pillars of any business:

- **Acquisition vs. retention cost**: acquiring a new customer costs between 5x and 25x more than retaining an existing one.
- **Revenue impact**: a 5% reduction in churn rate can increase profits by 25% to 95% (Harvard Business Review).
- **Cascade effect**: customers who leave generate negative word of mouth and drag others along.
- **Reactivity**: most companies only act when the customer has already left, when the recovery cost is at its maximum.

The problem is not knowing *how many* customers leave, but identifying *which ones* are at risk **before** they make the decision.

## Proposed Solution

Predictive machine learning model that analyzes customers' historical behavior to assign a churn probability to each one, enabling proactive and focused retention actions.

### Technical Approach

```
Customer historical data
        |
        v
  EDA (Exploratory Data Analysis)
        |
        v
  Feature Engineering
  (tenure_bucket, monthly_charges_per_service,
   contract_value, engagement_score, ...)
        |
        v
  Hyperparameter optimization (Optuna)
        |
        v
  XGBoost / LightGBM (ensemble)
        |
        v
  SHAP: interpretability
  ("why is this customer at risk?")
        |
        v
  Prediction API + Dashboard
```

### Dataset

- **Telco Customer Churn**: reference dataset from Kaggle ([link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).
- 7,043 customers with 21 variables (demographics, contracted services, billing, tenure).
- Target: `Churn` column (Yes/No).

## Expected Results

| Metric | Value |
|---------|-------|
| AUC-ROC | >0.85 |
| Precision (Churn class) | >0.75 |
| Recall (Churn class) | >0.80 |
| F1-Score | >0.77 |

Additionally, the model provides:
- **Top 5 risk factors** for each customer (via SHAP).
- **Risk segmentation**: high, medium, low.
- **Actionable insights**: which levers to pull to reduce risk for each segment.

## Technologies

- **XGBoost / LightGBM**: high-performance gradient boosting models
- **SHAP**: model interpretability and explainability
- **Optuna**: Bayesian hyperparameter optimization
- **scikit-learn**: preprocessing and metrics
- **pandas / numpy**: data manipulation
- **matplotlib / seaborn**: visualization
- **FastAPI**: real-time prediction API

## How to Run

### 1. Installation

```bash
cd portfolio/03-churn-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### 3. Train the Model

```bash
python src/train.py \
    --data_path data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --output_dir models/ \
    --n_trials 50
```

This will generate:
- `models/churn_model.joblib` - trained model
- `models/roc_curve.png` - ROC curve
- `models/pr_curve.png` - Precision-Recall curve
- `models/shap_summary.png` - SHAP feature importance summary
- `models/classification_report.txt` - classification report

### 4. Launch the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8002
```

### 5. Predict Customer Churn

```bash
curl -X POST "http://localhost:8002/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "tenure": 12,
        "monthly_charges": 79.5,
        "total_charges": 954.0,
        "contract": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "tech_support": "No",
        "num_services": 4
    }'
```

Response:
```json
{
    "churn_probability": 0.82,
    "risk_level": "high",
    "top_risk_factors": [
        {"feature": "contract_Month-to-month", "impact": 0.23},
        {"feature": "tenure", "impact": -0.18},
        {"feature": "online_security_No", "impact": 0.15}
    ]
}
```

## How to Present It: Client Pitch

### Value Proposition

> "With this model you can focus your retention campaigns on the 20% of customers with the highest risk, knowing exactly *why* each customer is at risk and *what* actions to take to retain them."

### Estimated ROI

**Scenario**: telecommunications company with 50,000 customers, average ticket 60 EUR/month, current churn rate 2% monthly.

| Item | Before | After |
|----------|-------|---------|
| Customers lost/month | 1,000 | 700 (-30%) |
| Revenue lost/month | 60,000 EUR | 42,000 EUR |
| Retention campaign cost | 0 EUR | 5,000 EUR (targeted) |
| **Net monthly savings** | - | **13,000 EUR** |

**Estimated annual savings: ~156,000 EUR**, in recovered revenue from retention alone. Not counting savings in new customer acquisition costs.

### Key Points for the Presentation

1. **Personalized demo**: if the client provides anonymized data, train the model with their real data and show results at the meeting.
2. **Interpretability**: SHAP allows explaining each prediction in business language ("this customer is high risk because they have only been with us for 3 months, have a monthly contract, and have no tech support").
3. **Actionability**: the model not only tells *who* is leaving, but *why*, enabling the design of specific interventions.
4. **CRM integration**: predictions can be integrated directly into Salesforce, HubSpot, or any CRM via API.
5. **Continuous improvement**: the model is retrained periodically with new data to maintain its accuracy.

### Frequently Asked Client Questions

- **"Our data is different"** - The approach is sector-agnostic. It adapts to any business with historical customer data.
- **"How much data do we need?"** - A minimum of 5,000 historical customers with at least 6 months of data. The more, the better.
- **"Is it a black box?"** - No. SHAP provides complete explanations for each prediction. It complies with regulatory explainability requirements.
- **"How often does it need retraining?"** - We recommend monthly or quarterly retraining, depending on how fast the business changes.

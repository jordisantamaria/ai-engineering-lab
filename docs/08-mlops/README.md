# MLOps

> **ML without MLOps = a demo that never reaches production.**
> MLOps is the discipline that turns ML experiments into reliable, reproducible, and maintainable systems in production.

---

## Table of Contents

- [What is MLOps and Why It Matters](#what-is-mlops-and-why-it-matters)
- [MLOps Maturity Levels](#mlops-maturity-levels)
- [Experiment Tracking](#experiment-tracking)
- [Data Versioning](#data-versioning)
- [Feature Store](#feature-store)
- [ML Pipelines](#ml-pipelines)
- [Model Registry](#model-registry)
- [Automatic Retraining](#automatic-retraining)
- [Infrastructure](#infrastructure)
- [What You Really Need](#what-you-really-need-for-consulting)

---

## What is MLOps and Why It Matters

MLOps (Machine Learning Operations) applies DevOps practices to the ML lifecycle. The goal is to take models from experimentation to production in a reproducible, automated, and monitored way.

**Without MLOps:**

```
Scientist trains model → Sends .pkl via email → Engineer tries to deploy it
→ "It doesn't work on my machine" → 3 weeks debugging → New data already arrived
→ The model is no longer useful → Start over
```

**With MLOps:**

```
Commit to main → Automatic pipeline → Train → Evaluate → Register model
→ Test in staging → Automatic deploy → Monitoring → Retrain if performance drops
```

**Why it matters for consulting:**

| Without MLOps | With MLOps |
|-----------|-----------|
| "Our data scientist left and nobody knows which model we use" | Versioned and documented models |
| "We don't know what data was used for training" | Versioned data, reproducible pipeline |
| "The model stopped working and we didn't notice" | Automatic drift alerts |
| "Retraining takes 2 weeks of manual work" | One-click retraining (or automatic) |
| 6-month project that dies upon delivery | System the client can maintain |

---

## MLOps Maturity Levels

Google defines three maturity levels for MLOps. It is the most referenced framework in the industry.

### Levels Table

| Aspect | Level 0: Manual | Level 1: ML Pipeline | Level 2: CI/CD + Pipeline |
|---------|----------------|---------------------|--------------------------|
| **Training** | Manual in notebook | Automated pipeline | Pipeline + CI/CD |
| **Deploy** | Manual (copy-paste) | Semi-automatic | Automatic |
| **Tracking** | Nothing or spreadsheet | MLflow/W&B | MLflow/W&B integrated in pipeline |
| **Testing** | None | Basic model validation | Data, model, and code tests |
| **Monitoring** | None | Basic metrics | Drift detection + alerts |
| **Retraining** | Whenever someone remembers | Triggered (schedule/manual) | Automatic by drift/schedule |
| **Reproducibility** | None | Partial (fixed pipeline) | Full (code + data + config) |
| **Deploy time** | Weeks | Days | Hours/minutes |

### Level 0: Manual (Where most clients are)

```
Jupyter Notebook
    ↓ (manual)
Model .pkl in local folder
    ↓ (manual)
"Hey, send me the model on Slack"
    ↓ (manual)
Manual deploy on a server
    ↓ (nobody monitors)
The model silently degrades
```

**Characteristics:**
- Everything is manual and ad-hoc
- No automated pipeline
- Unversioned data and models
- Knowledge is in the data scientist's head
- Reproducing an experiment is practically impossible

### Level 1: ML Pipeline Automation

```
New data (trigger)
    ↓ (automatic)
Pipeline: fetch → preprocess → train → evaluate
    ↓ (automatic)
Model registered in MLflow with metrics
    ↓ (manual/semi-auto)
Deploy if it exceeds threshold
    ↓ (automatic)
Basic monitoring
```

**The big leap:** The pipeline is reproducible. Anyone can run it and get the same result.

### Level 2: CI/CD Pipeline Automation

```
Push to main
    ↓ (automatic CI)
Code + data + model tests
    ↓ (automatic CD)
Full pipeline: train → evaluate → register → deploy
    ↓ (automatic)
Canary deployment → monitoring → rollback if it fails
    ↓ (automatic)
Retraining via drift detection
```

> **For consulting:** Most clients are at Level 0. Taking them to Level 1 is already a **big win** that justifies the project. Level 2 is rarely needed in a first phase.

---

## Experiment Tracking

### Why It's Essential

Without experiment tracking:
- "What hyperparameters did I use in that model that worked well?"
- "What was the accuracy of the model from 3 weeks ago?"
- "What version of the data did I use?"

With experiment tracking:
- Every experiment documented automatically
- Visual comparison of metrics
- Full reproducibility
- Ability to show the client the project's progress

### MLflow

MLflow is the open-source standard for experiment tracking. Four key concepts:

| Concept | What it is | Example |
|----------|--------|---------|
| **Experiment** | Group of related runs | "fraud-classifier-v2" |
| **Run** | A single training execution | "run with lr=0.01, epochs=50" |
| **Parameters** | Run configuration | learning_rate, batch_size, model_type |
| **Metrics** | Measurable results | accuracy, f1, loss |
| **Artifacts** | Generated files | model.pkl, confusion_matrix.png |

#### Full Code: Training with MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import json

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # or "sqlite:///mlflow.db" for local
mlflow.set_experiment("churn-classifier")

# Load data
df = pd.read_csv("data/churn_dataset.csv")
X = df.drop("churn", axis=1)
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters
params = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}

# Train with tracking
with mlflow.start_run(run_name="rf-baseline-v2"):
    # Log parameters
    mlflow.log_params(params)
    mlflow.log_param("dataset_version", "2024-01-15")
    mlflow.log_param("n_samples_train", len(X_train))
    mlflow.log_param("n_features", X_train.shape[1])

    # Train
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_weighted", f1)

    # Log artifacts
    report = classification_report(y_test, y_pred, output_dict=True)
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact("classification_report.json")

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = model.feature_importances_
    indices = importances.argsort()[-15:]
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([X.columns[i] for i in indices])
    ax.set_title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

    # Log the model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="churn-classifier",
    )

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1: {f1:.4f}")
```

#### MLflow UI: Compare Runs

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Open http://localhost:5000 in the browser
# - View all runs in an experiment
# - Compare metrics side by side
# - View evolution charts
# - Download artifacts
```

#### Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-classifier"
)

# Transition to staging
client.transition_model_version_stage(
    name="churn-classifier",
    version=result.version,
    stage="Staging",
)

# After tests, promote to production
client.transition_model_version_stage(
    name="churn-classifier",
    version=result.version,
    stage="Production",
)

# Load production model
model = mlflow.sklearn.load_model("models:/churn-classifier/Production")
```

#### Serving with MLflow

```bash
# Serve model directly from MLflow
mlflow models serve \
    -m "models:/churn-classifier/Production" \
    -p 5001 \
    --no-conda

# Make a prediction
curl -X POST http://localhost:5001/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'
```

### Weights & Biases (W&B)

W&B is the cloud-first alternative to MLflow. Better UI, better collaboration, more features but with a paid tier.

```python
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Initialize (creates a run on wandb.ai)
wandb.init(
    project="churn-classifier",
    name="rf-baseline-v2",
    config={
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "dataset_version": "2024-01-15",
    },
)

# Train
model = RandomForestClassifier(**wandb.config)
model.fit(X_train, y_train)

# Evaluate and log
y_pred = model.predict(X_test)
wandb.log({
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
})

# Log data table for analysis
table = wandb.Table(columns=["true", "predicted"])
for true, pred in zip(y_test[:100], y_pred[:100]):
    table.add_data(true, pred)
wandb.log({"predictions": table})

# Finish
wandb.finish()
```

#### W&B Sweeps (Hyperparameter Tuning)

```python
# sweep_config.yaml
sweep_config = {
    "method": "bayes",  # bayesian optimization
    "metric": {"name": "f1_weighted", "goal": "maximize"},
    "parameters": {
        "n_estimators": {"values": [100, 200, 500]},
        "max_depth": {"min": 3, "max": 20},
        "min_samples_split": {"min": 2, "max": 10},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.001,
            "max": 0.1,
        },
    },
}

def train_sweep():
    wandb.init()
    config = wandb.config

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    wandb.log({"f1_weighted": f1_score(y_test, y_pred, average="weighted")})

# Launch sweep
sweep_id = wandb.sweep(sweep_config, project="churn-classifier")
wandb.agent(sweep_id, function=train_sweep, count=50)  # 50 experiments
```

### MLflow vs W&B

| Aspect | MLflow | W&B |
|---------|--------|-----|
| **Hosting** | Self-hosted (or Databricks managed) | Cloud (wandb.ai) |
| **Cost** | Free (open source) | Free tier + $50/user/month pro |
| **UI** | Functional, basic | Excellent, interactive |
| **Experiment tracking** | Complete | Complete + tables, media |
| **Model registry** | Included | Included (Model Registry) |
| **Hyperparameter tuning** | Not native | Sweeps (bayesian, grid, random) |
| **Collaboration** | Limited | Excellent (teams, reports) |
| **Integration** | Universal | Universal + PyTorch/HF native |
| **Setup** | More work (self-host) | 1 line of code |
| **Data control** | Full (your servers) | Data on W&B cloud |
| **Ideal for** | Enterprise, on-prem, regulated industries | Startups, distributed teams |

> **Recommendation for consulting:** Start with MLflow (free, self-hosted, no external dependencies). If the client already uses W&B or needs advanced collaboration, use W&B.

---

## Data Versioning

### The Problem

```
"We trained the model with January data. Now the February data
is different. We don't know what changed. The new model is worse.
Can we go back to the January data?"

→ Without data versioning, the answer is "probably not".
```

### DVC (Data Version Control)

DVC works like git but for large files (data, models). Heavy files are stored in remote storage (S3, GCS) and only a pointer (.dvc file) is committed to git.

```bash
# Install DVC
pip install dvc dvc-s3  # or dvc-gs, dvc-azure

# Initialize in a git repo
dvc init

# Add a data file
dvc add data/training_data.csv
# This creates:
#   data/training_data.csv.dvc  (pointer, committed to git)
#   data/.gitignore             (the actual file is ignored in git)

git add data/training_data.csv.dvc data/.gitignore
git commit -m "Add training data v1"

# Configure remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc push  # Upload data to S3

# When data changes
dvc add data/training_data.csv
git add data/training_data.csv.dvc
git commit -m "Update training data v2"
dvc push

# Go back to a previous version
git checkout HEAD~1 -- data/training_data.csv.dvc
dvc checkout  # Downloads the previous version from S3

# On another machine, get the data
git clone <repo>
dvc pull  # Download data from S3
```

**How it works:**

```
Git repo:
├── data/
│   ├── training_data.csv.dvc   ← Pointer (file hash)
│   └── .gitignore               ← Ignores the actual CSV
├── models/
│   └── model.pkl.dvc            ← Pointer to model
└── dvc.yaml                     ← Pipeline definition

S3 bucket (dvc remote):
├── ab/cdef1234...               ← training_data.csv (v1)
├── 12/3456abcd...               ← training_data.csv (v2)
└── ff/eedd1122...               ← model.pkl
```

### Simple Alternative: Snapshots with Timestamp

For small projects where DVC is overkill:

```python
# scripts/snapshot_data.py
import boto3
from datetime import datetime

s3 = boto3.client("s3")
BUCKET = "ml-project-data"

def create_snapshot(local_path: str, dataset_name: str):
    """Upload a data snapshot with timestamp."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    s3_key = f"snapshots/{dataset_name}/{timestamp}/{dataset_name}.csv"

    s3.upload_file(local_path, BUCKET, s3_key)
    print(f"Snapshot created: s3://{BUCKET}/{s3_key}")
    return s3_key

def list_snapshots(dataset_name: str):
    """List available snapshots."""
    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=f"snapshots/{dataset_name}/",
        Delimiter="/",
    )
    for prefix in response.get("CommonPrefixes", []):
        print(prefix["Prefix"])

# Usage
create_snapshot("data/training.csv", "training")
list_snapshots("training")
```

---

## Feature Store

### What It Is

A feature store is a centralized repository of features (variables) prepared for ML. It stores data transformations ready to use in training and inference.

```
Without Feature Store:
  Model A: computes "avg_purchases_30d" in its pipeline
  Model B: computes "avg_purchases_30d" in its pipeline (different)
  Model C: computes "avg_purchases_30d" in its pipeline (yet another)
  → 3 different implementations of the same feature, possible inconsistencies

With Feature Store:
  Feature Store: has "avg_purchases_30d" computed once
  Model A, B, C: read the same feature → guaranteed consistency
```

### Feast (Open Source)

```python
# feature_store.yaml
project: my_ml_project
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
```

```python
# features.py
from feast import Entity, Feature, FeatureView, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# Data source
customer_source = FileSource(
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp",
)

# Entity
customer = Entity(
    name="customer_id",
    join_keys=["customer_id"],
)

# Feature view
customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    schema=[
        Feature(name="avg_purchases_30d", dtype=Float32),
        Feature(name="total_spend", dtype=Float32),
        Feature(name="days_since_last_purchase", dtype=Int64),
    ],
    source=customer_source,
    ttl=timedelta(days=1),
)
```

```python
# Use features for training
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Get historical features for training
training_df = store.get_historical_features(
    entity_df=entity_df,  # DataFrame with customer_id + timestamp
    features=[
        "customer_features:avg_purchases_30d",
        "customer_features:total_spend",
        "customer_features:days_since_last_purchase",
    ],
).to_df()

# Get online features for inference
feature_vector = store.get_online_features(
    features=[
        "customer_features:avg_purchases_30d",
        "customer_features:total_spend",
    ],
    entity_rows=[{"customer_id": 12345}],
).to_dict()
```

### Do You Need a Feature Store?

| Situation | Need a Feature Store? |
|-----------|------------------------|
| 1 model, small team | No. Pandas/SQL is enough. |
| 2-3 models, shared features | Probably not. Shared Python modules. |
| 5+ models, many shared features | Probably yes. |
| Need train/serve consistency | Yes, this is the strongest case. |
| Large team, many data scientists | Yes. |

> **For most consulting projects: you do NOT need a feature store.** A Python module with feature engineering functions is enough. When you have 5+ models sharing features, evaluate it.

---

## ML Pipelines

### What They Are

An ML pipeline is a DAG (Directed Acyclic Graph) that defines the steps from raw data to deployed model.

```
Raw data → Preprocess → Feature Engineering → Train → Evaluate → Deploy
    │              │                │                │         │         │
    ▼              ▼                ▼                ▼         ▼         ▼
   S3         clean data      features.csv    model.pkl  metrics   endpoint
```

### Complexity Levels

#### Simple: Python Scripts + Makefile

```makefile
# Makefile
.PHONY: all data features train evaluate deploy

all: data features train evaluate

data:
	python scripts/01_fetch_data.py

features: data
	python scripts/02_feature_engineering.py

train: features
	python scripts/03_train.py

evaluate: train
	python scripts/04_evaluate.py

deploy: evaluate
	python scripts/05_deploy.py

clean:
	rm -rf data/processed/ models/*.pkl
```

```bash
# Run the full pipeline
make all

# Only retrain (assumes data is already processed)
make train evaluate
```

#### Intermediate: Prefect

```python
# pipeline.py
from prefect import flow, task
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import joblib

@task(retries=2, retry_delay_seconds=60)
def fetch_data(source: str) -> pd.DataFrame:
    """Extract data from source."""
    return pd.read_sql(f"SELECT * FROM {source}", engine)

@task
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform data."""
    df = df.dropna(subset=["target"])
    df = df.fillna(df.median(numeric_only=True))
    return df

@task
def train_model(df: pd.DataFrame, params: dict):
    """Train model with tracking."""
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        return model, accuracy

@task
def evaluate_and_deploy(model, accuracy: float, threshold: float = 0.85):
    """Deploy if accuracy exceeds threshold."""
    if accuracy >= threshold:
        joblib.dump(model, "models/model_latest.pkl")
        print(f"Model deployed with accuracy {accuracy:.4f}")
    else:
        print(f"Model rejected: accuracy {accuracy:.4f} < {threshold}")

@flow(name="ML Training Pipeline")
def training_pipeline(source: str = "customers", params: dict = None):
    """Full training pipeline."""
    if params is None:
        params = {"n_estimators": 200, "max_depth": 10}

    df = fetch_data(source)
    df_clean = preprocess(df)
    model, accuracy = train_model(df_clean, params)
    evaluate_and_deploy(model, accuracy)

if __name__ == "__main__":
    training_pipeline()
```

#### Advanced: Kubeflow / SageMaker Pipelines

For enterprise scale. Each step is an independent container, scalable, with dedicated resources.

| Tool | Complexity | Scalability | Hosting | Ideal For |
|-------------|------------|--------------|---------|-----------|
| Scripts + Makefile | Low | Low | Local/server | POC, small projects |
| Scripts + cron | Low | Low | Server | Simple batch scheduling |
| Prefect | Medium | Medium | Cloud/self-hosted | Mid-size teams |
| Airflow | Medium-High | High | Self-hosted/Cloud | Complex pipelines, data eng |
| Kubeflow | High | Very high | Kubernetes | Enterprise, many models |
| SageMaker Pipelines | Medium | High | AWS managed | Full AWS ecosystem |
| Vertex AI Pipelines | Medium | High | GCP managed | Full GCP ecosystem |

> **Recommendation for consulting:** Start with scripts + Makefile (or cron). If you need retries, scheduling, and UI, move up to Prefect or Airflow. Kubeflow only if the client already has Kubernetes and needs serious scale.

---

## Model Registry

### Concept

The Model Registry is a centralized catalog of models with versioning and lifecycle management. Think of it as an internal "app store" for ML models.

```
Model stages:

None → Staging → Production → Archived
               ↑            │
               └────────────┘ (rollback if it fails)
```

### Full Workflow with MLflow Model Registry

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 1. TRAIN: Train and register
with mlflow.start_run():
    model = train_my_model(X_train, y_train)
    accuracy = evaluate(model, X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)

    # Register automatically
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="churn-classifier",
    )

# 2. STAGING: Promote version to staging
latest_version = client.get_latest_versions("churn-classifier", stages=["None"])[0]
client.transition_model_version_stage(
    name="churn-classifier",
    version=latest_version.version,
    stage="Staging",
)

# 3. TEST: Evaluate in staging
staging_model = mlflow.sklearn.load_model("models:/churn-classifier/Staging")
staging_accuracy = evaluate(staging_model, X_staging_test, y_staging_test)

# 4. PRODUCTION: If validation passes, promote
if staging_accuracy >= PRODUCTION_THRESHOLD:
    # Archive current production model
    prod_versions = client.get_latest_versions("churn-classifier", stages=["Production"])
    for v in prod_versions:
        client.transition_model_version_stage(
            name="churn-classifier",
            version=v.version,
            stage="Archived",
        )

    # Promote new model
    client.transition_model_version_stage(
        name="churn-classifier",
        version=latest_version.version,
        stage="Production",
    )
    print(f"Model v{latest_version.version} in production")

# 5. SERVE: Load production model
production_model = mlflow.sklearn.load_model("models:/churn-classifier/Production")
```

---

## Automatic Retraining

### Retraining Triggers

| Trigger | When | How to Detect |
|---------|--------|---------------|
| **Schedule** | Every week/month | Cron, Airflow schedule |
| **Data drift** | Input distribution changes | KS test, Evidently AI |
| **Performance drop** | Accuracy drops in production | Metrics monitoring |
| **New data** | New labeled data arrives | Event trigger (S3, DB) |
| **Manual** | The team decides | Button in UI, API call |

### Retraining Pipeline

```python
# retrain_pipeline.py
import mlflow
from datetime import datetime, timedelta

def should_retrain() -> tuple[bool, str]:
    """Decide whether retraining is needed."""
    reasons = []

    # Check 1: Time since last training
    client = mlflow.tracking.MlflowClient()
    prod_model = client.get_latest_versions("my-model", stages=["Production"])[0]
    last_trained = datetime.fromtimestamp(prod_model.creation_timestamp / 1000)
    if datetime.utcnow() - last_trained > timedelta(days=30):
        reasons.append("More than 30 days since last training")

    # Check 2: Data drift
    drift_score = check_data_drift()  # Your drift detection function
    if drift_score > 0.3:
        reasons.append(f"Data drift detected: {drift_score:.2f}")

    # Check 3: Performance drop
    current_accuracy = get_production_accuracy()  # From your monitoring
    baseline_accuracy = float(prod_model.tags.get("accuracy", "0.9"))
    if current_accuracy < baseline_accuracy - 0.05:
        reasons.append(
            f"Accuracy dropped from {baseline_accuracy:.3f} to {current_accuracy:.3f}"
        )

    return len(reasons) > 0, "; ".join(reasons)

def retrain_and_evaluate():
    """Full retraining pipeline."""
    should, reason = should_retrain()

    if not should:
        print("No retraining needed")
        return

    print(f"Retraining: {reason}")

    # 1. Fetch new data
    X_train, X_test, y_train, y_test = fetch_latest_data()

    # 2. Train new model
    with mlflow.start_run(run_name=f"retrain-{datetime.utcnow().isoformat()}"):
        new_model = train(X_train, y_train)
        new_accuracy = evaluate(new_model, X_test, y_test)

        mlflow.log_metric("accuracy", new_accuracy)
        mlflow.log_param("retrain_reason", reason)

        # 3. Compare with production model
        prod_model = mlflow.sklearn.load_model("models:/my-model/Production")
        prod_accuracy = evaluate(prod_model, X_test, y_test)

        print(f"New model: {new_accuracy:.4f} vs Production: {prod_accuracy:.4f}")

        # 4. Deploy only if better
        if new_accuracy > prod_accuracy:
            mlflow.sklearn.log_model(
                new_model, "model",
                registered_model_name="my-model",
            )
            print("New model registered. Ready to promote to production.")
        else:
            print("New model does not outperform current one. Not deploying.")
```

### Shadow Deployment and Canary

```
Shadow Deployment:
  Request → Model A (production) → Response to user
         → Model B (shadow)     → Log for comparison (not sent to user)

  Period: 1-2 weeks
  Goal: validate Model B without risk
  Decide: if B is consistently better → promote

Canary Deployment:
  Request → [90%] Model A (production)  → Response
         → [10%] Model B (canary)       → Response

  Period: gradual (10% → 25% → 50% → 100%)
  Goal: detect problems with limited real traffic
  Rollback: if B fails → go back to 100% A
```

---

## Infrastructure

### Decision Table

| Scale | Recommended Infra | Monthly Cost | Complexity |
|--------|------------------|---------------|-------------|
| **POC / MVP** | Laptop + Docker | $0 | Low |
| **1-2 models, low traffic** | Single server (EC2/VM) + Docker | $50-200 | Low |
| **2-5 models, medium traffic** | ECS/Cloud Run + managed DB | $200-1000 | Medium |
| **5+ models, high traffic** | Kubernetes + Seldon/KServe | $1000-5000 | High |
| **Enterprise** | SageMaker/Vertex AI (managed) | $2000+ | Medium (managed) |
| **Edge / IoT** | ONNX/TFLite on devices | Variable | High |

### Typical Infrastructure Diagram

```
Consulting - Typical Setup:

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   GitHub     │────▶│GitHub Actions│────▶│  Docker Hub  │
│   (code)     │     │  (CI/CD)     │     │  / ECR       │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌────────────────────────────▼──────────┐
                     │          Cloud (AWS/GCP/Azure)         │
                     │                                        │
                     │  ┌─────────┐     ┌─────────────────┐  │
                     │  │ FastAPI │     │  PostgreSQL/     │  │
                     │  │ + Docker│◄───▶│  Redis           │  │
                     │  │ (ECS)   │     │                  │  │
                     │  └────┬────┘     └─────────────────┘  │
                     │       │                                │
                     │  ┌────▼────┐     ┌─────────────────┐  │
                     │  │CloudWatch│    │  S3 (models,    │  │
                     │  │/Grafana │     │  data, logs)     │  │
                     │  └─────────┘     └─────────────────┘  │
                     └────────────────────────────────────────┘
```

---

## What You Really Need for Consulting

Most AI consulting projects do not need a complex MLOps stack. This is the pragmatic stack that covers 90% of cases:

### The Minimum Viable Stack

| Need | Tool | Why |
|-----------|-------------|---------|
| **Code** | Git + GitHub | Universal standard, no debate |
| **Experiments** | MLflow | Free, open source, experiment tracking + model registry |
| **Packaging** | Docker | Reproducibility, "works on any machine" |
| **Model serving** | FastAPI | Fast, typed, automatic documentation |
| **CI/CD** | GitHub Actions | Integrated with GitHub, free for public repos |
| **Monitoring** | CloudWatch + Prometheus | Infra metrics + custom metrics |

### When to Add More

```
"Do I need X?"

Feature Store → Do you have 5+ models sharing features? → If not, NO.
Kubernetes   → Do you have 10+ microservices? → If not, NO.
Kubeflow     → Do you have 20+ ML pipelines? → If not, NO.
Airflow      → Do you have 5+ pipelines with dependencies? → If not, scripts + cron.
Kafka        → Do you need to process 10k+ events/sec? → If not, SQS or batch.
Terraform    → Does your infra have 20+ cloud resources? → If not, console/CLI.
```

### MLOps Checklist for Consulting

```
Phase 1: Project starts
  [x] Git repo
  [x] requirements.txt / pyproject.toml
  [x] README with setup instructions
  [x] Local MLflow for tracking
  [ ] (Optional) DVC if data is large

Phase 2: Model ready
  [x] Model registered in MLflow
  [x] Baseline metrics documented
  [x] Basic model tests
  [x] Working Dockerfile
  [ ] (Optional) GitHub Actions for tests

Phase 3: In production
  [x] FastAPI endpoint deployed
  [x] CI/CD configured (build + test + deploy)
  [x] Latency and error monitoring
  [x] Health check endpoint
  [ ] (Optional) Drift detection

Phase 4: Maintenance
  [x] Retraining pipeline (at least manual)
  [x] Alerts configured
  [x] Documentation for the client's team
  [ ] (Optional) Automatic retraining
  [ ] (Optional) A/B testing
```

---

> **Key takeaway:** MLOps is not about installing every tool in the ecosystem. It is about having a reproducible and maintainable process for taking models to production. For consulting, Git + MLflow + Docker + FastAPI + GitHub Actions give you 90% of what you need. Add complexity only when the problem justifies it.

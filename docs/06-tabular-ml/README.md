# Tabular ML

## Why Tabular ML is 80% of Consulting Projects

Most enterprise data lives in tables: SQL databases, spreadsheets, CSVs, data warehouses. When a client says "I want to predict X", almost always X depends on tabular data: transactions, customer characteristics, operational metrics, logs.

```
Data in typical consulting:

70-80%  ->  Tables (SQL, CSV, Excel)  ->  Gradient Boosting
10-15%  ->  Text (emails, tickets)    ->  NLP / LLMs
5-10%   ->  Images (defects, docs)    ->  Computer Vision
1-5%    ->  Time series               ->  Time Series models
```

> **Reality:** Mastering XGBoost/LightGBM + feature engineering solves the majority of real ML consulting projects.

---

## Gradient Boosting: The King of Tabular

### Intuition

Gradient Boosting builds an ensemble of weak decision trees, where each new tree **learns from the errors of the previous one**.

```
Boosting process:

Original data: y_real = [10, 20, 30, 40, 50]

Step 1: Tree 1 predicts (simple, weak)
  y_pred_1 = [12, 18, 28, 35, 48]
  residuals = [10-12, 20-18, 30-28, 40-35, 50-48]
            = [-2, 2, 2, 5, 2]

Step 2: Tree 2 is trained on the RESIDUALS
  Tries to predict: [-2, 2, 2, 5, 2]
  y_pred_2 = [-1.5, 1.8, 2.1, 4.5, 1.9]

Step 3: Combine (with learning rate = 0.1)
  y_final = y_pred_1 + 0.1 * y_pred_2
           = [12, 18, 28, 35, 48] + 0.1 * [-1.5, 1.8, 2.1, 4.5, 1.9]
           = [11.85, 18.18, 28.21, 35.45, 48.19]

  Closer to [10, 20, 30, 40, 50]!

Step 4: Compute new residuals, train Tree 3...
Step 5: Repeat N times (n_estimators)

Each tree corrects a bit of the accumulated errors.
The learning rate controls how much it "trusts" each new tree.
```

```
Process diagram:

  Data -----> [Tree 1] -----> Prediction 1
                                     |
                              Compute Residuals
                                     |
  Residuals_1 -> [Tree 2] -----> Prediction 2
                                     |
                              Compute Residuals
                                     |
  Residuals_2 -> [Tree 3] -----> Prediction 3
                                     |
                                   ...
                                     |
  Final Prediction = Pred_1 + lr*Pred_2 + lr*Pred_3 + ... + lr*Pred_N
```

---

### XGBoost

XGBoost (eXtreme Gradient Boosting) is the most popular and robust implementation.

**How it works (intuition):**

Unlike classic gradient boosting, XGBoost:
- Uses a **regularized objective function** (prevents overfitting)
- Builds trees in a **level-wise** fashion (level by level)
- Handles missing values natively
- Parallelizes tree construction

**Key Hyperparameters:**

| Parameter | Typical Range | What it controls | Effect |
|---|---|---|---|
| `n_estimators` | 100-10000 | Number of trees | More = better until overfitting |
| `max_depth` | 3-10 | Maximum tree depth | More = more complex, more overfit |
| `learning_rate` (eta) | 0.01-0.3 | How much each tree contributes | Lower = more trees needed, better generalization |
| `subsample` | 0.5-1.0 | Fraction of rows per tree | Lower = more regularization |
| `colsample_bytree` | 0.5-1.0 | Fraction of columns per tree | Lower = more regularization |
| `reg_alpha` (L1) | 0-10 | L1 regularization | Higher = more sparsity |
| `reg_lambda` (L2) | 0-10 | L2 regularization | Higher = smaller weights |
| `min_child_weight` | 1-10 | Min leaf weight | Higher = more conservative |
| `gamma` | 0-5 | Min loss reduction for split | Higher = fewer splits |

**Complete Code Example:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# Load data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create model
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # For imbalance
    random_state=42,
    n_jobs=-1,
    eval_metric="auc",
    early_stopping_rounds=50,
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Best iteration
print(f"Best iteration: {model.best_iteration}")
```

**Feature Importance:**

```python
import matplotlib.pyplot as plt

# Three types of importance in XGBoost
# 1. Gain: average loss reduction when the feature is used
# 2. Weight (frequency): how many times the feature is used in splits
# 3. Cover: average number of samples affected by splits of this feature

# Gain is the most informative
importances = model.get_booster().get_score(importance_type="gain")
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

for feature, importance in sorted_imp[:10]:
    print(f"{feature:30s}: {importance:.4f}")

# Plot
xgb.plot_importance(model, importance_type="gain", max_num_features=15)
plt.tight_layout()
plt.show()
```

---

### LightGBM

LightGBM (Light Gradient Boosting Machine) from Microsoft. Generally faster than XGBoost.

**Differences from XGBoost:**

```
XGBoost (level-wise):          LightGBM (leaf-wise):
Grows ALL nodes at the          Grows the node with the GREATEST
same level                      loss reduction

       [root]                         [root]
      /      \                       /      \
    [A]      [B]                   [A]      [B]
   /   \    /   \                 /   \
 [C]  [D] [E]  [F]             [C]  [D]

More uniform,                   More accurate but
less overfit                    can overfit on
                                small data
```

**Advantages of LightGBM:**

| Advantage | Description |
|---|---|
| **GOSS** | Gradient-based One-Side Sampling: keeps samples with large gradient, samples those with small gradient |
| **EFB** | Exclusive Feature Bundling: bundles mutually exclusive features (reduces dimensionality) |
| **Native categoricals** | No need for one-hot encoding, LightGBM handles categoricals directly |
| **Speed** | 2-10x faster than XGBoost on large datasets |

**Key LightGBM Hyperparameters:**

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    max_depth=-1,              # No limit (leaf-wise is controlled by num_leaves)
    num_leaves=31,             # Max leaves per tree (KEY in LightGBM)
    learning_rate=0.1,
    subsample=0.8,             # Called 'bagging_fraction' internally
    colsample_bytree=0.8,     # Called 'feature_fraction' internally
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,      # Minimum samples in leaf
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100),
    ],
    categorical_feature=["city", "product_type"],  # Native categoricals!
)
```

### When to Choose LightGBM vs XGBoost

| Criterion | XGBoost | LightGBM |
|---|---|---|
| **Large dataset (>100K rows)** | Slow | Much faster |
| **Small dataset (<10K rows)** | Less overfit | Can overfit |
| **Many categoricals** | Needs encoding | Native (faster, sometimes better) |
| **Feature importance** | Good | Good |
| **GPU support** | Yes | Yes |
| **Community/docs** | More mature | Very good too |
| **Kaggle competitions** | Popular | Very popular |
| **Enterprise production** | Very stable | Very stable |

> **Practical advice:** Try both. The accuracy difference is usually <1%. LightGBM is faster for iterating. For the final delivery, use whichever gives the best cross-validation result.

### CatBoost

CatBoost (from Yandex) is another solid alternative:

- **Excellent with categoricals** without needing encoding (better than LightGBM for high-cardinality categoricals)
- **Less tuning needed** - defaults are good
- **Ordered boosting** - reduces data leakage during training
- Generally slightly slower than LightGBM

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.1,
    cat_features=["city", "product_type"],  # Indicate which are categoricals
    verbose=100,
)
model.fit(X_train, y_train, eval_set=(X_test, y_test))
```

---

## Advanced Feature Engineering for Tabular

Feature engineering is what has the most impact on tabular model performance. An XGBoost with good features > any model with raw features.

### Aggregation Features (GroupBy Stats)

```python
# For each customer, compute statistics of their transactions
agg_features = df.groupby("customer_id").agg(
    total_transactions=("amount", "count"),
    avg_amount=("amount", "mean"),
    max_amount=("amount", "max"),
    std_amount=("amount", "std"),
    days_as_customer=("date", lambda x: (x.max() - x.min()).days),
    unique_categories=("category", "nunique"),
).reset_index()

# Merge with the original dataset
df = df.merge(agg_features, on="customer_id", how="left")
```

### Interaction Features

```python
# Create features that capture relationships between variables
df["income_per_age"] = df["annual_income"] / (df["age"] + 1)
df["debt_to_income"] = df["total_debt"] / (df["annual_income"] + 1)
df["avg_balance_per_account"] = df["total_balance"] / (df["num_accounts"] + 1)

# Differences and ratios are very useful
df["price_difference"] = df["current_price"] - df["previous_price"]
df["price_ratio"] = df["current_price"] / (df["previous_price"] + 1)
```

### Time-Based Features

```python
# Extract temporal components
df["day_of_week"] = df["date"].dt.dayofweek       # 0=Monday, 6=Sunday
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["day_of_month"] = df["date"].dt.day
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["hour"] = df["date"].dt.hour
df["is_business_hours"] = df["hour"].between(9, 18).astype(int)

# Lag features (past values)
df = df.sort_values(["customer_id", "date"])
df["previous_amount"] = df.groupby("customer_id")["amount"].shift(1)
df["amount_7_days_ago"] = df.groupby("customer_id")["amount"].shift(7)

# Rolling statistics
df["amount_mean_7d"] = (
    df.groupby("customer_id")["amount"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)
df["amount_std_30d"] = (
    df.groupby("customer_id")["amount"]
    .transform(lambda x: x.rolling(30, min_periods=1).std())
)
```

### Categorical Encoding

```python
# Frequency Encoding: replace category by its frequency
freq_encoding = df["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq_encoding)

# Target Encoding (WITH CARE - data leakage!)
# Only use with proper cross-validation
from sklearn.model_selection import KFold

def target_encode_cv(df, col, target, n_splits=5):
    """Target encoding with cross-validation to avoid leakage."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    df[f"{col}_target_enc"] = 0.0
    global_mean = df[target].mean()

    for train_idx, val_idx in kf.split(df):
        means = df.iloc[train_idx].groupby(col)[target].mean()
        df.loc[df.index[val_idx], f"{col}_target_enc"] = (
            df.iloc[val_idx][col].map(means).fillna(global_mean)
        )
    return df

# Feature Crosses
df["city_x_type"] = df["city"].astype(str) + "_" + df["product_type"].astype(str)
```

### Feature Engineering Tips

| Technique | When it adds value |
|---|---|
| **Aggregation features** | Transactional data with an entity (customer, product) |
| **Interaction features** | When two features together are more informative |
| **Time features** | Data with a temporal component |
| **Frequency encoding** | High-cardinality categoricals |
| **Target encoding** | Categoricals correlated with the target (beware leakage) |
| **Feature crosses** | When the combination of categories matters |
| **Polynomial features** | Simple nonlinear relationships (beware dimensionality) |

---

## Handling Common Problems

### Imbalanced Classes

The most common problem in enterprise classification (fraud 1%, churn 5%, defects 2%).

```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Option 1: SMOTE (Synthetic oversampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# Caution: only apply to train, NEVER to test/validation

# Option 2: Class weights (simpler, recommended)
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
weight_dict = dict(zip(np.unique(y_train), weights))

# In XGBoost
model = xgb.XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
)

# In LightGBM
model = lgb.LGBMClassifier(
    class_weight="balanced",
    # Or manually:
    # is_unbalance=True,
)

# Option 3: Threshold tuning
from sklearn.metrics import precision_recall_curve

y_prob = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Find threshold that maximizes F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred_custom = (y_prob >= best_threshold).astype(int)
```

**Metrics for imbalanced data:**

| Metric | Use? | Why |
|---|---|---|
| **Accuracy** | NO | 99% accuracy predicting always "no fraud" with 1% fraud |
| **F1-score** | YES | Balances precision and recall |
| **AUC-PR** | YES | Better than AUC-ROC for severe imbalance |
| **AUC-ROC** | Partial | Can be optimistic with heavy imbalance |
| **Precision@K** | YES | "Of the top K alerts, how many are real?" |

### Missing Values

```python
# XGBoost and LightGBM handle NaN natively - often it's BETTER
# to leave NaN as-is than to impute

# If you need to impute (for other models or new features):
from sklearn.impute import SimpleImputer

# Numeric: median (robust to outliers)
num_imputer = SimpleImputer(strategy="median")

# Categorical: mode or special value
cat_imputer = SimpleImputer(strategy="constant", fill_value="MISSING")

# Create "missing" features (they can be informative!)
df["income_missing"] = df["income"].isna().astype(int)
```

### High Cardinality Categoricals

When a categorical variable has many unique values (cities, zip codes, product IDs).

```python
# Problem: One-hot encoding of 10,000 cities = 10,000 columns

# Solution 1: Target encoding (see previous section)

# Solution 2: Frequency encoding
df["city_freq"] = df["city"].map(df["city"].value_counts(normalize=True))

# Solution 3: Group rare categories
threshold = 0.01  # Categories with <1% frequency
freq = df["city"].value_counts(normalize=True)
df["city_grouped"] = df["city"].apply(
    lambda x: x if freq[x] >= threshold else "OTHER"
)

# Solution 4: LightGBM/CatBoost with native categoricals (the best)
# Just pass the list of categorical columns and the model handles them
```

### Feature Leakage

**What it is:** when your model has access to information it wouldn't have in production, artificially inflating metrics.

```
Common sources of leakage in business data:

1. Future features:
   Predicting March churn using April data
   -> Ensure features only use data BEFORE the event

2. Target leakage:
   Predicting if a patient has diabetes using "diabetes_medication"
   -> The medication is a CONSEQUENCE of the diagnosis

3. Train-test contamination:
   Normalizing BEFORE the split (the test "sees" training statistics)
   -> Always fit on train, transform on test

4. Group leakage:
   Same customer in train and test (the model "remembers" the customer)
   -> Split by GROUP, not by row
```

**How to detect leakage:**

```python
# Warning sign: AUC > 0.99 on your first model
# Check feature importance: if one feature dominates, investigate

importance = model.feature_importances_
for feat, imp in sorted(zip(X.columns, importance), key=lambda x: -x[1])[:5]:
    print(f"{feat}: {imp:.4f}")

# If the top feature has disproportionate importance,
# investigate whether there's leakage
```

---

## Production Pipeline with scikit-learn

### ColumnTransformer + Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# Define column types
numeric_features = ["age", "income", "balance", "num_transactions"]
categorical_features = ["city", "account_type", "segment"]

# Pipeline for numerics
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),  # Not necessary for tree models, but yes for others
])

# Pipeline for categoricals
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",  # Drop unlisted columns
)

# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
    )),
])

# Train
pipeline.fit(X_train, y_train)

# Predict (preprocessing is applied automatically)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
```

### Serialization with joblib

```python
import joblib

# Save complete pipeline (preprocessing + model)
joblib.dump(pipeline, "churn_model_v1.joblib")

# Load in production
pipeline_loaded = joblib.load("churn_model_v1.joblib")

# Predict with new data (same columns as in training)
new_customer = pd.DataFrame({
    "age": [35],
    "income": [50000],
    "balance": [12000],
    "num_transactions": [45],
    "city": ["Madrid"],
    "account_type": ["premium"],
    "segment": ["retail"],
})

churn_probability = pipeline_loaded.predict_proba(new_customer)[:, 1]
print(f"Churn probability: {churn_probability[0]:.2%}")
```

---

## Hyperparameter Tuning with Optuna

### Why Optuna

| Method | Efficiency | Implementation |
|---|---|---|
| **Grid Search** | Poor (explores everything) | Simple |
| **Random Search** | Acceptable | Simple |
| **Optuna** (Bayesian) | Very good | Moderate |

Optuna uses **Bayesian optimization**: it learns from previous runs to intelligently choose the next hyperparameters.

### Complete Example with XGBoost + Optuna

```python
import optuna
from sklearn.model_selection import cross_val_score
import xgboost as xgb

def objective(trial):
    """Objective function that Optuna optimizes."""

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
    }

    model = xgb.XGBClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=50,
    )

    # Cross-validation with early stopping
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="roc_auc",
        fit_params={
            "eval_set": [(X_test, y_test)],
            "verbose": False,
        }
    )

    return scores.mean()

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Best hyperparameters
print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Train final model with best parameters
best_model = xgb.XGBClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
```

### Tuning Tips

**Order of importance for tuning:**

```
1. learning_rate + n_estimators  (most impactful)
   -> Start with lr=0.1, n_estimators=1000 + early stopping

2. max_depth / num_leaves  (tree complexity)
   -> max_depth=6 is a good start

3. subsample + colsample_bytree  (regularization via sampling)
   -> 0.8 is a good start for both

4. reg_alpha + reg_lambda  (explicit regularization)
   -> Usually small values

5. min_child_weight / min_child_samples  (leaf control)
   -> Increase if there's overfitting
```

---

## Interpretability

In consulting, interpretability is **as important as accuracy**. The client asks "why does the model predict this?" and you need to answer.

### Feature Importance (Built-in)

```python
# Tree models have built-in feature importance
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print(importance.head(10))
```

Limitation: global feature importance tells you which features are important ON AVERAGE, but not why a specific INDIVIDUAL prediction is what it is.

### SHAP Values

SHAP (SHapley Additive exPlanations) tells you **how much each feature contributes to each individual prediction**.

**Intuition:**

```
Prediction for Customer X: churn probability = 78%
Baseline (dataset average): 25%

SHAP decomposes the difference (78% - 25% = 53%):

  tenure_months = 3        -> +20% (short time as customer)
  num_complaints = 5       -> +15% (many complaints)
  monthly_app_usage = 2    -> +12% (uses the app very little)
  income = 80000           -> -5%  (high income reduces churn)
  account_type = premium   -> -3%  (premium accounts churn less)
  ... other features ...   -> +14%
  -----------------------------------------
  Total SHAP:                 +53% (25% base + 53% = 78%)

Each SHAP value is interpreted as:
"This feature pushed the prediction X points up/down
compared to the dataset average"
```

**Code Example:**

```python
import shap

# Create explainer (uses TreeExplainer for tree models - VERY fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 1. Summary Plot: global importance + direction of effect
shap.summary_plot(shap_values, X_test)
# Shows features ordered by importance
# Each point is a sample
# Color = feature value (red=high, blue=low)
# Horizontal position = effect on prediction

# 2. Waterfall Plot: explanation of ONE prediction
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],         # First sample
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns.tolist(),
))

# 3. Force Plot: another way to see an individual prediction
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
)

# 4. Dependence Plot: how ONE feature affects predictions
shap.dependence_plot("tenure_months", shap_values, X_test)
# Shows the relationship between feature value and its SHAP effect
```

### Partial Dependence Plots (PDP)

```python
from sklearn.inspection import PartialDependenceDisplay

# PDP shows the MARGINAL average effect of a feature
PartialDependenceDisplay.from_estimator(
    model, X_test,
    features=["tenure_months", "num_transactions"],
    kind="both",  # Shows ICE lines + PDP average
)
```

Difference SHAP vs PDP:
- **SHAP:** effect of each feature for each prediction (local)
- **PDP:** average effect of a feature over the entire dataset (global)
- In consulting, **SHAP** is more useful for explaining individual predictions to the client.

### LIME

LIME (Local Interpretable Model-agnostic Explanations) creates a simple (linear) model that approximates the complex model's behavior around a specific prediction. Useful when you can't use SHAP (non-tree models), but SHAP is more robust.

---

## When to Use Deep Learning for Tabular

### DL Models for Tabular

| Model | Description | Performance |
|---|---|---|
| **TabNet** | Attention + feature selection | Similar to GBM, sometimes better |
| **FT-Transformer** | Feature Tokenizer + Transformer | Competitive with GBM |
| **TabTransformer** | Categorical embeddings + Transformer | Good with many categoricals |

### Reality

```
Typical benchmark on tabular data:

XGBoost/LightGBM:  AUC = 0.892
TabNet:             AUC = 0.887
FT-Transformer:    AUC = 0.890
Neural Network:     AUC = 0.875

Difference: marginal or nonexistent
Complexity: MUCH higher for DL
Development time: 5x more for DL
```

> **Practical rule:** Gradient boosting wins on pure tabular data in >95% of cases. The exception is when you have **multimodal** data (table + images + text) where DL allows fusing everything in an end-to-end model.

---

## Typical Consulting Case Step by Step

### Scenario: Churn Prediction for a Telco

#### 1. Meeting with the Client

```
Key questions:
- What do they consider "churn"? (cancellation, no usage in 30 days?)
- What actions will they take with predictions? (retention offers?)
- What data do they have available?
- What is the cost of a false positive vs false negative?
- How often do they need predictions? (daily, monthly?)
```

#### 2. Explore Data (EDA)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("telco_data.csv")

# Quick summary
print(f"Shape: {df.shape}")
print(f"\nTarget distribution:\n{df['churn'].value_counts(normalize=True)}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDtypes:\n{df.dtypes.value_counts()}")

# Correlations with the target
numeric_cols = df.select_dtypes(include="number").columns
correlations = df[numeric_cols].corrwith(df["churn"]).abs().sort_values(ascending=False)
print(f"\nCorrelations with churn:\n{correlations.head(10)}")
```

#### 3. Define Success Metric

```
Agreed with the client:
- Primary metric: F1-score (balance between precision and recall)
- Secondary metric: AUC-PR (due to imbalance)
- Goal: F1 > 0.70 (data team's current baseline: 0.55)
- Constraint: Recall > 0.75 (don't miss more than 25% of churners)
```

#### 4. Baseline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# Simple baseline
baseline = LogisticRegression(class_weight="balanced", max_iter=1000)
baseline.fit(X_train_processed, y_train)
y_pred_baseline = baseline.predict(X_test_processed)

print("=== BASELINE (Logistic Regression) ===")
print(classification_report(y_test, y_pred_baseline))
# Positive class F1: 0.58
```

#### 5. Feature Engineering

```python
# Behavior features
df["usage_trend"] = df["current_month_usage"] - df["previous_month_usage"]
df["usage_ratio"] = df["current_month_usage"] / (df["previous_month_usage"] + 1)
df["complaints_per_month"] = df["total_complaints"] / (df["tenure_months"] + 1)

# Engagement features
df["uses_app"] = (df["monthly_app_logins"] > 0).astype(int)
df["days_without_activity"] = (pd.Timestamp.now() - df["last_activity"]).dt.days

# Aggregations
agg = df.groupby("plan_id").agg(
    plan_churn_rate=("churn", "mean"),
    plan_avg_tenure=("tenure_months", "mean"),
).reset_index()
df = df.merge(agg, on="plan_id", how="left")
```

#### 6. XGBoost / LightGBM

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

y_pred = model.predict(X_test)
print("=== LightGBM ===")
print(classification_report(y_test, y_pred))
# Positive class F1: 0.74 (significant improvement!)
```

#### 7. Interpret Results with SHAP

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For the client presentation:
# 1. Global factors that cause churn
shap.summary_plot(shap_values[1], X_test)  # [1] = positive class (churn)

# 2. Individual example: "Why does this customer have high risk?"
idx = 42  # A customer with high churn probability
shap.waterfall_plot(shap.Explanation(
    values=shap_values[1][idx],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[idx],
    feature_names=X_test.columns.tolist(),
))
```

#### 8. Present to the Client

```
Presentation structure:

1. Reminder of the objective and agreed metric
2. Results:
   - "Our model correctly identifies 78% of customers who will churn"
   - "Of every 10 alerts, 7 are real churners"
3. Key churn factors (SHAP summary plot):
   - Decreasing usage trend
   - Recent unresolved complaints
   - Low tenure
4. Concrete example (SHAP waterfall):
   - "This customer has 82% churn probability BECAUSE..."
5. Business recommendations:
   - Prioritize retention for customers with decreasing usage
   - Resolving complaints in <48h reduces churn by 30%
6. Next steps: deployment, monitoring
```

#### 9. Deployment

```python
import joblib

# Save model
joblib.dump(pipeline, "churn_model_v1.joblib")

# In production (daily script):
def predict_churn_batch(new_data_path):
    """Predict churn for all active customers."""
    model = joblib.load("churn_model_v1.joblib")
    df = pd.read_csv(new_data_path)

    # Feature engineering (same transformations as in training)
    df = create_features(df)

    # Predict
    df["churn_probability"] = model.predict_proba(df[feature_columns])[:, 1]
    df["churn_risk"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["low", "medium", "high"]
    )

    # Export for the retention team
    high_risk = df[df["churn_risk"] == "high"].sort_values(
        "churn_probability", ascending=False
    )
    high_risk.to_csv("high_risk_customers.csv", index=False)

    return high_risk
```

---

## Tabular ML Project Checklist

```
PHASE 1: Understand the problem
[ ] Meet with the client and understand the business problem
[ ] Define the target variable precisely
[ ] Agree on success metric
[ ] Identify data sources

PHASE 2: Data
[ ] EDA: target distribution, missing values, outliers
[ ] Identify and handle data leakage
[ ] Define temporal or group-based split (not random if there are dependencies)
[ ] Feature engineering

PHASE 3: Modeling
[ ] Simple baseline (logistic regression or rules)
[ ] XGBoost / LightGBM with reasonable defaults
[ ] Feature selection (SHAP, importance, remove noise)
[ ] Hyperparameter tuning with Optuna
[ ] Cross-validation (5-fold or temporal)

PHASE 4: Interpretation
[ ] SHAP: global factors and individual explanations
[ ] Validate with domain experts (do the factors make sense?)
[ ] Error analysis (where does the model fail?)

PHASE 5: Delivery
[ ] Reproducible pipeline (ColumnTransformer + Pipeline)
[ ] Serialize model (joblib)
[ ] Document features and transformations
[ ] Present results to client (with interpretability)
[ ] Production monitoring plan
```

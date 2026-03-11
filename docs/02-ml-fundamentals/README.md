# Machine Learning Fundamentals

## Table of Contents

- [What is Machine Learning](#what-is-machine-learning)
- [Types of Learning](#types-of-learning)
- [The Complete ML Pipeline](#the-complete-ml-pipeline)
- [Feature Engineering](#feature-engineering)
- [Evaluation Metrics](#evaluation-metrics)
- [Underfitting vs Overfitting](#underfitting-vs-overfitting)
- [Cross-Validation](#cross-validation)
- [Key scikit-learn Algorithms](#key-scikit-learn-algorithms)
- [Baseline Models](#baseline-models)

---

## What is Machine Learning

### Traditional programming vs Machine Learning

```
Traditional programming:
  Data + Rules  -->  [ Program ]  -->  Result
  "If temperature > 30, then turn on AC"

Machine Learning:
  Data + Results  -->  [ Learning algorithm ]  -->  Model (learned rules)
  "Here are 10,000 houses with their prices; learn to predict the price"
```

The fundamental difference: in traditional programming you write the rules. In ML, the algorithm discovers the rules from the data.

### ML vs Deep Learning vs AI

```
+-----------------------------------------------------+
|                                                     |
|   Artificial Intelligence (AI)                      |
|   Any system that simulates intelligence            |
|                                                     |
|   +---------------------------------------------+   |
|   |                                             |   |
|   |   Machine Learning                          |   |
|   |   Learns from data, without programming     |   |
|   |   rules                                     |   |
|   |                                             |   |
|   |   +-------------------------------------+   |   |
|   |   |                                     |   |   |
|   |   |   Deep Learning                     |   |   |
|   |   |   ML with deep neural networks      |   |   |
|   |   |   (many layers)                     |   |   |
|   |   |                                     |   |   |
|   |   |   +-----------------------------+   |   |   |
|   |   |   | LLMs (GPT, Claude, etc.)    |   |   |   |
|   |   |   | DL with Transformers        |   |   |   |
|   |   |   +-----------------------------+   |   |   |
|   |   +-------------------------------------+   |   |
|   +---------------------------------------------+   |
+-----------------------------------------------------+
```

| | Classical ML | Deep Learning |
|---|---|---|
| Data | Hundreds to thousands | Thousands to millions |
| Features | You design them (feature engineering) | The model learns them |
| Interpretability | High (you can explain decisions) | Low (black box) |
| Compute | CPU sufficient | GPU required |
| Development time | Fast | Slow |
| **When to use** | **Tabular data, few data, need to explain** | **Images, text, audio, lots of data** |

---

## Types of Learning

### Supervised Learning

The model learns from labeled examples: for each input (X), you tell it the correct output (y).

#### Classification: predict a category

```
Input (features)           -->  Output (label)
Email [words, sender]      -->  Spam / Not spam
Image [pixels]             -->  Cat / Dog
Transaction [amount, time] -->  Fraud / Normal
```

#### Regression: predict a number

```
Input (features)                      -->  Output (number)
House [sqm, bedrooms, area]           -->  Price: 350,000
Customer [age, history, income]       -->  Churn probability: 0.73
Product [category, season]            -->  Demand: 1,240 units
```

### Unsupervised Learning

No labels. The model discovers patterns and structure in the data by itself.

#### Clustering: group similar data

```
Customer data -->  [ K-Means ]  -->  Group A (young, high spending)
                                     Group B (older, savers)
                                     Group C (families, medium spending)
```

#### Dimensionality reduction: compress data

```
Data with 100 features  -->  [ PCA ]  -->  Data with 10 features
                                           (keeping 95% of the information)
```

### Other types (mention)

| Type | What it is | Example |
|---|---|---|
| **Semi-supervised** | Few labeled data + many unlabeled | Classify 1M images with only 1K labeled |
| **Self-supervised** | The model creates its own labels from the input | GPT: predict the next word from the text |
| **Reinforcement Learning** | Learn by trial and error with rewards | AlphaGo, robots, RLHF in LLMs |

---

## The Complete ML Pipeline

```
1. Business  -->  2. Data  -->  3. Features  -->  4. Split  -->  5. Model
   problem         EDA        engineering      train/val/test   selection

                                                                    |
                                                                    v
8. Deploy  <--  7. Evaluation  <--  6. Training
   production     metrics          fit(X_train, y_train)
```

### Step 1: Define the business problem

Before touching code, answer these questions:

- What business problem am I solving?
- What type of problem is it? (classification, regression, clustering)
- What is the business success metric? (not just ML metric)
- What data do I have? Is it sufficient?
- Is there a human baseline or simple rule?
- What is the cost of being wrong? (false positive vs false negative)

> **Key point for consulting:** 80% of the value is in defining the problem well and choosing the right metric. A perfect model optimizing the wrong metric is useless.

### Step 2: EDA (Exploratory Data Analysis)

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# General summary
print(f"Shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nStatistics:\n{df.describe()}")

# Target distribution
print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True)}")

# Correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Feature distributions
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
for ax, col in zip(axes.flatten(), df.select_dtypes(include=[np.number]).columns):
    sns.histplot(data=df, x=col, hue='target', kde=True, ax=ax)
plt.tight_layout()
plt.show()
```

### Step 3: Feature Engineering

(See detailed section below)

### Step 4: Split train/val/test

```python
from sklearn.model_selection import train_test_split

# Basic split (80/10/10)
X = df.drop('target', axis=1)
y = df['target']

# First: separate test (don't touch until the end)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Second: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.11, random_state=42, stratify=y_temp
)
# 0.11 of 0.9 is ~0.1 of the total

print(f"Train: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Val:   {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test:  {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.0f}%)")
```

#### Types of split depending on the problem

| Split type | When to use | Example |
|---|---|---|
| **Random split** | i.i.d. data, general classification | Classify emails as spam |
| **Stratified split** | Imbalanced classes | Fraud detection (1% fraud) |
| **Temporal split** | Time series, dated data | Predict future sales |
| **Group split** | Grouped data (same users, patients) | Per-patient prediction |

```python
# Stratified split (maintains class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Temporal split (dated data)
df = df.sort_values('date')
split_date = '2024-01-01'
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]
# NEVER use future data to train

# Group split (a user only in train or test, not both)
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['user_id']))
```

> **Key point:** The test set is sacred. It is only used ONCE at the end to report results. If you use it to make decisions during development, you contaminated your evaluation.

### Step 5-6: Model selection and training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Quickly try several models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"{name}: Accuracy = {acc:.4f}")
```

### Step 7: Evaluation

(See detailed metrics section below)

### Step 8: Deployment

```python
import joblib

# Save model
joblib.dump(best_model, 'models/model_v1.joblib')

# Load model
model = joblib.load('models/model_v1.joblib')

# Predict with new data
prediction = model.predict(X_new)
```

---

## Feature Engineering

Feature engineering is the art of transforming raw data into features that a model can use effectively. In many cases, good feature engineering matters more than choosing the right model.

### Numeric features

#### Scaling (normalization)

Many models (SVM, KNN, neural networks, logistic regression) are sensitive to feature scale. A feature with range [0, 1000] will dominate over one with range [0, 1].

| Scaler | What it does | Intuitive formula | When to use |
|---|---|---|---|
| **StandardScaler** | Mean 0, std 1 | (x - mean) / std | Default for most cases |
| **MinMaxScaler** | Scale to [0, 1] | (x - min) / (max - min) | When you need a fixed range |
| **RobustScaler** | Uses median and IQR | (x - median) / IQR | Data with outliers |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (the most used)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform on train
X_val_scaled = scaler.transform(X_val)           # only transform on val/test
X_test_scaled = scaler.transform(X_test)         # only transform on val/test

# IMPORTANT: fit only on train, transform on everything
# If you fit on everything, there is data leakage
```

> **Key point:** Tree-based models (Random Forest, XGBoost) do NOT need scaling. Trees are based on splits, not distances. But linear models, SVM, and neural networks DO need it.

#### Log transform

Useful for features with a right-skewed distribution: prices, salaries, counts.

```python
# Log transform (compresses large values, expands small ones)
# Before:  [100, 200, 500, 10000, 50000]  -> very skewed
# After:   [4.6, 5.3, 6.2, 9.2, 10.8]    -> more uniform

df['log_salary'] = np.log1p(df['salary'])  # log(1+x) to avoid log(0)
# To undo: np.expm1(df['log_salary'])
```

#### Binning (discretization)

Convert a continuous value into categories. Useful when the relationship is not linear.

```python
# Binning by quantiles
df['age_bin'] = pd.qcut(df['age'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# Binning by fixed ranges
df['age_group'] = pd.cut(df['age'],
    bins=[0, 18, 30, 45, 60, 100],
    labels=['minor', 'young', 'adult', 'mature', 'senior']
)
```

### Categorical features

| Method | What it does | When to use | Example |
|---|---|---|---|
| **One-hot** | Creates binary column per value | Few categories (<20), no order | Country, color |
| **Label encoding** | Assigns number to each value | Trees, ordinal categories | Size (S=0, M=1, L=2) |
| **Target encoding** | Replaces with target mean | Many categories (>20) | Zip code |
| **Frequency encoding** | Replaces with frequency | Many categories, no leakage | Product category |

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['color']])
# color='red'   -> [1, 0, 0]
# color='blue'  -> [0, 1, 0]
# color='green' -> [0, 0, 1]

# Or with Pandas (simpler)
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')

# Label encoding (for trees or ordinal categories)
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])

# Target encoding (beware of leakage, use fold-based)
# Each category is replaced with the target mean for that category
means = df.groupby('city')['target'].mean()
df['city_target_enc'] = df['city'].map(means)

# Frequency encoding
freq = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq)
```

### Missing values: imputation strategies

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Imputer with median (robust to outliers)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# KNN imputer (more sophisticated, uses nearest neighbors)
knn_imputer = KNNImputer(n_neighbors=5)
X_train_imputed = knn_imputer.fit_transform(X_train)
```

### Feature selection

Not all features help. Some introduce noise or are correlated with each other.

```python
# 1. Remove features with zero or near-zero variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# 2. Correlation: remove features highly correlated with each other
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
df = df.drop(columns=to_drop)

# 3. Model importance (feature importance with Random Forest)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).head(20).plot(kind='barh')
plt.title('Feature Importance (Random Forest)')
plt.show()

# 4. Mutual Information
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=X_train.columns)
mi_series.sort_values(ascending=False).head(20).plot(kind='barh')
plt.title('Mutual Information Scores')
plt.show()
```

---

## Evaluation Metrics

Choosing the right metric is as important as choosing the model. In consulting, the metric must align with the business objective.

### Classification Metrics

#### Confusion Matrix explained

```
                     Model Prediction
                    Positive       Negative
               +--------------+--------------+
    Positive   |     TP       |     FN       |
Actual         |   True       |   False      |
    (label)    |  Positive    |  Negative    |
               +--------------+--------------+
    Negative   |     FP       |     TN       |
               |   False      |   True       |
               |  Positive    |  Negative    |
               +--------------+--------------+

TP = You said "yes" and it was "yes"     (correct)
TN = You said "no" and it was "no"       (correct)
FP = You said "yes" but it was "no"      (false alarm)
FN = You said "no" but it was "yes"      (missed it)
```

#### Derived metrics

| Metric | Intuitive formula | What it measures | Range |
|---|---|---|---|
| **Accuracy** | Correct / Total | Overall correct proportion | [0, 1] |
| **Precision** | TP / (TP + FP) | Of those I said positive, how many actually were | [0, 1] |
| **Recall** | TP / (TP + FN) | Of the actual positives, how many I detected | [0, 1] |
| **F1** | 2 * (Prec * Rec) / (Prec + Rec) | Balance between precision and recall | [0, 1] |
| **AUC-ROC** | Area under ROC curve | Ability to separate classes | [0.5, 1] |
| **AUC-PR** | Area under Precision-Recall curve | Performance on positive class | [0, 1] |
| **Log Loss** | -mean(y*log(p) + (1-y)*log(1-p)) | Quality of probabilities | [0, inf) |

#### When to use each metric

| My dataset has... | Use this metric | Why |
|---|---|---|
| Balanced classes (50/50) | Accuracy, F1 | Accuracy works well when classes are balanced |
| Imbalanced classes (95/5) | AUC-PR, F1, Recall | Accuracy is misleading (always predicting "negative" gives 95%) |
| High cost of FN (cancer, fraud) | **Recall** | We don't want to miss a real case |
| High cost of FP (spam, recommendations) | **Precision** | We don't want to bother with false alarms |
| Need calibrated probabilities | Log Loss | Measures quality of probabilities, not just the decision |
| Need a single number to compare | AUC-ROC or F1 | Summarizes performance in a single value |

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    classification_report
)

# Full report
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities

print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Individual metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"AUC-PR:    {average_precision_score(y_test, y_prob):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_prob):.4f}")
```

### Regression Metrics

| Metric | Intuitive formula | Interpretation | Units |
|---|---|---|---|
| **MAE** | mean(\|actual - pred\|) | Average absolute error | Same as y |
| **MSE** | mean((actual - pred)^2) | Penalizes large errors | Units^2 |
| **RMSE** | sqrt(MSE) | Typical error | Same as y |
| **R²** | 1 - (MSE model / MSE mean) | % of variance explained | Unitless |
| **MAPE** | mean(\|actual-pred\| / \|actual\|) * 100 | Mean percentage error | % |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE:  {mae:.2f}")        # "We are off by an average of 5,000 EUR"
print(f"RMSE: {rmse:.2f}")       # "Typical error of 7,000 EUR"
print(f"R²:   {r2:.4f}")         # "We explain 87% of the price variance"
print(f"MAPE: {mape:.1f}%")      # "Relative error of 8.3%"
```

| When to use | Metric | Reason |
|---|---|---|
| Want to interpret the error easily | MAE | Intuitive units, robust to outliers |
| Want to penalize large errors | RMSE | Large errors weigh more |
| Want to compare models across different datasets | R² | Normalized, comparable |
| Want relative error | MAPE | Percentage, easy to communicate |

---

## Underfitting vs Overfitting

### Bias-Variance Tradeoff

```
Total error = Bias² + Variance + Irreducible noise

                    Underfitting               Just right             Overfitting
                    (high bias)               (balanced)             (high variance)

Real data:          *  *                     *  *                    *  *
                  *      *                 *      *                *    * *
                *          *             *    --    *             *  /\   \  *
                  *      *              *  --    --  *            * /  \   \ *
                    *  *                 --          --             /    \___\

Model:            ----------             -- curve --               /\/\/\/\/\
                  (straight line)        (fits well)              (memorizes data)

Train error:       HIGH                    LOW                    VERY LOW
Val error:         HIGH                    LOW                    HIGH
Diagnosis:       Model too simple       Model adequate          Model too complex
```

| | Underfitting | Overfitting |
|---|---|---|
| **Symptom** | Train and val error high | Train error low, val error high |
| **Cause** | Model too simple | Model too complex |
| **Solution** | More complex model, more features | Regularization, more data, simpler model |

### Regularization techniques

| Technique | What it does | Where it is used |
|---|---|---|
| **L1 (Lasso)** | Penalizes sum of absolute weight values, can make weights = 0 | Linear regression, feature selection |
| **L2 (Ridge)** | Penalizes sum of squared weights, reduces large weights | Linear regression, neural networks |
| **Elastic Net** | Combines L1 and L2 | Linear regression |
| **Dropout** | Randomly deactivates neurons during training | Neural networks |
| **Early stopping** | Stop training when val_loss increases | Any iterative model |
| **Data augmentation** | Create synthetic data (rotate images, etc.) | Images, text |
| **Max depth, min samples** | Limit tree complexity | Decision trees, Random Forest |

```python
# L1 (Lasso) - does automatic feature selection
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)  # alpha controls regularization strength

# L2 (Ridge)
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)

# Elastic Net (mix L1 + L2)
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 0.5 = half L1, half L2

# Early stopping with XGBoost
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=1000,
    early_stopping_rounds=50,  # Stop if val_loss doesn't improve in 50 rounds
    eval_metric='logloss'
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)
```

### Learning curves: how to diagnose

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation')
plt.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
plt.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
```

**How to read learning curves:**

```
Underfitting:                   Overfitting:                   Just right:
  Score                          Score                          Score
  1.0 |                         1.0 |-- train                  1.0 |-- train
      |                             |                               |-- val
  0.8 |-- train                  0.8 |                          0.8 |=======
      |-- val                        |                               |
  0.6 |=======                   0.6 |          val             0.6 |
      |  both low                    |----------                    |
  0.4 |                          0.4 |  large gap               0.4 |
      +---------- data              +---------- data              +---------- data

  Both curves converge            Large distance between        Both curves converge
  but to a low value.             train and val.                to a high value.
  -> More complex model           -> More data or regularize    -> All good
```

---

## Cross-Validation

Cross-validation gives you a more robust estimate of performance than a single train/val split.

### K-Fold Cross-Validation

```
Data: [========================================]

Fold 1: [VVVV][================================]  -> Score 1
Fold 2: [====][VVVV][============================]  -> Score 2
Fold 3: [========][VVVV][========================]  -> Score 3
Fold 4: [============][VVVV][====================]  -> Score 4
Fold 5: [================][VVVV][================]  -> Score 5

V = Validation (test for that fold)
= = Training

Final score = mean(Score 1..5) +/- std(Score 1..5)
```

### Types of Cross-Validation

| Type | When to use | Example |
|---|---|---|
| **K-Fold** (k=5 or 10) | Medium datasets, uniform distribution | General classification |
| **Stratified K-Fold** | Imbalanced classes | Fraud detection, medical diagnosis |
| **Time Series Split** | Temporal data | Sales prediction, prices |
| **Group K-Fold** | Grouped data (users, patients) | A user cannot be in train AND test |
| **Leave-One-Out** | Very small datasets (<100) | Rarely used in practice |
| **Repeated K-Fold** | Want very robust estimation | Reporting results in papers |

```python
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, TimeSeriesSplit, GroupKFold
)

# Basic K-Fold
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

# Stratified K-Fold (maintains class proportions in each fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
print(f"F1: {scores.mean():.4f} +/- {scores.std():.4f}")

# Time Series Split (respects temporal order)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
print(f"MSE: {-scores.mean():.4f} +/- {scores.std():.4f}")

# Group K-Fold (groups don't mix between folds)
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring='accuracy')
```

#### Time Series Split visualized

```
Temporal data: [Jan][Feb][Mar][Apr][May][Jun][Jul][Aug]

Fold 1: [Train: Jan-Feb        ]  [Test: Mar     ]
Fold 2: [Train: Jan-Feb-Mar    ]  [Test: Apr     ]
Fold 3: [Train: Jan-Feb-Mar-Apr]  [Test: May     ]
Fold 4: [Train: Jan-....-May   ]  [Test: Jun     ]
Fold 5: [Train: Jan-....-Jun   ]  [Test: Jul     ]

Training always comes BEFORE test.
We never use future data to train.
```

---

## Key scikit-learn Algorithms

### Summary table

| Algorithm | Type | Complexity | Interpretable | Needs scaling | Best for |
|---|---|---|---|---|---|
| **Linear Regression** | Regression | Low | High | Yes | Baseline, linear relationships |
| **Logistic Regression** | Classification | Low | High | Yes | Baseline, text (with TF-IDF) |
| **Decision Tree** | Both | Medium | High | No | Explain decisions |
| **Random Forest** | Both | Medium | Medium | No | **Default for tabular** |
| **Gradient Boosting** (XGBoost, LightGBM) | Both | High | Low | No | **Best tabular performance** |
| **SVM** | Both | High | Low | Yes | Small datasets, high dim |
| **KNN** | Both | Low (train) | Medium | Yes | Small datasets, simple |
| **K-Means** | Clustering | Low | Medium | Yes | Customer segmentation |
| **DBSCAN** | Clustering | Medium | Medium | Yes | Irregularly shaped clusters |
| **PCA** | Dim. reduction | Low | Medium | Yes | Visualization, preprocessing |

### Quick examples

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Logistic regression (always a good baseline)
lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_train, y_train)

# Random Forest (robust, little tuning)
rf = RandomForestClassifier(
    n_estimators=100,    # number of trees
    max_depth=10,        # maximum depth
    min_samples_leaf=5,  # minimum samples per leaf
    random_state=42,
    n_jobs=-1            # use all cores
)
rf.fit(X_train, y_train)

# XGBoost (generally the best for tabular data)
import xgboost as xgb
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
```

### Dimensionality reduction: PCA vs t-SNE vs UMAP

| Method | Speed | Preserves | Primary use |
|---|---|---|---|
| **PCA** | Fast | Global structure (variance) | Preprocessing, reduce dimensions before model |
| **t-SNE** | Slow | Local structure (neighbors) | 2D cluster visualization |
| **UMAP** | Fast | Global + local | 2D visualization (better than t-SNE) |

```python
from sklearn.manifold import TSNE
import umap  # pip install umap-learn

# PCA (2 components to visualize)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Visualize all three
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, X_reduced, title in zip(axes, [X_pca, X_tsne, X_umap], ['PCA', 't-SNE', 'UMAP']):
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='Set2', alpha=0.6, s=10)
    ax.set_title(title)
    ax.legend(*scatter.legend_elements(), title="Class")
plt.tight_layout()
plt.show()
```

---

## Baseline Models

### Why always start with a simple baseline

A baseline is the simplest model you can build. It serves as a reference: if your complex model doesn't beat the baseline, something is wrong.

| Problem type | Baseline | Implementation |
|---|---|---|
| Binary classification | Always predict majority class | `DummyClassifier(strategy='most_frequent')` |
| Classification | Logistic Regression | `LogisticRegression()` |
| Regression | Always predict the mean | `DummyRegressor(strategy='mean')` |
| Regression | Linear Regression | `LinearRegression()` |
| Time series | Predict the previous value | `y_pred = y_test.shift(1)` |
| NLP | TF-IDF + Logistic Regression | Simple pipeline |
| Images | Transfer learning (pretrained model) | `torchvision.models.resnet18(pretrained=True)` |

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Trivial baseline: always predict the most common class
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
print(f"Baseline (dummy): {dummy.score(X_test, y_test):.4f}")

# Reasonable baseline: logistic regression with scaling
baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
baseline.fit(X_train, y_train)
print(f"Baseline (LR): {baseline.score(X_test, y_test):.4f}")

# Now compare with the complex model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest: {rf.score(X_test, y_test):.4f}")

# If Random Forest doesn't significantly beat LR,
# maybe you don't need additional complexity.
```

### Recommended flow in consulting

```
1. Dummy baseline           -> "Always predicting 'no fraud' gives 99% accuracy"
                                (this demonstrates that accuracy is not the right metric)

2. Simple baseline          -> "Logistic Regression with F1=0.45"
                                (this is the real reference)

3. Medium model             -> "Random Forest with F1=0.72"
                                (clear improvement, worth the effort)

4. Complex model            -> "Tuned XGBoost with F1=0.75"
                                (marginal improvement, evaluate if the extra
                                 complexity is worth it)

5. Deep Learning            -> "Neural network with F1=0.76"
                                (very little improvement, probably not worth
                                 the operational complexity)
```

> **Key point for consulting:** Always present results as improvement over the baseline. "Our model improves fraud detection by 60% compared to the current system" is much more impactful than "Our model has F1=0.75".

---

> **Summary:** Machine Learning is a systematic process, not magic. Start simple (baseline), iterate fast, and make sure the metric you optimize is aligned with the business objective. Feature engineering and correct metric selection usually matter more than the algorithm you choose.

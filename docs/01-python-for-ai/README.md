# Python for AI/ML

## Table of Contents

- [NumPy Essentials](#numpy-essentials)
- [Pandas Essentials](#pandas-essentials)
- [Matplotlib and Seaborn](#matplotlib-and-seaborn)
- [Performance Tips](#performance-tips)

---

## NumPy Essentials

NumPy is the foundation of the entire AI/ML ecosystem in Python. Every library (Pandas, scikit-learn, PyTorch) uses NumPy internally or shares its interface.

### Arrays, shapes and dtypes

```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3])                    # 1D - vector
b = np.array([[1, 2], [3, 4]])             # 2D - matrix
c = np.zeros((3, 4, 5))                    # 3D - tensor

# Shape: the "form" of the array
print(a.shape)   # (3,)       - vector of 3 elements
print(b.shape)   # (2, 2)     - 2x2 matrix
print(c.shape)   # (3, 4, 5)  - 3D tensor

# Understanding shapes is CRITICAL in deep learning
# Example: batch of images
images = np.random.randn(32, 3, 224, 224)
# 32 images, 3 channels (RGB), 224x224 pixels
# PyTorch convention: (batch, channels, height, width)
```

#### Important dtypes for ML

| dtype | Bytes | Range | Usage |
|---|---|---|---|
| `float32` | 4 | ~7 decimal digits | **Default in ML/DL** |
| `float16` | 2 | ~3 decimal digits | Mixed precision training |
| `float64` | 8 | ~15 decimal digits | Scientific computing |
| `int64` | 8 | -2^63 to 2^63-1 | Labels, indices |
| `int8` | 1 | -128 to 127 | Model quantization |
| `bool` | 1 | True/False | Masks |

```python
# Control dtype (important for memory)
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 12 bytes
y = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # 24 bytes (double)

# Convert
x_16 = x.astype(np.float16)  # Reduces memory by half
```

### Broadcasting

Broadcasting is the rule NumPy uses to operate on arrays of different shapes. It is fundamental to understand because PyTorch uses the same rules.

**Rule:** NumPy compares shapes from right to left. Dimensions are compatible if they are equal or if one of them is 1.

```
Shape A:     (4, 3)
Shape B:        (3,)    ->  expands to (1, 3) -> then to (4, 3)
Result:      (4, 3)     OK

Shape A:     (4, 3)
Shape B:     (4, 1)     ->  expands to (4, 3)
Result:      (4, 3)     OK

Shape A:     (4, 3)
Shape B:     (4, 2)     ->  ERROR: 3 != 2 and neither is 1
```

#### Broadcasting visualization

```
Example: add a vector to each row of a matrix

Matrix (3, 4):              Vector (4,):
+-----------------+         +-----------------+
|  1   2   3   4  |    +    | 10  20  30  40  |
|  5   6   7   8  |         +-----------------+
|  9  10  11  12  |              |
+-----------------+              | broadcast (copy to each row)
                                 v
                            +-----------------+
         +                  | 10  20  30  40  |
                            | 10  20  30  40  |
                            | 10  20  30  40  |
                            +-----------------+

Result (3, 4):
+-----------------+
| 11  22  33  44  |
| 15  26  37  48  |
| 19  30  41  52  |
+-----------------+
```

```python
# Code
matrix = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
vector = np.array([10, 20, 30, 40])

result = matrix + vector  # Automatic broadcasting
```

#### Common use in ML: normalize features

```
Features (1000, 5):         Mean (5,):
+-----------------+         +-----------------+
| x1  x2  x3 ... |    -    | m1  m2  m3 ...  |  <- mean per column
| ...             |         +-----------------+
| ...             |
+-----------------+

Each column has its own mean subtracted.
```

```python
# Normalize features (mean 0, standard deviation 1)
X = np.random.randn(1000, 5)  # 1000 samples, 5 features
mean = X.mean(axis=0)          # Shape: (5,)
std = X.std(axis=0)            # Shape: (5,)

X_normalized = (X - mean) / std  # Broadcasting: (1000,5) - (5,) / (5,)
```

### Vectorized operations vs loops

```python
import time

n = 1_000_000

# BAD: Python loop (SLOW)
a = list(range(n))
b = list(range(n))

start = time.time()
c = [a[i] + b[i] for i in range(n)]
print(f"Python loop: {time.time() - start:.4f}s")

# GOOD: vectorized with NumPy (FAST)
a_np = np.arange(n)
b_np = np.arange(n)

start = time.time()
c_np = a_np + b_np
print(f"NumPy vectorized: {time.time() - start:.4f}s")

# Typical result:
# Python loop:        0.1500s
# NumPy vectorized:   0.0015s  (100x faster)
```

> **Key point:** Never use Python loops to operate on arrays. Always use NumPy vectorized operations. The difference can be 100x or more.

### Slicing, indexing and reshaping

```python
# Slicing (same as Python lists, but multidimensional)
matrix = np.arange(20).reshape(4, 5)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19]])

matrix[0, :]      # First row: [0, 1, 2, 3, 4]
matrix[:, 0]      # First column: [0, 5, 10, 15]
matrix[1:3, 2:4]  # Submatrix: [[7, 8], [12, 13]]
matrix[:, -1]     # Last column: [4, 9, 14, 19]

# Boolean indexing (VERY useful for filtering data)
values = np.array([1, -2, 3, -4, 5])
mask = values > 0
positives = values[mask]  # [1, 3, 5]

# Fancy indexing
indices = np.array([0, 3, 4])
values[indices]  # [1, -4, 5]
```

#### Reshaping

```python
# reshape: change the shape without changing the data
a = np.arange(12)          # Shape: (12,)
b = a.reshape(3, 4)        # Shape: (3, 4)
c = a.reshape(2, 2, 3)    # Shape: (2, 2, 3)
d = a.reshape(-1, 4)       # Shape: (3, 4) - NumPy calculates the -1

# Common shape operations in DL
x = np.random.randn(32, 784)     # Batch of 32 flattened images
x_img = x.reshape(32, 1, 28, 28)  # Restore to 28x28 images

# flatten: flatten to 1D
x_flat = x_img.reshape(32, -1)    # (32, 784)
# or equivalently
x_flat = x_img.flatten()           # (25088,) - everything flattened

# squeeze / unsqueeze (expand/reduce dimensions of size 1)
a = np.array([[1, 2, 3]])   # Shape: (1, 3)
a.squeeze()                  # Shape: (3,)

b = np.array([1, 2, 3])     # Shape: (3,)
b[np.newaxis, :]             # Shape: (1, 3)  - add batch dimension
b[:, np.newaxis]             # Shape: (3, 1)  - add feature dimension
```

### Linear algebra

You don't need to prove theorems, but you do need intuition about these operations because they appear constantly in ML and DL.

#### Dot product

Measures the "similarity" between two vectors. The basis of attention scores, embeddings, etc.

```
a = [1, 2, 3]
b = [4, 5, 6]

dot(a, b) = 1*4 + 2*5 + 3*6 = 32

Intuition: if two vectors point in the same direction,
their dot product is large and positive.
If they are perpendicular, it is 0.
If they point in opposite directions, it is large and negative.
```

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 32
```

#### Matrix multiplication

It is simply many dot products organized together. The basis of neural networks: `output = input @ weights`.

```
A (2, 3) @ B (3, 2) = C (2, 2)

A:              B:              C:
[1, 2, 3]      [7, 8]          [1*7+2*9+3*11,  1*8+2*10+3*12]     [58,  64]
[4, 5, 6]  @   [9, 10]    =    [4*7+5*9+6*11,  4*8+5*10+6*12]  =  [139, 154]
                [11, 12]

Rule: (m, n) @ (n, p) = (m, p)
      The inner dimensions (n) must match.
```

```python
A = np.array([[1,2,3], [4,5,6]])     # (2, 3)
B = np.array([[7,8], [9,10], [11,12]])  # (3, 2)

C = A @ B      # (2, 2)  - @ operator is matmul
# equivalent: C = np.matmul(A, B)
# equivalent: C = np.dot(A, B)     # Only for 2D
```

#### Eigenvalues

Intuition: the eigenvectors of a matrix are the "principal directions" along which the transformation only stretches or compresses (does not rotate). The eigenvalue tells you how much it stretches in that direction.

Use in ML: PCA (Principal Component Analysis) uses eigenvalues to find the directions of maximum variance.

```python
# Example: manual PCA
from numpy.linalg import eig

# 2D data
X = np.random.randn(100, 2)
X[:, 1] = X[:, 0] * 2 + np.random.randn(100) * 0.1  # Strong correlation

# Covariance matrix
cov_matrix = np.cov(X.T)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(cov_matrix)
print(f"Eigenvalues: {eigenvalues}")
# The largest eigenvalue indicates the direction of maximum variance
# That direction is the first principal component
```

### Random: seeds and distributions

```python
# SEED: for reproducibility
np.random.seed(42)  # Old way (global)

# Modern way (recommended): local generator
rng = np.random.default_rng(seed=42)

# Common distributions in ML
rng.uniform(0, 1, size=10)         # Uniform [0, 1]
rng.normal(0, 1, size=(3, 4))      # Normal (Gaussian) - mean 0, std 1
rng.integers(0, 10, size=5)         # Random integers [0, 10)
rng.choice([1,2,3,4,5], size=3)    # Choose from array
rng.permutation(10)                 # Random permutation of 0..9
rng.shuffle(my_array)               # Shuffle in-place

# Full reproducibility in a script
def set_seed(seed=42):
    np.random.seed(seed)
    # If using PyTorch:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
```

---

## Pandas Essentials

Pandas is the primary tool for manipulating tabular data. Every ML project starts with Pandas for EDA and feature engineering.

### DataFrame and Series

```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'name': ['Ana', 'Bob', 'Carlos', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000],
    'department': ['IT', 'HR', 'IT', 'Marketing']
})

# Basic operations
df.shape          # (4, 4) - rows, columns
df.dtypes         # Type of each column
df.info()         # Full summary
df.describe()     # Descriptive statistics
df.head(3)        # First 3 rows
df.columns        # Column names
df.nunique()      # Unique values per column
```

### Reading data

```python
# CSV (the most common)
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', sep=';', encoding='utf-8')

# Parquet (RECOMMENDED for large datasets - faster, smaller)
df = pd.read_parquet('data.parquet')
df.to_parquet('data.parquet', index=False)

# JSON
df = pd.read_json('data.json')
df = pd.read_json('data.json', lines=True)  # JSON Lines

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table', conn)

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```

> **Key point:** Use Parquet whenever you can. It is 5-10x faster than CSV and takes up less space. Additionally, it preserves data types.

### Key operations

#### Filtering

```python
# Filter rows
df[df['age'] > 30]
df[df['department'] == 'IT']
df[(df['age'] > 25) & (df['salary'] > 55000)]  # AND
df[df['department'].isin(['IT', 'HR'])]
```

#### GroupBy

```python
# Group and aggregate
df.groupby('department')['salary'].mean()
df.groupby('department').agg({
    'salary': ['mean', 'median', 'std'],
    'age': ['min', 'max']
})

# ML example: metrics by category
results = pd.DataFrame({
    'model': ['RF', 'RF', 'XGB', 'XGB'],
    'fold': [1, 2, 1, 2],
    'accuracy': [0.85, 0.87, 0.90, 0.88]
})
results.groupby('model')['accuracy'].agg(['mean', 'std'])
```

#### Merge (JOIN)

```python
# Equivalent to SQL JOIN
customers = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Ana', 'Bob', 'Carlos']
})
orders = pd.DataFrame({
    'customer_id': [1, 1, 2, 4],
    'product': ['A', 'B', 'C', 'D'],
    'amount': [100, 200, 150, 300]
})

# Inner join (only where there is a match)
pd.merge(customers, orders, left_on='id', right_on='customer_id', how='inner')

# Left join (all customers, even those without orders)
pd.merge(customers, orders, left_on='id', right_on='customer_id', how='left')
```

#### Pivot

```python
# Pivot table (like Excel pivot table)
sales = pd.DataFrame({
    'month': ['Jan', 'Jan', 'Feb', 'Feb'],
    'product': ['A', 'B', 'A', 'B'],
    'sales': [100, 200, 150, 250]
})

table = sales.pivot_table(
    values='sales',
    index='month',
    columns='product',
    aggfunc='sum'
)
```

#### Apply

```python
# Apply function to each row/column
df['annual_salary'] = df['salary'].apply(lambda x: x * 12)

# Apply per row (slower, avoid if you can vectorize)
df['category'] = df.apply(
    lambda row: 'senior' if row['age'] > 30 else 'junior',
    axis=1
)

# BETTER: vectorized
df['category'] = np.where(df['age'] > 30, 'senior', 'junior')
```

### Handling missing values

```python
# Detect
df.isnull().sum()              # Count of NaN per column
df.isnull().mean() * 100       # Percentage of NaN

# Imputation strategies for ML
# 1. Drop rows (only if few NaN)
df_clean = df.dropna()

# 2. Drop columns with many NaN (>50%)
threshold = 0.5
cols_to_drop = df.columns[df.isnull().mean() > threshold]
df = df.drop(columns=cols_to_drop)

# 3. Fill with mean/median (numeric)
df['age'] = df['age'].fillna(df['age'].median())

# 4. Fill with mode (categorical)
df['department'] = df['department'].fillna(df['department'].mode()[0])

# 5. Forward fill (time series)
df['value'] = df['value'].ffill()

# 6. Create missing indicator feature (sometimes NaN is informative)
df['age_missing'] = df['age'].isnull().astype(int)
df['age'] = df['age'].fillna(df['age'].median())
```

| Strategy | When to use | Caution |
|---|---|---|
| Drop rows | Few NaN (<5%), abundant data | Loses data, possible bias |
| Mean/median | Numeric, normal/skewed distribution | Does not capture variability |
| Mode | Categorical | May bias toward most common value |
| Forward fill | Time series | Do not use on non-temporal data |
| Indicator + impute | When missingness has meaning | Doubles features |
| Model (KNN, iterative) | Many NaN with patterns | More complex, risk of data leakage |

### Basic feature engineering with Pandas

```python
# Create new features
df['salary_age_ratio'] = df['salary'] / df['age']
df['log_salary'] = np.log1p(df['salary'])  # log(1+x) to avoid log(0)
df['salary_binned'] = pd.cut(df['salary'], bins=3, labels=['low', 'medium', 'high'])

# Temporal features
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['hour'] = df['date'].dt.hour

# Categorical encoding
df['department_encoded'] = df['department'].map({
    'IT': 0, 'HR': 1, 'Marketing': 2
})

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['department'], prefix='dept')

# Aggregation features (example: purchases per customer)
df['total_customer_purchases'] = df.groupby('customer_id')['amount'].transform('sum')
df['avg_customer_purchases'] = df.groupby('customer_id')['amount'].transform('mean')
df['num_customer_purchases'] = df.groupby('customer_id')['amount'].transform('count')
```

---

## Matplotlib and Seaborn

### Basic setup

```python
import matplotlib.pyplot as plt
import seaborn as sns

# General style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
```

### Essential plots for ML

#### Histogram: feature distribution

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Simple histogram
axes[0].hist(df['age'], bins=20, edgecolor='black', alpha=0.7)
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')

# With Seaborn (prettier)
sns.histplot(data=df, x='salary', kde=True, ax=axes[1])
axes[1].set_title('Salary Distribution')

# Compare distributions by class
sns.histplot(data=df, x='feature', hue='target', kde=True, ax=axes[2])
axes[2].set_title('Feature by Class')

plt.tight_layout()
plt.savefig('distributions.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### Scatter plot: relationship between features

```python
# Simple scatter
plt.scatter(df['feature_1'], df['feature_2'], c=df['target'], cmap='RdBu', alpha=0.6)
plt.colorbar(label='Class')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Relationship Between Features')
plt.show()

# With Seaborn
sns.scatterplot(data=df, x='feature_1', y='feature_2', hue='target', palette='Set2')
```

#### Correlation heatmap

```python
# Correlation between numeric features
correlation_matrix = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,        # Show values
    fmt='.2f',         # Decimal format
    cmap='coolwarm',   # Colormap
    center=0,          # Center at 0
    vmin=-1, vmax=1,
    square=True
)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
```

#### Box plot: detect outliers

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot of a single feature
sns.boxplot(data=df, y='salary', ax=axes[0])
axes[0].set_title('Salary Distribution')

# Box plot by category
sns.boxplot(data=df, x='department', y='salary', ax=axes[1])
axes[1].set_title('Salary by Department')

plt.tight_layout()
plt.show()
```

### Training curves

```python
# Simulate training data
history = {
    'train_loss': [0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08],
    'val_loss': [0.95, 0.75, 0.6, 0.5, 0.48, 0.50, 0.55, 0.60],
    'train_acc': [0.5, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.97],
    'val_acc': [0.48, 0.62, 0.72, 0.78, 0.80, 0.79, 0.78, 0.77],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss per Epoch')
axes[0].legend()
axes[0].grid(True)
# Mark where overfitting starts
axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Overfitting')

# Accuracy
axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy per Epoch')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

### Confusion matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Example data
y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# With absolute numbers
disp1 = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
disp1.plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix (Counts)')

# Normalized (percentages)
cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=['Negative', 'Positive'])
disp2.plot(ax=axes[1], cmap='Blues', values_format='.2%')
axes[1].set_title('Confusion Matrix (Normalized)')

plt.tight_layout()
plt.show()
```

### Subplots and customization

```python
# Complex plot grid
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Complete Dataset EDA', fontsize=16, y=1.02)

# Iterate over features
features = ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6']
for i, (ax, feat) in enumerate(zip(axes.flatten(), features)):
    sns.histplot(data=df, x=feat, hue='target', kde=True, ax=ax)
    ax.set_title(f'Distribution of {feat}')

plt.tight_layout()
plt.savefig('complete_eda.png', dpi=150, bbox_inches='tight')
plt.show()
```

```python
# Professional customization
fig, ax = plt.subplots(figsize=(10, 6))

# Plot
ax.plot(x, y, color='#2196F3', linewidth=2, label='Model A')
ax.fill_between(x, y-std, y+std, alpha=0.2, color='#2196F3')  # Confidence band

# Customize
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Performance Tips

### Speed hierarchy in Pandas

```
Vectorized NumPy/Pandas  >>  .apply()  >>  itertuples()  >>  iterrows()
      (fastest)                                              (slowest)
```

```python
# BEST: vectorized (microseconds)
df['double'] = df['value'] * 2

# OK: apply (milliseconds)
df['double'] = df['value'].apply(lambda x: x * 2)

# BAD: iterrows (seconds)
for idx, row in df.iterrows():
    df.at[idx, 'double'] = row['value'] * 2
```

#### Real benchmark (1 million rows)

```python
import pandas as pd
import numpy as np
import time

n = 1_000_000
df = pd.DataFrame({'a': np.random.randn(n), 'b': np.random.randn(n)})

# Vectorized: ~2ms
start = time.time()
df['c'] = df['a'] + df['b']
print(f"Vectorized: {time.time() - start:.4f}s")

# Apply: ~500ms
start = time.time()
df['c'] = df.apply(lambda row: row['a'] + row['b'], axis=1)
print(f"Apply: {time.time() - start:.4f}s")

# Iterrows: ~30s
start = time.time()
for idx, row in df.iterrows():
    df.at[idx, 'c'] = row['a'] + row['b']
print(f"Iterrows: {time.time() - start:.4f}s")

# Approximate result:
# Vectorized:  0.002s   (1x)
# Apply:       0.500s   (250x slower)
# Iterrows:   30.000s   (15000x slower)
```

### Efficient data types

```python
# Before optimizing
df.info(memory_usage='deep')

# Optimize numeric types
df['age'] = df['age'].astype('int8')         # -128 to 127
df['zip_code'] = df['zip_code'].astype('int32')
df['price'] = df['price'].astype('float32')   # Sufficient precision for ML

# Optimize categorical
df['country'] = df['country'].astype('category')  # If few unique values
df['gender'] = df['gender'].astype('category')

# Automatic dtype optimization function
def optimize_dtypes(df):
    """Reduce memory usage by optimizing dtypes."""
    mem_before = df.memory_usage(deep=True).sum() / 1e6

    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')

    mem_after = df.memory_usage(deep=True).sum() / 1e6
    print(f"Memory: {mem_before:.1f} MB -> {mem_after:.1f} MB "
          f"({(1 - mem_after/mem_before)*100:.0f}% reduction)")
    return df
```

### Polars vs Pandas

| Feature | Pandas | Polars |
|---|---|---|
| Speed | Baseline | 5-50x faster |
| Memory | High | Efficient |
| Multithreading | No (GIL) | Yes (native Rust) |
| API | Familiar, flexible | Expressive, functional |
| Lazy evaluation | No | Yes |
| Ecosystem | Huge | Growing |
| **When to use** | **EDA, prototyping, teams** | **Large datasets, pipelines** |

```python
import polars as pl

# Reading (faster than Pandas)
df_pl = pl.read_csv('data.csv')
df_pl = pl.read_parquet('data.parquet')

# Similar API but with lazy evaluation
result = (
    df_pl.lazy()
    .filter(pl.col('age') > 30)
    .group_by('department')
    .agg([
        pl.col('salary').mean().alias('avg_salary'),
        pl.col('salary').std().alias('salary_std'),
        pl.count().alias('n'),
    ])
    .sort('avg_salary', descending=True)
    .collect()  # Executes everything at once, optimized
)

# Convert between Pandas and Polars
df_pandas = df_pl.to_pandas()
df_polars = pl.from_pandas(df_pandas)
```

> **Key point:** Use Pandas for interactive EDA and datasets that fit in memory. Consider Polars when Pandas becomes slow (millions of rows) or in production pipelines. You don't need to rewrite everything; convert between both as needed.

---

> **Summary:** Mastering NumPy, Pandas, Matplotlib and Seaborn is the foundation for any AI/ML work. Prioritize vectorized operations over loops, use Parquet over CSV, and know efficient dtypes for working with large datasets.

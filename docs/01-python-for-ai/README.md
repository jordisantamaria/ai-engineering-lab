# Python para AI/ML

## Tabla de Contenidos

- [NumPy Esencial](#numpy-esencial)
- [Pandas Esencial](#pandas-esencial)
- [Matplotlib y Seaborn](#matplotlib-y-seaborn)
- [Tips de Rendimiento](#tips-de-rendimiento)

---

## NumPy Esencial

NumPy es la base de todo el ecosistema de AI/ML en Python. Cada libreria (Pandas, scikit-learn, PyTorch) usa NumPy internamente o comparte su interfaz.

### Arrays, shapes y dtypes

```python
import numpy as np

# Crear arrays
a = np.array([1, 2, 3])                    # 1D - vector
b = np.array([[1, 2], [3, 4]])             # 2D - matrix
c = np.zeros((3, 4, 5))                    # 3D - tensor

# Shape: la "forma" del array
print(a.shape)   # (3,)       - vector de 3 elementos
print(b.shape)   # (2, 2)     - matrix 2x2
print(c.shape)   # (3, 4, 5)  - tensor 3D

# Entender shapes es CRITICO en deep learning
# Ejemplo: batch de imagenes
imagenes = np.random.randn(32, 3, 224, 224)
# 32 imagenes, 3 canales (RGB), 224x224 pixeles
# Convencion PyTorch: (batch, channels, height, width)
```

#### dtypes importantes para ML

| dtype | Bytes | Rango | Uso |
|---|---|---|---|
| `float32` | 4 | ~7 digitos decimales | **Default en ML/DL** |
| `float16` | 2 | ~3 digitos decimales | Mixed precision training |
| `float64` | 8 | ~15 digitos decimales | Calculo cientifico |
| `int64` | 8 | -2^63 a 2^63-1 | Labels, indices |
| `int8` | 1 | -128 a 127 | Cuantizacion de modelos |
| `bool` | 1 | True/False | Mascaras |

```python
# Controlar dtype (importante para memoria)
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 12 bytes
y = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # 24 bytes (doble)

# Convertir
x_16 = x.astype(np.float16)  # Reduce memoria a la mitad
```

### Broadcasting

Broadcasting es la regla que NumPy usa para operar arrays de diferentes shapes. Es fundamental entenderlo porque PyTorch usa las mismas reglas.

**Regla:** NumPy compara shapes de derecha a izquierda. Las dimensiones son compatibles si son iguales o si una de ellas es 1.

```
Forma A:     (4, 3)
Forma B:        (3,)    ->  se expande a (1, 3) -> luego a (4, 3)
Resultado:   (4, 3)     OK

Forma A:     (4, 3)
Forma B:     (4, 1)     ->  se expande a (4, 3)
Resultado:   (4, 3)     OK

Forma A:     (4, 3)
Forma B:     (4, 2)     ->  ERROR: 3 != 2 y ninguno es 1
```

#### Visualizacion de broadcasting

```
Ejemplo: sumar un vector a cada fila de una matriz

Matrix (3, 4):              Vector (4,):
┌─────────────────┐         ┌─────────────────┐
│  1   2   3   4  │    +    │ 10  20  30  40  │
│  5   6   7   8  │         └─────────────────┘
│  9  10  11  12  │              │
└─────────────────┘              │ broadcast (copiar a cada fila)
                                 ▼
                            ┌─────────────────┐
         +                  │ 10  20  30  40  │
                            │ 10  20  30  40  │
                            │ 10  20  30  40  │
                            └─────────────────┘

Resultado (3, 4):
┌─────────────────┐
│ 11  22  33  44  │
│ 15  26  37  48  │
│ 19  30  41  52  │
└─────────────────┘
```

```python
# Codigo
matrix = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
vector = np.array([10, 20, 30, 40])

resultado = matrix + vector  # Broadcasting automatico
```

#### Uso comun en ML: normalizar features

```
Features (1000, 5):         Media (5,):
┌─────────────────┐         ┌─────────────────┐
│ x1  x2  x3 ... │    -    │ m1  m2  m3 ...  │  <- media por columna
│ ...             │         └─────────────────┘
│ ...             │
└─────────────────┘

Cada columna se le resta su propia media.
```

```python
# Normalizar features (media 0, desviacion 1)
X = np.random.randn(1000, 5)  # 1000 muestras, 5 features
mean = X.mean(axis=0)          # Shape: (5,)
std = X.std(axis=0)            # Shape: (5,)

X_normalized = (X - mean) / std  # Broadcasting: (1000,5) - (5,) / (5,)
```

### Operaciones vectorizadas vs loops

```python
import time

n = 1_000_000

# MAL: loop de Python (LENTO)
a = list(range(n))
b = list(range(n))

start = time.time()
c = [a[i] + b[i] for i in range(n)]
print(f"Loop Python: {time.time() - start:.4f}s")

# BIEN: vectorizado con NumPy (RAPIDO)
a_np = np.arange(n)
b_np = np.arange(n)

start = time.time()
c_np = a_np + b_np
print(f"NumPy vectorizado: {time.time() - start:.4f}s")

# Resultado tipico:
# Loop Python:       0.1500s
# NumPy vectorizado:  0.0015s  (100x mas rapido)
```

> **Punto clave:** Nunca uses loops de Python para operar sobre arrays. Siempre usa operaciones vectorizadas de NumPy. La diferencia puede ser de 100x o mas.

### Slicing, indexing y reshaping

```python
# Slicing (igual que listas de Python, pero multidimensional)
matrix = np.arange(20).reshape(4, 5)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19]])

matrix[0, :]      # Primera fila: [0, 1, 2, 3, 4]
matrix[:, 0]      # Primera columna: [0, 5, 10, 15]
matrix[1:3, 2:4]  # Submatriz: [[7, 8], [12, 13]]
matrix[:, -1]     # Ultima columna: [4, 9, 14, 19]

# Boolean indexing (MUY util para filtrar datos)
valores = np.array([1, -2, 3, -4, 5])
mask = valores > 0
positivos = valores[mask]  # [1, 3, 5]

# Fancy indexing
indices = np.array([0, 3, 4])
valores[indices]  # [1, -4, 5]
```

#### Reshaping

```python
# reshape: cambiar la forma sin cambiar los datos
a = np.arange(12)          # Shape: (12,)
b = a.reshape(3, 4)        # Shape: (3, 4)
c = a.reshape(2, 2, 3)    # Shape: (2, 2, 3)
d = a.reshape(-1, 4)       # Shape: (3, 4) - NumPy calcula el -1

# Operaciones de forma comunes en DL
x = np.random.randn(32, 784)     # Batch de 32 imagenes aplanadas
x_img = x.reshape(32, 1, 28, 28)  # Restaurar a imagenes 28x28

# flatten: aplanar a 1D
x_flat = x_img.reshape(32, -1)    # (32, 784)
# o equivalentemente
x_flat = x_img.flatten()           # (25088,) - todo aplanado

# squeeze / unsqueeze (expandir/reducir dimensiones de tamano 1)
a = np.array([[1, 2, 3]])   # Shape: (1, 3)
a.squeeze()                  # Shape: (3,)

b = np.array([1, 2, 3])     # Shape: (3,)
b[np.newaxis, :]             # Shape: (1, 3)  - agregar dimension de batch
b[:, np.newaxis]             # Shape: (3, 1)  - agregar dimension de feature
```

### Algebra lineal

No necesitas demostrar teoremas, pero si necesitas intuicion de estas operaciones porque aparecen constantemente en ML y DL.

#### Dot product (producto punto)

Mide la "similitud" entre dos vectores. Base de las attention scores, embeddings, etc.

```
a = [1, 2, 3]
b = [4, 5, 6]

dot(a, b) = 1*4 + 2*5 + 3*6 = 32

Intuicion: si dos vectores apuntan en la misma direccion,
su dot product es grande y positivo.
Si son perpendiculares, es 0.
Si apuntan en direcciones opuestas, es grande y negativo.
```

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))  # 32
```

#### Multiplicacion de matrices

Es simplemente muchos dot products organizados. La base de las redes neuronales: `output = input @ weights`.

```
A (2, 3) @ B (3, 2) = C (2, 2)

A:              B:              C:
[1, 2, 3]      [7, 8]          [1*7+2*9+3*11,  1*8+2*10+3*12]     [58,  64]
[4, 5, 6]  @   [9, 10]    =    [4*7+5*9+6*11,  4*8+5*10+6*12]  =  [139, 154]
                [11, 12]

Regla: (m, n) @ (n, p) = (m, p)
       Las dimensiones internas (n) deben coincidir.
```

```python
A = np.array([[1,2,3], [4,5,6]])     # (2, 3)
B = np.array([[7,8], [9,10], [11,12]])  # (3, 2)

C = A @ B      # (2, 2)  - operador @ es matmul
# equivalente: C = np.matmul(A, B)
# equivalente: C = np.dot(A, B)     # Solo para 2D
```

#### Eigenvalues (autovalores)

Intuicion: los eigenvectors de una matriz son las "direcciones principales" a lo largo de las cuales la transformacion solo estira o comprime (no rota). El eigenvalue dice cuanto estira en esa direccion.

Uso en ML: PCA (Principal Component Analysis) usa eigenvalues para encontrar las direcciones de maxima varianza.

```python
# Ejemplo: PCA manual
from numpy.linalg import eig

# Datos 2D
X = np.random.randn(100, 2)
X[:, 1] = X[:, 0] * 2 + np.random.randn(100) * 0.1  # Correlacion fuerte

# Matriz de covarianza
cov_matrix = np.cov(X.T)

# Eigenvalues y eigenvectors
eigenvalues, eigenvectors = eig(cov_matrix)
print(f"Eigenvalues: {eigenvalues}")
# El eigenvalue mas grande indica la direccion de mayor varianza
# Esa direccion es el primer componente principal
```

### Random: seeds y distribuciones

```python
# SEED: para reproducibilidad
np.random.seed(42)  # Forma antigua (global)

# Forma moderna (recomendada): generador local
rng = np.random.default_rng(seed=42)

# Distribuciones comunes en ML
rng.uniform(0, 1, size=10)         # Uniforme [0, 1]
rng.normal(0, 1, size=(3, 4))      # Normal (Gaussiana) - media 0, std 1
rng.integers(0, 10, size=5)         # Enteros aleatorios [0, 10)
rng.choice([1,2,3,4,5], size=3)    # Elegir del array
rng.permutation(10)                 # Permutacion aleatoria de 0..9
rng.shuffle(my_array)               # Shuffle in-place

# Reproducibilidad completa en un script
def set_seed(seed=42):
    np.random.seed(seed)
    # Si usas PyTorch:
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
```

---

## Pandas Esencial

Pandas es la herramienta principal para manipular datos tabulares. Todo proyecto de ML empieza con Pandas para EDA y feature engineering.

### DataFrame y Series

```python
import pandas as pd

# Crear DataFrame
df = pd.DataFrame({
    'nombre': ['Ana', 'Bob', 'Carlos', 'Diana'],
    'edad': [25, 30, 35, 28],
    'salario': [50000, 60000, 70000, 55000],
    'departamento': ['IT', 'HR', 'IT', 'Marketing']
})

# Operaciones basicas
df.shape          # (4, 4) - filas, columnas
df.dtypes         # Tipo de cada columna
df.info()         # Resumen completo
df.describe()     # Estadisticas descriptivas
df.head(3)        # Primeras 3 filas
df.columns        # Nombres de columnas
df.nunique()      # Valores unicos por columna
```

### Lectura de datos

```python
# CSV (el mas comun)
df = pd.read_csv('datos.csv')
df = pd.read_csv('datos.csv', sep=';', encoding='utf-8')

# Parquet (RECOMENDADO para datasets grandes - mas rapido, mas pequeno)
df = pd.read_parquet('datos.parquet')
df.to_parquet('datos.parquet', index=False)

# JSON
df = pd.read_json('datos.json')
df = pd.read_json('datos.json', lines=True)  # JSON Lines

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM tabla', conn)

# Excel
df = pd.read_excel('datos.xlsx', sheet_name='Hoja1')
```

> **Punto clave:** Usa Parquet siempre que puedas. Es 5-10x mas rapido que CSV y ocupa menos espacio. Ademas, conserva los tipos de datos.

### Operaciones clave

#### Filtrado

```python
# Filtrar filas
df[df['edad'] > 30]
df[df['departamento'] == 'IT']
df[(df['edad'] > 25) & (df['salario'] > 55000)]  # AND
df[df['departamento'].isin(['IT', 'HR'])]
```

#### GroupBy

```python
# Agrupar y agregar
df.groupby('departamento')['salario'].mean()
df.groupby('departamento').agg({
    'salario': ['mean', 'median', 'std'],
    'edad': ['min', 'max']
})

# Ejemplo ML: metricas por categoria
resultados = pd.DataFrame({
    'modelo': ['RF', 'RF', 'XGB', 'XGB'],
    'fold': [1, 2, 1, 2],
    'accuracy': [0.85, 0.87, 0.90, 0.88]
})
resultados.groupby('modelo')['accuracy'].agg(['mean', 'std'])
```

#### Merge (JOIN)

```python
# Equivalente a SQL JOIN
clientes = pd.DataFrame({
    'id': [1, 2, 3],
    'nombre': ['Ana', 'Bob', 'Carlos']
})
pedidos = pd.DataFrame({
    'cliente_id': [1, 1, 2, 4],
    'producto': ['A', 'B', 'C', 'D'],
    'monto': [100, 200, 150, 300]
})

# Inner join (solo donde hay match)
pd.merge(clientes, pedidos, left_on='id', right_on='cliente_id', how='inner')

# Left join (todos los clientes, aunque no tengan pedidos)
pd.merge(clientes, pedidos, left_on='id', right_on='cliente_id', how='left')
```

#### Pivot

```python
# Pivot table (como tabla dinamica de Excel)
ventas = pd.DataFrame({
    'mes': ['Ene', 'Ene', 'Feb', 'Feb'],
    'producto': ['A', 'B', 'A', 'B'],
    'ventas': [100, 200, 150, 250]
})

tabla = ventas.pivot_table(
    values='ventas',
    index='mes',
    columns='producto',
    aggfunc='sum'
)
```

#### Apply

```python
# Aplicar funcion a cada fila/columna
df['salario_anual'] = df['salario'].apply(lambda x: x * 12)

# Apply por fila (mas lento, evitar si se puede vectorizar)
df['categoria'] = df.apply(
    lambda row: 'senior' if row['edad'] > 30 else 'junior',
    axis=1
)

# MEJOR: vectorizado
df['categoria'] = np.where(df['edad'] > 30, 'senior', 'junior')
```

### Manejo de missing values

```python
# Detectar
df.isnull().sum()              # Conteo de NaN por columna
df.isnull().mean() * 100       # Porcentaje de NaN

# Estrategias de imputacion para ML
# 1. Eliminar filas (solo si pocos NaN)
df_clean = df.dropna()

# 2. Eliminar columnas con muchos NaN (>50%)
threshold = 0.5
cols_to_drop = df.columns[df.isnull().mean() > threshold]
df = df.drop(columns=cols_to_drop)

# 3. Rellenar con media/mediana (numericas)
df['edad'] = df['edad'].fillna(df['edad'].median())

# 4. Rellenar con moda (categoricas)
df['departamento'] = df['departamento'].fillna(df['departamento'].mode()[0])

# 5. Forward fill (series temporales)
df['valor'] = df['valor'].ffill()

# 6. Crear feature indicadora de missing (a veces el NaN es informativo)
df['edad_missing'] = df['edad'].isnull().astype(int)
df['edad'] = df['edad'].fillna(df['edad'].median())
```

| Estrategia | Cuando usarla | Cuidado |
|---|---|---|
| Eliminar filas | Pocos NaN (<5%), datos abundantes | Pierde datos, posible sesgo |
| Media/mediana | Numericas, distribucion normal/sesgada | No captura variabilidad |
| Moda | Categoricas | Puede sesgar hacia valor mas comun |
| Forward fill | Series temporales | No usar en datos no temporales |
| Indicadora + imputar | Cuando el missing tiene significado | Duplica features |
| Modelo (KNN, iterativo) | Muchos NaN con patrones | Mas complejo, riesgo de data leakage |

### Feature engineering basico con Pandas

```python
# Crear nuevas features
df['ratio_salario_edad'] = df['salario'] / df['edad']
df['log_salario'] = np.log1p(df['salario'])  # log(1+x) para evitar log(0)
df['salario_binned'] = pd.cut(df['salario'], bins=3, labels=['bajo', 'medio', 'alto'])

# Features temporales
df['fecha'] = pd.to_datetime(df['fecha'])
df['dia_semana'] = df['fecha'].dt.dayofweek
df['mes'] = df['fecha'].dt.month
df['es_fin_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
df['hora'] = df['fecha'].dt.hour

# Encoding de categoricas
df['departamento_encoded'] = df['departamento'].map({
    'IT': 0, 'HR': 1, 'Marketing': 2
})

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['departamento'], prefix='dept')

# Aggregation features (ejemplo: compras por cliente)
df['total_compras_cliente'] = df.groupby('cliente_id')['monto'].transform('sum')
df['media_compras_cliente'] = df.groupby('cliente_id')['monto'].transform('mean')
df['num_compras_cliente'] = df.groupby('cliente_id')['monto'].transform('count')
```

---

## Matplotlib y Seaborn

### Setup basico

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo general
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
```

### Plots esenciales para ML

#### Histograma: distribucion de features

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Histograma simple
axes[0].hist(df['edad'], bins=20, edgecolor='black', alpha=0.7)
axes[0].set_title('Distribucion de Edad')
axes[0].set_xlabel('Edad')
axes[0].set_ylabel('Frecuencia')

# Con Seaborn (mas bonito)
sns.histplot(data=df, x='salario', kde=True, ax=axes[1])
axes[1].set_title('Distribucion de Salario')

# Comparar distribuciones por clase
sns.histplot(data=df, x='feature', hue='target', kde=True, ax=axes[2])
axes[2].set_title('Feature por Clase')

plt.tight_layout()
plt.savefig('distribuciones.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### Scatter plot: relacion entre features

```python
# Scatter simple
plt.scatter(df['feature_1'], df['feature_2'], c=df['target'], cmap='RdBu', alpha=0.6)
plt.colorbar(label='Clase')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Relacion entre Features')
plt.show()

# Con Seaborn
sns.scatterplot(data=df, x='feature_1', y='feature_2', hue='target', palette='Set2')
```

#### Heatmap de correlacion

```python
# Correlacion entre features numericas
correlation_matrix = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,        # Mostrar valores
    fmt='.2f',         # Formato decimal
    cmap='coolwarm',   # Colormap
    center=0,          # Centrar en 0
    vmin=-1, vmax=1,
    square=True
)
plt.title('Matriz de Correlacion')
plt.tight_layout()
plt.show()
```

#### Box plot: detectar outliers

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot de una feature
sns.boxplot(data=df, y='salario', ax=axes[0])
axes[0].set_title('Distribucion de Salario')

# Box plot por categoria
sns.boxplot(data=df, x='departamento', y='salario', ax=axes[1])
axes[1].set_title('Salario por Departamento')

plt.tight_layout()
plt.show()
```

### Curvas de training

```python
# Simular datos de entrenamiento
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
axes[0].set_title('Loss por Epoch')
axes[0].legend()
axes[0].grid(True)
# Marcar donde empieza overfitting
axes[0].axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Overfitting')

# Accuracy
axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy por Epoch')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

### Confusion matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Datos de ejemplo
y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1, 1, 1, 1, 0]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualizacion
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Con numeros absolutos
disp1 = ConfusionMatrixDisplay(cm, display_labels=['Negativo', 'Positivo'])
disp1.plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix (Counts)')

# Normalizada (porcentajes)
cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
disp2 = ConfusionMatrixDisplay(cm_norm, display_labels=['Negativo', 'Positivo'])
disp2.plot(ax=axes[1], cmap='Blues', values_format='.2%')
axes[1].set_title('Confusion Matrix (Normalizada)')

plt.tight_layout()
plt.show()
```

### Subplots y customizacion

```python
# Grid de plots complejo
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('EDA Completo del Dataset', fontsize=16, y=1.02)

# Iterar sobre features
features = ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6']
for i, (ax, feat) in enumerate(zip(axes.flatten(), features)):
    sns.histplot(data=df, x=feat, hue='target', kde=True, ax=ax)
    ax.set_title(f'Distribucion de {feat}')

plt.tight_layout()
plt.savefig('eda_completo.png', dpi=150, bbox_inches='tight')
plt.show()
```

```python
# Customizacion profesional
fig, ax = plt.subplots(figsize=(10, 6))

# Plot
ax.plot(x, y, color='#2196F3', linewidth=2, label='Modelo A')
ax.fill_between(x, y-std, y+std, alpha=0.2, color='#2196F3')  # Banda de confianza

# Customizar
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Comparacion de Modelos', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Tips de Rendimiento

### Jerarquia de velocidad en Pandas

```
Vectorizado NumPy/Pandas  >>  .apply()  >>  itertuples()  >>  iterrows()
      (mas rapido)                                            (mas lento)
```

```python
# MEJOR: vectorizado (microsegundos)
df['doble'] = df['valor'] * 2

# OK: apply (milisegundos)
df['doble'] = df['valor'].apply(lambda x: x * 2)

# MAL: iterrows (segundos)
for idx, row in df.iterrows():
    df.at[idx, 'doble'] = row['valor'] * 2
```

#### Benchmark real (1 millon de filas)

```python
import pandas as pd
import numpy as np
import time

n = 1_000_000
df = pd.DataFrame({'a': np.random.randn(n), 'b': np.random.randn(n)})

# Vectorizado: ~2ms
start = time.time()
df['c'] = df['a'] + df['b']
print(f"Vectorizado: {time.time() - start:.4f}s")

# Apply: ~500ms
start = time.time()
df['c'] = df.apply(lambda row: row['a'] + row['b'], axis=1)
print(f"Apply: {time.time() - start:.4f}s")

# Iterrows: ~30s
start = time.time()
for idx, row in df.iterrows():
    df.at[idx, 'c'] = row['a'] + row['b']
print(f"Iterrows: {time.time() - start:.4f}s")

# Resultado aproximado:
# Vectorizado:  0.002s   (1x)
# Apply:        0.500s   (250x mas lento)
# Iterrows:    30.000s   (15000x mas lento)
```

### Tipos de datos eficientes

```python
# Antes de optimizar
df.info(memory_usage='deep')

# Optimizar tipos numericos
df['edad'] = df['edad'].astype('int8')         # -128 a 127
df['codigo_postal'] = df['codigo_postal'].astype('int32')
df['precio'] = df['precio'].astype('float32')   # Suficiente precision para ML

# Optimizar categoricas
df['pais'] = df['pais'].astype('category')  # Si pocos valores unicos
df['genero'] = df['genero'].astype('category')

# Funcion de optimizacion automatica
def optimize_dtypes(df):
    """Reduce el uso de memoria optimizando dtypes."""
    mem_before = df.memory_usage(deep=True).sum() / 1e6

    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Menos de 50% valores unicos
            df[col] = df[col].astype('category')

    mem_after = df.memory_usage(deep=True).sum() / 1e6
    print(f"Memoria: {mem_before:.1f} MB -> {mem_after:.1f} MB "
          f"({(1 - mem_after/mem_before)*100:.0f}% reduccion)")
    return df
```

### Polars vs Pandas

| Caracteristica | Pandas | Polars |
|---|---|---|
| Velocidad | Base | 5-50x mas rapido |
| Memoria | Alta | Eficiente |
| Multithreading | No (GIL) | Si (Rust nativo) |
| API | Familiar, flexible | Expresiva, funcional |
| Lazy evaluation | No | Si |
| Ecosistema | Enorme | Creciendo |
| **Cuando usar** | **EDA, prototipado, equipos** | **Datasets grandes, pipelines** |

```python
import polars as pl

# Lectura (mas rapida que Pandas)
df_pl = pl.read_csv('datos.csv')
df_pl = pl.read_parquet('datos.parquet')

# API similar pero con lazy evaluation
result = (
    df_pl.lazy()
    .filter(pl.col('edad') > 30)
    .group_by('departamento')
    .agg([
        pl.col('salario').mean().alias('salario_medio'),
        pl.col('salario').std().alias('salario_std'),
        pl.count().alias('n'),
    ])
    .sort('salario_medio', descending=True)
    .collect()  # Ejecuta todo de una vez, optimizado
)

# Convertir entre Pandas y Polars
df_pandas = df_pl.to_pandas()
df_polars = pl.from_pandas(df_pandas)
```

> **Punto clave:** Usa Pandas para EDA interactivo y datasets que caben en memoria. Considera Polars cuando Pandas se vuelve lento (millones de filas) o en pipelines de produccion. No necesitas reescribir todo; convierte entre ambos segun necesites.

---

> **Resumen:** Dominar NumPy, Pandas, Matplotlib y Seaborn es la base para cualquier trabajo de AI/ML. Prioriza operaciones vectorizadas sobre loops, usa Parquet sobre CSV, y conoce los dtypes eficientes para trabajar con datasets grandes.

# Fundamentos de Machine Learning

## Tabla de Contenidos

- [Que es Machine Learning](#que-es-machine-learning)
- [Tipos de Aprendizaje](#tipos-de-aprendizaje)
- [El Pipeline de ML Completo](#el-pipeline-de-ml-completo)
- [Feature Engineering](#feature-engineering)
- [Metricas de Evaluacion](#metricas-de-evaluacion)
- [Underfitting vs Overfitting](#underfitting-vs-overfitting)
- [Cross-Validation](#cross-validation)
- [Algoritmos Clave de scikit-learn](#algoritmos-clave-de-scikit-learn)
- [Baseline Models](#baseline-models)

---

## Que es Machine Learning

### Programacion tradicional vs Machine Learning

```
Programacion tradicional:
  Datos + Reglas  -->  [ Programa ]  -->  Resultado
  "Si temperatura > 30, entonces encender AC"

Machine Learning:
  Datos + Resultados  -->  [ Algoritmo de aprendizaje ]  -->  Modelo (reglas aprendidas)
  "Aqui tienes 10,000 casas con sus precios; aprende a predecir el precio"
```

La diferencia fundamental: en programacion tradicional tu escribes las reglas. En ML, el algoritmo descubre las reglas a partir de los datos.

### ML vs Deep Learning vs AI

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Inteligencia Artificial (AI)                      │
│   Cualquier sistema que simula inteligencia         │
│                                                     │
│   ┌─────────────────────────────────────────────┐   │
│   │                                             │   │
│   │   Machine Learning                          │   │
│   │   Aprende de datos, sin programar reglas    │   │
│   │                                             │   │
│   │   ┌─────────────────────────────────────┐   │   │
│   │   │                                     │   │   │
│   │   │   Deep Learning                     │   │   │
│   │   │   ML con redes neuronales           │   │   │
│   │   │   profundas (muchas capas)          │   │   │
│   │   │                                     │   │   │
│   │   │   ┌─────────────────────────────┐   │   │   │
│   │   │   │ LLMs (GPT, Claude, etc.)    │   │   │   │
│   │   │   │ DL con Transformers         │   │   │   │
│   │   │   └─────────────────────────────┘   │   │   │
│   │   └─────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

| | ML Clasico | Deep Learning |
|---|---|---|
| Datos | Cientos a miles | Miles a millones |
| Features | Las diseñas tu (feature engineering) | Las aprende el modelo |
| Interpretabilidad | Alta (puedes explicar decisiones) | Baja (caja negra) |
| Compute | CPU suficiente | GPU necesaria |
| Tiempo de desarrollo | Rapido | Lento |
| **Cuando usarlo** | **Datos tabulares, pocos datos, necesitas explicar** | **Imagenes, texto, audio, muchos datos** |

---

## Tipos de Aprendizaje

### Aprendizaje Supervisado

El modelo aprende de ejemplos etiquetados: para cada input (X), le dices el output correcto (y).

#### Clasificacion: predecir una categoria

```
Input (features)           -->  Output (etiqueta)
Email [palabras, remitente] -->  Spam / No spam
Imagen [pixeles]            -->  Gato / Perro
Transaccion [monto, hora]  -->  Fraude / Normal
```

#### Regresion: predecir un numero

```
Input (features)                      -->  Output (numero)
Casa [metros2, habitaciones, zona]    -->  Precio: 350,000
Cliente [edad, historial, ingresos]   -->  Probabilidad de churn: 0.73
Producto [categoria, temporada]       -->  Demanda: 1,240 unidades
```

### Aprendizaje No Supervisado

No hay etiquetas. El modelo descubre patrones y estructura en los datos por si solo.

#### Clustering: agrupar datos similares

```
Datos de clientes -->  [ K-Means ]  -->  Grupo A (jovenes, alto gasto)
                                         Grupo B (mayores, ahorradores)
                                         Grupo C (familias, gasto medio)
```

#### Reduccion de dimensionalidad: comprimir datos

```
Datos con 100 features  -->  [ PCA ]  -->  Datos con 10 features
                                           (manteniendo 95% de la informacion)
```

### Otros tipos (mencion)

| Tipo | Que es | Ejemplo |
|---|---|---|
| **Semi-supervisado** | Pocos datos etiquetados + muchos sin etiquetar | Clasificar 1M de imagenes con solo 1K etiquetadas |
| **Self-supervised** | El modelo crea sus propias etiquetas del input | GPT: predecir la siguiente palabra del texto |
| **Reinforcement Learning** | Aprender por prueba y error con recompensas | AlphaGo, robots, RLHF en LLMs |

---

## El Pipeline de ML Completo

```
1. Problema  -->  2. Datos  -->  3. Features  -->  4. Split  -->  5. Modelo
   de negocio       EDA        engineering      train/val/test   seleccion

                                                                    │
                                                                    ▼
8. Deploy  <--  7. Evaluacion  <--  6. Entrenamiento
   produccion     metricas          fit(X_train, y_train)
```

### Paso 1: Definir el problema de negocio

Antes de tocar codigo, responde estas preguntas:

- Que problema de negocio estoy resolviendo?
- Que tipo de problema es? (clasificacion, regresion, clustering)
- Cual es la metrica de exito del negocio? (no solo metrica ML)
- Que datos tengo? Son suficientes?
- Existe un baseline humano o regla simple?
- Cual es el coste de equivocarse? (falso positivo vs falso negativo)

> **Punto clave para consultoria:** El 80% del valor esta en definir bien el problema y elegir la metrica correcta. Un modelo perfecto optimizando la metrica equivocada no sirve de nada.

### Paso 2: EDA (Exploratory Data Analysis)

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('datos.csv')

# Resumen general
print(f"Shape: {df.shape}")
print(f"\nTipos de datos:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nEstadisticas:\n{df.describe()}")

# Distribucion del target
print(f"\nDistribucion del target:\n{df['target'].value_counts(normalize=True)}")

# Correlaciones
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlacion')
plt.show()

# Distribuciones de features
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
for ax, col in zip(axes.flatten(), df.select_dtypes(include=[np.number]).columns):
    sns.histplot(data=df, x=col, hue='target', kde=True, ax=ax)
plt.tight_layout()
plt.show()
```

### Paso 3: Feature Engineering

(Ver seccion detallada mas abajo)

### Paso 4: Split train/val/test

```python
from sklearn.model_selection import train_test_split

# Split basico (80/10/10)
X = df.drop('target', axis=1)
y = df['target']

# Primero: separar test (no tocar hasta el final)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Segundo: separar train y validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.11, random_state=42, stratify=y_temp
)
# 0.11 de 0.9 es ~0.1 del total

print(f"Train: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Val:   {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test:  {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.0f}%)")
```

#### Tipos de split segun el problema

| Tipo de split | Cuando usarlo | Ejemplo |
|---|---|---|
| **Random split** | Datos i.i.d., clasificacion general | Clasificar emails como spam |
| **Stratified split** | Clases desbalanceadas | Deteccion de fraude (1% fraude) |
| **Temporal split** | Series temporales, datos con fecha | Predecir ventas futuras |
| **Group split** | Datos agrupados (mismos usuarios, pacientes) | Prediccion por paciente |

```python
# Split estratificado (mantiene proporciones de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split temporal (datos con fecha)
df = df.sort_values('fecha')
split_date = '2024-01-01'
train = df[df['fecha'] < split_date]
test = df[df['fecha'] >= split_date]
# NUNCA usar datos futuros para entrenar

# Group split (un usuario solo en train o test, no ambos)
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['user_id']))
```

> **Punto clave:** El test set es sagrado. Solo se usa UNA VEZ al final para reportar resultados. Si lo usas para tomar decisiones durante el desarrollo, contaminaste tu evaluacion.

### Paso 5-6: Seleccion de modelo y entrenamiento

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Probar varios modelos rapido
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"{nombre}: Accuracy = {acc:.4f}")
```

### Paso 7: Evaluacion

(Ver seccion detallada de metricas mas abajo)

### Paso 8: Deployment

```python
import joblib

# Guardar modelo
joblib.dump(mejor_modelo, 'models/modelo_v1.joblib')

# Cargar modelo
modelo = joblib.load('models/modelo_v1.joblib')

# Predecir con datos nuevos
prediccion = modelo.predict(X_nuevo)
```

---

## Feature Engineering

Feature engineering es el arte de transformar datos crudos en features que un modelo pueda usar eficazmente. En muchos casos, un buen feature engineering importa mas que elegir el modelo correcto.

### Features numericas

#### Scaling (normalizacion)

Muchos modelos (SVM, KNN, redes neuronales, regresion logistica) son sensibles a la escala de los features. Un feature con rango [0, 1000] dominara sobre uno con rango [0, 1].

| Scaler | Que hace | Formula intuitiva | Cuando usarlo |
|---|---|---|---|
| **StandardScaler** | Media 0, std 1 | (x - media) / std | Default para la mayoria |
| **MinMaxScaler** | Escala a [0, 1] | (x - min) / (max - min) | Cuando necesitas rango fijo |
| **RobustScaler** | Usa mediana e IQR | (x - mediana) / IQR | Datos con outliers |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (el mas usado)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform en train
X_val_scaled = scaler.transform(X_val)           # solo transform en val/test
X_test_scaled = scaler.transform(X_test)         # solo transform en val/test

# IMPORTANTE: fit solo en train, transform en todo
# Si haces fit en todo, hay data leakage
```

> **Punto clave:** Tree-based models (Random Forest, XGBoost) NO necesitan scaling. Los arboles se basan en splits, no en distancias. Pero modelos lineales, SVM y redes neuronales SI lo necesitan.

#### Log transform

Util para features con distribucion sesgada (skewed) a la derecha: precios, salarios, conteos.

```python
# Log transform (comprime valores grandes, expande pequeños)
# Antes:  [100, 200, 500, 10000, 50000]  -> muy sesgado
# Despues: [4.6, 5.3, 6.2, 9.2, 10.8]    -> mas uniforme

df['log_salario'] = np.log1p(df['salario'])  # log(1+x) para evitar log(0)
# Para deshacer: np.expm1(df['log_salario'])
```

#### Binning (discretizacion)

Convertir un valor continuo en categorias. Util cuando la relacion no es lineal.

```python
# Binning por cuantiles
df['edad_bin'] = pd.qcut(df['edad'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# Binning por rangos fijos
df['edad_grupo'] = pd.cut(df['edad'],
    bins=[0, 18, 30, 45, 60, 100],
    labels=['menor', 'joven', 'adulto', 'maduro', 'senior']
)
```

### Features categoricas

| Metodo | Que hace | Cuando usarlo | Ejemplo |
|---|---|---|---|
| **One-hot** | Crea columna binaria por valor | Pocas categorias (<20), sin orden | Pais, color |
| **Label encoding** | Asigna numero a cada valor | Arboles, categorias ordinales | Talla (S=0, M=1, L=2) |
| **Target encoding** | Reemplaza con media del target | Muchas categorias (>20) | Codigo postal |
| **Frequency encoding** | Reemplaza con frecuencia | Muchas categorias, sin leakage | Categoria de producto |

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['color']])
# color='rojo'  -> [1, 0, 0]
# color='azul'  -> [0, 1, 0]
# color='verde' -> [0, 0, 1]

# O con Pandas (mas simple)
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')

# Label encoding (para arboles o categorias ordinales)
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])

# Target encoding (cuidado con leakage, usar fold-based)
# Cada categoria se reemplaza con la media del target para esa categoria
means = df.groupby('ciudad')['target'].mean()
df['ciudad_target_enc'] = df['ciudad'].map(means)

# Frequency encoding
freq = df['ciudad'].value_counts(normalize=True)
df['ciudad_freq'] = df['ciudad'].map(freq)
```

### Missing values: estrategias de imputacion

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Imputer con mediana (robusto a outliers)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# KNN imputer (mas sofisticado, usa vecinos cercanos)
knn_imputer = KNNImputer(n_neighbors=5)
X_train_imputed = knn_imputer.fit_transform(X_train)
```

### Feature selection

No todos los features ayudan. Algunos introducen ruido o estan correlacionados entre si.

```python
# 1. Eliminar features con varianza cero o casi cero
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)

# 2. Correlacion: eliminar features muy correlacionados entre si
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
df = df.drop(columns=to_drop)

# 3. Importancia del modelo (feature importance con Random Forest)
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

## Metricas de Evaluacion

Elegir la metrica correcta es tan importante como elegir el modelo. En consultoria, la metrica debe alinearse con el objetivo de negocio.

### Metricas de Clasificacion

#### Confusion Matrix explicada

```
                     Prediccion del Modelo
                    Positivo       Negativo
               ┌──────────────┬──────────────┐
    Positivo   │     TP       │     FN       │
Real           │  Verdadero   │   Falso      │
    (etiqueta) │  Positivo    │  Negativo    │
               ├──────────────┼──────────────┤
    Negativo   │     FP       │     TN       │
               │   Falso      │  Verdadero   │
               │  Positivo    │  Negativo    │
               └──────────────┴──────────────┘

TP = Dijiste "si" y era "si"     (acierto)
TN = Dijiste "no" y era "no"     (acierto)
FP = Dijiste "si" pero era "no"  (falsa alarma)
FN = Dijiste "no" pero era "si"  (se te escapo)
```

#### Metricas derivadas

| Metrica | Formula (intuitiva) | Que mide | Rango |
|---|---|---|---|
| **Accuracy** | Aciertos / Total | Proporcion correcta global | [0, 1] |
| **Precision** | TP / (TP + FP) | De los que dije positivo, cuantos lo eran | [0, 1] |
| **Recall** | TP / (TP + FN) | De los reales positivos, cuantos detecte | [0, 1] |
| **F1** | 2 * (Prec * Rec) / (Prec + Rec) | Balance entre precision y recall | [0, 1] |
| **AUC-ROC** | Area bajo curva ROC | Capacidad de separar clases | [0.5, 1] |
| **AUC-PR** | Area bajo curva Precision-Recall | Rendimiento en clase positiva | [0, 1] |
| **Log Loss** | -media(y*log(p) + (1-y)*log(1-p)) | Calidad de las probabilidades | [0, inf) |

#### Cuando usar cada metrica

| Mi dataset tiene... | Usa esta metrica | Por que |
|---|---|---|
| Clases balanceadas (50/50) | Accuracy, F1 | Accuracy funciona bien si las clases son equilibradas |
| Clases desbalanceadas (95/5) | AUC-PR, F1, Recall | Accuracy es enganosa (predecir siempre "negativo" da 95%) |
| El coste de FN es alto (cancer, fraude) | **Recall** | No queremos que se nos escape un caso real |
| El coste de FP es alto (spam, recomendaciones) | **Precision** | No queremos molestar con falsas alarmas |
| Necesito probabilidades calibradas | Log Loss | Mide la calidad de las probabilidades, no solo la decision |
| Necesito un numero unico para comparar | AUC-ROC o F1 | Resume el rendimiento en un solo valor |

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    classification_report
)

# Reporte completo
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]  # Probabilidades

print(classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo']))

# Metricas individuales
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"AUC-PR:    {average_precision_score(y_test, y_prob):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_prob):.4f}")
```

### Metricas de Regresion

| Metrica | Formula (intuitiva) | Interpretacion | Unidades |
|---|---|---|---|
| **MAE** | media(\|real - pred\|) | Error promedio absoluto | Mismas que y |
| **MSE** | media((real - pred)^2) | Penaliza errores grandes | Unidades^2 |
| **RMSE** | sqrt(MSE) | Error tipico | Mismas que y |
| **R²** | 1 - (MSE modelo / MSE media) | % de varianza explicada | Sin unidades |
| **MAPE** | media(\|real-pred\| / \|real\|) * 100 | Error porcentual medio | % |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE:  {mae:.2f}")        # "Nos equivocamos en promedio 5,000 EUR"
print(f"RMSE: {rmse:.2f}")       # "Error tipico de 7,000 EUR"
print(f"R²:   {r2:.4f}")         # "Explicamos el 87% de la varianza del precio"
print(f"MAPE: {mape:.1f}%")      # "Error relativo del 8.3%"
```

| Cuando usar | Metrica | Razon |
|---|---|---|
| Quiero interpretar el error facilmente | MAE | Unidades intuitivas, robusto a outliers |
| Quiero penalizar errores grandes | RMSE | Errores grandes pesan mas |
| Quiero comparar modelos en datasets distintos | R² | Normalizado, comparable |
| Quiero error relativo | MAPE | Porcentual, facil de comunicar |

---

## Underfitting vs Overfitting

### Bias-Variance Tradeoff

```
Error total = Bias² + Varianza + Ruido irreducible

                    Underfitting               Justo                 Overfitting
                    (alto bias)              (equilibrio)           (alta varianza)

Datos reales:       •  •                     •  •                    •  •
                  •      •                 •      •                •    • •
                •          •             •    ──    •             •  /\   \  •
                  •      •              •  ──    ──  •            • /  \   \ •
                    •  •                 ──          ──             /    \___\

Modelo:           ──────────             ── curva ──               /\/\/\/\/\
                  (linea recta)        (se ajusta bien)          (memoriza datos)

Train error:       ALTO                    BAJO                    MUY BAJO
Val error:         ALTO                    BAJO                    ALTO
Diagnostico:     Modelo muy simple       Modelo adecuado        Modelo muy complejo
```

| | Underfitting | Overfitting |
|---|---|---|
| **Sintoma** | Train y val error altos | Train error bajo, val error alto |
| **Causa** | Modelo demasiado simple | Modelo demasiado complejo |
| **Solucion** | Modelo mas complejo, mas features | Regularizacion, mas datos, modelo mas simple |

### Tecnicas de regularizacion

| Tecnica | Que hace | Donde se usa |
|---|---|---|
| **L1 (Lasso)** | Penaliza la suma de valores absolutos de pesos, puede hacer pesos = 0 | Regresion lineal, seleccion de features |
| **L2 (Ridge)** | Penaliza la suma de cuadrados de pesos, reduce pesos grandes | Regresion lineal, redes neuronales |
| **Elastic Net** | Combina L1 y L2 | Regresion lineal |
| **Dropout** | Desactiva neuronas al azar durante training | Redes neuronales |
| **Early stopping** | Parar de entrenar cuando val_loss sube | Cualquier modelo iterativo |
| **Data augmentation** | Crear datos sinteticos (rotar imagenes, etc.) | Imagenes, texto |
| **Max depth, min samples** | Limitar complejidad del arbol | Decision trees, Random Forest |

```python
# L1 (Lasso) - hace feature selection automatica
from sklearn.linear_model import Lasso
modelo = Lasso(alpha=0.1)  # alpha controla la fuerza de regularizacion

# L2 (Ridge)
from sklearn.linear_model import Ridge
modelo = Ridge(alpha=1.0)

# Elastic Net (mezcla L1 + L2)
from sklearn.linear_model import ElasticNet
modelo = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 0.5 = mitad L1, mitad L2

# Early stopping con XGBoost
import xgboost as xgb
modelo = xgb.XGBClassifier(
    n_estimators=1000,
    early_stopping_rounds=50,  # Parar si val_loss no mejora en 50 rounds
    eval_metric='logloss'
)
modelo.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)
```

### Learning curves: como diagnosticar

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    modelo, X, y,
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
plt.xlabel('Tamano del training set')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()
```

**Como leer las learning curves:**

```
Underfitting:                   Overfitting:                   Justo:
  Score                          Score                          Score
  1.0 ┤                         1.0 ┤── train                  1.0 ┤── train
      │                             │                               │── val
  0.8 ┤── train                  0.8 ┤                          0.8 ┤═══════
      │── val                        │                               │
  0.6 ┤═══════                   0.6 ┤          val             0.6 ┤
      │  ambos bajos                 │──────────                    │
  0.4 ┤                          0.4 ┤  gran gap                0.4 ┤
      └────────── datos             └────────── datos              └────────── datos

  Ambas curvas convergen         Gran distancia entre          Ambas curvas convergen
  pero a un valor bajo.          train y val.                  a un valor alto.
  -> Modelo mas complejo         -> Mas datos o regularizar    -> Todo bien
```

---

## Cross-Validation

Cross-validation te da una estimacion mas robusta del rendimiento que un unico train/val split.

### K-Fold Cross-Validation

```
Datos: [████████████████████████████████████████]

Fold 1: [VVVV][████████████████████████████████]  -> Score 1
Fold 2: [████][VVVV][████████████████████████████]  -> Score 2
Fold 3: [████████][VVVV][████████████████████████]  -> Score 3
Fold 4: [████████████][VVVV][████████████████████]  -> Score 4
Fold 5: [████████████████][VVVV][████████████████]  -> Score 5

V = Validation (test de ese fold)
█ = Training

Score final = media(Score 1..5) ± std(Score 1..5)
```

### Tipos de Cross-Validation

| Tipo | Cuando usarlo | Ejemplo |
|---|---|---|
| **K-Fold** (k=5 o 10) | Datasets medianos, distribucion uniforme | Clasificacion general |
| **Stratified K-Fold** | Clases desbalanceadas | Deteccion de fraude, diagnostico medico |
| **Time Series Split** | Datos temporales | Prediccion de ventas, precios |
| **Group K-Fold** | Datos agrupados (usuarios, pacientes) | Un usuario no puede estar en train Y test |
| **Leave-One-Out** | Datasets muy pequenos (<100) | Raramente usado en practica |
| **Repeated K-Fold** | Quieres estimacion muy robusta | Reportar resultados en papers |

```python
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, TimeSeriesSplit, GroupKFold
)

# K-Fold basico
scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Stratified K-Fold (mantiene proporciones de clases en cada fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(modelo, X, y, cv=skf, scoring='f1')
print(f"F1: {scores.mean():.4f} ± {scores.std():.4f}")

# Time Series Split (respeta el orden temporal)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(modelo, X, y, cv=tscv, scoring='neg_mean_squared_error')
print(f"MSE: {-scores.mean():.4f} ± {scores.std():.4f}")

# Group K-Fold (grupos no se mezclan entre folds)
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(modelo, X, y, cv=gkf, groups=groups, scoring='accuracy')
```

#### Time Series Split visualizado

```
Datos temporales: [Ene][Feb][Mar][Abr][May][Jun][Jul][Ago]

Fold 1: [Train: Ene-Feb        ]  [Test: Mar     ]
Fold 2: [Train: Ene-Feb-Mar    ]  [Test: Abr     ]
Fold 3: [Train: Ene-Feb-Mar-Abr]  [Test: May     ]
Fold 4: [Train: Ene-....-May   ]  [Test: Jun     ]
Fold 5: [Train: Ene-....-Jun   ]  [Test: Jul     ]

El training siempre viene ANTES del test.
Nunca usamos datos futuros para entrenar.
```

---

## Algoritmos Clave de scikit-learn

### Tabla resumen

| Algoritmo | Tipo | Complejidad | Interpretable | Necesita scaling | Mejor para |
|---|---|---|---|---|---|
| **Linear Regression** | Regresion | Baja | Alta | Si | Baseline, relaciones lineales |
| **Logistic Regression** | Clasificacion | Baja | Alta | Si | Baseline, texto (con TF-IDF) |
| **Decision Tree** | Ambos | Media | Alta | No | Explicar decisiones |
| **Random Forest** | Ambos | Media | Media | No | **Default para tabular** |
| **Gradient Boosting** (XGBoost, LightGBM) | Ambos | Alta | Baja | No | **Mejor rendimiento tabular** |
| **SVM** | Ambos | Alta | Baja | Si | Datasets pequeños, alta dim |
| **KNN** | Ambos | Baja (train) | Media | Si | Datasets pequeños, simple |
| **K-Means** | Clustering | Baja | Media | Si | Segmentacion de clientes |
| **DBSCAN** | Clustering | Media | Media | Si | Clusters de forma irregular |
| **PCA** | Dim. reduction | Baja | Media | Si | Visualizacion, preproceso |

### Ejemplos rapidos

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Regresion logistica (siempre es un buen baseline)
lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_train, y_train)

# Random Forest (robusto, poco tuning)
rf = RandomForestClassifier(
    n_estimators=100,    # numero de arboles
    max_depth=10,        # profundidad maxima
    min_samples_leaf=5,  # minimo de muestras por hoja
    random_state=42,
    n_jobs=-1            # usar todos los cores
)
rf.fit(X_train, y_train)

# XGBoost (generalmente el mejor para datos tabulares)
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

# PCA para visualizacion
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")
```

### Reduccion de dimensionalidad: PCA vs t-SNE vs UMAP

| Metodo | Velocidad | Preserva | Uso principal |
|---|---|---|---|
| **PCA** | Rapido | Estructura global (varianza) | Preproceso, reducir dimensiones antes de modelo |
| **t-SNE** | Lento | Estructura local (vecinos) | Visualizacion 2D de clusters |
| **UMAP** | Rapido | Global + local | Visualizacion 2D (mejor que t-SNE) |

```python
from sklearn.manifold import TSNE
import umap  # pip install umap-learn

# PCA (2 componentes para visualizar)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Visualizar los tres
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, X_reduced, title in zip(axes, [X_pca, X_tsne, X_umap], ['PCA', 't-SNE', 'UMAP']):
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='Set2', alpha=0.6, s=10)
    ax.set_title(title)
    ax.legend(*scatter.legend_elements(), title="Clase")
plt.tight_layout()
plt.show()
```

---

## Baseline Models

### Por que siempre empezar con un baseline simple

Un baseline es el modelo mas simple que puedas construir. Sirve como referencia: si tu modelo complejo no supera al baseline, algo esta mal.

| Tipo de problema | Baseline | Implementacion |
|---|---|---|
| Clasificacion binaria | Predecir siempre la clase mayoritaria | `DummyClassifier(strategy='most_frequent')` |
| Clasificacion | Logistic Regression | `LogisticRegression()` |
| Regresion | Predecir siempre la media | `DummyRegressor(strategy='mean')` |
| Regresion | Linear Regression | `LinearRegression()` |
| Series temporales | Predecir el valor anterior | `y_pred = y_test.shift(1)` |
| NLP | TF-IDF + Logistic Regression | Pipeline simple |
| Imagenes | Transfer learning (modelo preentrenado) | `torchvision.models.resnet18(pretrained=True)` |

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Baseline trivial: predecir siempre la clase mas comun
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
print(f"Baseline (dummy): {dummy.score(X_test, y_test):.4f}")

# Baseline razonable: logistic regression con scaling
baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
baseline.fit(X_train, y_train)
print(f"Baseline (LR): {baseline.score(X_test, y_test):.4f}")

# Ahora comparar con el modelo complejo
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest: {rf.score(X_test, y_test):.4f}")

# Si Random Forest no supera significativamente a LR,
# quizas no necesitas complejidad adicional.
```

### Flujo recomendado en consultoria

```
1. Baseline dummy           -> "Predecir siempre 'no fraude' da 99% accuracy"
                                (esto demuestra que accuracy no es la metrica correcta)

2. Baseline simple          -> "Logistic Regression con F1=0.45"
                                (esta es la referencia real)

3. Modelo medio             -> "Random Forest con F1=0.72"
                                (mejora clara, vale la pena el esfuerzo)

4. Modelo complejo          -> "XGBoost tuneado con F1=0.75"
                                (mejora marginal, evaluar si vale la pena
                                 la complejidad adicional)

5. Deep Learning            -> "Red neuronal con F1=0.76"
                                (muy poca mejora, probablemente no vale la pena
                                 la complejidad operacional)
```

> **Punto clave para consultoria:** Siempre presenta resultados como mejora sobre el baseline. "Nuestro modelo mejora la deteccion de fraude en un 60% respecto al sistema actual" es mucho mas impactante que "Nuestro modelo tiene F1=0.75".

---

> **Resumen:** Machine Learning es un proceso sistematico, no magia. Empieza simple (baseline), itera rapido, y asegurate de que la metrica que optimizas esta alineada con el objetivo de negocio. El feature engineering y la seleccion de metrica correcta suelen importar mas que el algoritmo que elijas.

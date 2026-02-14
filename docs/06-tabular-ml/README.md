# Tabular ML

## Por Que Tabular ML es el 80% de Proyectos de Consultoria

La mayoria de datos empresariales viven en tablas: bases de datos SQL, hojas de calculo, CSVs, data warehouses. Cuando un cliente dice "quiero predecir X", casi siempre X depende de datos tabulares: transacciones, caracteristicas de clientes, metricas operativas, logs.

```
Datos en consultoria tipica:

70-80%  ->  Tablas (SQL, CSV, Excel)  ->  Gradient Boosting
10-15%  ->  Texto (emails, tickets)   ->  NLP / LLMs
5-10%   ->  Imagenes (defectos, docs) ->  Computer Vision
1-5%    ->  Series temporales          ->  Time Series models
```

> **Realidad:** Dominar XGBoost/LightGBM + feature engineering te resuelve la mayoria de proyectos reales de ML en consultoria.

---

## Gradient Boosting: El Rey de Tabular

### Intuicion

Gradient Boosting construye un ensemble de arboles de decision debiles, donde cada arbol nuevo **aprende de los errores del anterior**.

```
Proceso de Boosting:

Datos originales: y_real = [10, 20, 30, 40, 50]

Paso 1: Arbol 1 predice (simple, debil)
  y_pred_1 = [12, 18, 28, 35, 48]
  residuos = [10-12, 20-18, 30-28, 40-35, 50-48]
            = [-2, 2, 2, 5, 2]

Paso 2: Arbol 2 se entrena sobre los RESIDUOS
  Intenta predecir: [-2, 2, 2, 5, 2]
  y_pred_2 = [-1.5, 1.8, 2.1, 4.5, 1.9]

Paso 3: Combinar (con learning rate = 0.1)
  y_final = y_pred_1 + 0.1 * y_pred_2
           = [12, 18, 28, 35, 48] + 0.1 * [-1.5, 1.8, 2.1, 4.5, 1.9]
           = [11.85, 18.18, 28.21, 35.45, 48.19]

  Mas cerca de [10, 20, 30, 40, 50]!

Paso 4: Calcular nuevos residuos, entrenar Arbol 3...
Paso 5: Repetir N veces (n_estimators)

Cada arbol corrige un poco los errores acumulados.
El learning rate controla cuanto "confia" en cada arbol nuevo.
```

```
Diagrama del proceso:

  Datos -----> [Arbol 1] -----> Prediccion 1
                                     |
                              Calcular Residuos
                                     |
  Residuos_1 -> [Arbol 2] -----> Prediccion 2
                                     |
                              Calcular Residuos
                                     |
  Residuos_2 -> [Arbol 3] -----> Prediccion 3
                                     |
                                   ...
                                     |
  Prediccion Final = Pred_1 + lr*Pred_2 + lr*Pred_3 + ... + lr*Pred_N
```

---

### XGBoost

XGBoost (eXtreme Gradient Boosting) es la implementacion mas popular y robusta.

**Como funciona (intuicion):**

A diferencia de gradient boosting clasico, XGBoost:
- Usa una **funcion objetivo regularizada** (evita overfitting)
- Construye arboles de forma **level-wise** (nivel por nivel)
- Maneja valores missing de forma nativa
- Paraleliza la construccion de arboles

**Hiperparametros Clave:**

| Parametro | Rango tipico | Que controla | Efecto |
|---|---|---|---|
| `n_estimators` | 100-10000 | Numero de arboles | Mas = mejor hasta overfitting |
| `max_depth` | 3-10 | Profundidad maxima del arbol | Mas = mas complejo, mas overfit |
| `learning_rate` (eta) | 0.01-0.3 | Cuanto aporta cada arbol | Menor = mas arboles necesarios, mejor generalizacion |
| `subsample` | 0.5-1.0 | Fraccion de filas por arbol | Menor = mas regularizacion |
| `colsample_bytree` | 0.5-1.0 | Fraccion de columnas por arbol | Menor = mas regularizacion |
| `reg_alpha` (L1) | 0-10 | Regularizacion L1 | Mayor = mas sparsity |
| `reg_lambda` (L2) | 0-10 | Regularizacion L2 | Mayor = pesos mas pequenos |
| `min_child_weight` | 1-10 | Min peso de hoja | Mayor = mas conservador |
| `gamma` | 0-5 | Min reduccion de loss para split | Mayor = menos splits |

**Ejemplo de Codigo Completo:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# Cargar datos
df = pd.read_csv("datos.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Crear modelo
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Para desbalance
    random_state=42,
    n_jobs=-1,
    eval_metric="auc",
    early_stopping_rounds=50,
)

# Entrenar con early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)

# Evaluar
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Best iteration
print(f"Mejor iteracion: {model.best_iteration}")
```

**Feature Importance:**

```python
import matplotlib.pyplot as plt

# Tres tipos de importancia en XGBoost
# 1. Gain: reduccion promedio de la loss cuando se usa la feature
# 2. Weight (frequency): cuantas veces se usa la feature en splits
# 3. Cover: numero promedio de muestras afectadas por splits de esta feature

# Gain es la mas informativa
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

LightGBM (Light Gradient Boosting Machine) de Microsoft. Generalmente mas rapido que XGBoost.

**Diferencias con XGBoost:**

```
XGBoost (level-wise):          LightGBM (leaf-wise):
Crece TODOS los nodos           Crece el nodo con MAYOR
del mismo nivel                  reduccion de loss

       [root]                         [root]
      /      \                       /      \
    [A]      [B]                   [A]      [B]
   /   \    /   \                 /   \
 [C]  [D] [E]  [F]             [C]  [D]

Mas uniforme,                   Mas preciso pero
menos overfit                   puede hacer overfit
                                en datos pequenos
```

**Ventajas de LightGBM:**

| Ventaja | Descripcion |
|---|---|
| **GOSS** | Gradient-based One-Side Sampling: mantiene muestras con gradiente grande, samplea las de gradiente pequeno |
| **EFB** | Exclusive Feature Bundling: agrupa features mutuamente exclusivas (reduce dimensionalidad) |
| **Categoricals nativo** | No necesitas one-hot encoding, LightGBM maneja categoricals directamente |
| **Velocidad** | 2-10x mas rapido que XGBoost en datasets grandes |

**Hiperparametros Clave de LightGBM:**

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    max_depth=-1,              # Sin limite (leaf-wise lo controla num_leaves)
    num_leaves=31,             # Maximo hojas por arbol (CLAVE en LightGBM)
    learning_rate=0.1,
    subsample=0.8,             # Llamado 'bagging_fraction' internamente
    colsample_bytree=0.8,     # Llamado 'feature_fraction' internamente
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,      # Minimo muestras en hoja
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Entrenar con early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100),
    ],
    categorical_feature=["ciudad", "tipo_producto"],  # Categoricals nativo!
)
```

### Cuando Elegir LightGBM vs XGBoost

| Criterio | XGBoost | LightGBM |
|---|---|---|
| **Dataset grande (>100K filas)** | Lento | Mucho mas rapido |
| **Dataset pequeno (<10K filas)** | Menos overfit | Puede hacer overfit |
| **Muchas categoricals** | Necesita encoding | Nativo (mas rapido, a veces mejor) |
| **Feature importance** | Buena | Buena |
| **GPU support** | Si | Si |
| **Comunidad/docs** | Mas madura | Muy buena tambien |
| **Competiciones Kaggle** | Popular | Muy popular |
| **Produccion empresarial** | Muy estable | Muy estable |

> **Consejo practico:** Prueba ambos. La diferencia en accuracy suele ser <1%. LightGBM es mas rapido para iterar. Para la entrega final, usa el que de mejor resultado en cross-validation.

### CatBoost

CatBoost (de Yandex) es otra alternativa solida:

- **Excelente con categoricals** sin necesidad de encoding (mejor que LightGBM para categoricals de alta cardinalidad)
- **Menos tuning necesario** - los defaults son buenos
- **Ordered boosting** - reduce data leakage en el entrenamiento
- Generalmente un poco mas lento que LightGBM

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.1,
    cat_features=["ciudad", "tipo_producto"],  # Indicas cuales son categoricals
    verbose=100,
)
model.fit(X_train, y_train, eval_set=(X_test, y_test))
```

---

## Feature Engineering Avanzado para Tabular

Feature engineering es lo que mas impacto tiene en el rendimiento de modelos tabulares. Un XGBoost con buenas features > cualquier modelo con features crudas.

### Aggregation Features (GroupBy Stats)

```python
# Para cada cliente, calcular estadisticas de sus transacciones
agg_features = df.groupby("customer_id").agg(
    total_transacciones=("amount", "count"),
    monto_promedio=("amount", "mean"),
    monto_maximo=("amount", "max"),
    monto_std=("amount", "std"),
    dias_como_cliente=("date", lambda x: (x.max() - x.min()).days),
    categorias_unicas=("category", "nunique"),
).reset_index()

# Merge con el dataset original
df = df.merge(agg_features, on="customer_id", how="left")
```

### Interaction Features

```python
# Crear features que capturen relaciones entre variables
df["ingreso_por_edad"] = df["ingreso_anual"] / (df["edad"] + 1)
df["deuda_sobre_ingreso"] = df["deuda_total"] / (df["ingreso_anual"] + 1)
df["balance_promedio_por_cuenta"] = df["balance_total"] / (df["num_cuentas"] + 1)

# Diferencias y ratios son muy utiles
df["diferencia_precio"] = df["precio_actual"] - df["precio_anterior"]
df["ratio_precio"] = df["precio_actual"] / (df["precio_anterior"] + 1)
```

### Time-Based Features

```python
# Extraer componentes temporales
df["dia_semana"] = df["fecha"].dt.dayofweek       # 0=Lunes, 6=Domingo
df["mes"] = df["fecha"].dt.month
df["trimestre"] = df["fecha"].dt.quarter
df["dia_del_mes"] = df["fecha"].dt.day
df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
df["hora"] = df["fecha"].dt.hour
df["es_horario_laboral"] = df["hora"].between(9, 18).astype(int)

# Lag features (valores pasados)
df = df.sort_values(["customer_id", "fecha"])
df["monto_anterior"] = df.groupby("customer_id")["amount"].shift(1)
df["monto_hace_7_dias"] = df.groupby("customer_id")["amount"].shift(7)

# Rolling statistics
df["monto_media_7d"] = (
    df.groupby("customer_id")["amount"]
    .transform(lambda x: x.rolling(7, min_periods=1).mean())
)
df["monto_std_30d"] = (
    df.groupby("customer_id")["amount"]
    .transform(lambda x: x.rolling(30, min_periods=1).std())
)
```

### Encoding de Categoricals

```python
# Frequency Encoding: reemplazar categoria por su frecuencia
freq_encoding = df["ciudad"].value_counts(normalize=True)
df["ciudad_freq"] = df["ciudad"].map(freq_encoding)

# Target Encoding (CON CUIDADO - data leakage!)
# Usar solo con cross-validation apropiado
from sklearn.model_selection import KFold

def target_encode_cv(df, col, target, n_splits=5):
    """Target encoding con cross-validation para evitar leakage."""
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
df["ciudad_x_tipo"] = df["ciudad"].astype(str) + "_" + df["tipo_producto"].astype(str)
```

### Tips de Feature Engineering

| Tecnica | Cuando aporta valor |
|---|---|
| **Aggregation features** | Datos transaccionales con entidad (cliente, producto) |
| **Interaction features** | Cuando dos features juntas son mas informativas |
| **Time features** | Datos con componente temporal |
| **Frequency encoding** | Categoricals de alta cardinalidad |
| **Target encoding** | Categoricals correlacionadas con el target (cuidado con leakage) |
| **Feature crosses** | Cuando la combinacion de categorias importa |
| **Polynomial features** | Relaciones no lineales simples (cuidado con dimension) |

---

## Manejo de Problemas Comunes

### Clases Desbalanceadas

El problema mas comun en clasificacion empresarial (fraude 1%, churn 5%, defectos 2%).

```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Opcion 1: SMOTE (Oversampling sintetico)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# Cuidado: solo aplicar a train, NUNCA a test/validation

# Opcion 2: Class weights (mas simple, recomendado)
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
weight_dict = dict(zip(np.unique(y_train), weights))

# En XGBoost
model = xgb.XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
)

# En LightGBM
model = lgb.LGBMClassifier(
    class_weight="balanced",
    # O manualmente:
    # is_unbalance=True,
)

# Opcion 3: Threshold tuning
from sklearn.metrics import precision_recall_curve

y_prob = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# Encontrar threshold que maximice F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
y_pred_custom = (y_prob >= best_threshold).astype(int)
```

**Metricas para datos desbalanceados:**

| Metrica | Usar? | Por que |
|---|---|---|
| **Accuracy** | NO | 99% accuracy prediciendo siempre "no fraude" con 1% fraude |
| **F1-score** | SI | Balancea precision y recall |
| **AUC-PR** | SI | Mejor que AUC-ROC para desbalance severo |
| **AUC-ROC** | Parcial | Puede ser optimista con mucho desbalance |
| **Precision@K** | SI | "De los top K alertas, cuantas son reales?" |

### Missing Values

```python
# XGBoost y LightGBM manejan NaN nativamente - muchas veces es MEJOR
# dejar los NaN como estan que imputar

# Si necesitas imputar (para otros modelos o features nuevas):
from sklearn.impute import SimpleImputer

# Numerico: mediana (robusta a outliers)
num_imputer = SimpleImputer(strategy="median")

# Categorico: moda o valor especial
cat_imputer = SimpleImputer(strategy="constant", fill_value="MISSING")

# Crear features de "missing" (pueden ser informativas!)
df["ingreso_missing"] = df["ingreso"].isna().astype(int)
```

### High Cardinality Categoricals

Cuando una variable categorica tiene muchos valores unicos (ciudades, codigos postales, IDs de producto).

```python
# Problema: One-hot encoding de 10,000 ciudades = 10,000 columnas

# Solucion 1: Target encoding (ver seccion anterior)

# Solucion 2: Frequency encoding
df["ciudad_freq"] = df["ciudad"].map(df["ciudad"].value_counts(normalize=True))

# Solucion 3: Agrupar categorias raras
threshold = 0.01  # Categorias con <1% de frecuencia
freq = df["ciudad"].value_counts(normalize=True)
df["ciudad_agrupada"] = df["ciudad"].apply(
    lambda x: x if freq[x] >= threshold else "OTROS"
)

# Solucion 4: LightGBM/CatBoost con categoricals nativos (lo mejor)
# Solo pasas la lista de columnas categoricas y el modelo las maneja
```

### Feature Leakage

**Que es:** cuando tu modelo tiene acceso a informacion que no tendria en produccion, inflando metricas artificialmente.

```
Fuentes comunes de leakage en datos de negocio:

1. Features del futuro:
   Predecir churn de marzo usando datos de abril
   -> Asegurar que las features solo usan datos ANTERIORES al evento

2. Target leakage:
   Predecir si un paciente tiene diabetes usando "medicacion_diabetes"
   -> La medicacion es CONSECUENCIA del diagnostico

3. Train-test contamination:
   Normalizar ANTES del split (el test "ve" estadisticas del train)
   -> Siempre hacer fit en train, transform en test

4. Group leakage:
   Mismo cliente en train y test (el modelo "recuerda" al cliente)
   -> Hacer split por GRUPO, no por fila
```

**Como detectar leakage:**

```python
# Senal de alarma: AUC > 0.99 en tu primer modelo
# Verificar feature importance: si una feature domina, investigar

importance = model.feature_importances_
for feat, imp in sorted(zip(X.columns, importance), key=lambda x: -x[1])[:5]:
    print(f"{feat}: {imp:.4f}")

# Si la feature top tiene importancia desproporcionada,
# investigar si hay leakage
```

---

## Pipeline de Produccion con scikit-learn

### ColumnTransformer + Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb

# Definir tipos de columnas
numeric_features = ["edad", "ingreso", "balance", "num_transacciones"]
categorical_features = ["ciudad", "tipo_cuenta", "segmento"]

# Pipeline para numericas
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),  # No necesario para tree models, pero si para otros
])

# Pipeline para categoricas
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combinar
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",  # Descartar columnas no listadas
)

# Pipeline completo
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
    )),
])

# Entrenar
pipeline.fit(X_train, y_train)

# Predecir (el preprocessing se aplica automaticamente)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
```

### Serializacion con joblib

```python
import joblib

# Guardar pipeline completo (preprocessing + modelo)
joblib.dump(pipeline, "modelo_churn_v1.joblib")

# Cargar en produccion
pipeline_loaded = joblib.load("modelo_churn_v1.joblib")

# Predecir con datos nuevos (mismas columnas que en entrenamiento)
nuevo_cliente = pd.DataFrame({
    "edad": [35],
    "ingreso": [50000],
    "balance": [12000],
    "num_transacciones": [45],
    "ciudad": ["Madrid"],
    "tipo_cuenta": ["premium"],
    "segmento": ["retail"],
})

probabilidad_churn = pipeline_loaded.predict_proba(nuevo_cliente)[:, 1]
print(f"Probabilidad de churn: {probabilidad_churn[0]:.2%}")
```

---

## Hyperparameter Tuning con Optuna

### Por Que Optuna

| Metodo | Eficiencia | Implementacion |
|---|---|---|
| **Grid Search** | Mala (explora todo) | Simple |
| **Random Search** | Aceptable | Simple |
| **Optuna** (Bayesian) | Muy buena | Moderada |

Optuna usa **optimizacion bayesiana**: aprende de ejecuciones anteriores para elegir los proximos hiperparametros de forma inteligente.

### Ejemplo Completo con XGBoost + Optuna

```python
import optuna
from sklearn.model_selection import cross_val_score
import xgboost as xgb

def objective(trial):
    """Funcion objetivo que Optuna optimiza."""

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

    # Cross-validation con early stopping
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

# Ejecutar optimizacion
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Mejores hiperparametros
print(f"Mejor AUC: {study.best_value:.4f}")
print(f"Mejores params: {study.best_params}")

# Entrenar modelo final con los mejores parametros
best_model = xgb.XGBClassifier(**study.best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)
```

### Tips de Tuning

**Orden de importancia para tunear:**

```
1. learning_rate + n_estimators  (los mas impactantes)
   -> Empieza con lr=0.1, n_estimators=1000 + early stopping

2. max_depth / num_leaves  (complejidad del arbol)
   -> max_depth=6 es un buen inicio

3. subsample + colsample_bytree  (regularizacion por sampling)
   -> 0.8 es un buen inicio para ambos

4. reg_alpha + reg_lambda  (regularizacion explicita)
   -> Usualmente valores pequenos

5. min_child_weight / min_child_samples  (control de hojas)
   -> Incrementar si hay overfitting
```

---

## Interpretabilidad

En consultoria, la interpretabilidad es **tan importante como la accuracy**. El cliente pregunta "por que el modelo predice esto?" y necesitas responder.

### Feature Importance (Built-in)

```python
# Los tree models tienen feature importance integrada
importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print(importance.head(10))
```

Limitacion: la feature importance global te dice que features son importantes EN PROMEDIO, pero no por que una prediccion INDIVIDUAL es como es.

### SHAP Values

SHAP (SHapley Additive exPlanations) te dice **cuanto contribuye cada feature a cada prediccion individual**.

**Intuicion:**

```
Prediccion para Cliente X: probabilidad de churn = 78%
Baseline (promedio del dataset): 25%

SHAP descompone la diferencia (78% - 25% = 53%):

  antiguedad_meses = 3     -> +20% (poco tiempo como cliente)
  num_quejas = 5           -> +15% (muchas quejas)
  uso_app_mensual = 2      -> +12% (usa poco la app)
  ingreso = 80000          -> -5%  (ingreso alto reduce churn)
  tipo_cuenta = premium    -> -3%  (las premium churnan menos)
  ... otras features ...   -> +14%
  -----------------------------------------
  Total SHAP:                 +53% (25% base + 53% = 78%)

Cada valor SHAP se interpreta como:
"Esta feature empujo la prediccion X puntos hacia arriba/abajo
comparado con el promedio del dataset"
```

**Ejemplo de Codigo:**

```python
import shap

# Crear explainer (usa TreeExplainer para modelos de arboles - MUY rapido)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 1. Summary Plot: importancia global + direccion del efecto
shap.summary_plot(shap_values, X_test)
# Muestra las features ordenadas por importancia
# Cada punto es una muestra
# Color = valor de la feature (rojo=alto, azul=bajo)
# Posicion horizontal = efecto en la prediccion

# 2. Waterfall Plot: explicacion de UNA prediccion
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],         # Primera muestra
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns.tolist(),
))

# 3. Force Plot: otra forma de ver una prediccion individual
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
)

# 4. Dependence Plot: como afecta UNA feature a las predicciones
shap.dependence_plot("antiguedad_meses", shap_values, X_test)
# Muestra la relacion entre el valor de la feature y su efecto SHAP
```

### Partial Dependence Plots (PDP)

```python
from sklearn.inspection import PartialDependenceDisplay

# PDP muestra el efecto MARGINAL promedio de una feature
PartialDependenceDisplay.from_estimator(
    model, X_test,
    features=["antiguedad_meses", "num_transacciones"],
    kind="both",  # Muestra ICE lines + PDP promedio
)
```

Diferencia SHAP vs PDP:
- **SHAP:** efecto de cada feature para cada prediccion (local)
- **PDP:** efecto promedio de una feature sobre todo el dataset (global)
- En consultoria, **SHAP** es mas util para explicar predicciones individuales al cliente.

### LIME

LIME (Local Interpretable Model-agnostic Explanations) crea un modelo simple (lineal) que aproxima el comportamiento del modelo complejo alrededor de una prediccion especifica. Util cuando no puedes usar SHAP (modelos no-tree), pero SHAP es mas robusto.

---

## Cuando Usar Deep Learning para Tabular

### Modelos de DL para Tabular

| Modelo | Descripcion | Performance |
|---|---|---|
| **TabNet** | Atencion + feature selection | Similar a GBM, a veces mejor |
| **FT-Transformer** | Feature Tokenizer + Transformer | Competitivo con GBM |
| **TabTransformer** | Embeddings categoricos + Transformer | Bueno con muchas categoricals |

### Realidad

```
Benchmark tipico en datos tabulares:

XGBoost/LightGBM:  AUC = 0.892
TabNet:             AUC = 0.887
FT-Transformer:    AUC = 0.890
Neural Network:     AUC = 0.875

Diferencia: marginal o inexistente
Complejidad: MUCHO mayor para DL
Tiempo de desarrollo: 5x mas para DL
```

> **Regla practica:** Gradient boosting gana en tabular puro en >95% de los casos. La excepcion es cuando tienes datos **multimodales** (tabla + imagenes + texto) donde DL permite fusionar todo en un modelo end-to-end.

---

## Caso Tipo de Consultoria Paso a Paso

### Escenario: Prediccion de Churn para una Telco

#### 1. Reunion con el Cliente

```
Preguntas clave:
- Que consideran "churn"? (cancelacion, no uso en 30 dias?)
- Que acciones tomaran con las predicciones? (ofertas de retencion?)
- Que datos tienen disponibles?
- Cual es el costo de un falso positivo vs falso negativo?
- Cada cuanto necesitan predicciones? (diario, mensual?)
```

#### 2. Explorar Datos (EDA)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("telco_data.csv")

# Resumen rapido
print(f"Shape: {df.shape}")
print(f"\nTarget distribution:\n{df['churn'].value_counts(normalize=True)}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nDtypes:\n{df.dtypes.value_counts()}")

# Correlaciones con el target
numeric_cols = df.select_dtypes(include="number").columns
correlations = df[numeric_cols].corrwith(df["churn"]).abs().sort_values(ascending=False)
print(f"\nCorrelaciones con churn:\n{correlations.head(10)}")
```

#### 3. Definir Metrica de Exito

```
Con el cliente se acuerda:
- Metrica primaria: F1-score (balance entre precision y recall)
- Metrica secundaria: AUC-PR (por el desbalance)
- Objetivo: F1 > 0.70 (baseline actual del equipo de datos: 0.55)
- Constraint: Recall > 0.75 (no perder mas del 25% de churners)
```

#### 4. Baseline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# Baseline simple
baseline = LogisticRegression(class_weight="balanced", max_iter=1000)
baseline.fit(X_train_processed, y_train)
y_pred_baseline = baseline.predict(X_test_processed)

print("=== BASELINE (Logistic Regression) ===")
print(classification_report(y_test, y_pred_baseline))
# F1 de clase positiva: 0.58
```

#### 5. Feature Engineering

```python
# Features de comportamiento
df["tendencia_uso"] = df["uso_mes_actual"] - df["uso_mes_anterior"]
df["ratio_uso"] = df["uso_mes_actual"] / (df["uso_mes_anterior"] + 1)
df["quejas_por_mes"] = df["total_quejas"] / (df["meses_antiguedad"] + 1)

# Features de engagement
df["usa_app"] = (df["logins_app_mensual"] > 0).astype(int)
df["dias_sin_actividad"] = (pd.Timestamp.now() - df["ultima_actividad"]).dt.days

# Aggregations
agg = df.groupby("plan_id").agg(
    churn_rate_plan=("churn", "mean"),
    avg_antiguedad_plan=("meses_antiguedad", "mean"),
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
# F1 de clase positiva: 0.74 (mejora significativa!)
```

#### 7. Interpretar Resultados con SHAP

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Para la presentacion al cliente:
# 1. Factores globales que causan churn
shap.summary_plot(shap_values[1], X_test)  # [1] = clase positiva (churn)

# 2. Ejemplo individual: "Por que este cliente tiene alto riesgo?"
idx = 42  # Un cliente con alta probabilidad de churn
shap.waterfall_plot(shap.Explanation(
    values=shap_values[1][idx],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[idx],
    feature_names=X_test.columns.tolist(),
))
```

#### 8. Presentar al Cliente

```
Estructura de la presentacion:

1. Recordatorio del objetivo y la metrica acordada
2. Resultados:
   - "Nuestro modelo identifica correctamente el 78% de clientes que van a churnar"
   - "De cada 10 alertas, 7 son churners reales"
3. Factores clave de churn (SHAP summary plot):
   - Tendencia de uso decreciente
   - Quejas recientes sin resolver
   - Baja antiguedad
4. Ejemplo concreto (SHAP waterfall):
   - "Este cliente tiene 82% probabilidad de churn PORQUE..."
5. Recomendaciones de negocio:
   - Priorizar retencion en clientes con uso decreciente
   - Resolver quejas en <48h reduce churn un 30%
6. Proximos pasos: deployment, monitoreo
```

#### 9. Deployment

```python
import joblib

# Guardar modelo
joblib.dump(pipeline, "churn_model_v1.joblib")

# En produccion (script diario):
def predict_churn_batch(new_data_path):
    """Predecir churn para todos los clientes activos."""
    model = joblib.load("churn_model_v1.joblib")
    df = pd.read_csv(new_data_path)

    # Feature engineering (mismas transformaciones que en entrenamiento)
    df = create_features(df)

    # Predecir
    df["churn_probability"] = model.predict_proba(df[feature_columns])[:, 1]
    df["churn_risk"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["bajo", "medio", "alto"]
    )

    # Exportar para el equipo de retencion
    high_risk = df[df["churn_risk"] == "alto"].sort_values(
        "churn_probability", ascending=False
    )
    high_risk.to_csv("clientes_alto_riesgo.csv", index=False)

    return high_risk
```

---

## Checklist de Proyecto Tabular ML

```
FASE 1: Entender el problema
[ ] Reunirse con el cliente y entender el problema de negocio
[ ] Definir la variable target con precision
[ ] Acordar metrica de exito
[ ] Identificar fuentes de datos

FASE 2: Datos
[ ] EDA: distribucion del target, missing values, outliers
[ ] Identificar y manejar data leakage
[ ] Definir split temporal o por grupos (no random si hay dependencias)
[ ] Feature engineering

FASE 3: Modelado
[ ] Baseline simple (logistic regression o reglas)
[ ] XGBoost / LightGBM con defaults razonables
[ ] Feature selection (SHAP, importancia, eliminar ruido)
[ ] Hyperparameter tuning con Optuna
[ ] Validacion cruzada (5-fold o temporal)

FASE 4: Interpretacion
[ ] SHAP: factores globales y explicaciones individuales
[ ] Validar con expertos del dominio (los factores tienen sentido?)
[ ] Analisis de errores (donde falla el modelo?)

FASE 5: Entrega
[ ] Pipeline reproducible (ColumnTransformer + Pipeline)
[ ] Serializar modelo (joblib)
[ ] Documentar features y transformaciones
[ ] Presentar resultados al cliente (con interpretabilidad)
[ ] Plan de monitoreo en produccion
```

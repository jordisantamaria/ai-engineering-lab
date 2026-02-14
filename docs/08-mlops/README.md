# MLOps

> **ML sin MLOps = una demo que nunca llega a producción.**
> MLOps es la disciplina que convierte experimentos de ML en sistemas confiables, reproducibles y mantenibles en producción.

---

## Tabla de Contenidos

- [Qué es MLOps y Por Qué Importa](#qué-es-mlops-y-por-qué-importa)
- [Niveles de Madurez MLOps](#niveles-de-madurez-mlops)
- [Experiment Tracking](#experiment-tracking)
- [Data Versioning](#data-versioning)
- [Feature Store](#feature-store)
- [ML Pipelines](#ml-pipelines)
- [Model Registry](#model-registry)
- [Reentrenamiento Automático](#reentrenamiento-automático)
- [Infraestructura](#infraestructura)
- [Lo Que Realmente Necesitas](#lo-que-realmente-necesitas-para-consultoría)

---

## Qué es MLOps y Por Qué Importa

MLOps (Machine Learning Operations) aplica prácticas de DevOps al ciclo de vida de ML. El objetivo es llevar modelos de la experimentación a producción de forma reproducible, automatizada y monitorizada.

**Sin MLOps:**

```
Científico entrena modelo → Envía .pkl por email → Ingeniero intenta deployarlo
→ "No funciona en mi máquina" → 3 semanas depurando → Ya hay datos nuevos
→ El modelo ya no sirve → Vuelta a empezar
```

**Con MLOps:**

```
Commit en main → Pipeline automático → Train → Evaluar → Registrar modelo
→ Test en staging → Deploy automático → Monitoring → Reentrenar si baja performance
```

**Por qué importa para consultoría:**

| Sin MLOps | Con MLOps |
|-----------|-----------|
| "Nuestro data scientist se fue y nadie sabe qué modelo usamos" | Modelos versionados y documentados |
| "No sabemos con qué datos se entrenó" | Datos versionados, pipeline reproducible |
| "El modelo dejó de funcionar y no nos dimos cuenta" | Alertas automáticas por drift |
| "Reentrenar tarda 2 semanas de trabajo manual" | Reentrenamiento con un click (o automático) |
| Proyecto de 6 meses que muere al entregar | Sistema que el cliente puede mantener |

---

## Niveles de Madurez MLOps

Google define tres niveles de madurez en MLOps. Es el framework más referenciado en la industria.

### Tabla de Niveles

| Aspecto | Level 0: Manual | Level 1: ML Pipeline | Level 2: CI/CD + Pipeline |
|---------|----------------|---------------------|--------------------------|
| **Entrenamiento** | Manual en notebook | Pipeline automatizado | Pipeline + CI/CD |
| **Deploy** | Manual (copy-paste) | Semi-automático | Automático |
| **Tracking** | Nada o spreadsheet | MLflow/W&B | MLflow/W&B integrado en pipeline |
| **Testing** | Ninguno | Validación básica de modelo | Tests de datos, modelo, código |
| **Monitoring** | Ninguno | Métricas básicas | Drift detection + alertas |
| **Reentrenamiento** | Cuando alguien se acuerda | Triggered (schedule/manual) | Automático por drift/schedule |
| **Reproducibilidad** | Ninguna | Parcial (pipeline fijo) | Total (código + datos + config) |
| **Tiempo de deploy** | Semanas | Días | Horas/minutos |

### Level 0: Manual (Donde están la mayoría de clientes)

```
Notebook Jupyter
    ↓ (manual)
Modelo .pkl en carpeta local
    ↓ (manual)
"Oye, pásame el modelo por Slack"
    ↓ (manual)
Deploy manual en un servidor
    ↓ (nadie monitoriza)
El modelo se degrada silenciosamente
```

**Características:**
- Todo es manual y ad-hoc
- No hay pipeline automatizado
- Datos y modelos sin versionar
- El conocimiento está en la cabeza del data scientist
- Reproducir un experimento es prácticamente imposible

### Level 1: ML Pipeline Automation

```
Datos nuevos (trigger)
    ↓ (automático)
Pipeline: fetch → preprocess → train → evaluate
    ↓ (automático)
Modelo registrado en MLflow con métricas
    ↓ (manual/semi-auto)
Deploy si supera threshold
    ↓ (automático)
Monitoring básico
```

**El gran salto:** El pipeline es reproducible. Cualquier persona puede ejecutarlo y obtener el mismo resultado.

### Level 2: CI/CD Pipeline Automation

```
Push a main
    ↓ (CI automático)
Tests de código + datos + modelo
    ↓ (CD automático)
Pipeline completo: train → evaluate → register → deploy
    ↓ (automático)
Canary deployment → monitoring → rollback si falla
    ↓ (automático)
Reentrenamiento por drift detection
```

> **Para consultoría:** La mayoría de clientes están en Level 0. Llevarlos a Level 1 ya es un **gran win** que justifica el proyecto. Level 2 raramente es necesario en una primera fase.

---

## Experiment Tracking

### Por Qué Es Esencial

Sin experiment tracking:
- "Qué hiperparámetros usé en ese modelo que funcionaba bien?"
- "Cuál era el accuracy del modelo de hace 3 semanas?"
- "Qué versión de los datos usé?"

Con experiment tracking:
- Cada experimento documentado automáticamente
- Comparación visual de métricas
- Reproducibilidad total
- Poder mostrar al cliente el progreso del proyecto

### MLflow

MLflow es el estándar open-source para experiment tracking. Cuatro conceptos clave:

| Concepto | Qué es | Ejemplo |
|----------|--------|---------|
| **Experiment** | Grupo de runs relacionados | "clasificador-fraude-v2" |
| **Run** | Una ejecución de entrenamiento | "run con lr=0.01, epochs=50" |
| **Parameters** | Configuración del run | learning_rate, batch_size, model_type |
| **Metrics** | Resultados medibles | accuracy, f1, loss |
| **Artifacts** | Archivos generados | modelo.pkl, confusion_matrix.png |

#### Código Completo: Training con MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import json

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # o "sqlite:///mlflow.db" para local
mlflow.set_experiment("clasificador-churn")

# Cargar datos
df = pd.read_csv("data/churn_dataset.csv")
X = df.drop("churn", axis=1)
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparámetros
params = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}

# Entrenar con tracking
with mlflow.start_run(run_name="rf-baseline-v2"):
    # Loguear parámetros
    mlflow.log_params(params)
    mlflow.log_param("dataset_version", "2024-01-15")
    mlflow.log_param("n_samples_train", len(X_train))
    mlflow.log_param("n_features", X_train.shape[1])

    # Entrenar
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Predecir y evaluar
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Loguear métricas
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_weighted", f1)

    # Loguear artefactos
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

    # Loguear el modelo
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="churn-classifier",
    )

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1: {f1:.4f}")
```

#### MLflow UI: Comparar Runs

```bash
# Iniciar servidor MLflow
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Abrir http://localhost:5000 en el navegador
# - Ver todos los runs de un experiment
# - Comparar métricas lado a lado
# - Ver gráficos de evolución
# - Descargar artefactos
```

#### Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Registrar un modelo
result = mlflow.register_model(
    f"runs:/{run_id}/model",
    "churn-classifier"
)

# Transicionar a staging
client.transition_model_version_stage(
    name="churn-classifier",
    version=result.version,
    stage="Staging",
)

# Después de tests, promover a producción
client.transition_model_version_stage(
    name="churn-classifier",
    version=result.version,
    stage="Production",
)

# Cargar modelo de producción
model = mlflow.sklearn.load_model("models:/churn-classifier/Production")
```

#### Serving con MLflow

```bash
# Servir modelo directamente desde MLflow
mlflow models serve \
    -m "models:/churn-classifier/Production" \
    -p 5001 \
    --no-conda

# Hacer predicción
curl -X POST http://localhost:5001/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [[1.0, 2.0, 3.0, 4.0, 5.0]]}'
```

### Weights & Biases (W&B)

W&B es la alternativa cloud-first a MLflow. Mejor UI, mejor colaboración, más features pero con tier de pago.

```python
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Inicializar (crea un run en wandb.ai)
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

# Entrenar
model = RandomForestClassifier(**wandb.config)
model.fit(X_train, y_train)

# Evaluar y loguear
y_pred = model.predict(X_test)
wandb.log({
    "accuracy": accuracy_score(y_test, y_pred),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
})

# Loguear tabla de datos para análisis
table = wandb.Table(columns=["true", "predicted"])
for true, pred in zip(y_test[:100], y_pred[:100]):
    table.add_data(true, pred)
wandb.log({"predictions": table})

# Finalizar
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

# Lanzar sweep
sweep_id = wandb.sweep(sweep_config, project="churn-classifier")
wandb.agent(sweep_id, function=train_sweep, count=50)  # 50 experimentos
```

### MLflow vs W&B

| Aspecto | MLflow | W&B |
|---------|--------|-----|
| **Hosting** | Self-hosted (o Databricks managed) | Cloud (wandb.ai) |
| **Coste** | Gratis (open source) | Free tier + $50/user/mes pro |
| **UI** | Funcional, básica | Excelente, interactiva |
| **Experiment tracking** | Completo | Completo + tablas, media |
| **Model registry** | Incluido | Incluido (Model Registry) |
| **Hyperparameter tuning** | No nativo | Sweeps (bayesian, grid, random) |
| **Colaboración** | Limitada | Excelente (teams, reports) |
| **Integración** | Universal | Universal + PyTorch/HF nativo |
| **Setup** | Más trabajo (self-host) | 1 línea de código |
| **Control de datos** | Total (tus servidores) | Datos en cloud de W&B |
| **Ideal para** | Enterprise, on-prem, regulación | Startups, equipos distribuidos |

> **Recomendación para consultoría:** Empieza con MLflow (gratis, self-hosted, sin dependencias externas). Si el cliente ya usa W&B o necesita colaboración avanzada, usa W&B.

---

## Data Versioning

### El Problema

```
"Entrenamos el modelo con los datos de enero. Ahora los datos de febrero
son diferentes. No sabemos qué cambió. El modelo nuevo es peor.
¿Podemos volver a los datos de enero?"

→ Sin data versioning, la respuesta es "probablemente no".
```

### DVC (Data Version Control)

DVC funciona como git pero para archivos grandes (datos, modelos). Los archivos pesados se guardan en storage remoto (S3, GCS) y en git solo se guarda un puntero (.dvc file).

```bash
# Instalar DVC
pip install dvc dvc-s3  # o dvc-gs, dvc-azure

# Inicializar en un repo git
dvc init

# Agregar un archivo de datos
dvc add data/training_data.csv
# Esto crea:
#   data/training_data.csv.dvc  (puntero, se commitea a git)
#   data/.gitignore             (el archivo real se ignora en git)

git add data/training_data.csv.dvc data/.gitignore
git commit -m "Add training data v1"

# Configurar remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc push  # Sube datos a S3

# Cuando cambian los datos
dvc add data/training_data.csv
git add data/training_data.csv.dvc
git commit -m "Update training data v2"
dvc push

# Volver a una versión anterior
git checkout HEAD~1 -- data/training_data.csv.dvc
dvc checkout  # Descarga la versión anterior de S3

# En otra máquina, obtener los datos
git clone <repo>
dvc pull  # Descarga datos desde S3
```

**Cómo funciona:**

```
Git repo:
├── data/
│   ├── training_data.csv.dvc   ← Puntero (hash del archivo)
│   └── .gitignore               ← Ignora el CSV real
├── models/
│   └── model.pkl.dvc            ← Puntero al modelo
└── dvc.yaml                     ← Pipeline definition

S3 bucket (dvc remote):
├── ab/cdef1234...               ← training_data.csv (v1)
├── 12/3456abcd...               ← training_data.csv (v2)
└── ff/eedd1122...               ← model.pkl
```

### Alternativa Simple: Snapshots con Timestamp

Para proyectos pequeños donde DVC es overkill:

```python
# scripts/snapshot_data.py
import boto3
from datetime import datetime

s3 = boto3.client("s3")
BUCKET = "ml-project-data"

def create_snapshot(local_path: str, dataset_name: str):
    """Subir snapshot de datos con timestamp."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    s3_key = f"snapshots/{dataset_name}/{timestamp}/{dataset_name}.csv"

    s3.upload_file(local_path, BUCKET, s3_key)
    print(f"Snapshot creado: s3://{BUCKET}/{s3_key}")
    return s3_key

def list_snapshots(dataset_name: str):
    """Listar snapshots disponibles."""
    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=f"snapshots/{dataset_name}/",
        Delimiter="/",
    )
    for prefix in response.get("CommonPrefixes", []):
        print(prefix["Prefix"])

# Uso
create_snapshot("data/training.csv", "training")
list_snapshots("training")
```

---

## Feature Store

### Qué Es

Un feature store es un repositorio centralizado de features (variables) preparadas para ML. Almacena las transformaciones de datos listas para usar en entrenamiento e inference.

```
Sin Feature Store:
  Modelo A: calcula "avg_purchases_30d" en su pipeline
  Modelo B: calcula "avg_purchases_30d" en su pipeline (diferente)
  Modelo C: calcula "avg_purchases_30d" en su pipeline (otra diferente)
  → 3 implementaciones diferentes de la misma feature, posibles inconsistencias

Con Feature Store:
  Feature Store: tiene "avg_purchases_30d" calculada una vez
  Modelo A, B, C: leen la misma feature → consistencia garantizada
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

# Fuente de datos
customer_source = FileSource(
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp",
)

# Entidad
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
# Usar features para entrenamiento
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Obtener features históricas para training
training_df = store.get_historical_features(
    entity_df=entity_df,  # DataFrame con customer_id + timestamp
    features=[
        "customer_features:avg_purchases_30d",
        "customer_features:total_spend",
        "customer_features:days_since_last_purchase",
    ],
).to_df()

# Obtener features online para inference
feature_vector = store.get_online_features(
    features=[
        "customer_features:avg_purchases_30d",
        "customer_features:total_spend",
    ],
    entity_rows=[{"customer_id": 12345}],
).to_dict()
```

### Necesitas un Feature Store?

| Situación | Necesitas Feature Store? |
|-----------|------------------------|
| 1 modelo, equipo pequeno | No. Pandas/SQL es suficiente. |
| 2-3 modelos, features compartidas | Probablemente no. Módulos Python compartidos. |
| 5+ modelos, muchas features compartidas | Probablemente sí. |
| Necesitas consistencia train/serve | Sí, es el caso más fuerte. |
| Equipo grande, muchos data scientists | Sí. |

> **Para la mayoría de proyectos de consultoría: NO necesitas un feature store.** Un módulo Python con funciones de feature engineering es suficiente. Cuando tengas 5+ modelos compartiendo features, evalúalo.

---

## ML Pipelines

### Qué Son

Un ML pipeline es un DAG (Directed Acyclic Graph) que define los pasos desde datos crudos hasta modelo deployado.

```
Datos crudos → Preprocesar → Feature Engineering → Train → Evaluate → Deploy
    │              │                │                │         │         │
    ▼              ▼                ▼                ▼         ▼         ▼
   S3         datos limpios     features.csv    model.pkl  metrics   endpoint
```

### Niveles de Complejidad

#### Simple: Scripts Python + Makefile

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
# Ejecutar pipeline completo
make all

# Solo reentrenar (asume datos ya procesados)
make train evaluate
```

#### Intermedio: Prefect

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
    """Extraer datos de la fuente."""
    return pd.read_sql(f"SELECT * FROM {source}", engine)

@task
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Limpiar y transformar datos."""
    df = df.dropna(subset=["target"])
    df = df.fillna(df.median(numeric_only=True))
    return df

@task
def train_model(df: pd.DataFrame, params: dict):
    """Entrenar modelo con tracking."""
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
    """Deploy si accuracy supera threshold."""
    if accuracy >= threshold:
        joblib.dump(model, "models/model_latest.pkl")
        print(f"Modelo deployado con accuracy {accuracy:.4f}")
    else:
        print(f"Modelo rechazado: accuracy {accuracy:.4f} < {threshold}")

@flow(name="ML Training Pipeline")
def training_pipeline(source: str = "customers", params: dict = None):
    """Pipeline completo de entrenamiento."""
    if params is None:
        params = {"n_estimators": 200, "max_depth": 10}

    df = fetch_data(source)
    df_clean = preprocess(df)
    model, accuracy = train_model(df_clean, params)
    evaluate_and_deploy(model, accuracy)

if __name__ == "__main__":
    training_pipeline()
```

#### Avanzado: Kubeflow / SageMaker Pipelines

Para escala enterprise. Cada paso es un container independiente, escalable, con recursos dedicados.

| Herramienta | Complejidad | Escalabilidad | Hosting | Ideal Para |
|-------------|------------|--------------|---------|-----------|
| Scripts + Makefile | Baja | Baja | Local/servidor | POC, proyectos pequenos |
| Scripts + cron | Baja | Baja | Servidor | Batch scheduling simple |
| Prefect | Media | Media | Cloud/self-hosted | Equipos medianos |
| Airflow | Media-Alta | Alta | Self-hosted/Cloud | Pipelines complejos, data eng |
| Kubeflow | Alta | Muy alta | Kubernetes | Enterprise, muchos modelos |
| SageMaker Pipelines | Media | Alta | AWS managed | Full AWS ecosystem |
| Vertex AI Pipelines | Media | Alta | GCP managed | Full GCP ecosystem |

> **Recomendación para consultoría:** Empieza con scripts + Makefile (o cron). Si necesitas retries, scheduling, y UI, sube a Prefect o Airflow. Kubeflow solo si el cliente ya tiene Kubernetes y necesita escala seria.

---

## Model Registry

### Concepto

El Model Registry es un catálogo centralizado de modelos con versionado y ciclo de vida. Piénsalo como un "app store" interno para modelos ML.

```
Stages de un modelo:

None → Staging → Production → Archived
               ↑            │
               └────────────┘ (rollback si falla)
```

### Workflow Completo con MLflow Model Registry

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 1. TRAIN: Entrenar y registrar
with mlflow.start_run():
    model = train_my_model(X_train, y_train)
    accuracy = evaluate(model, X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)

    # Registrar automáticamente
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="churn-classifier",
    )

# 2. STAGING: Promover versión a staging
latest_version = client.get_latest_versions("churn-classifier", stages=["None"])[0]
client.transition_model_version_stage(
    name="churn-classifier",
    version=latest_version.version,
    stage="Staging",
)

# 3. TEST: Evaluar en staging
staging_model = mlflow.sklearn.load_model("models:/churn-classifier/Staging")
staging_accuracy = evaluate(staging_model, X_staging_test, y_staging_test)

# 4. PRODUCTION: Si pasa validación, promover
if staging_accuracy >= PRODUCTION_THRESHOLD:
    # Archivar modelo actual en producción
    prod_versions = client.get_latest_versions("churn-classifier", stages=["Production"])
    for v in prod_versions:
        client.transition_model_version_stage(
            name="churn-classifier",
            version=v.version,
            stage="Archived",
        )

    # Promover nuevo modelo
    client.transition_model_version_stage(
        name="churn-classifier",
        version=latest_version.version,
        stage="Production",
    )
    print(f"Modelo v{latest_version.version} en producción")

# 5. SERVE: Cargar modelo de producción
production_model = mlflow.sklearn.load_model("models:/churn-classifier/Production")
```

---

## Reentrenamiento Automático

### Triggers para Reentrenamiento

| Trigger | Cuándo | Cómo Detectar |
|---------|--------|---------------|
| **Schedule** | Cada semana/mes | Cron, Airflow schedule |
| **Data drift** | Distribución de inputs cambia | KS test, Evidently AI |
| **Performance drop** | Accuracy baja en producción | Monitoring de métricas |
| **Nuevos datos** | Llegan datos etiquetados nuevos | Trigger por evento (S3, DB) |
| **Manual** | El equipo decide | Botón en UI, API call |

### Pipeline de Reentrenamiento

```python
# retrain_pipeline.py
import mlflow
from datetime import datetime, timedelta

def should_retrain() -> tuple[bool, str]:
    """Decidir si hay que reentrenar."""
    reasons = []

    # Check 1: Tiempo desde último entrenamiento
    client = mlflow.tracking.MlflowClient()
    prod_model = client.get_latest_versions("my-model", stages=["Production"])[0]
    last_trained = datetime.fromtimestamp(prod_model.creation_timestamp / 1000)
    if datetime.utcnow() - last_trained > timedelta(days=30):
        reasons.append("Han pasado más de 30 días desde el último entrenamiento")

    # Check 2: Data drift
    drift_score = check_data_drift()  # Tu función de drift detection
    if drift_score > 0.3:
        reasons.append(f"Data drift detectado: {drift_score:.2f}")

    # Check 3: Performance drop
    current_accuracy = get_production_accuracy()  # De tu monitoring
    baseline_accuracy = float(prod_model.tags.get("accuracy", "0.9"))
    if current_accuracy < baseline_accuracy - 0.05:
        reasons.append(
            f"Accuracy bajó de {baseline_accuracy:.3f} a {current_accuracy:.3f}"
        )

    return len(reasons) > 0, "; ".join(reasons)

def retrain_and_evaluate():
    """Pipeline completo de reentrenamiento."""
    should, reason = should_retrain()

    if not should:
        print("No es necesario reentrenar")
        return

    print(f"Reentrenando: {reason}")

    # 1. Fetch datos nuevos
    X_train, X_test, y_train, y_test = fetch_latest_data()

    # 2. Entrenar nuevo modelo
    with mlflow.start_run(run_name=f"retrain-{datetime.utcnow().isoformat()}"):
        new_model = train(X_train, y_train)
        new_accuracy = evaluate(new_model, X_test, y_test)

        mlflow.log_metric("accuracy", new_accuracy)
        mlflow.log_param("retrain_reason", reason)

        # 3. Comparar con modelo en producción
        prod_model = mlflow.sklearn.load_model("models:/my-model/Production")
        prod_accuracy = evaluate(prod_model, X_test, y_test)

        print(f"Nuevo modelo: {new_accuracy:.4f} vs Producción: {prod_accuracy:.4f}")

        # 4. Deploy solo si es mejor
        if new_accuracy > prod_accuracy:
            mlflow.sklearn.log_model(
                new_model, "model",
                registered_model_name="my-model",
            )
            print("Nuevo modelo registrado. Listo para promover a producción.")
        else:
            print("Modelo nuevo no supera al actual. No se deploya.")
```

### Shadow Deployment y Canary

```
Shadow Deployment:
  Request → Modelo A (producción) → Respuesta al usuario
         → Modelo B (shadow)     → Log para comparar (no se envía al usuario)

  Período: 1-2 semanas
  Objetivo: validar Modelo B sin riesgo
  Decidir: si B es consistentemente mejor → promover

Canary Deployment:
  Request → [90%] Modelo A (producción)  → Respuesta
         → [10%] Modelo B (canary)       → Respuesta

  Período: gradual (10% → 25% → 50% → 100%)
  Objetivo: detectar problemas con tráfico real limitado
  Rollback: si B falla → volver a 100% A
```

---

## Infraestructura

### Tabla de Decisión

| Escala | Infra Recomendada | Coste Mensual | Complejidad |
|--------|------------------|---------------|-------------|
| **POC / MVP** | Laptop + Docker | $0 | Baja |
| **1-2 modelos, bajo tráfico** | Single server (EC2/VM) + Docker | $50-200 | Baja |
| **2-5 modelos, tráfico medio** | ECS/Cloud Run + managed DB | $200-1000 | Media |
| **5+ modelos, alto tráfico** | Kubernetes + Seldon/KServe | $1000-5000 | Alta |
| **Enterprise** | SageMaker/Vertex AI (managed) | $2000+ | Media (managed) |
| **Edge / IoT** | ONNX/TFLite en dispositivos | Variable | Alta |

### Diagrama de Infraestructura Típica

```
Consultoría - Setup Típico:

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   GitHub     │────▶│GitHub Actions│────▶│  Docker Hub  │
│   (código)   │     │  (CI/CD)     │     │  / ECR       │
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
                     │  │CloudWatch│    │  S3 (modelos,   │  │
                     │  │/Grafana │     │  datos, logs)    │  │
                     │  └─────────┘     └─────────────────┘  │
                     └────────────────────────────────────────┘
```

---

## Lo Que Realmente Necesitas para Consultoría

La mayoría de proyectos de consultoría AI no necesitan un stack MLOps complejo. Este es el stack pragmático que cubre el 90% de los casos:

### El Stack Mínimo Viable

| Necesidad | Herramienta | Por Qué |
|-----------|-------------|---------|
| **Código** | Git + GitHub | Estándar universal, sin discusión |
| **Experimentos** | MLflow | Gratis, open source, experiment tracking + model registry |
| **Empaquetado** | Docker | Reproducibilidad, "funciona en cualquier máquina" |
| **Servir modelo** | FastAPI | Rápido, tipado, documentación automática |
| **CI/CD** | GitHub Actions | Integrado con GitHub, gratis para repos públicos |
| **Monitoring** | CloudWatch + Prometheus | Métricas de infra + custom metrics |

### Cuándo Añadir Más

```
"¿Necesito X?"

Feature Store → ¿Tienes 5+ modelos compartiendo features? → Si no, NO.
Kubernetes   → ¿Tienes 10+ microservicios? → Si no, NO.
Kubeflow     → ¿Tienes 20+ pipelines ML? → Si no, NO.
Airflow      → ¿Tienes 5+ pipelines con dependencias? → Si no, scripts + cron.
Kafka        → ¿Necesitas procesar 10k+ eventos/seg? → Si no, SQS o batch.
Terraform    → ¿Tu infra tiene 20+ recursos cloud? → Si no, consola/CLI.
```

### Checklist de MLOps para Consultoría

```
Fase 1: Proyecto empieza
  [x] Repo Git
  [x] requirements.txt / pyproject.toml
  [x] README con instrucciones de setup
  [x] MLflow local para tracking
  [ ] (Opcional) DVC si datos son grandes

Fase 2: Modelo listo
  [x] Modelo registrado en MLflow
  [x] Métricas baseline documentadas
  [x] Tests básicos del modelo
  [x] Dockerfile funcional
  [ ] (Opcional) GitHub Actions para tests

Fase 3: En producción
  [x] FastAPI endpoint deployado
  [x] CI/CD configurado (build + test + deploy)
  [x] Monitoring de latencia y errores
  [x] Health check endpoint
  [ ] (Opcional) Drift detection

Fase 4: Mantenimiento
  [x] Pipeline de reentrenamiento (al menos manual)
  [x] Alertas configuradas
  [x] Documentación para el equipo del cliente
  [ ] (Opcional) Reentrenamiento automático
  [ ] (Opcional) A/B testing
```

---

> **Resumen para llevar:** MLOps no es instalar todas las herramientas del ecosistema. Es tener un proceso reproducible y mantenible para llevar modelos a producción. Para consultoría, Git + MLflow + Docker + FastAPI + GitHub Actions te dan el 90% de lo que necesitas. Añade complejidad solo cuando el problema lo justifique.

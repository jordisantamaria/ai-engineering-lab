# Deployment de Modelos ML

> **Realidad incómoda:** El 87% de los modelos ML nunca llegan a producción.
> El deployment es donde mueren la mayoría de proyectos. Saber entrenar un modelo es solo la mitad del camino.

---

## Tabla de Contenidos

- [De Notebook a Producción](#de-notebook-a-producción)
- [Servir Modelos con FastAPI](#servir-modelos-con-fastapi)
- [Optimización de Inference](#optimización-de-inference)
- [Docker para ML](#docker-para-ml)
- [Deployment en Cloud](#deployment-en-cloud)
- [Serverless ML](#serverless-ml)
- [Streaming vs Batch Inference](#streaming-vs-batch-inference)
- [CI/CD para ML](#cicd-para-ml)
- [Monitoring en Producción](#monitoring-en-producción)
- [Tips de Consultoría](#tips-de-consultoría)

---

## Por Qué Deployment Es Donde Fallan los Proyectos ML

El gap entre "funciona en mi notebook" y "funciona en producción" es enorme:

| Notebook | Producción |
|----------|-----------|
| Código lineal, sin estructura | Módulos, clases, funciones reutilizables |
| Datos estáticos (CSV local) | Datos dinámicos (APIs, DBs, streams) |
| Sin manejo de errores | Errores gestionados, logging, retries |
| Una ejecución manual | Miles de requests concurrentes 24/7 |
| "Funciona en mi máquina" | Funciona en cualquier máquina (Docker) |
| Sin tests | Unit tests, integration tests, load tests |
| Un usuario (tú) | Múltiples usuarios, SLAs, latencia |

**Punto clave:** Un modelo con 95% de accuracy que no está en producción tiene valor cero para el negocio. Un modelo con 85% de accuracy en producción genera valor real.

---

## De Notebook a Producción

### El Gap

El notebook es perfecto para exploración. Pero el código de un notebook típico tiene:

- Variables globales por todos lados
- Celdas que dependen del orden de ejecución
- Código duplicado
- Sin manejo de errores
- Dependencias no documentadas

### Refactorizar a Módulos Python

**Estructura de proyecto recomendada:**

```
ml-project/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Carga de datos
│   │   └── preprocessing.py   # Limpieza, transformaciones
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py     # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py           # Entrenamiento
│   │   ├── predict.py         # Inferencia
│   │   └── evaluate.py        # Métricas
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Configuración
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI app
│   ├── schemas.py             # Pydantic models
│   └── dependencies.py        # Dependencias (modelo cargado)
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── models/                    # Modelos serializados (.pkl, .onnx)
├── notebooks/                 # Notebooks de exploración
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Testing de Modelos ML

```python
# tests/test_model.py
import pytest
import numpy as np
from src.models.predict import ModelPredictor

class TestModelPredictor:
    """Tests unitarios para el modelo."""

    @pytest.fixture
    def predictor(self):
        return ModelPredictor(model_path="models/model_v1.pkl")

    def test_prediction_shape(self, predictor):
        """La predicción debe tener la forma correcta."""
        input_data = np.random.rand(1, 10)
        prediction = predictor.predict(input_data)
        assert prediction.shape == (1,)

    def test_prediction_range(self, predictor):
        """Las predicciones deben estar en rango válido."""
        input_data = np.random.rand(5, 10)
        predictions = predictor.predict(input_data)
        assert all(0 <= p <= 1 for p in predictions)

    def test_prediction_deterministic(self, predictor):
        """Misma entrada → misma salida."""
        input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0,
                                6.0, 7.0, 8.0, 9.0, 10.0]])
        pred1 = predictor.predict(input_data)
        pred2 = predictor.predict(input_data)
        np.testing.assert_array_equal(pred1, pred2)

    def test_handles_missing_values(self, predictor):
        """El modelo debe manejar NaN correctamente."""
        input_data = np.array([[1.0, np.nan, 3.0, 4.0, 5.0,
                                6.0, 7.0, 8.0, 9.0, 10.0]])
        # No debe lanzar excepción
        prediction = predictor.predict(input_data)
        assert not np.isnan(prediction).any()
```

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.mark.asyncio
async def test_predict_endpoint():
    """Test del endpoint de predicción."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={
            "features": [1.0, 2.0, 3.0, 4.0, 5.0,
                         6.0, 7.0, 8.0, 9.0, 10.0]
        })
    assert response.status_code == 200
    assert "prediction" in response.json()

@pytest.mark.asyncio
async def test_predict_invalid_input():
    """Input inválido debe retornar 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={
            "features": "not a list"
        })
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_health_check():
    """Health check debe retornar 200."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

---

## Servir Modelos con FastAPI

FastAPI es la opción preferida para servir modelos ML en Python: rápido, tipado, documentación automática con OpenAPI/Swagger.

### Estructura Completa

```python
# api/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class PredictionRequest(BaseModel):
    """Schema de entrada para predicciones."""
    features: list[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        description="Vector de 10 features numéricas"
    )
    model_version: Optional[str] = Field(
        default="latest",
        description="Versión del modelo a usar"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("Todas las features deben ser numéricas")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [1.0, 2.0, 3.0, 4.0, 5.0,
                                 6.0, 7.0, 8.0, 9.0, 10.0],
                    "model_version": "v1"
                }
            ]
        }
    }

class PredictionResponse(BaseModel):
    """Schema de salida."""
    prediction: float
    confidence: float
    model_version: str

class BatchPredictionRequest(BaseModel):
    """Schema para predicciones en batch."""
    instances: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de vectores de features"
    )

class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
```

```python
# api/main.py
import time
import logging
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from api.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse
)

logger = logging.getLogger(__name__)

# Estado global de la app
class AppState:
    model = None
    model_version = "unknown"
    start_time = None

state = AppState()

def load_model():
    """Carga el modelo al iniciar la aplicación."""
    import joblib
    try:
        state.model = joblib.load("models/model_latest.pkl")
        state.model_version = "v1.2.0"
        logger.info(f"Modelo {state.model_version} cargado correctamente")
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: cargar modelo al startup, cleanup al shutdown."""
    # Startup
    state.start_time = time.time()
    load_model()
    logger.info("Aplicación iniciada")
    yield
    # Shutdown
    logger.info("Aplicación apagándose")

app = FastAPI(
    title="ML Prediction API",
    description="API para servir predicciones del modelo XYZ",
    version="1.0.0",
    lifespan=lifespan,
)

# ---- Endpoints ----

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check: verificar que el modelo está cargado."""
    return HealthResponse(
        status="healthy" if state.model else "unhealthy",
        model_loaded=state.model is not None,
        model_version=state.model_version,
        uptime_seconds=time.time() - state.start_time,
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predicción individual."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = state.model.predict(features)[0]

        # Si el modelo soporta predict_proba
        confidence = 0.0
        if hasattr(state.model, "predict_proba"):
            proba = state.model.predict_proba(features)[0]
            confidence = float(max(proba))

        return PredictionResponse(
            prediction=float(prediction),
            confidence=confidence,
            model_version=state.model_version,
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predicciones en batch (hasta 100 instancias)."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        features = np.array(request.instances)
        predictions = state.model.predict(features)

        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            model_version=state.model_version,
        )
    except Exception as e:
        logger.error(f"Error en batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Async vs Sync para Inference

| Aspecto | Sync | Async |
|---------|------|-------|
| Inference CPU-bound (sklearn, XGBoost) | Funciona bien con workers | `run_in_executor` necesario |
| Inference GPU (PyTorch, TF) | Bloquea el event loop | Usar `asyncio.to_thread` |
| I/O (cargar datos, llamar APIs) | Bloquea el thread | Nativo async, ideal |
| Recomendación general | Usar Gunicorn con workers | Para I/O heavy |

```python
# Para modelos CPU-bound, usar run_in_executor
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict")
async def predict(request: PredictionRequest):
    loop = asyncio.get_event_loop()
    # Ejecutar inference en thread pool para no bloquear event loop
    prediction = await loop.run_in_executor(
        executor,
        lambda: state.model.predict(np.array(request.features).reshape(1, -1))
    )
    return {"prediction": float(prediction[0])}
```

**Regla general:** Si la inference tarda menos de 100ms, sync con Gunicorn workers está bien. Si tarda más, usar async con thread pool.

---

## Optimización de Inference

### ONNX Runtime

ONNX (Open Neural Network Exchange) es un formato intermedio que permite optimizar la inference independientemente del framework de entrenamiento.

```python
# Convertir PyTorch a ONNX
import torch
import torch.onnx

model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Input de ejemplo para trazar el grafo
dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
print("Modelo exportado a ONNX")
```

```python
# Inference con ONNX Runtime
import onnxruntime as ort
import numpy as np

# Crear sesión (solo una vez al startup)
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Inference
input_data = np.random.rand(1, 10).astype(np.float32)
outputs = session.run(None, {"input": input_data})
prediction = outputs[0]
```

### TorchScript

```python
# Opción 1: torch.jit.trace (para modelos sin control flow dinámico)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")

# Opción 2: torch.jit.script (para modelos con if/for dinámicos)
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Cargar e inferir
loaded_model = torch.jit.load("model_traced.pt")
output = loaded_model(torch.randn(1, 10))
```

### Quantization

```python
import torch.quantization

# Dynamic Quantization (más fácil, menos speedup)
# Funciona bien para modelos con muchas capas lineales (NLP)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)

# El modelo ahora es ~2-4x más pequeño y ~1.5-2x más rápido
torch.save(quantized_model.state_dict(), "model_quantized.pth")
```

| Tipo | Cómo funciona | Speedup | Pérdida de accuracy |
|------|--------------|---------|-------------------|
| Dynamic | Pesos cuantizados, activaciones en runtime | 1.5-2x | Mínima |
| Static | Pesos + activaciones cuantizados con calibración | 2-3x | Baja |
| QAT (Quantization-Aware Training) | Simulación de quantization durante training | 2-4x | Muy baja |

### Tabla Resumen de Optimización

| Técnica | Speedup Típico | Complejidad | Mejor Para |
|---------|---------------|-------------|-----------|
| ONNX Runtime | 1.5-3x | Baja | Cualquier modelo, producción |
| TorchScript | 1.2-2x | Baja | Modelos PyTorch |
| Dynamic Quantization | 1.5-2x | Baja | NLP, modelos lineales |
| Static Quantization | 2-3x | Media | CV, modelos convolucionales |
| Batching de requests | 2-5x (throughput) | Media | Alto tráfico |
| Model pruning | 1.5-3x | Alta | Modelos grandes |
| Distillation | 2-10x | Alta | Cuando puedes reentrenar |
| GPU inference | 5-50x | Media | Modelos deep learning |
| TensorRT (NVIDIA) | 2-6x vs PyTorch | Alta | GPU NVIDIA en producción |

### Batching de Requests

```python
import asyncio
from collections import deque

class BatchPredictor:
    """Acumula requests y las procesa en batch para mayor throughput."""

    def __init__(self, model, batch_size=32, max_wait_ms=50):
        self.model = model
        self.batch_size = batch_size
        self.max_wait = max_wait_ms / 1000
        self.queue = deque()

    async def predict(self, features):
        future = asyncio.get_event_loop().create_future()
        self.queue.append((features, future))

        if len(self.queue) >= self.batch_size:
            await self._process_batch()
        else:
            # Esperar un poco por más requests
            await asyncio.sleep(self.max_wait)
            if not future.done():
                await self._process_batch()

        return await future

    async def _process_batch(self):
        batch = []
        futures = []
        while self.queue and len(batch) < self.batch_size:
            features, future = self.queue.popleft()
            if not future.done():
                batch.append(features)
                futures.append(future)

        if batch:
            import numpy as np
            batch_array = np.array(batch)
            predictions = self.model.predict(batch_array)
            for pred, future in zip(predictions, futures):
                future.set_result(float(pred))
```

### Model Warmup

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar modelo
    load_model()

    # Warmup: hacer predicciones dummy para cargar todo en memoria/cache
    logger.info("Warming up modelo...")
    dummy = np.random.rand(10, 10).astype(np.float32)
    for _ in range(5):
        state.model.predict(dummy)
    logger.info("Warmup completado")

    yield
```

---

## Docker para ML

### Dockerfile Multi-Stage

```dockerfile
# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Instalar dependencias de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (capa cacheada si no cambian)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copiar solo las dependencias instaladas
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copiar código de la aplicación
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose para Desarrollo

```yaml
# docker-compose.yml
version: "3.8"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models    # Montar modelos para desarrollo
    environment:
      - MODEL_PATH=/app/models/model_latest.pkl
      - LOG_LEVEL=DEBUG
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Opcional: para caching de predicciones
  api-with-cache:
    build: .
    ports:
      - "8001:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - CACHE_TTL=3600
    depends_on:
      - redis

volumes:
  redis_data:
```

### Tips para Reducir Tamaño de Imagen

| Técnica | Ahorro Típico | Ejemplo |
|---------|--------------|---------|
| Multi-stage build | 40-60% | Separar builder de runtime |
| `python:3.11-slim` en vez de `python:3.11` | ~800MB | Base image mínima |
| `--no-cache-dir` en pip | 10-20% | `pip install --no-cache-dir` |
| `.dockerignore` | Variable | Excluir notebooks, datos, .git |
| Instalar solo lo necesario | 20-50% | No incluir dev dependencies |
| Limpiar apt cache | 50-100MB | `rm -rf /var/lib/apt/lists/*` |

**Ejemplo `.dockerignore`:**

```
notebooks/
data/
*.ipynb
.git/
__pycache__/
*.pyc
.env
tests/
docs/
```

### GPU en Docker

```bash
# Instalar nvidia-container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Ejecutar con acceso a GPU
docker run --gpus all -p 8000:8000 ml-api:latest

# Docker Compose con GPU
```

```yaml
# docker-compose.gpu.yml
services:
  api:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
```

---

## Deployment en Cloud

### AWS

**ECS/Fargate (Containers serverless):**

```bash
# 1. Crear repositorio ECR
aws ecr create-repository --repository-name ml-api

# 2. Build y push
docker build -t ml-api .
docker tag ml-api:latest <account>.dkr.ecr.<region>.amazonaws.com/ml-api:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/ml-api:latest

# 3. Crear task definition y servicio en ECS
# (usar la consola o Terraform para configurar)
```

**SageMaker Endpoints (Managed):**

```python
import sagemaker
from sagemaker.sklearn import SKLearnModel

model = SKLearnModel(
    model_data="s3://bucket/model.tar.gz",
    role="arn:aws:iam::role/SageMakerRole",
    framework_version="1.2-1",
    entry_point="inference.py",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="my-ml-endpoint",
)

# Auto-scaling
client = boto3.client("application-autoscaling")
client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/{endpoint_name}/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=10,
)
```

**Lambda (para modelos ligeros):**

```python
# lambda_handler.py
import json
import onnxruntime as ort
import numpy as np

# Cargar modelo fuera del handler (reutilizado entre invocaciones)
session = ort.InferenceSession("/opt/model.onnx")

def handler(event, context):
    body = json.loads(event["body"])
    features = np.array(body["features"], dtype=np.float32).reshape(1, -1)

    outputs = session.run(None, {"input": features})
    prediction = float(outputs[0][0])

    return {
        "statusCode": 200,
        "body": json.dumps({"prediction": prediction}),
    }
```

### Tabla Comparativa de Servicios Cloud

| Servicio | Coste Mensual (aprox) | Auto-scaling | Complejidad | GPU | Mejor Para |
|----------|----------------------|-------------|-------------|-----|-----------|
| **AWS ECS/Fargate** | $30-200 | Si | Media | No (Fargate) | APIs containerizadas |
| **AWS SageMaker** | $50-500+ | Si (managed) | Baja | Si | Equipos ML enterprise |
| **AWS Lambda** | $0-50 | Automático | Baja | No | Modelos ligeros, tráfico variable |
| **GCP Cloud Run** | $0-150 | Si | Baja | Si (limitado) | APIs containerizadas |
| **GCP Vertex AI** | $50-500+ | Si | Media | Si | Equipos ML en GCP |
| **Azure Container Apps** | $30-200 | Si | Media | Si | APIs containerizadas en Azure |
| **Azure ML** | $50-500+ | Si | Media | Si | Enterprise en Azure |

---

## Serverless ML

### Cuándo Tiene Sentido

- Modelos pequenos (< 250MB empaquetado)
- Inference rápida (< 10 segundos)
- Tráfico variable (picos y valles)
- Budget limitado (pagas solo por uso)

### El Problema del Cold Start

| Factor | Impacto en Cold Start |
|--------|---------------------|
| Tamaño del paquete | Mayor paquete = más tiempo |
| Runtime (Python) | ~500ms-1s base |
| Carga del modelo | La parte más lenta (1-10s) |
| VPC config | +2-5s si está en VPC |

**Soluciones:**

```python
# 1. Provisioned Concurrency (AWS Lambda)
# Mantener N instancias siempre calientes
# Coste: ~$15/mes por instancia

# 2. Modelo en formato ONNX (más rápido de cargar)
# 3. Modelo en EFS (no descargarlo cada vez)
# 4. Ping periódico (CloudWatch Events cada 5 min)
# 5. Usar layers para dependencias (se cachean)
```

### Lambda + ONNX para Computer Vision

```python
# Estructura del Lambda Layer:
# layer/
# ├── python/
# │   ├── onnxruntime/
# │   ├── numpy/
# │   └── pillow/

# handler.py
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from io import BytesIO
import base64

session = ort.InferenceSession("/opt/model.onnx")

def handler(event, context):
    # Decodificar imagen base64
    img_bytes = base64.b64decode(event["body"])
    img = Image.open(BytesIO(img_bytes)).resize((224, 224))

    # Preprocessing
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_array = np.expand_dims(img_array, 0)         # Batch dim

    # Inference
    outputs = session.run(None, {"input": img_array})
    class_id = int(np.argmax(outputs[0]))

    return {
        "statusCode": 200,
        "body": json.dumps({"class_id": class_id}),
    }
```

---

## Streaming vs Batch Inference

### Tabla de Decisión

| Modo | Latencia | Caso de Uso | Herramientas | Coste |
|------|---------|-------------|-------------|-------|
| **Batch** | Minutos-horas | Scoring nocturno, reports | Airflow, cron, Spark | Bajo |
| **Real-time** | Milisegundos | API de predicción, chatbots | FastAPI, SageMaker | Medio-Alto |
| **Near real-time** | Segundos | Fraud detection, recommendations | Kafka, SQS + consumer | Medio |
| **Streaming** | Milisegundos (continuo) | Sensor data, logs en vivo | Kafka Streams, Flink | Alto |

### Batch Inference

```python
# batch_predict.py - Ejecutar con Airflow o cron
import pandas as pd
import joblib
from datetime import datetime

def run_batch_predictions():
    # Cargar datos nuevos
    df = pd.read_sql("SELECT * FROM customers WHERE scored_at IS NULL", engine)

    # Cargar modelo
    model = joblib.load("models/model_latest.pkl")

    # Predecir
    features = df[feature_columns].values
    df["prediction"] = model.predict(features)
    df["scored_at"] = datetime.utcnow()

    # Guardar resultados
    df[["customer_id", "prediction", "scored_at"]].to_sql(
        "predictions", engine, if_exists="append", index=False
    )
    print(f"Procesados {len(df)} registros")

if __name__ == "__main__":
    run_batch_predictions()
```

### Near Real-Time con Message Queue

```python
# consumer.py - Consumidor de SQS/Kafka
import json
import boto3
import numpy as np
import joblib

sqs = boto3.client("sqs")
model = joblib.load("models/model_latest.pkl")
QUEUE_URL = "https://sqs.region.amazonaws.com/account/ml-requests"

def process_messages():
    while True:
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,  # Batch de hasta 10
            WaitTimeSeconds=20,       # Long polling
        )

        messages = response.get("Messages", [])
        if not messages:
            continue

        # Procesar en batch
        features_batch = []
        for msg in messages:
            body = json.loads(msg["Body"])
            features_batch.append(body["features"])

        predictions = model.predict(np.array(features_batch))

        # Guardar resultados y eliminar mensajes
        for msg, pred in zip(messages, predictions):
            body = json.loads(msg["Body"])
            save_prediction(body["request_id"], float(pred))
            sqs.delete_message(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=msg["ReceiptHandle"],
            )

if __name__ == "__main__":
    process_messages()
```

### Cuándo Usar Cada Modo

```
¿Necesitas respuesta inmediata (< 1s)?
├── Sí → Real-time API (FastAPI)
└── No
    ├── ¿Datos llegan continuamente?
    │   ├── Sí → ¿Necesitas resultado en < 30s?
    │   │   ├── Sí → Streaming (Kafka Streams)
    │   │   └── No → Near real-time (SQS + consumer)
    │   └── No → Batch (Airflow/cron)
    └── ¿Procesamiento periódico?
        └── Sí → Batch (Airflow/cron)
```

---

## CI/CD para ML

### GitHub Actions Pipeline

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Run linting
        run: |
          ruff check src/ api/ tests/

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov=api

      - name: Run model validation
        run: python scripts/validate_model.py

  build-and-push:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: eu-west-1

      - name: Login to ECR
        id: ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push Docker image
        run: |
          docker build -t ${{ steps.ecr.outputs.registry }}/ml-api:${{ github.sha }} .
          docker push ${{ steps.ecr.outputs.registry }}/ml-api:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster ml-cluster \
            --service ml-api-service \
            --force-new-deployment
```

### Model Validation antes de Deploy

```python
# scripts/validate_model.py
"""
Validar que el nuevo modelo es al menos tan bueno como el de producción.
Se ejecuta en CI antes de deploy.
"""
import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Cargar test set estándar
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Cargar modelo nuevo (candidato)
new_model = joblib.load("models/model_latest.pkl")
new_preds = new_model.predict(X_test)

# Métricas del nuevo modelo
new_accuracy = accuracy_score(y_test, new_preds)
new_f1 = f1_score(y_test, new_preds, average="weighted")

# Cargar métricas del modelo en producción
with open("models/production_metrics.json") as f:
    prod_metrics = json.load(f)

print(f"Modelo nuevo    - Accuracy: {new_accuracy:.4f}, F1: {new_f1:.4f}")
print(f"Modelo en prod  - Accuracy: {prod_metrics['accuracy']:.4f}, F1: {prod_metrics['f1']:.4f}")

# Validar: el nuevo modelo no puede ser peor
TOLERANCE = 0.01  # 1% de tolerancia
if new_f1 < prod_metrics["f1"] - TOLERANCE:
    print("FALLO: El nuevo modelo es significativamente peor que producción")
    exit(1)

print("OK: Modelo validado, listo para deploy")
```

---

## Monitoring en Producción

### Data Drift

El data drift ocurre cuando la distribución de los datos de entrada cambia respecto a los datos de entrenamiento. El modelo fue entrenado con cierta distribución y puede no funcionar bien con datos diferentes.

```python
# monitoring/drift_detector.py
import numpy as np
from scipy import stats

class DriftDetector:
    """Detectar drift comparando distribuciones."""

    def __init__(self, reference_data: np.ndarray, threshold: float = 0.05):
        self.reference = reference_data
        self.threshold = threshold

    def check_drift(self, current_data: np.ndarray) -> dict:
        """Kolmogorov-Smirnov test por feature."""
        results = {}
        for i in range(self.reference.shape[1]):
            stat, p_value = stats.ks_2samp(
                self.reference[:, i],
                current_data[:, i],
            )
            results[f"feature_{i}"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "drift_detected": p_value < self.threshold,
            }

        n_drifted = sum(1 for r in results.values() if r["drift_detected"])
        results["summary"] = {
            "total_features": self.reference.shape[1],
            "features_with_drift": n_drifted,
            "overall_drift": n_drifted > self.reference.shape[1] * 0.3,
        }
        return results
```

### Métricas a Monitorizar

| Categoría | Métrica | Alerta Si |
|-----------|---------|----------|
| **Latencia** | P50, P95, P99 response time | P95 > 500ms |
| **Throughput** | Requests/segundo | Caída > 50% |
| **Errores** | Error rate (4xx, 5xx) | > 1% |
| **Modelo** | Prediction distribution | Shift significativo |
| **Modelo** | Confidence score distribution | Media baja > 10% |
| **Datos** | Input feature distributions | KS test p < 0.05 |
| **Datos** | Missing values rate | Aumento > 5% |
| **Infra** | CPU/Memory usage | > 80% sostenido |
| **Negocio** | Conversion rate, revenue impact | Caída > X% |

### Herramientas de Monitoring

| Herramienta | Tipo | Coste | Mejor Para |
|-------------|------|-------|-----------|
| **Evidently AI** | Data/model drift | Open source | Dashboards de drift |
| **WhyLabs** | ML monitoring | Free tier + paid | Monitoring continuo |
| **Prometheus + Grafana** | Métricas infra | Open source | Latencia, throughput |
| **CloudWatch** (AWS) | Infra + custom | Pay per use | Si ya estás en AWS |
| **Arize AI** | ML observability | Free tier + paid | Debugging de modelos |

### Cuándo Reentrenar

```
Checklist para decidir reentrenamiento:
- [ ] Data drift detectado en > 30% de features
- [ ] Accuracy en producción baja > 5% vs baseline
- [ ] Han pasado > 3 meses desde último entrenamiento
- [ ] Hay nuevos datos etiquetados disponibles
- [ ] El negocio reporta degradación en resultados

Si >= 2 de estos son true → Reentrenar
```

---

## Tips de Consultoría

### Enfoque Pragmático de Deployment

```
Fase 1 (Semana 1-2): MVP
├── Modelo entrenado en notebook
├── Script batch que lee CSV → genera predicciones → guarda CSV
├── Cliente valida resultados manualmente
└── Coste: $0

Fase 2 (Semana 3-4): API
├── FastAPI + Docker
├── Endpoint simple de predicción
├── Deploy en un servidor (EC2, Cloud Run)
├── Documentación con Swagger
└── Coste: $30-100/mes

Fase 3 (Mes 2-3): Producción
├── CI/CD con GitHub Actions
├── Monitoring básico (latencia, errores)
├── Auto-scaling si es necesario
├── Data drift detection
└── Coste: $100-500/mes

Fase 4 (Mes 4+): Escalar
├── MLOps pipeline completo
├── Reentrenamiento automático
├── A/B testing
├── Multi-model serving
└── Coste: $500+/mes
```

### Reglas de Oro

1. **No sobre-engineerear la primera versión.** Un CSV con predicciones que el cliente puede revisar vale más que una API perfecta que nadie usa.

2. **FastAPI te regala documentación.** El Swagger UI generado automáticamente (disponible en `/docs`) es perfecto para que el cliente entienda y pruebe la API sin documentación adicional.

3. **Docker desde el día 1.** Aunque el deploy inicial sea simple, tener un Dockerfile elimina el "funciona en mi máquina" desde el principio.

4. **Monitoring > más accuracy.** Saber cuándo tu modelo falla es más valioso que mejorar la accuracy un 2%.

5. **Empezar con batch.** Si el cliente no necesita respuesta en milisegundos, un proceso batch nocturno es 10x más fácil de mantener que una API.

---

> **Resumen para llevar:** Deployment no es un paso final, es una disciplina. El camino más seguro es iterativo: batch primero, API después, escalar cuando se justifique. FastAPI + Docker + GitHub Actions te dan el 80% de lo que necesitas para el 90% de los proyectos de consultoría.

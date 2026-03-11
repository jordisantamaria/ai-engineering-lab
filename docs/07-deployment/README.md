# ML Model Deployment

> **Uncomfortable reality:** 87% of ML models never make it to production.
> Deployment is where most projects die. Knowing how to train a model is only half the battle.

---

## Table of Contents

- [From Notebook to Production](#from-notebook-to-production)
- [Serving Models with FastAPI](#serving-models-with-fastapi)
- [Inference Optimization](#inference-optimization)
- [Docker for ML](#docker-for-ml)
- [Cloud Deployment](#cloud-deployment)
- [Serverless ML](#serverless-ml)
- [Streaming vs Batch Inference](#streaming-vs-batch-inference)
- [CI/CD for ML](#cicd-for-ml)
- [Production Monitoring](#production-monitoring)
- [Consulting Tips](#consulting-tips)

---

## Why Deployment Is Where ML Projects Fail

The gap between "works in my notebook" and "works in production" is enormous:

| Notebook | Production |
|----------|-----------|
| Linear code, no structure | Modules, classes, reusable functions |
| Static data (local CSV) | Dynamic data (APIs, DBs, streams) |
| No error handling | Managed errors, logging, retries |
| One manual execution | Thousands of concurrent requests 24/7 |
| "Works on my machine" | Works on any machine (Docker) |
| No tests | Unit tests, integration tests, load tests |
| One user (you) | Multiple users, SLAs, latency |

**Key point:** A model with 95% accuracy that's not in production has zero value to the business. A model with 85% accuracy in production generates real value.

---

## From Notebook to Production

### The Gap

The notebook is perfect for exploration. But typical notebook code has:

- Global variables everywhere
- Cells that depend on execution order
- Duplicated code
- No error handling
- Undocumented dependencies

### Refactor to Python Modules

**Recommended project structure:**

```
ml-project/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading
│   │   └── preprocessing.py   # Cleaning, transformations
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py     # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py           # Training
│   │   ├── predict.py         # Inference
│   │   └── evaluate.py        # Metrics
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Configuration
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI app
│   ├── schemas.py             # Pydantic models
│   └── dependencies.py        # Dependencies (loaded model)
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── models/                    # Serialized models (.pkl, .onnx)
├── notebooks/                 # Exploration notebooks
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

### ML Model Testing

```python
# tests/test_model.py
import pytest
import numpy as np
from src.models.predict import ModelPredictor

class TestModelPredictor:
    """Unit tests for the model."""

    @pytest.fixture
    def predictor(self):
        return ModelPredictor(model_path="models/model_v1.pkl")

    def test_prediction_shape(self, predictor):
        """Prediction must have the correct shape."""
        input_data = np.random.rand(1, 10)
        prediction = predictor.predict(input_data)
        assert prediction.shape == (1,)

    def test_prediction_range(self, predictor):
        """Predictions must be in a valid range."""
        input_data = np.random.rand(5, 10)
        predictions = predictor.predict(input_data)
        assert all(0 <= p <= 1 for p in predictions)

    def test_prediction_deterministic(self, predictor):
        """Same input -> same output."""
        input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0,
                                6.0, 7.0, 8.0, 9.0, 10.0]])
        pred1 = predictor.predict(input_data)
        pred2 = predictor.predict(input_data)
        np.testing.assert_array_equal(pred1, pred2)

    def test_handles_missing_values(self, predictor):
        """The model must handle NaN correctly."""
        input_data = np.array([[1.0, np.nan, 3.0, 4.0, 5.0,
                                6.0, 7.0, 8.0, 9.0, 10.0]])
        # Should not raise an exception
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
    """Test the prediction endpoint."""
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
    """Invalid input should return 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json={
            "features": "not a list"
        })
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_health_check():
    """Health check should return 200."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

---

## Serving Models with FastAPI

FastAPI is the preferred option for serving ML models in Python: fast, typed, automatic documentation with OpenAPI/Swagger.

### Complete Structure

```python
# api/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class PredictionRequest(BaseModel):
    """Input schema for predictions."""
    features: list[float] = Field(
        ...,
        min_length=10,
        max_length=10,
        description="Vector of 10 numeric features"
    )
    model_version: Optional[str] = Field(
        default="latest",
        description="Model version to use"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("All features must be numeric")
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
    """Output schema."""
    prediction: float
    confidence: float
    model_version: str

class BatchPredictionRequest(BaseModel):
    """Schema for batch predictions."""
    instances: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of feature vectors"
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

# App global state
class AppState:
    model = None
    model_version = "unknown"
    start_time = None

state = AppState()

def load_model():
    """Load the model at application startup."""
    import joblib
    try:
        state.model = joblib.load("models/model_latest.pkl")
        state.model_version = "v1.2.0"
        logger.info(f"Model {state.model_version} loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: load model at startup, cleanup at shutdown."""
    # Startup
    state.start_time = time.time()
    load_model()
    logger.info("Application started")
    yield
    # Shutdown
    logger.info("Application shutting down")

app = FastAPI(
    title="ML Prediction API",
    description="API for serving predictions from model XYZ",
    version="1.0.0",
    lifespan=lifespan,
)

# ---- Endpoints ----

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check: verify the model is loaded."""
    return HealthResponse(
        status="healthy" if state.model else "unhealthy",
        model_loaded=state.model is not None,
        model_version=state.model_version,
        uptime_seconds=time.time() - state.start_time,
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Individual prediction."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = state.model.predict(features)[0]

        # If the model supports predict_proba
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
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch predictions (up to 100 instances)."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        features = np.array(request.instances)
        predictions = state.model.predict(features)

        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            model_version=state.model_version,
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Async vs Sync for Inference

| Aspect | Sync | Async |
|---------|------|-------|
| CPU-bound inference (sklearn, XGBoost) | Works well with workers | `run_in_executor` needed |
| GPU inference (PyTorch, TF) | Blocks the event loop | Use `asyncio.to_thread` |
| I/O (loading data, calling APIs) | Blocks the thread | Native async, ideal |
| General recommendation | Use Gunicorn with workers | For I/O heavy |

```python
# For CPU-bound models, use run_in_executor
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict")
async def predict(request: PredictionRequest):
    loop = asyncio.get_event_loop()
    # Run inference in thread pool to not block event loop
    prediction = await loop.run_in_executor(
        executor,
        lambda: state.model.predict(np.array(request.features).reshape(1, -1))
    )
    return {"prediction": float(prediction[0])}
```

**General rule:** If inference takes less than 100ms, sync with Gunicorn workers is fine. If it takes longer, use async with thread pool.

---

## Inference Optimization

### ONNX Runtime

ONNX (Open Neural Network Exchange) is an intermediate format that allows optimizing inference independently of the training framework.

```python
# Convert PyTorch to ONNX
import torch
import torch.onnx

model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Example input to trace the graph
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
print("Model exported to ONNX")
```

```python
# Inference with ONNX Runtime
import onnxruntime as ort
import numpy as np

# Create session (only once at startup)
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
# Option 1: torch.jit.trace (for models without dynamic control flow)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")

# Option 2: torch.jit.script (for models with dynamic if/for)
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Load and infer
loaded_model = torch.jit.load("model_traced.pt")
output = loaded_model(torch.randn(1, 10))
```

### Quantization

```python
import torch.quantization

# Dynamic Quantization (easiest, less speedup)
# Works well for models with many linear layers (NLP)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)

# The model is now ~2-4x smaller and ~1.5-2x faster
torch.save(quantized_model.state_dict(), "model_quantized.pth")
```

| Type | How it works | Speedup | Accuracy loss |
|------|--------------|---------|-------------------|
| Dynamic | Quantized weights, activations at runtime | 1.5-2x | Minimal |
| Static | Quantized weights + activations with calibration | 2-3x | Low |
| QAT (Quantization-Aware Training) | Quantization simulation during training | 2-4x | Very low |

### Optimization Summary Table

| Technique | Typical Speedup | Complexity | Best For |
|---------|---------------|-------------|-----------|
| ONNX Runtime | 1.5-3x | Low | Any model, production |
| TorchScript | 1.2-2x | Low | PyTorch models |
| Dynamic Quantization | 1.5-2x | Low | NLP, linear models |
| Static Quantization | 2-3x | Medium | CV, convolutional models |
| Request Batching | 2-5x (throughput) | Medium | High traffic |
| Model pruning | 1.5-3x | High | Large models |
| Distillation | 2-10x | High | When you can retrain |
| GPU inference | 5-50x | Medium | Deep learning models |
| TensorRT (NVIDIA) | 2-6x vs PyTorch | High | NVIDIA GPUs in production |

### Request Batching

```python
import asyncio
from collections import deque

class BatchPredictor:
    """Accumulates requests and processes them in batch for higher throughput."""

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
            # Wait a bit for more requests
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
    # Load model
    load_model()

    # Warmup: make dummy predictions to load everything into memory/cache
    logger.info("Warming up model...")
    dummy = np.random.rand(10, 10).astype(np.float32)
    for _ in range(5):
        state.model.predict(dummy)
    logger.info("Warmup completed")

    yield
```

---

## Docker for ML

### Multi-Stage Dockerfile

```dockerfile
# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cached layer if they don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only installed dependencies
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose for Development

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
      - ./models:/app/models    # Mount models for development
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

  # Optional: for prediction caching
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

### Tips for Reducing Image Size

| Technique | Typical Savings | Example |
|---------|--------------|---------|
| Multi-stage build | 40-60% | Separate builder from runtime |
| `python:3.11-slim` instead of `python:3.11` | ~800MB | Minimal base image |
| `--no-cache-dir` in pip | 10-20% | `pip install --no-cache-dir` |
| `.dockerignore` | Variable | Exclude notebooks, data, .git |
| Install only what's needed | 20-50% | Don't include dev dependencies |
| Clean apt cache | 50-100MB | `rm -rf /var/lib/apt/lists/*` |

**Example `.dockerignore`:**

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

### GPU in Docker

```bash
# Install nvidia-container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Run with GPU access
docker run --gpus all -p 8000:8000 ml-api:latest

# Docker Compose with GPU
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

## Cloud Deployment

### AWS

**ECS/Fargate (Serverless Containers):**

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name ml-api

# 2. Build and push
docker build -t ml-api .
docker tag ml-api:latest <account>.dkr.ecr.<region>.amazonaws.com/ml-api:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/ml-api:latest

# 3. Create task definition and service in ECS
# (use the console or Terraform to configure)
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

**Lambda (for lightweight models):**

```python
# lambda_handler.py
import json
import onnxruntime as ort
import numpy as np

# Load model outside the handler (reused between invocations)
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

### Cloud Services Comparison Table

| Service | Monthly Cost (approx) | Auto-scaling | Complexity | GPU | Best For |
|----------|----------------------|-------------|-------------|-----|-----------|
| **AWS ECS/Fargate** | $30-200 | Yes | Medium | No (Fargate) | Containerized APIs |
| **AWS SageMaker** | $50-500+ | Yes (managed) | Low | Yes | Enterprise ML teams |
| **AWS Lambda** | $0-50 | Automatic | Low | No | Lightweight models, variable traffic |
| **GCP Cloud Run** | $0-150 | Yes | Low | Yes (limited) | Containerized APIs |
| **GCP Vertex AI** | $50-500+ | Yes | Medium | Yes | ML teams on GCP |
| **Azure Container Apps** | $30-200 | Yes | Medium | Yes | Containerized APIs on Azure |
| **Azure ML** | $50-500+ | Yes | Medium | Yes | Enterprise on Azure |

---

## Serverless ML

### When It Makes Sense

- Small models (< 250MB packaged)
- Fast inference (< 10 seconds)
- Variable traffic (peaks and valleys)
- Limited budget (pay only for usage)

### The Cold Start Problem

| Factor | Impact on Cold Start |
|--------|---------------------|
| Package size | Larger package = more time |
| Runtime (Python) | ~500ms-1s base |
| Model loading | The slowest part (1-10s) |
| VPC config | +2-5s if in VPC |

**Solutions:**

```python
# 1. Provisioned Concurrency (AWS Lambda)
# Keep N instances always warm
# Cost: ~$15/month per instance

# 2. Model in ONNX format (faster to load)
# 3. Model on EFS (don't download it every time)
# 4. Periodic ping (CloudWatch Events every 5 min)
# 5. Use layers for dependencies (they're cached)
```

### Lambda + ONNX for Computer Vision

```python
# Lambda Layer structure:
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
    # Decode base64 image
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

### Decision Table

| Mode | Latency | Use Case | Tools | Cost |
|------|---------|-------------|-------------|-------|
| **Batch** | Minutes-hours | Nightly scoring, reports | Airflow, cron, Spark | Low |
| **Real-time** | Milliseconds | Prediction API, chatbots | FastAPI, SageMaker | Medium-High |
| **Near real-time** | Seconds | Fraud detection, recommendations | Kafka, SQS + consumer | Medium |
| **Streaming** | Milliseconds (continuous) | Sensor data, live logs | Kafka Streams, Flink | High |

### Batch Inference

```python
# batch_predict.py - Run with Airflow or cron
import pandas as pd
import joblib
from datetime import datetime

def run_batch_predictions():
    # Load new data
    df = pd.read_sql("SELECT * FROM customers WHERE scored_at IS NULL", engine)

    # Load model
    model = joblib.load("models/model_latest.pkl")

    # Predict
    features = df[feature_columns].values
    df["prediction"] = model.predict(features)
    df["scored_at"] = datetime.utcnow()

    # Save results
    df[["customer_id", "prediction", "scored_at"]].to_sql(
        "predictions", engine, if_exists="append", index=False
    )
    print(f"Processed {len(df)} records")

if __name__ == "__main__":
    run_batch_predictions()
```

### Near Real-Time with Message Queue

```python
# consumer.py - SQS/Kafka consumer
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
            MaxNumberOfMessages=10,  # Batch of up to 10
            WaitTimeSeconds=20,       # Long polling
        )

        messages = response.get("Messages", [])
        if not messages:
            continue

        # Process in batch
        features_batch = []
        for msg in messages:
            body = json.loads(msg["Body"])
            features_batch.append(body["features"])

        predictions = model.predict(np.array(features_batch))

        # Save results and delete messages
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

### When to Use Each Mode

```
Do you need an immediate response (< 1s)?
├── Yes -> Real-time API (FastAPI)
└── No
    ├── Does data arrive continuously?
    │   ├── Yes -> Do you need result in < 30s?
    │   │   ├── Yes -> Streaming (Kafka Streams)
    │   │   └── No  -> Near real-time (SQS + consumer)
    │   └── No  -> Batch (Airflow/cron)
    └── Periodic processing?
        └── Yes -> Batch (Airflow/cron)
```

---

## CI/CD for ML

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

### Model Validation before Deploy

```python
# scripts/validate_model.py
"""
Validate that the new model is at least as good as the production one.
Runs in CI before deploy.
"""
import json
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load standard test set
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Load new model (candidate)
new_model = joblib.load("models/model_latest.pkl")
new_preds = new_model.predict(X_test)

# New model metrics
new_accuracy = accuracy_score(y_test, new_preds)
new_f1 = f1_score(y_test, new_preds, average="weighted")

# Load production model metrics
with open("models/production_metrics.json") as f:
    prod_metrics = json.load(f)

print(f"New model      - Accuracy: {new_accuracy:.4f}, F1: {new_f1:.4f}")
print(f"Prod model     - Accuracy: {prod_metrics['accuracy']:.4f}, F1: {prod_metrics['f1']:.4f}")

# Validate: the new model cannot be worse
TOLERANCE = 0.01  # 1% tolerance
if new_f1 < prod_metrics["f1"] - TOLERANCE:
    print("FAIL: New model is significantly worse than production")
    exit(1)

print("OK: Model validated, ready for deploy")
```

---

## Production Monitoring

### Data Drift

Data drift occurs when the distribution of input data changes compared to the training data. The model was trained with a certain distribution and may not work well with different data.

```python
# monitoring/drift_detector.py
import numpy as np
from scipy import stats

class DriftDetector:
    """Detect drift by comparing distributions."""

    def __init__(self, reference_data: np.ndarray, threshold: float = 0.05):
        self.reference = reference_data
        self.threshold = threshold

    def check_drift(self, current_data: np.ndarray) -> dict:
        """Kolmogorov-Smirnov test per feature."""
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

### Metrics to Monitor

| Category | Metric | Alert If |
|-----------|---------|----------|
| **Latency** | P50, P95, P99 response time | P95 > 500ms |
| **Throughput** | Requests/second | Drop > 50% |
| **Errors** | Error rate (4xx, 5xx) | > 1% |
| **Model** | Prediction distribution | Significant shift |
| **Model** | Confidence score distribution | Mean drops > 10% |
| **Data** | Input feature distributions | KS test p < 0.05 |
| **Data** | Missing values rate | Increase > 5% |
| **Infra** | CPU/Memory usage | > 80% sustained |
| **Business** | Conversion rate, revenue impact | Drop > X% |

### Monitoring Tools

| Tool | Type | Cost | Best For |
|-------------|------|-------|-----------|
| **Evidently AI** | Data/model drift | Open source | Drift dashboards |
| **WhyLabs** | ML monitoring | Free tier + paid | Continuous monitoring |
| **Prometheus + Grafana** | Infra metrics | Open source | Latency, throughput |
| **CloudWatch** (AWS) | Infra + custom | Pay per use | If you're already on AWS |
| **Arize AI** | ML observability | Free tier + paid | Model debugging |

### When to Retrain

```
Checklist for deciding retraining:
- [ ] Data drift detected in > 30% of features
- [ ] Production accuracy dropped > 5% vs baseline
- [ ] More than 3 months since last training
- [ ] New labeled data is available
- [ ] Business reports degradation in results

If >= 2 of these are true -> Retrain
```

---

## Consulting Tips

### Pragmatic Deployment Approach

```
Phase 1 (Week 1-2): MVP
├── Model trained in notebook
├── Batch script that reads CSV -> generates predictions -> saves CSV
├── Client validates results manually
└── Cost: $0

Phase 2 (Week 3-4): API
├── FastAPI + Docker
├── Simple prediction endpoint
├── Deploy on a server (EC2, Cloud Run)
├── Documentation with Swagger
└── Cost: $30-100/month

Phase 3 (Month 2-3): Production
├── CI/CD with GitHub Actions
├── Basic monitoring (latency, errors)
├── Auto-scaling if needed
├── Data drift detection
└── Cost: $100-500/month

Phase 4 (Month 4+): Scale
├── Complete MLOps pipeline
├── Automatic retraining
├── A/B testing
├── Multi-model serving
└── Cost: $500+/month
```

### Golden Rules

1. **Don't over-engineer the first version.** A CSV with predictions that the client can review is worth more than a perfect API that nobody uses.

2. **FastAPI gives you free documentation.** The automatically generated Swagger UI (available at `/docs`) is perfect for the client to understand and test the API without additional documentation.

3. **Docker from day 1.** Even if the initial deploy is simple, having a Dockerfile eliminates "works on my machine" from the start.

4. **Monitoring > more accuracy.** Knowing when your model fails is more valuable than improving accuracy by 2%.

5. **Start with batch.** If the client doesn't need a response in milliseconds, a nightly batch process is 10x easier to maintain than an API.

---

> **Takeaway:** Deployment is not a final step, it's a discipline. The safest path is iterative: batch first, API next, scale when justified. FastAPI + Docker + GitHub Actions gives you 80% of what you need for 90% of consulting projects.

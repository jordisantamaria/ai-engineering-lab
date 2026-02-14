# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source code and model artifacts
COPY src/ ./src/
COPY models/ ./models/

# Expose the API port
EXPOSE 8000

# Environment variables (can be overridden at runtime)
ENV MODEL_PATH=models/best_model.pth
ENV NUM_CLASSES=2

# Run the API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

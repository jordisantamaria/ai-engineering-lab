"""
FastAPI application for defect detection inference.

Provides REST endpoints to upload product images and receive
defect classification results in real time.
"""

import io
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from model import DefectClassifier, get_inference_transform, load_model


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Schema for the /predict endpoint response."""
    predicted_class: str
    class_index: int
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Schema for the /health endpoint response."""
    status: str
    model_loaded: bool
    device: str
    num_classes: int


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

# Global references populated during startup
_model: Optional[DefectClassifier] = None
_device: Optional[torch.device] = None
_transform = None
_class_names: list = ["good", "defect"]

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "2"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the application starts up."""
    global _model, _device, _transform, _class_names

    if os.path.exists(MODEL_PATH):
        _model, _device = load_model(MODEL_PATH, num_classes=NUM_CLASSES)
        _transform = get_inference_transform()

        # Try to recover class names from the checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=_device)
        if isinstance(checkpoint, dict) and "class_names" in checkpoint:
            _class_names = checkpoint["class_names"]

        print(
            f"Model loaded from {MODEL_PATH} on {_device} "
            f"with classes {_class_names}"
        )
    else:
        print(
            f"WARNING: No model found at {MODEL_PATH}. "
            f"/predict will return 503 until a model is available."
        )

    yield  # Application runs here

    # Cleanup (if needed) goes after yield
    print("Shutting down defect detection API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Defect Detection API",
    description="Real-time product defect classification using EfficientNet-B0.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return the current health status of the service."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        device=str(_device) if _device else "n/a",
        num_classes=NUM_CLASSES,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Classify an uploaded image as defective or non-defective.

    Accepts JPEG or PNG images. Returns the predicted class,
    confidence score, per-class probabilities, and inference time.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure a valid checkpoint exists.",
        )

    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = _transform(image).unsqueeze(0).to(_device)

        # Run inference
        start = time.time()
        with torch.no_grad():
            logits = _model(input_tensor)
            probs = torch.softmax(logits, dim=1)
        inference_time = (time.time() - start) * 1000

        confidence, predicted_idx = torch.max(probs, dim=1)
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()

        probabilities = {
            _class_names[i]: round(probs[0][i].item(), 4)
            for i in range(len(_class_names))
        }

        return PredictionResponse(
            predicted_class=_class_names[predicted_idx],
            class_index=predicted_idx,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            inference_time_ms=round(inference_time, 2),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

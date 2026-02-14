"""
FastAPI application for churn prediction.

Accepts customer features as JSON and returns the churn probability,
risk level, and top risk factors explained via SHAP.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model import ChurnModel


# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """Schema for customer feature input."""
    tenure: int = Field(..., description="Number of months the customer has been with the company")
    monthly_charges: float = Field(..., description="Current monthly charges")
    total_charges: float = Field(..., description="Total charges since joining")
    contract: str = Field("Month-to-month", description="Contract type")
    payment_method: str = Field("Electronic check", description="Payment method")
    internet_service: str = Field("Fiber optic", description="Internet service type")
    online_security: str = Field("No", description="Has online security service")
    tech_support: str = Field("No", description="Has tech support service")
    num_services: int = Field(1, description="Number of active services")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 12,
                    "monthly_charges": 79.5,
                    "total_charges": 954.0,
                    "contract": "Month-to-month",
                    "payment_method": "Electronic check",
                    "internet_service": "Fiber optic",
                    "online_security": "No",
                    "tech_support": "No",
                    "num_services": 4,
                }
            ]
        }
    }


class RiskFactor(BaseModel):
    """A single risk factor from SHAP explanation."""
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    """Schema for the /predict endpoint response."""
    churn_probability: float
    risk_level: str
    top_risk_factors: List[RiskFactor]


class HealthResponse(BaseModel):
    """Schema for the /health endpoint response."""
    status: str
    model_loaded: bool
    num_features: int


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")
_model: Optional[ChurnModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the churn model on startup."""
    global _model

    if os.path.exists(MODEL_PATH):
        _model = ChurnModel.load(MODEL_PATH)
        print(f"Churn model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: No model found at {MODEL_PATH}.")

    yield

    print("Shutting down churn prediction API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Predict customer churn probability and identify top risk factors "
        "using XGBoost with SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return the current health status of the service."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        num_features=len(_model.feature_names) if _model and _model.feature_names else 0,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn probability for a customer.

    Accepts customer features and returns the churn probability,
    a risk level classification (alto/medio/bajo), and the top
    risk factors from SHAP analysis.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure a trained model exists.",
        )

    try:
        # Convert input to a feature dictionary
        features = customer.model_dump()

        # Map input fields to model feature names
        # (the model expects one-hot encoded features from training)
        model_features = _build_model_features(features)

        result = _model.predict_single(model_features)

        return PredictionResponse(
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            top_risk_factors=[
                RiskFactor(feature=f["feature"], impact=f["impact"])
                for f in result["top_risk_factors"]
            ],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


def _build_model_features(raw_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw API input into model-compatible feature format.

    Handles the mapping from user-friendly field names to the one-hot
    encoded feature names expected by the trained model.
    """
    features = {}

    # Numeric features
    features["tenure"] = raw_features.get("tenure", 0)
    features["MonthlyCharges"] = raw_features.get("monthly_charges", 0)
    features["TotalCharges"] = raw_features.get("total_charges", 0)
    features["num_services"] = raw_features.get("num_services", 1)

    # Derived features
    tenure = features["tenure"]
    monthly = features["MonthlyCharges"]
    num_svc = features["num_services"]

    features["contract_value"] = monthly * tenure
    features["monthly_charges_per_service"] = (
        monthly / num_svc if num_svc > 0 else 0
    )
    features["avg_monthly_spend"] = (
        features["TotalCharges"] / tenure if tenure > 0 else monthly
    )
    features["charge_increase_ratio"] = (
        monthly / features["avg_monthly_spend"]
        if features["avg_monthly_spend"] > 0
        else 1.0
    )

    # One-hot encoded categoricals
    contract = raw_features.get("contract", "Month-to-month")
    features["Contract_One year"] = 1 if contract == "One year" else 0
    features["Contract_Two year"] = 1 if contract == "Two year" else 0

    payment = raw_features.get("payment_method", "")
    features["PaymentMethod_Credit card (automatic)"] = (
        1 if "Credit card" in payment else 0
    )
    features["PaymentMethod_Electronic check"] = (
        1 if "Electronic check" in payment else 0
    )
    features["PaymentMethod_Mailed check"] = (
        1 if "Mailed check" in payment else 0
    )

    internet = raw_features.get("internet_service", "")
    features["InternetService_Fiber optic"] = (
        1 if internet == "Fiber optic" else 0
    )
    features["InternetService_No"] = 1 if internet == "No" else 0

    features["OnlineSecurity_Yes"] = (
        1 if raw_features.get("online_security") == "Yes" else 0
    )
    features["TechSupport_Yes"] = (
        1 if raw_features.get("tech_support") == "Yes" else 0
    )
    features["has_premium_support"] = (
        1
        if features["OnlineSecurity_Yes"] or features["TechSupport_Yes"]
        else 0
    )

    return features


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)

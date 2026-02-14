"""
FastAPI application for the recommendation engine.

Provides endpoints for user-based recommendations and item-based
similarity lookups using the hybrid recommender.
"""

import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from collaborative import CollaborativeRecommender
from content_based import ContentBasedRecommender
from hybrid import HybridRecommender


# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Schema for the /recommend endpoint request."""
    user_id: str = Field(..., description="User ID to generate recommendations for")
    n: int = Field(10, ge=1, le=100, description="Number of recommendations")


class SimilarRequest(BaseModel):
    """Schema for the /similar endpoint request."""
    product_id: str = Field(..., description="Product ID to find similar products for")
    n: int = Field(10, ge=1, le=100, description="Number of similar products")


class RecommendationItem(BaseModel):
    """Schema for a single recommendation."""
    product_id: str
    score: float
    source: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None


class RecommendResponse(BaseModel):
    """Schema for recommendation endpoint responses."""
    recommendations: List[RecommendationItem]
    count: int


class HealthResponse(BaseModel):
    """Schema for the /health endpoint response."""
    status: str
    num_products: int
    num_users: int


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

PRODUCTS_PATH = os.getenv("PRODUCTS_PATH", "data/products.csv")
INTERACTIONS_PATH = os.getenv("INTERACTIONS_PATH", "data/interactions.csv")

_recommender: Optional[HybridRecommender] = None
_num_products: int = 0
_num_users: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and initialize recommenders on startup."""
    global _recommender, _num_products, _num_users

    if os.path.exists(PRODUCTS_PATH) and os.path.exists(INTERACTIONS_PATH):
        products = pd.read_csv(PRODUCTS_PATH)
        interactions = pd.read_csv(INTERACTIONS_PATH)

        _num_products = len(products)
        _num_users = interactions["user_id"].nunique()

        # Build content-based recommender
        content = ContentBasedRecommender()
        content.fit(products)

        # Build item metadata dict for collaborative recommender
        item_meta = {}
        for _, row in products.iterrows():
            pid = str(row.get("product_id", ""))
            item_meta[pid] = {
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "price": row.get("price", 0),
            }

        # Build collaborative recommender
        collab = CollaborativeRecommender()
        collab.fit(interactions, item_metadata=item_meta)

        # Create hybrid recommender
        _recommender = HybridRecommender(content, collab)

        print(
            f"Recommender initialized: {_num_products} products, "
            f"{_num_users} users"
        )
    else:
        print(
            f"WARNING: Data files not found. "
            f"Expected {PRODUCTS_PATH} and {INTERACTIONS_PATH}"
        )

    yield

    print("Shutting down recommendation engine API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Recommendation Engine API",
    description="Hybrid product recommendation system combining content-based and collaborative filtering.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return the current health status of the service."""
    return HealthResponse(
        status="healthy" if _recommender else "no_data",
        num_products=_num_products,
        num_users=_num_users,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Get personalized product recommendations for a user.

    Uses the hybrid recommender that blends collaborative filtering
    (user behavior) with content-based similarity (product descriptions).
    Handles cold-start automatically for new users.
    """
    if _recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not initialized. Ensure data files exist.",
        )

    try:
        recs = _recommender.recommend(request.user_id, n=request.n)

        items = [
            RecommendationItem(
                product_id=r["product_id"],
                score=r["score"],
                source=r.get("source"),
                name=r.get("name"),
                category=r.get("category"),
                price=r.get("price"),
            )
            for r in recs
        ]

        return RecommendResponse(recommendations=items, count=len(items))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(exc)}",
        )


@app.post("/similar", response_model=RecommendResponse)
async def find_similar(request: SimilarRequest):
    """
    Find products similar to a given product.

    Uses the hybrid approach combining semantic similarity (from
    product descriptions) with collaborative similarity (items
    frequently bought together).
    """
    if _recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not initialized. Ensure data files exist.",
        )

    try:
        recs = _recommender.recommend_similar(request.product_id, n=request.n)

        items = [
            RecommendationItem(
                product_id=r["product_id"],
                score=r["score"],
                source=r.get("source"),
                name=r.get("name"),
                category=r.get("category"),
                price=r.get("price"),
            )
            for r in recs
        ]

        return RecommendResponse(recommendations=items, count=len(items))

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Similarity search failed: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)

# Product Recommendation Engine

## Business Problem

Product recommendations are one of the most powerful growth drivers in e-commerce and retail:

- **Conversion**: users who interact with recommendations have a 2-5x higher conversion rate.
- **Average ticket**: recommendations like "you might also like" and "frequently bought together" increase the average order value by 10-30%.
- **Engagement**: personalized recommendations increase time spent on the platform and visit frequency.
- **Discovery**: 35% of Amazon's sales come from its recommendation engine (McKinsey).

Without a recommendation system, the same catalog is shown to all users, missing opportunities for cross-selling and personalization.

## Proposed Solution

Hybrid recommendation system that combines two complementary approaches:

1. **Content-Based Filtering**: recommends similar products based on product descriptions and attributes (uses semantic embeddings).
2. **Collaborative Filtering**: recommends products based on the behavior of similar users (ALS - Alternating Least Squares).
3. **Hybrid approach**: combines both with adjustable weighting to obtain the best recommendations, including cold start handling.

### Architecture

```
                    +---------------------------+
                    |    Recommendation API      |
                    +---------------------------+
                       /                    \
                      v                      v
    +--------------------+      +------------------------+
    | Content-Based      |      | Collaborative          |
    | (SentenceTransf.)  |      | (ALS / Implicit)       |
    +--------------------+      +------------------------+
    | - Product embeddings|      | - User-item matrix    |
    | - Cosine similarity |      | - Factorization       |
    | - New products      |      | - Similar users       |
    +--------------------+      +------------------------+
                      \                      /
                       v                    v
                    +---------------------------+
                    |    Hybrid Recommender      |
                    |  (weighted ensemble)       |
                    +---------------------------+
                              |
                              v
                    Final recommendations
                    (with cold start handling)
```

## Expected Results

| Metric | Value |
|---------|-------|
| Precision@10 | >0.15 |
| Recall@10 | >0.10 |
| NDCG@10 | >0.20 |
| Catalog coverage | >60% |
| Response time | <200ms |

## Technologies

- **sentence-transformers**: semantic embeddings of product descriptions
- **implicit**: ALS library for collaborative filtering
- **scikit-learn**: similarity metrics and evaluation
- **scipy**: sparse matrices for user-product interactions
- **FastAPI**: high-performance REST API
- **pandas / numpy**: data manipulation

## How to Run

### 1. Installation

```bash
cd portfolio/04-recommendation-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data

The system expects two CSV files:

- `data/products.csv`: columns `product_id`, `name`, `description`, `category`, `price`
- `data/interactions.csv`: columns `user_id`, `product_id`, `rating` (or `purchase_count`)

### 3. Launch the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8003
```

### 4. Get Recommendations

```bash
# Recommendations for a user
curl -X POST "http://localhost:8003/recommend" \
    -H "Content-Type: application/json" \
    -d '{"user_id": "user_123", "n": 10}'

# Similar products
curl -X POST "http://localhost:8003/similar" \
    -H "Content-Type: application/json" \
    -d '{"product_id": "prod_456", "n": 5}'
```

## How to Present It: Client Pitch

### Value Proposition

> "Companies with personalized recommendations see a 10-30% increase in revenue. Our hybrid system works even with new users (cold start) and newly added products, adapting in real time to your customers' behavior."

### Estimated ROI

**Scenario**: e-commerce with 100,000 active users/month, average ticket 45 EUR, current conversion rate 2.5%.

| Item | Before | After |
|----------|-------|---------|
| Conversion rate | 2.5% | 3.5% (+40%) |
| Average ticket | 45 EUR | 52 EUR (+15%) |
| Orders/month | 2,500 | 3,500 |
| Monthly revenue | 112,500 EUR | 182,000 EUR |

**Estimated increase: ~69,500 EUR/month** in additional revenue thanks to recommendations.

### Key Points for the Presentation

1. **Interactive demo**: load the client's catalog and show live recommendations.
2. **A/B testing**: the system can be evaluated with A/B tests before full implementation.
3. **Cold start solved**: new users receive recommendations from day one (content-based), which improve as they interact (collaborative).
4. **Real time**: recommendations are served in <200ms, compatible with any frontend.
5. **Privacy**: user data never leaves the client's infrastructure.

### Frequently Asked Client Questions

- **"We have little user data"** - We start with content-based (only needs the product catalog) and transition to hybrid when there are enough interactions.
- **"We already have basic recommendations (best sellers)"** - Personalized recommendations systematically outperform popular ones in engagement and conversion.
- **"How do we measure the impact?"** - Controlled A/B test: group A with recommendations, group B without. We measure conversion, average ticket, and engagement.

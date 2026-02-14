"""
Hybrid recommendation engine combining content-based and collaborative filtering.

Uses a weighted ensemble of both approaches and handles cold-start scenarios:
- New user with no history -> content-based recommendations
- New item not in interaction data -> content-based similarity
- Established user with history -> hybrid blend
"""

from typing import Dict, List, Optional

from collaborative import CollaborativeRecommender
from content_based import ContentBasedRecommender


class HybridRecommender:
    """
    Hybrid recommender that blends content-based and collaborative scores.

    Gracefully handles cold-start scenarios by falling back to the
    appropriate single-method approach when the other is unavailable.
    """

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        collaborative_recommender: CollaborativeRecommender,
        content_weight: float = 0.4,
        collaborative_weight: float = 0.6,
    ):
        """
        Initialize the hybrid recommender.

        Args:
            content_recommender: Fitted ContentBasedRecommender instance.
            collaborative_recommender: Fitted CollaborativeRecommender instance.
            content_weight: Weight for content-based scores (0-1).
            collaborative_weight: Weight for collaborative scores (0-1).
        """
        self.content = content_recommender
        self.collaborative = collaborative_recommender
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

        # Normalize weights to sum to 1
        total = self.content_weight + self.collaborative_weight
        self.content_weight /= total
        self.collaborative_weight /= total

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        content_weight: Optional[float] = None,
        collaborative_weight: Optional[float] = None,
    ) -> List[Dict]:
        """
        Generate hybrid recommendations for a user.

        Handles cold-start scenarios automatically:
        - Unknown user: falls back to popular/content-based items.
        - Known user: blends collaborative and content-based scores.

        Args:
            user_id: The user to recommend for.
            n: Number of recommendations to return.
            content_weight: Override the default content weight.
            collaborative_weight: Override the default collaborative weight.

        Returns:
            List of recommendation dicts with product_id, score,
            source (hybrid/content/collaborative), and metadata.
        """
        cw = content_weight if content_weight is not None else self.content_weight
        clw = collaborative_weight if collaborative_weight is not None else self.collaborative_weight

        # Normalize overridden weights
        total = cw + clw
        cw /= total
        clw /= total

        # Check if user exists in collaborative data
        user_known = self.collaborative.has_user(user_id)

        if not user_known:
            # Cold start: new user -> content-based only
            return self._cold_start_user(n)

        # Get recommendations from both systems (request extra to allow merging)
        fetch_n = n * 3

        try:
            collab_recs = self.collaborative.recommend_for_user(user_id, n=fetch_n)
        except Exception:
            collab_recs = []

        # Build a score dictionary from collaborative results
        collab_scores: Dict[str, float] = {}
        collab_meta: Dict[str, Dict] = {}
        for rec in collab_recs:
            pid = rec["product_id"]
            collab_scores[pid] = rec["score"]
            collab_meta[pid] = {
                k: v for k, v in rec.items() if k not in ("product_id", "score")
            }

        # For content-based: get similar items to the user's top collaborative picks
        content_scores: Dict[str, float] = {}
        content_meta: Dict[str, Dict] = {}

        # Use top collaborative items as seeds for content-based expansion
        seed_items = [rec["product_id"] for rec in collab_recs[:5]]
        for seed_id in seed_items:
            try:
                similar = self.content.recommend_similar(seed_id, n=fetch_n // 5)
                for rec in similar:
                    pid = rec["product_id"]
                    # Keep the highest content score for each product
                    if pid not in content_scores or rec["score"] > content_scores[pid]:
                        content_scores[pid] = rec["score"]
                        content_meta[pid] = {
                            k: v
                            for k, v in rec.items()
                            if k not in ("product_id", "score")
                        }
            except (ValueError, RuntimeError):
                continue

        # Merge scores with weighted combination
        all_product_ids = set(collab_scores.keys()) | set(content_scores.keys())
        merged: Dict[str, float] = {}

        for pid in all_product_ids:
            c_score = content_scores.get(pid, 0.0)
            cl_score = collab_scores.get(pid, 0.0)
            merged[pid] = cw * c_score + clw * cl_score

        # Sort by merged score and return top-N
        sorted_products = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for pid, score in sorted_products[:n]:
            # Determine source
            in_content = pid in content_scores
            in_collab = pid in collab_scores
            if in_content and in_collab:
                source = "hybrid"
            elif in_collab:
                source = "collaborative"
            else:
                source = "content"

            meta = collab_meta.get(pid, content_meta.get(pid, {}))
            recommendations.append(
                {
                    "product_id": pid,
                    "score": round(score, 4),
                    "source": source,
                    **meta,
                }
            )

        return recommendations

    def recommend_similar(
        self,
        product_id: str,
        n: int = 10,
    ) -> List[Dict]:
        """
        Find products similar to a given product using hybrid approach.

        Blends content-based similarity (embeddings) with collaborative
        similarity (co-purchase patterns).

        Args:
            product_id: The product to find similar items for.
            n: Number of similar products to return.

        Returns:
            List of recommendation dicts with product_id, score, and source.
        """
        fetch_n = n * 2

        # Content-based similarity (always available if product has description)
        content_recs = []
        try:
            content_recs = self.content.recommend_similar(product_id, n=fetch_n)
        except (ValueError, RuntimeError):
            pass

        # Collaborative similarity (only if product is in interaction data)
        collab_recs = []
        if self.collaborative.has_item(product_id):
            try:
                collab_recs = self.collaborative.find_similar_items(
                    product_id, n=fetch_n
                )
            except Exception:
                pass

        # If only one source is available, return it directly
        if not collab_recs:
            return content_recs[:n]
        if not content_recs:
            return collab_recs[:n]

        # Merge scores
        content_scores = {r["product_id"]: r["score"] for r in content_recs}
        collab_scores = {r["product_id"]: r["score"] for r in collab_recs}
        meta = {}
        for r in content_recs + collab_recs:
            pid = r["product_id"]
            if pid not in meta:
                meta[pid] = {
                    k: v for k, v in r.items() if k not in ("product_id", "score")
                }

        all_ids = set(content_scores.keys()) | set(collab_scores.keys())
        merged = {}
        for pid in all_ids:
            c = content_scores.get(pid, 0.0)
            cl = collab_scores.get(pid, 0.0)
            merged[pid] = self.content_weight * c + self.collaborative_weight * cl

        sorted_products = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for pid, score in sorted_products[:n]:
            source = "hybrid"
            if pid not in content_scores:
                source = "collaborative"
            elif pid not in collab_scores:
                source = "content"

            recommendations.append(
                {
                    "product_id": pid,
                    "score": round(score, 4),
                    "source": source,
                    **meta.get(pid, {}),
                }
            )

        return recommendations

    def _cold_start_user(self, n: int) -> List[Dict]:
        """
        Handle cold-start for a new user with no interaction history.

        Falls back to the most popular items from the collaborative
        model or general content-based recommendations.

        Args:
            n: Number of items to return.

        Returns:
            List of recommendation dicts.
        """
        # Strategy: recommend diverse popular items
        # Use the first few items from the collaborative model's most popular
        if (
            self.collaborative.user_item_matrix is not None
            and self.collaborative.user_item_matrix.shape[0] > 0
        ):
            # Sum interactions per item to find popular items
            item_popularity = self.collaborative.user_item_matrix.sum(axis=0).A1
            popular_indices = item_popularity.argsort()[::-1][:n]

            recommendations = []
            for idx in popular_indices:
                pid = self.collaborative.item_ids[idx]
                recommendations.append(
                    {
                        "product_id": pid,
                        "score": round(float(item_popularity[idx]), 4),
                        "source": "popular",
                        **self.collaborative.item_metadata.get(pid, {}),
                    }
                )

            return recommendations

        return []

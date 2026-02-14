"""
Collaborative filtering recommender using Alternating Least Squares (ALS).

Builds a user-item interaction matrix from purchase/rating data and
factorizes it using the implicit library to learn latent user and item
factors for recommendation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


class CollaborativeRecommender:
    """
    Collaborative filtering recommender using ALS matrix factorization.

    Learns latent factor representations for users and items from
    implicit feedback (purchases, clicks) or explicit ratings.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 50,
        alpha: float = 40.0,
    ):
        """
        Initialize the collaborative recommender.

        Args:
            factors: Number of latent factors in the factorization.
            regularization: L2 regularization weight.
            iterations: Number of ALS iterations.
            alpha: Confidence scaling factor for implicit feedback.
        """
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

        self._model = None
        self.user_item_matrix: Optional[sparse.csr_matrix] = None
        self.user_ids: List[str] = []
        self.item_ids: List[str] = []
        self._user_to_idx: Dict[str, int] = {}
        self._item_to_idx: Dict[str, int] = {}
        self.item_metadata: Dict[str, Dict] = {}

    def fit(
        self,
        interactions: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "product_id",
        value_col: str = "rating",
        item_metadata: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """
        Build the user-item matrix and train the ALS model.

        Args:
            interactions: DataFrame with user-item interactions.
            user_col: Column name for user IDs.
            item_col: Column name for item (product) IDs.
            value_col: Column name for interaction values (ratings/counts).
            item_metadata: Optional dict mapping item IDs to metadata dicts.
        """
        from implicit.als import AlternatingLeastSquares

        # Build user and item ID mappings
        self.user_ids = interactions[user_col].astype(str).unique().tolist()
        self.item_ids = interactions[item_col].astype(str).unique().tolist()
        self._user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self._item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}

        if item_metadata:
            self.item_metadata = item_metadata

        # Build sparse user-item matrix
        rows = interactions[user_col].astype(str).map(self._user_to_idx).values
        cols = interactions[item_col].astype(str).map(self._item_to_idx).values
        values = interactions[value_col].values.astype(np.float32)

        self.user_item_matrix = sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )

        # Train ALS model (implicit expects item-user matrix)
        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42,
        )

        # Scale by alpha for implicit feedback confidence weighting
        confidence_matrix = (self.user_item_matrix * self.alpha).astype(np.float32)
        self._model.fit(confidence_matrix)

        print(
            f"Trained ALS model: {len(self.user_ids)} users, "
            f"{len(self.item_ids)} items, {self.factors} factors"
        )

    def recommend_for_user(
        self,
        user_id: str,
        n: int = 10,
        filter_already_interacted: bool = True,
    ) -> List[Dict]:
        """
        Generate item recommendations for a specific user.

        Args:
            user_id: The user ID to generate recommendations for.
            n: Number of recommendations to return.
            filter_already_interacted: Whether to exclude items the user
                                      has already interacted with.

        Returns:
            List of dicts with product_id, score, and metadata.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if user_id not in self._user_to_idx:
            raise ValueError(f"User '{user_id}' not found in training data.")

        user_idx = self._user_to_idx[user_id]

        # Get recommendations from the ALS model
        item_indices, scores = self._model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=n,
            filter_already_liked_items=filter_already_interacted,
        )

        recommendations = []
        for item_idx, score in zip(item_indices, scores):
            item_id = self.item_ids[item_idx]
            recommendations.append(
                {
                    "product_id": item_id,
                    "score": round(float(score), 4),
                    **self.item_metadata.get(item_id, {}),
                }
            )

        return recommendations

    def find_similar_items(
        self,
        item_id: str,
        n: int = 10,
    ) -> List[Dict]:
        """
        Find items similar to a given item based on learned embeddings.

        Args:
            item_id: The item ID to find similar items for.
            n: Number of similar items to return.

        Returns:
            List of dicts with product_id, score, and metadata.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if item_id not in self._item_to_idx:
            raise ValueError(f"Item '{item_id}' not found in training data.")

        item_idx = self._item_to_idx[item_id]

        # Get similar items from the ALS model
        similar_indices, scores = self._model.similar_items(
            item_idx, N=n + 1  # +1 because it includes the item itself
        )

        recommendations = []
        for idx, score in zip(similar_indices, scores):
            if idx == item_idx:
                continue  # Skip the query item itself
            iid = self.item_ids[idx]
            recommendations.append(
                {
                    "product_id": iid,
                    "score": round(float(score), 4),
                    **self.item_metadata.get(iid, {}),
                }
            )

        return recommendations[:n]

    def find_similar_users(
        self,
        user_id: str,
        n: int = 10,
    ) -> List[Dict]:
        """
        Find users similar to a given user based on learned factors.

        Args:
            user_id: The user ID to find similar users for.
            n: Number of similar users to return.

        Returns:
            List of dicts with user_id and similarity score.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if user_id not in self._user_to_idx:
            raise ValueError(f"User '{user_id}' not found in training data.")

        user_idx = self._user_to_idx[user_id]

        similar_indices, scores = self._model.similar_users(
            user_idx, N=n + 1
        )

        results = []
        for idx, score in zip(similar_indices, scores):
            if idx == user_idx:
                continue
            results.append(
                {
                    "user_id": self.user_ids[idx],
                    "score": round(float(score), 4),
                }
            )

        return results[:n]

    def has_user(self, user_id: str) -> bool:
        """Check if a user exists in the training data."""
        return user_id in self._user_to_idx

    def has_item(self, item_id: str) -> bool:
        """Check if an item exists in the training data."""
        return item_id in self._item_to_idx

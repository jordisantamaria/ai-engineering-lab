"""
Churn prediction model wrapper around XGBoost with SHAP explanations.

Provides training with cross-validation, prediction with probability
outputs, and SHAP-based feature importance explanations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score


class ChurnModel:
    """
    Churn prediction model backed by XGBoost.

    Wraps the training, prediction, and explanation logic into a single
    class suitable for both offline analysis and real-time serving.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the churn model.

        Args:
            params: XGBoost hyperparameters. If None, uses sensible defaults.
        """
        self.params = params or {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1,
            "random_state": 42,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self.explainer: Optional[shap.TreeExplainer] = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Train the model with cross-validation scoring.

        Args:
            X: Feature DataFrame.
            y: Target Series (0/1).
            params: Override hyperparameters (optional).
            cv_folds: Number of cross-validation folds.

        Returns:
            Dictionary with cross-validation metrics (mean and std).
        """
        if params:
            self.params.update(params)

        self.feature_names = X.columns.tolist()

        # Initialize and train the XGBoost classifier
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, verbose=False)

        # Cross-validation scoring
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_auc = cross_val_score(
            self.model, X, y, cv=cv, scoring="roc_auc"
        )
        cv_f1 = cross_val_score(
            self.model, X, y, cv=cv, scoring="f1"
        )

        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)

        metrics = {
            "cv_auc_mean": round(float(np.mean(cv_auc)), 4),
            "cv_auc_std": round(float(np.std(cv_auc)), 4),
            "cv_f1_mean": round(float(np.mean(cv_f1)), 4),
            "cv_f1_std": round(float(np.std(cv_f1)), 4),
        }

        print(f"Training complete.")
        print(f"  CV AUC-ROC: {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}")
        print(f"  CV F1:      {metrics['cv_f1_mean']:.4f} +/- {metrics['cv_f1_std']:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probabilities for the given features.

        Args:
            X: Feature DataFrame (must match training features).

        Returns:
            Array of churn probabilities (float between 0 and 1).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)[:, 1]

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn for a single customer and return explanation.

        Args:
            features: Dictionary of feature name -> value.

        Returns:
            Dictionary with churn_probability, risk_level, and
            top_risk_factors from SHAP.
        """
        df = pd.DataFrame([features])

        # Ensure column order matches training
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_names]

        probability = float(self.predict(df)[0])

        # Determine risk level
        if probability >= 0.7:
            risk_level = "alto"
        elif probability >= 0.4:
            risk_level = "medio"
        else:
            risk_level = "bajo"

        # Get SHAP explanation for this prediction
        top_factors = self.explain_single(df)

        return {
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "top_risk_factors": top_factors,
        }

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for the given features.

        Args:
            X: Feature DataFrame.

        Returns:
            SHAP values array of shape (n_samples, n_features).
        """
        if self.explainer is None:
            if self.model is None:
                raise RuntimeError("Model not trained.")
            self.explainer = shap.TreeExplainer(self.model)

        shap_values = self.explainer.shap_values(X)
        return shap_values

    def explain_single(
        self, X: pd.DataFrame, top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get the top SHAP-based risk factors for a single prediction.

        Args:
            X: Single-row DataFrame.
            top_n: Number of top factors to return.

        Returns:
            List of dicts with 'feature' and 'impact' keys, sorted
            by absolute impact descending.
        """
        shap_values = self.explain(X)

        if isinstance(shap_values, list):
            # For binary classification, take the positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        feature_names = self.feature_names or X.columns.tolist()
        values = shap_values[0] if len(shap_values.shape) > 1 else shap_values

        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(values))[::-1][:top_n]

        factors = []
        for idx in sorted_indices:
            factors.append(
                {
                    "feature": feature_names[idx],
                    "impact": round(float(values[idx]), 4),
                }
            )

        return factors

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores sorted by importance.

        Returns:
            Dictionary mapping feature names to importance scores,
            sorted descending by score.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        importance = self.model.feature_importances_
        feature_names = self.feature_names or [
            f"feature_{i}" for i in range(len(importance))
        ]

        importance_dict = dict(zip(feature_names, importance))

        # Sort by importance descending
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance

    def save(self, path: str) -> None:
        """
        Save the model, feature names, and parameters to disk.

        Args:
            path: File path for the saved model (.joblib).
        """
        save_dict = {
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
        }
        joblib.dump(save_dict, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ChurnModel":
        """
        Load a saved model from disk.

        Args:
            path: Path to the .joblib file.

        Returns:
            ChurnModel instance with the loaded model.
        """
        save_dict = joblib.load(path)
        instance = cls(params=save_dict["params"])
        instance.model = save_dict["model"]
        instance.feature_names = save_dict["feature_names"]
        instance.explainer = shap.TreeExplainer(instance.model)

        print(f"Model loaded from {path}")
        return instance

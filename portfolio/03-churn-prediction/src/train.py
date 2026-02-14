"""
Full training pipeline for the churn prediction model.

Loads data, engineers features, tunes hyperparameters with Optuna,
trains the final model, evaluates with standard metrics, generates
SHAP explanations, and saves all artifacts.

Usage:
    python src/train.py --data_path data/telco_churn.csv --output_dir models/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import shap
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from data_processing import clean_data, feature_engineering, load_data, prepare_features
from model import ChurnModel


def objective(trial, X_train, y_train) -> float:
    """
    Optuna objective function for hyperparameter tuning.

    Suggests hyperparameters, trains XGBoost with cross-validation,
    and returns the mean AUC-ROC score.
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
    }

    churn_model = ChurnModel(params=params)
    metrics = churn_model.train(X_train, y_train, cv_folds=5)

    return metrics["cv_auc_mean"]


def plot_roc_curve(y_true, y_prob, output_path: str) -> None:
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f"Model (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Churn Prediction", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {output_path}")


def plot_pr_curve(y_true, y_prob, output_path: str) -> None:
    """Plot and save the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve - Churn Prediction", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"PR curve saved to {output_path}")


def plot_shap_summary(model: ChurnModel, X, output_path: str) -> None:
    """Generate and save a SHAP summary plot."""
    shap_values = model.explain(X)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a churn prediction model with XGBoost + Optuna"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the customer churn CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save model and artifacts",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna hyperparameter search trials",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load and prepare data
    # -----------------------------------------------------------------------
    print("\n=== Step 1: Loading and preparing data ===\n")
    df = load_data(args.data_path)
    df = clean_data(df)
    df = feature_engineering(df)
    X, y, feature_names = prepare_features(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # -----------------------------------------------------------------------
    # Step 2: Hyperparameter tuning with Optuna
    # -----------------------------------------------------------------------
    print(f"\n=== Step 2: Hyperparameter tuning ({args.n_trials} trials) ===\n")

    # Suppress Optuna logging for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=args.n_trials,
    )

    best_params = study.best_params
    print(f"\nBest AUC-ROC: {study.best_value:.4f}")
    print(f"Best parameters: {best_params}")

    # -----------------------------------------------------------------------
    # Step 3: Train final model with best parameters
    # -----------------------------------------------------------------------
    print("\n=== Step 3: Training final model ===\n")

    final_model = ChurnModel(params=best_params)
    cv_metrics = final_model.train(X_train, y_train, cv_folds=5)

    # -----------------------------------------------------------------------
    # Step 4: Evaluate on test set
    # -----------------------------------------------------------------------
    print("\n=== Step 4: Test set evaluation ===\n")

    y_prob = final_model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    print(report)

    # Save classification report
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Churn Prediction - Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write(f"\nAUC-ROC: {roc_auc_score(y_test, y_prob):.4f}\n")
        f.write(f"\nCV Metrics: {cv_metrics}\n")
        f.write(f"\nBest Optuna Parameters: {best_params}\n")
    print(f"Classification report saved to {report_path}")

    # -----------------------------------------------------------------------
    # Step 5: Generate visualizations
    # -----------------------------------------------------------------------
    print("\n=== Step 5: Generating plots ===\n")

    plot_roc_curve(y_test, y_prob, os.path.join(args.output_dir, "roc_curve.png"))
    plot_pr_curve(y_test, y_prob, os.path.join(args.output_dir, "pr_curve.png"))
    plot_shap_summary(final_model, X_test, os.path.join(args.output_dir, "shap_summary.png"))

    # Print top feature importances
    print("\nTop 10 Feature Importances (XGBoost):")
    importance = final_model.get_feature_importance()
    for i, (feat, score) in enumerate(list(importance.items())[:10], 1):
        print(f"  {i:2d}. {feat:40s} {score:.4f}")

    # -----------------------------------------------------------------------
    # Step 6: Save the model
    # -----------------------------------------------------------------------
    print("\n=== Step 6: Saving model ===\n")

    model_path = os.path.join(args.output_dir, "churn_model.joblib")
    final_model.save(model_path)

    print("\n=== Training pipeline complete ===")


if __name__ == "__main__":
    main()

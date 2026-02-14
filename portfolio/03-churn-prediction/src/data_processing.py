"""
Data loading, cleaning, and feature engineering for churn prediction.

Handles the Telco Customer Churn dataset from Kaggle, but the pipeline
is general enough to adapt to other customer datasets.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load the churn dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame as loaded from the file.
    """
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset: fix types, handle missing values, standardize.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()

    # TotalCharges has some blank strings -> convert to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Fill missing TotalCharges with MonthlyCharges * tenure
        mask = df["TotalCharges"].isna()
        df.loc[mask, "TotalCharges"] = (
            df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
        )

    # Convert SeniorCitizen from 0/1 to No/Yes for consistency
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Drop customerID (not useful for modeling)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Standardize target column
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Fill any remaining missing values
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown")

    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    print(f"Cleaned data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Missing values: {df.isna().sum().sum()}")

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that capture customer behavior patterns.

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with additional engineered features.
    """
    df = df.copy()

    # Tenure buckets: categorize customer lifetime
    if "tenure" in df.columns:
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=[0, 6, 12, 24, 48, 72, float("inf")],
            labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y", "6y+"],
        ).astype(str)

    # Monthly charges per service: normalize cost by number of services
    service_columns = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    existing_services = [c for c in service_columns if c in df.columns]

    if existing_services and "MonthlyCharges" in df.columns:
        # Count active services (Yes or Fiber optic/DSL)
        df["num_services"] = sum(
            (df[col] != "No") & (df[col] != "No internet service")
            & (df[col] != "No phone service")
            for col in existing_services
        ).astype(int)

        # Avoid division by zero
        df["monthly_charges_per_service"] = np.where(
            df["num_services"] > 0,
            df["MonthlyCharges"] / df["num_services"],
            0,
        )

    # Contract value: estimated lifetime value so far
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["contract_value"] = df["MonthlyCharges"] * df["tenure"]

    # Average monthly spend: total / tenure
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_spend"] = np.where(
            df["tenure"] > 0,
            df["TotalCharges"] / df["tenure"],
            df["MonthlyCharges"],
        )

    # Charge increase indicator: current monthly vs. average
    if "avg_monthly_spend" in df.columns and "MonthlyCharges" in df.columns:
        df["charge_increase_ratio"] = np.where(
            df["avg_monthly_spend"] > 0,
            df["MonthlyCharges"] / df["avg_monthly_spend"],
            1.0,
        )

    # Has premium support: combines security + tech support
    if "OnlineSecurity" in df.columns and "TechSupport" in df.columns:
        df["has_premium_support"] = (
            (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
        ).astype(int)

    print(f"Feature engineering complete: {len(df.columns)} columns")

    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "Churn",
    exclude_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare the feature matrix and target vector for model training.

    Encodes categorical variables using one-hot encoding and separates
    features from the target variable.

    Args:
        df: DataFrame after cleaning and feature engineering.
        target_col: Name of the target column.
        exclude_cols: Columns to exclude from features (besides target).

    Returns:
        Tuple of (X, y, feature_names):
            - X: Feature DataFrame with encoded categoricals.
            - y: Target Series (0/1).
            - feature_names: List of feature column names.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Separate target
    y = df[target_col].astype(int)

    # Drop target and excluded columns
    cols_to_drop = [target_col] + exclude_cols
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # One-hot encode categorical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Ensure all values are numeric
    X = X.astype(float)

    feature_names = X.columns.tolist()

    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    return X, y, feature_names


# Allow Optional to be used in type hints above
from typing import Optional

"""
models/xgboost_model.py — XGBoost training & prediction
Recommended for: Acute, Antibiotic categories
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def train_and_predict_xgboost(
    df: pd.DataFrame,
    feature_cols: List[str],
    forecast_year: int = 2025,
    future_features_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[float], object]:
    """
    Train XGBoost on historical data (year < forecast_year) and predict
    all 12 months of forecast_year.

    Args:
        df: DataFrame with [date, units_sold, ...feature_cols]
        feature_cols: list of feature column names to use
        forecast_year: year to forecast
        future_features_df: pre-built future feature rows (12 rows)

    Returns:
        (predictions_list, model)
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost package not installed. Run: pip install xgboost")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Training data ─────────────────────────────────────────────────────
    train = df[df["date"].dt.year < forecast_year].copy()
    train = train.dropna(subset=feature_cols + ["units_sold"])

    if len(train) < 12:
        raise ValueError(f"Insufficient training data: {len(train)} rows (need ≥ 12)")

    X_train = train[feature_cols].fillna(0)
    y_train = train["units_sold"]

    # ── Model ─────────────────────────────────────────────────────────────
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # ── Future features ───────────────────────────────────────────────────
    if future_features_df is None:
        raise ValueError("future_features_df must be provided for XGBoost prediction")

    X_future = future_features_df[feature_cols].fillna(0)
    predictions = model.predict(X_future).clip(min=0).tolist()

    logger.info(f"XGBoost forecast complete: {len(predictions)} months")
    return predictions, model

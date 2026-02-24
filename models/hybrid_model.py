"""
models/hybrid_model.py — Prophet + XGBoost blended forecast
Recommended for: Acute, Gastro categories
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def train_and_predict_hybrid(
    df: pd.DataFrame,
    feature_cols: List[str],
    forecast_year: int = 2025,
    future_features_df: Optional[pd.DataFrame] = None,
    prophet_weight: float = 0.5,
) -> Tuple[List[float], dict]:
    """
    Blend Prophet and XGBoost predictions.

    Prophet captures trend + seasonality.
    XGBoost captures residuals and external feature effects.
    Final = prophet_weight * prophet_pred + (1 - prophet_weight) * xgb_pred

    Args:
        df: DataFrame with [date, units_sold, ...feature_cols]
        feature_cols: feature columns for XGBoost
        forecast_year: year to forecast
        future_features_df: 12-row future feature DataFrame for XGBoost
        prophet_weight: blend weight for Prophet (0–1)

    Returns:
        (blended_predictions, {"prophet": ..., "xgboost": ..., "models": {...}})
    """
    from models.prophet_model import train_and_predict_prophet
    from models.xgboost_model import train_and_predict_xgboost

    # ── Prophet component ─────────────────────────────────────────────────
    prophet_preds, prophet_model = train_and_predict_prophet(df, forecast_year)

    # ── XGBoost component ─────────────────────────────────────────────────
    xgb_preds, xgb_model = train_and_predict_xgboost(
        df, feature_cols, forecast_year, future_features_df
    )

    # ── Blend ─────────────────────────────────────────────────────────────
    blended = [
        max(0, prophet_weight * p + (1 - prophet_weight) * x)
        for p, x in zip(prophet_preds, xgb_preds)
    ]

    logger.info(
        f"Hybrid forecast complete: prophet_weight={prophet_weight}, "
        f"xgb_weight={1 - prophet_weight}"
    )

    return blended, {
        "prophet_predictions": prophet_preds,
        "xgboost_predictions": xgb_preds,
        "prophet_weight": prophet_weight,
        "xgboost_weight": 1 - prophet_weight,
        "models": {"prophet": prophet_model, "xgboost": xgb_model},
    }

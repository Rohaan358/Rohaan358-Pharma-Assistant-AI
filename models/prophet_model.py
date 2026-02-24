"""
models/prophet_model.py — Prophet training & prediction
Recommended for: Chronic, Gastro categories
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def train_and_predict_prophet(
    df: pd.DataFrame,
    forecast_year: int = 2025,
) -> Tuple[List[float], object]:
    """
    Train Prophet on historical data (year < forecast_year) and predict
    all 12 months of forecast_year.

    Args:
        df: DataFrame with columns [date, units_sold]
        forecast_year: year to forecast (default 2025)

    Returns:
        (predictions_list, model)
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("prophet package not installed. Run: pip install prophet")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Training data: strictly before forecast_year ──────────────────────
    train = df[df["date"].dt.year < forecast_year].copy()
    train = train.rename(columns={"date": "ds", "units_sold": "y"})
    train = train[["ds", "y"]].dropna()

    if len(train) < 12:
        raise ValueError(f"Insufficient training data: {len(train)} rows (need ≥ 12)")

    # ── Model ─────────────────────────────────────────────────────────────
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )
    model.fit(train)

    # ── Future dataframe for forecast_year ────────────────────────────────
    future_dates = pd.date_range(
        start=f"{forecast_year}-01-01", periods=12, freq="MS"
    )
    future = pd.DataFrame({"ds": future_dates})

    forecast = model.predict(future)
    predictions = forecast["yhat"].clip(lower=0).tolist()

    logger.info(f"Prophet forecast complete: {len(predictions)} months")
    return predictions, model

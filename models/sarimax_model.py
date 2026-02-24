"""
models/sarimax_model.py — SARIMAX training & prediction
Recommended for: Antibiotic, Chronic categories
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def train_and_predict_sarimax(
    df: pd.DataFrame,
    forecast_year: int = 2025,
    exog_cols: Optional[List[str]] = None,
    future_exog_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[float], object]:
    """
    Train SARIMAX on historical data (year < forecast_year) and predict
    all 12 months of forecast_year.

    Handles exogenous variables by filling future values with historical means
    if future_exog_df is not provided.

    Args:
        df: DataFrame with [date, units_sold, ...optional exog cols]
        forecast_year: year to forecast
        exog_cols: exogenous feature columns to include (optional)
        future_exog_df: 12-row DataFrame of future exogenous values

    Returns:
        (predictions_list, model_result)
    """
    df = df.copy()
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        pass  # might already be datetime
    
    df = df.sort_values("date")

    # ── Training data ─────────────────────────────────────────────────────
    train = df[df["date"].dt.year < forecast_year].copy()
    train = train.set_index("date")
    # Set frequency explicitly if possible, else rely on index
    try:
        train.index = pd.DatetimeIndex(train.index).to_period("M")
    except Exception:
        pass

    y_train = train["units_sold"]
    if hasattr(y_train, "fillna"):
         y_train = y_train.fillna(method="ffill").fillna(0) # Deprecated but works, or use ffill()

    if len(y_train) < 24:
        raise ValueError(f"Insufficient training data: {len(y_train)} rows (need ≥ 24)")

    # ── Exogenous features ────────────────────────────────────────────────
    exog_train = None
    exog_future = None
    
    if exog_cols:
        available_exog = [c for c in exog_cols if c in train.columns]
        if available_exog:
            exog_train_df = train[available_exog].fillna(0)
            exog_train = exog_train_df.values
            
            # Prepare future exog
            if future_exog_df is not None:
                # Use provided future values
                # Ensure same columns and order
                missing = [c for c in available_exog if c not in future_exog_df.columns]
                if not missing:
                    exog_future = future_exog_df[available_exog].fillna(0).values
                else:
                    logger.warning(f"Missing cols in future exog: {missing}. Using mean imputation.")
                    means = exog_train_df.mean()
                    exog_future = np.tile(means.values, (12, 1))
            else:
                # Option B: Fill with historical mean
                logger.info("No future exog provided. Filling with historical means.")
                means = exog_train_df.mean()
                exog_future = np.tile(means.values, (12, 1))

    # ── Model Training ────────────────────────────────────────────────────
    # Default order
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)

    try:
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
    except Exception as e:
        logger.warning(f"SARIMAX (1,1,1)(1,1,1,12) failed: {e}. Trying simpler order (1,0,0).")
        try:
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=(1, 0, 0),
                seasonal_order=(1, 0, 0, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)
        except Exception as e2:
            # Fallback Option A: Try without exog if exog caused the issue?
            # But normally we want to catch this upstream in fallback logic.
            # Re-raising for specific error message
            raise RuntimeError(f"SARIMAX training failed: {e} | {e2}")

    # ── Forecast ──────────────────────────────────────────────────────────
    # Steps needs to be 12.
    # Exog must match steps.
    try:
        if exog_train is not None and exog_future is None:
             # Should be handled above, but double check
             # If for some reason exog_future is None but exog_train is not, forecast() will fail
             raise ValueError("Out-of-sample forecast requires exog.")
             
        forecast = result.forecast(steps=12, exog=exog_future)
        
        # Handle index alignment issues if returned as series
        if hasattr(forecast, "values"):
            preds = forecast.values
        else:
            preds = forecast
            
        predictions = np.maximum(preds, 0).tolist()
        
    except Exception as e:
        logger.error(f"SARIMAX forecast step prediction failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")

    logger.info(f"SARIMAX forecast complete: {len(predictions)} months")
    return predictions, result

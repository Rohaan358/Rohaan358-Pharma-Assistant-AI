"""
forecasting/feature_engineering.py — Category-wise feature builder
"""
import pandas as pd
import numpy as np
from typing import Tuple, List


CATEGORY_FEATURES = {
    "antibiotic": [
        "month", "quarter", "year",
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_std_3",
        "month_sin", "month_cos",
        "is_flu_season",          # Nov–Feb
        "disease_index",          # from external_features if available
        "prescription_rate",      # from external_features if available
        "temperature_avg",        # from external_features if available
    ],
    "acute": [
        "month", "quarter", "year",
        "lag_1", "lag_2", "lag_3", "lag_6",
        "rolling_mean_3", "rolling_mean_6", "rolling_std_3",
        "month_sin", "month_cos",
        "promotion_flag",         # from external_features if available
        "is_monsoon",             # Jul–Sep
        "weather_index",          # from external_features if available
        "week_of_year",
    ],
    "chronic": [
        "month", "quarter", "year",
        "lag_1", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
        "rolling_std_6",
        "month_sin", "month_cos",
        "trend_index",            # linear trend component
        "refill_cycle",           # from external_features if available
        "patient_adherence",      # from external_features if available
    ],
    "gastro": [
        "month", "quarter", "year",
        "lag_1", "lag_2", "lag_3", "lag_6",
        "rolling_mean_3", "rolling_mean_6", "rolling_std_3",
        "month_sin", "month_cos",
        "is_festival_period",     # Eid, Ramzan, etc.
        "dietary_index",          # from external_features if available
        "is_monsoon",             # Jul–Sep
    ],
    "other": [
        "month", "quarter", "year",
        "lag_1", "lag_3", "lag_6",
        "rolling_mean_3", "rolling_mean_6",
        "month_sin", "month_cos",
    ],
}


def build_features(df: pd.DataFrame, category: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build category-specific features for a product's time series.

    Args:
        df: DataFrame with columns [date, units_sold, external_features (optional)]
        category: product category string

    Returns:
        (feature_df, feature_names)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ── Time features ──────────────────────────────────────────────────────
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Lag features ──────────────────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["units_sold"].shift(lag)

    # ── Rolling features ──────────────────────────────────────────────────
    for window in [3, 6, 12]:
        df[f"rolling_mean_{window}"] = (
            df["units_sold"].shift(1).rolling(window).mean()
        )
        df[f"rolling_std_{window}"] = (
            df["units_sold"].shift(1).rolling(window).std()
        )

    # ── Trend index ───────────────────────────────────────────────────────
    df["trend_index"] = np.arange(len(df))

    # ── Category-specific binary flags ────────────────────────────────────
    df["is_flu_season"] = df["month"].isin([11, 12, 1, 2]).astype(int)
    df["is_monsoon"] = df["month"].isin([7, 8, 9]).astype(int)
    df["is_festival_period"] = df["month"].isin([3, 4, 6]).astype(int)  # approx Ramzan/Eid

    # ── External features (if present) ────────────────────────────────────
    ext_cols = [
        "disease_index", "prescription_rate", "temperature_avg",
        "promotion_flag", "weather_index", "refill_cycle",
        "patient_adherence", "dietary_index",
    ]
    if "external_features" in df.columns:
        for col in ext_cols:
            df[col] = df["external_features"].apply(
                lambda x: x.get(col, np.nan) if isinstance(x, dict) else np.nan
            )
    else:
        for col in ext_cols:
            df[col] = np.nan

    # Fill external feature NaNs with 0
    df[ext_cols] = df[ext_cols].fillna(0)

    # ── Select category features ───────────────────────────────────────────
    from forecasting.model_selector import get_category_type
    cat = get_category_type(category)
    desired = CATEGORY_FEATURES.get(cat, CATEGORY_FEATURES["other"])
    available = [f for f in desired if f in df.columns]

    return df, available


def build_future_features(
    last_date: pd.Timestamp,
    n_months: int,
    category: str,
    last_known_values: pd.Series,
    external_future: dict = None,
) -> pd.DataFrame:
    """
    Build feature rows for future months (no actual units_sold available).
    Uses rolling/lag approximations from last known values.
    """
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1), periods=n_months, freq="MS"
    )
    rows = []
    history = list(last_known_values)

    for i, dt in enumerate(future_dates):
        row = {"date": dt}
        row["month"] = dt.month
        row["quarter"] = dt.quarter
        row["year"] = dt.year
        row["week_of_year"] = dt.isocalendar()[1]
        row["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * dt.month / 12)
        row["trend_index"] = len(last_known_values) + i

        # Lags from rolling history
        for lag in [1, 2, 3, 6, 12]:
            idx = -(lag)
            row[f"lag_{lag}"] = history[idx] if len(history) >= lag else np.nan

        # Rolling means/stds
        for window in [3, 6, 12]:
            if len(history) >= window:
                row[f"rolling_mean_{window}"] = np.mean(history[-window:])
                row[f"rolling_std_{window}"] = np.std(history[-window:])
            else:
                row[f"rolling_mean_{window}"] = np.mean(history) if history else 0
                row[f"rolling_std_{window}"] = 0

        # Binary flags
        row["is_flu_season"] = int(dt.month in [11, 12, 1, 2])
        row["is_monsoon"] = int(dt.month in [7, 8, 9])
        row["is_festival_period"] = int(dt.month in [3, 4, 6])

        # External future features
        ext_defaults = {
            "disease_index": 0, "prescription_rate": 0, "temperature_avg": 0,
            "promotion_flag": 0, "weather_index": 0, "refill_cycle": 0,
            "patient_adherence": 0, "dietary_index": 0,
        }
        if external_future:
            ext_defaults.update(external_future)
        row.update(ext_defaults)

        rows.append(row)
        # Append a placeholder (rolling mean) to history for next iteration
        history.append(row.get("rolling_mean_3", history[-1] if history else 0))

    return pd.DataFrame(rows)

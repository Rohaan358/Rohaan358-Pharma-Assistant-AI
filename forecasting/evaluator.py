"""
forecasting/evaluator.py — MAE, RMSE, MAPE calculation
"""
import numpy as np
from typing import List, Optional, Dict


def calculate_metrics(
    actual: List[Optional[float]],
    predicted: List[float],
) -> Dict[str, float | str]:
    """
    Calculate MAE, RMSE, MAPE between actual and predicted values.
    Ignores positions where actual is None or NaN.
    """
    pairs = [
        (a, p)
        for a, p in zip(actual, predicted)
        if a is not None and not np.isnan(float(a))
    ]

    if not pairs:
        return {"MAE": 0.0, "RMSE": 0.0, "MAPE": "N/A"}

    actuals = np.array([p[0] for p in pairs], dtype=float)
    preds = np.array([p[1] for p in pairs], dtype=float)

    mae = float(np.mean(np.abs(actuals - preds)))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))

    # MAPE — avoid division by zero
    nonzero_mask = actuals != 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((actuals[nonzero_mask] - preds[nonzero_mask]) / actuals[nonzero_mask])) * 100)
        mape_str = f"{mape:.2f}%"
    else:
        mape_str = "N/A"

    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": mape_str}

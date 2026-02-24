"""
routes/forecast_routes.py — /forecast/* endpoints
"""
import logging
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from database.mongo import (
    find_documents,
    upsert_document,
    aggregate_documents,
    is_connected,
)
from forecasting.feature_engineering import build_features, build_future_features
from forecasting.model_selector import select_model, get_fallback_list
from forecasting.evaluator import calculate_metrics
from schemas.pydantic_models import (
    ForecastRequest,
    ForecastResult,
    ForecastRunResponse,
    ForecastMetrics,
    ForecastCompareResponse,
)

router = APIRouter(prefix="/forecast", tags=["Forecasting"])
logger = logging.getLogger(__name__)

MONTH_LABELS = [
    "Jan-2025", "Feb-2025", "Mar-2025", "Apr-2025",
    "May-2025", "Jun-2025", "Jul-2025", "Aug-2025",
    "Sep-2025", "Oct-2025", "Nov-2025", "Dec-2025",
]


async def _load_product_data(product: str) -> pd.DataFrame:
    """Load all sales records for a product from MongoDB."""
    docs = await find_documents(
        "sales_data",
        {"product_name": product},
        projection={"_id": 0},
        sort_field="date",
    )
    if not docs:
        raise HTTPException(status_code=404, detail=f"No data found for product '{product}'")
    return pd.DataFrame(docs)


def _run_model(model_name: str, df: pd.DataFrame, feature_cols: List[str], future_df: pd.DataFrame, forecast_year: int):
    """Dispatch to the correct model and return predictions."""
    if model_name == "prophet":
        from models.prophet_model import train_and_predict_prophet
        preds, _ = train_and_predict_prophet(df, forecast_year)
        return preds

    elif model_name == "xgboost":
        from models.xgboost_model import train_and_predict_xgboost
        preds, _ = train_and_predict_xgboost(df, feature_cols, forecast_year, future_df)
        return preds

    elif model_name == "sarimax":
        from models.sarimax_model import train_and_predict_sarimax
        exog_cols = [c for c in feature_cols if c not in ["month", "quarter", "year", "trend_index"]]
        preds, _ = train_and_predict_sarimax(df, forecast_year, exog_cols if exog_cols else None)
        return preds

    elif model_name == "hybrid":
        from models.hybrid_model import train_and_predict_hybrid
        preds, _ = train_and_predict_hybrid(df, feature_cols, forecast_year, future_df)
        return preds

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─── POST /forecast/run ───────────────────────────────────────────────────────

@router.post("/run", response_model=ForecastRunResponse)
async def run_forecast(request: ForecastRequest):
    """
    Run forecast for a product.
    - Trains on data < forecast_year
    - Predicts all 12 months of forecast_year
    - Compares with actual data if available
    - Stores result in MongoDB
    """
    if not is_connected():
        raise HTTPException(status_code=400, detail="Not connected to MongoDB.")

    # ── Load data ─────────────────────────────────────────────────────────
    df = await _load_product_data(request.product)

    # ── Determine category ────────────────────────────────────────────────
    category = request.category
    if not category:
        category_val = df["product_category"].mode()
        category = str(category_val.iloc[0]) if not category_val.empty else "other"
    else:
        # Category is now a plain string from request, no .value
        category = category

    # ── Feature engineering ───────────────────────────────────────────────
    feat_df, feature_cols = build_features(df, category)

    # Build future features for XGBoost / Hybrid
    train_df = feat_df[feat_df["date"].dt.year < request.year]
    last_date = train_df["date"].max() if not train_df.empty else pd.Timestamp(f"{request.year - 1}-12-01")
    last_values = train_df["units_sold"].tail(12)
    future_df = build_future_features(last_date, 12, category, last_values)

    # ── Model selection ───────────────────────────────────────────────────
    model_name = select_model(category, request.model.value)

    # ── Run model with fallback ───────────────────────────────────────────
    try:
        predictions = _run_model(model_name, feat_df, feature_cols, future_df, request.year)
    except Exception as e:
        logger.warning(f"Primary model '{model_name}' failed: {e}. Trying fallback.")
        
        fallback_models = get_fallback_list(category)
        success = False
        errors = [f"Primary {model_name}: {str(e)}"]
        
        for fallback in fallback_models:
            print(f"Fallback attempt: {fallback}")
            try:
                predictions = _run_model(fallback, feat_df, feature_cols, future_df, request.year)
                model_name = f"{fallback} (fallback)"
                success = True
                break
            except Exception as e2:
                errors.append(f"Fallback {fallback}: {str(e2)}")
                logger.warning(f"Fallback '{fallback}' failed: {e2}")
        
        if not success:
            raise HTTPException(status_code=500, detail=f"All models failed. Details: {'; '.join(errors)}")

    # ── Load actual 2025 data ─────────────────────────────────────────────
    actual_docs = await find_documents(
        "sales_data",
        {"product_name": request.product, "date": {"$regex": f"^{request.year}"}},
        projection={"_id": 0, "date": 1, "units_sold": 1},
        sort_field="date",
    )
    actual_by_month = {}
    for doc in actual_docs:
        month_num = int(doc["date"][5:7])
        actual_by_month[month_num] = doc["units_sold"]

    actual = [actual_by_month.get(m) for m in range(1, 13)]

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics_dict = calculate_metrics(actual, predictions)
    metrics = ForecastMetrics(**metrics_dict) if metrics_dict.get("MAE") != 0 or any(a is not None for a in actual) else None

    # ── Build result ──────────────────────────────────────────────────────
    result = ForecastResult(
        product=request.product,
        category=category,
        model_used=model_name,
        months=MONTH_LABELS,
        actual=actual,
        predicted=[round(p, 2) for p in predictions],
        metrics=metrics,
        features_used=feature_cols,
    )

    # ── Persist to MongoDB ────────────────────────────────────────────────
    await upsert_document(
        "forecast_results",
        {"product": request.product, "year": request.year},
        result.model_dump(),
    )

    return ForecastRunResponse(status="success", result=result)


# ─── GET /forecast/results ────────────────────────────────────────────────────

@router.get("/results", response_model=List[ForecastResult])
async def get_forecast_results(
    product: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
):
    """Retrieve stored forecast results from MongoDB."""
    if not is_connected():
        raise HTTPException(status_code=400, detail="Not connected to MongoDB.")

    query = {}
    if product:
        query["product"] = product
    if category:
        query["category"] = category

    docs = await find_documents("forecast_results", query, projection={"_id": 0})
    return [ForecastResult(**d) for d in docs]


# ─── GET /forecast/compare ────────────────────────────────────────────────────

@router.get("/compare", response_model=List[ForecastCompareResponse])
async def compare_forecast(product: Optional[str] = Query(None)):
    """Actual vs Predicted comparison for forecast year."""
    if not is_connected():
        raise HTTPException(status_code=400, detail="Not connected to MongoDB.")

    query = {}
    if product:
        query["product"] = product

    docs = await find_documents("forecast_results", query, projection={"_id": 0})
    if not docs:
        raise HTTPException(status_code=404, detail="No forecast results found.")

    responses = []
    for doc in docs:
        months = doc.get("months", MONTH_LABELS)
        actual = doc.get("actual", [])
        predicted = doc.get("predicted", [])

        comparison = [
            {
                "month": m,
                "actual": a,
                "predicted": p,
                "difference": round(p - (a or 0), 2) if a is not None else None,
                "pct_error": (
                    f"{abs((p - a) / a * 100):.1f}%"
                    if a and a != 0
                    else "N/A"
                ),
            }
            for m, a, p in zip(months, actual, predicted)
        ]

        metrics = doc.get("metrics")
        responses.append(
            ForecastCompareResponse(
                product=doc["product"],
                category=doc.get("category", ""),
                model_used=doc.get("model_used", ""),
                comparison=comparison,
                metrics=ForecastMetrics(**metrics) if metrics else None,
            )
        )
    return responses


# ─── GET /forecast/plot ───────────────────────────────────────────────────────

@router.get("/plot")
async def get_plot_data(product: Optional[str] = Query(None)):
    """
    Return plot-ready JSON for actual vs predicted visualization.
    Format matches the specification in the project prompt.
    """
    if not is_connected():
        raise HTTPException(status_code=400, detail="Not connected to MongoDB.")

    query = {}
    if product:
        query["product"] = product

    docs = await find_documents("forecast_results", query, projection={"_id": 0})
    if not docs:
        raise HTTPException(status_code=404, detail="No forecast results found.")

    # Return in the exact plot format specified
    plots = []
    for doc in docs:
        metrics = doc.get("metrics", {})
        plots.append({
            "product": doc["product"],
            "category": doc.get("category", ""),
            "model_used": doc.get("model_used", ""),
            "months": doc.get("months", MONTH_LABELS),
            "actual": doc.get("actual", []),
            "predicted": doc.get("predicted", []),
            "metrics": {
                "MAE": metrics.get("MAE", 0),
                "RMSE": metrics.get("RMSE", 0),
                "MAPE": metrics.get("MAPE", "N/A"),
            },
        })

    return plots if len(plots) > 1 else plots[0] if plots else {}

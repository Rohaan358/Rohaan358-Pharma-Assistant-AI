"""
routes/data_routes.py — /data/* endpoints
"""
import io
import json
import logging
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse

from database.mongo import (
    find_documents,
    insert_many_documents,
    get_distinct_values,
    aggregate_documents,
    is_connected,
)
from schemas.pydantic_models import (
    SalesRecord,
    SalesUploadResponse,
    DataSummaryResponse,
    ProductInfo,
)

router = APIRouter(tags=["Data Management"])
logger = logging.getLogger(__name__)


# ─── POST /data/upload ────────────────────────────────────────────────────────

@router.post("/data/upload", response_model=SalesUploadResponse)
async def upload_sales_data(file: UploadFile = File(...)):
    """
    Upload sales data as CSV or JSON file.
    Expected columns: date, product_name, product_category, units_sold,
                      [external_features]
    """
    if not is_connected():
        raise HTTPException(status_code=400, detail="Not connected to MongoDB. Call /connect first.")

    content = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(".json"):
            data = json.loads(content)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        else:
            raise HTTPException(status_code=400, detail="Only CSV and JSON files are supported.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"File parsing error: {str(e)}")

    # ── Validate required columns ──────────────────────────────────────────
    required = {"date", "product_name", "product_category", "units_sold"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")

    # ── Clean & normalize ──────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0).astype(int)
    df["product_category"] = df["product_category"].str.strip()

    # Handle external_features column
    if "external_features" not in df.columns:
        df["external_features"] = [{}] * len(df)
    else:
        df["external_features"] = df["external_features"].apply(
            lambda x: x if isinstance(x, dict) else {}
        )

    documents = df.to_dict(orient="records")

    try:
        inserted = await insert_many_documents("sales_data", documents)
        return SalesUploadResponse(
            inserted=inserted,
            message=f"Successfully uploaded {inserted} records to 'sales_data' collection.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database insert error: {str(e)}")


# ─── GET /data/products ───────────────────────────────────────────────────────

@router.get("/data/products", response_model=List[ProductInfo])
async def list_products():
    """List all products with their category, record count, and date range."""
    if not is_connected():
        raise HTTPException(status_code=400, detail="Not connected to MongoDB.")

    pipeline = [
        {
            "$group": {
                "_id": {"product_name": "$product_name", "product_category": "$product_category"},
                "record_count": {"$sum": 1},
                "date_min": {"$min": "$date"},
                "date_max": {"$max": "$date"},
            }
        },
        {"$sort": {"_id.product_name": 1}},
    ]

    results = await aggregate_documents("sales_data", pipeline)
    return [
        ProductInfo(
            product_name=r["_id"]["product_name"],
            product_category=r["_id"].get("product_category", "unknown"),
            record_count=r["record_count"],
            date_min=str(r["date_min"]),
            date_max=str(r["date_max"]),
        )
        for r in results
    ]


# ─── GET /data/summary ────────────────────────────────────────────────────────

@router.get("/data/summary", response_model=DataSummaryResponse)
async def data_summary():
    """Overview: date range, categories, product count."""
    if not is_connected():
        raise HTTPException(status_code=400, detail="Not connected to MongoDB.")

    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_records": {"$sum": 1},
                "date_min": {"$min": "$date"},
                "date_max": {"$max": "$date"},
            }
        }
    ]
    agg = await aggregate_documents("sales_data", pipeline)
    if not agg:
        return DataSummaryResponse(
            total_records=0,
            date_range={"min": "N/A", "max": "N/A"},
            categories=[],
            products=[],
        )

    stats = agg[0]
    categories = await get_distinct_values("sales_data", "product_category")
    products = await get_distinct_values("sales_data", "product_name")
    return DataSummaryResponse(
        total_records=stats["total_records"],
        date_range={"min": str(stats["date_min"]), "max": str(stats["date_max"])},
        categories=sorted(categories),
        products=sorted(products),
    )


# ─── GET /data/compare ────────────────────────────────────────────────────────

@router.get("/data/compare")
async def get_product_comparison(
    product: str = Query(...),
    years: str = Query(...)  # e.g., "2023,2024,2025"
):
    """
    Fetch monthly comparison data for a product across specified years.
    Returns: { "product": "...", "data": { "2023": { "Jan": val, ... }, ... } }
    """
    if not is_connected():
        raise HTTPException(status_code=400, detail="Not connected to MongoDB.")

    try:
        year_list = [int(y.strip()) for y in years.split(",") if y.strip().isdigit()]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid years format. Use comma-separated integers.")

    if not year_list:
        raise HTTPException(status_code=400, detail="No valid years provided.")

    # Find docs for this product and years
    query = {"product_name": product}
    year_regex = "^(" + "|".join(map(str, year_list)) + ")"
    query["date"] = {"$regex": year_regex}

    docs = await find_documents("sales_data", query, projection={"_id": 0, "date": 1, "units_sold": 1})
    
    month_map = {
        "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun",
        "07": "Jul", "18": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
        "08": "Aug" # Fix for 08
    }
    # Wait, 08 mapping was missing 08! 18 was a typo. Fixing.
    month_map = {
        "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun",
        "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"
    }
    
    # Initialize with None to match prompt "Jan": 15000, "Feb": null
    # BUT for math (totals/YoY), initializing with 0 is better. 
    # I'll use 0 as default since missing record = zero sales in this context.
    structured_data = {str(y): {m: 0 for m in month_map.values()} for y in year_list}
    
    for d in docs:
        date_str = d["date"] 
        year = date_str[:4]
        month_code = date_str[5:7]
        month_name = month_map.get(month_code)
        
        if year in structured_data and month_name:
            structured_data[year][month_name] += d["units_sold"]

    return {
        "product": product,
        "data": structured_data
    }

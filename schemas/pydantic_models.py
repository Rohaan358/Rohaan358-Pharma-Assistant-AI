"""
schemas/pydantic_models.py — All Pydantic v2 request/response schemas
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import date
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class ForecastModel(str, Enum):
    prophet = "prophet"
    xgboost = "xgboost"
    sarimax = "sarimax"
    hybrid = "hybrid"
    auto = "auto"


# ─── Data Schemas ─────────────────────────────────────────────────────────────

class SalesRecord(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    product_name: str
    product_category: str
    units_sold: int = Field(..., ge=0)
    external_features: Optional[Dict[str, Any]] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")
        return v


class SalesUploadResponse(BaseModel):
    inserted: int
    message: str


class ProductInfo(BaseModel):
    product_name: str
    product_category: str
    record_count: int
    date_min: str
    date_max: str


class DataSummaryResponse(BaseModel):
    total_records: int
    date_range: Dict[str, str]
    categories: List[str]
    products: List[str]


# ─── Forecast Schemas ─────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    product: str
    category: Optional[str] = None
    model: ForecastModel = ForecastModel.auto
    year: int = 2025


class ForecastMetrics(BaseModel):
    MAE: float
    RMSE: float
    MAPE: str


class ForecastResult(BaseModel):
    product: str
    category: str
    model_used: str
    months: List[str]
    actual: List[Optional[float]]
    predicted: List[float]
    metrics: Optional[ForecastMetrics] = None
    features_used: Optional[List[str]] = None


class ForecastRunResponse(BaseModel):
    status: str
    result: ForecastResult


class ForecastCompareResponse(BaseModel):
    product: str
    category: str
    model_used: str
    comparison: List[Dict[str, Any]]
    metrics: Optional[ForecastMetrics] = None


# ─── Agent Schemas ────────────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    product: Optional[str] = None
    category: Optional[str] = None
    include_forecast_context: bool = True


class AgentQueryResponse(BaseModel):
    query: str
    response: str
    context_used: Optional[str] = None
    detected_product: Optional[str] = None
    detected_category: Optional[str] = None


class AgentAnalyzeRequest(BaseModel):
    product: Optional[str] = None
    category: Optional[str] = None
    analysis_type: Literal["trend", "anomaly", "comparison", "full"] = "full"


class AgentAnalyzeResponse(BaseModel):
    subject: str
    analysis_type: str
    insights: str
    recommendations: List[str]


# ─── Chat History Schemas ───────────────────────────────────────────────────

class ChatHistorySaveRequest(BaseModel):
    session_id: str
    role: str
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatHistoryEntry(BaseModel):
    role: str
    message: str
    timestamp: str
    context: Optional[Dict[str, Any]] = None


class ChatSessionSummary(BaseModel):
    session_id: str
    start_time: str
    message_count: int
    first_message: str
    context: Optional[Dict[str, Any]] = None

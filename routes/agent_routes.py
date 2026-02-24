"""
routes/agent_routes.py — /agent/* endpoints
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query

from datetime import datetime
from agent.llama_client import call_llama
from agent.prompts import build_system_prompt, build_insights_prompt, build_analysis_prompt, detect_context
from database.mongo import (
    find_documents, 
    aggregate_documents, 
    is_connected,
    save_chat,
    get_chat_history,
    get_all_sessions,
    delete_session
)
from schemas.pydantic_models import (
    AgentQueryRequest,
    AgentQueryResponse,
    AgentAnalyzeRequest,
    AgentAnalyzeResponse,
    ChatHistorySaveRequest,
    ChatHistoryEntry,
    ChatSessionSummary,
)

router = APIRouter(prefix="/agent", tags=["AI Agent"])
logger = logging.getLogger(__name__)


async def _get_data_summary() -> dict:
    """Fetch a lightweight data summary for context injection."""
    if not is_connected():
        return {}
    try:
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
        stats = agg[0] if agg else {}

        from database.mongo import get_distinct_values
        categories = await get_distinct_values("sales_data", "product_category")
        products = await get_distinct_values("sales_data", "product_name")

        return {
            "total_records": stats.get("total_records", 0),
            "date_range": {
                "min": str(stats.get("date_min", "N/A")),
                "max": str(stats.get("date_max", "N/A")),
            },
            "categories": sorted(categories),
            "products": sorted(products),
        }
    except Exception as e:
        logger.warning(f"Could not fetch data summary: {e}")
        return {}


async def _get_recent_forecasts(product: Optional[str] = None, category: Optional[str] = None) -> list:
    """Fetch recent forecast results for context injection."""
    if not is_connected():
        return []
    try:
        query = {}
        if product:
            query["product"] = product
        if category:
            query["category"] = category
        docs = await find_documents("forecast_results", query, projection={"_id": 0}, limit=10)
        return docs
    except Exception:
        return []


# ─── POST /agent/query ────────────────────────────────────────────────────────

@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    """
    Natural language query → Llama 3.3 70B response.
    Injects relevant MongoDB context into the system prompt.
    Auto-detects product/category if not manually selected.
    Auto-saves BOTH messages if session_id is provided.
    """
    context_parts = []
    
    # ── Auto-Detection ────────────────────────────────────────────────────
    data_summary = await _get_data_summary()
    all_prods = data_summary.get("products", [])
    
    det_p, det_c = detect_context(request.query, all_prods)
    
    final_p = request.product or det_p
    final_c = request.category or det_c
    
    if final_p: context_parts.append(f"Auto-detected Product: {final_p}" if not request.product else f"Product: {final_p}")
    if final_c: context_parts.append(f"Auto-detected Category: {final_c}" if not request.category else f"Category: {final_c}")

    if request.include_forecast_context:
        forecasts = await _get_recent_forecasts(final_p, final_c)
        system_prompt = build_system_prompt(
            data_summary=data_summary,
            forecast_results=forecasts,
            product=final_p,
            category=final_c,
        )
        if data_summary:
            context_parts.append(f"DB: {data_summary.get('total_records', 0)} records")
        if forecasts:
            context_parts.append(f"Forecasts: {len(forecasts)} loaded")
    else:
        system_prompt = build_system_prompt(product=final_p, category=final_c)

    try:
        response = await call_llama(system_prompt, request.query)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI agent error: {str(e)}")

    # ── Auto-save History ──────────────────────────────────────────────────
    if request.session_id:
        timestamp = datetime.now().isoformat()
        chat_context = {
            "product": final_p,
            "category": final_c,
            "include_forecast": request.include_forecast_context
        }
        
        # 1. Save User Message
        await save_chat({
            "session_id": request.session_id,
            "timestamp": timestamp,
            "role": "user",
            "message": request.query,
            "context": chat_context
        })
        
        # 2. Save Agent Response
        await save_chat({
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat(),
            "role": "assistant",
            "message": response,
            "context": chat_context
        })

    return AgentQueryResponse(
        query=request.query,
        response=response,
        context_used="; ".join(context_parts) if context_parts else None,
        detected_product=final_p,
        detected_category=final_c
    )


# ─── GET /agent/insights ──────────────────────────────────────────────────────

@router.get("/insights")
async def agent_insights():
    """
    Auto-generate insights on overall sales trends and forecast performance.
    """
    data_summary = await _get_data_summary()
    forecasts = await _get_recent_forecasts()

    if not data_summary and not forecasts:
        raise HTTPException(
            status_code=400,
            detail="No data available. Upload sales data and run forecasts first."
        )

    system_prompt, user_message = build_insights_prompt(data_summary, forecasts)

    try:
        response = await call_llama(system_prompt, user_message, temperature=0.4, max_tokens=2000)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI agent error: {str(e)}")

    return {
        "insights": response,
        "data_context": {
            "total_records": data_summary.get("total_records", 0),
            "products_analyzed": len(data_summary.get("products", [])),
            "forecasts_available": len(forecasts),
        },
    }


# ─── POST /agent/analyze ──────────────────────────────────────────────────────

@router.post("/analyze", response_model=AgentAnalyzeResponse)
async def agent_analyze(request: AgentAnalyzeRequest):
    """
    Deep analysis for a specific product or category.
    """
    if not request.product and not request.category:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: product, category"
        )

    subject = request.product or request.category
    data_summary = await _get_data_summary()
    forecasts = await _get_recent_forecasts(request.product, request.category)

    context_data = {
        "summary": data_summary,
        "forecasts": forecasts,
        "product": request.product,
        "category": request.category,
    }

    system_prompt, user_message = build_analysis_prompt(
        subject=subject,
        analysis_type=request.analysis_type,
        context_data=context_data,
    )

    try:
        response = await call_llama(system_prompt, user_message, temperature=0.3, max_tokens=2000)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI agent error: {str(e)}")

    # Parse recommendations from response (simple heuristic)
    lines = response.split("\n")
    recommendations = [
        line.lstrip("•-*123456789. ").strip()
        for line in lines
        if line.strip() and any(kw in line.lower() for kw in ["recommend", "suggest", "should", "consider", "action"])
    ][:5]

    return AgentAnalyzeResponse(
        subject=subject,
        analysis_type=request.analysis_type,
        insights=response,
        recommendations=recommendations if recommendations else ["See full analysis above."],
    )


# ─── History Endpoints ────────────────────────────────────────────────────────

@router.post("/history/save")
async def save_history_endpoint(request: ChatHistorySaveRequest):
    """Insert one document to chat_history collection."""
    if not is_connected():
        raise HTTPException(status_code=400, detail="Database not connected.")
    
    doc = {
        "session_id": request.session_id,
        "timestamp": datetime.now().isoformat(),
        "role": request.role,
        "message": request.message,
        "context": request.context
    }
    await save_chat(doc)
    return {"saved": True}


@router.get("/history/all", response_model=List[ChatSessionSummary])
async def list_all_sessions():
    """Return all sessions summary."""
    if not is_connected():
        raise HTTPException(status_code=400, detail="Database not connected.")
    return await get_all_sessions()


@router.get("/history/{session_id}", response_model=List[ChatHistoryEntry])
async def get_session_history(session_id: str):
    """Return all messages for this session sorted by timestamp ASC."""
    if not is_connected():
        raise HTTPException(status_code=400, detail="Database not connected.")
    return await get_chat_history(session_id)


@router.delete("/history/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Delete entire session from MongoDB."""
    if not is_connected():
        raise HTTPException(status_code=400, detail="Database not connected.")
    await delete_session(session_id)
    return {"deleted": True}

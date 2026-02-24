"""
main.py — FastAPI application entry point
Pharma Sales Forecasting Agent
"""
import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from routes.data_routes import router as data_router
from routes.forecast_routes import router as forecast_router
from routes.agent_routes import router as agent_router

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from database.mongo import connect_to_mongodb, close_mongo_connection
    
    # Auto-connect to MongoDB on startup
    await connect_to_mongodb()
    
    yield
    await close_mongo_connection()
    logger.info("Pharma Agent shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Pharma Sales Forecasting Agent",
    description=(
        "An intelligent backend system for pharmaceutical sales forecasting. "
        "Supports Prophet, XGBoost, SARIMAX, and Hybrid models with "
        "Meta Llama 3.3 70B AI-powered analysis."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(data_router)
app.include_router(forecast_router)
app.include_router(agent_router)


# ─── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Pharma Sales Forecasting Agent",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "data": [
                "POST /data/upload",
                "GET  /data/products",
                "GET  /data/summary",
            ],
            "forecasting": [
                "POST /forecast/run",
                "GET  /forecast/results",
                "GET  /forecast/compare",
                "GET  /forecast/plot",
            ],
            "agent": [
                "POST /agent/query",
                "GET  /agent/insights",
                "POST /agent/analyze",
            ],
        },
    }


@app.get("/health", tags=["Health"])
async def health():
    from database.mongo import is_connected
    return {
        "status": "healthy",
        "mongodb_connected": is_connected(),
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

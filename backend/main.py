"""
Quantitative Trading Dashboard API
===================================
FastAPI application entry point.

Features:
- RESTful API for trade tracking
- Portfolio metrics calculation
- Dashboard data aggregation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from .config import settings
from .database.connection import init_db
from .routers import trades_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 Starting Quant Trading Dashboard API...")
    init_db()
    print("✅ Database initialized")
    yield
    # Shutdown
    print("👋 Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Quantitative Trading Dashboard API
    
    A comprehensive API for tracking trades and portfolio performance.
    
    ### Features
    - **Trade Management**: Create, read, update, delete trades
    - **Portfolio Metrics**: Real-time P&L, win rate, Sharpe ratio
    - **Dashboard Data**: Aggregated summaries for visualization
    
    ### Trade Workflow
    1. Create trade with entry details
    2. Monitor open positions
    3. Close trade with exit price
    4. View performance metrics
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trades_router)


# Health check endpoint
@app.get("/health", tags=["health"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }


@app.get("/", tags=["root"])
def root():
    """Root endpoint with API information."""
    return {
        "message": "Quant Trading Dashboard API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


def start_server():
    """Start the API server."""
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )


if __name__ == "__main__":
    start_server()

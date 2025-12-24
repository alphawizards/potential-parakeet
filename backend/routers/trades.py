"""
Trade API Routes
================
RESTful API endpoints for trade operations.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from ..database.connection import get_db
from ..database.schemas import (
    TradeCreate, TradeUpdate, TradeResponse, TradeListResponse,
    PortfolioMetrics, DashboardSummary, TradeStatsResponse
)
from ..services.trade_service import TradeService

router = APIRouter(prefix="/api/trades", tags=["trades"])


# ============== Dependencies ==============

def get_trade_service(db: Session = Depends(get_db)) -> TradeService:
    """Dependency injection for TradeService."""
    return TradeService(db)


# ============== CRUD Endpoints ==============

@router.post("/", response_model=TradeResponse, status_code=status.HTTP_201_CREATED)
def create_trade(
    trade_data: TradeCreate,
    service: TradeService = Depends(get_trade_service)
):
    """
    Create a new trade.
    
    - **trade_id**: Unique identifier for the trade
    - **ticker**: Asset ticker symbol (e.g., SPY, QQQ)
    - **direction**: BUY or SELL
    - **quantity**: Number of units
    - **entry_price**: Price per unit at entry
    - **entry_date**: Date and time of trade entry
    """
    try:
        trade = service.create_trade(trade_data)
        return TradeResponse.model_validate(trade)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/", response_model=TradeListResponse)
def get_trades(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    status: Optional[str] = Query(None, description="Filter by status (OPEN, CLOSED, CANCELLED)"),
    strategy: Optional[str] = Query(None, description="Filter by strategy name"),
    start_date: Optional[datetime] = Query(None, description="Filter trades after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter trades before this date"),
    sort_by: str = Query("entry_date", description="Sort by field"),
    sort_desc: bool = Query(True, description="Sort descending"),
    service: TradeService = Depends(get_trade_service)
):
    """
    Get paginated list of trades with optional filters.
    
    Returns rolling history of all trades with pagination.
    """
    return service.get_trades(
        page=page,
        page_size=page_size,
        ticker=ticker,
        status=status,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_by,
        sort_desc=sort_desc
    )


@router.get("/{trade_id}", response_model=TradeResponse)
def get_trade(
    trade_id: int,
    service: TradeService = Depends(get_trade_service)
):
    """Get a single trade by ID."""
    trade = service.get_trade(trade_id)
    if not trade:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with ID {trade_id} not found"
        )
    return TradeResponse.model_validate(trade)


@router.patch("/{trade_id}", response_model=TradeResponse)
def update_trade(
    trade_id: int,
    trade_data: TradeUpdate,
    service: TradeService = Depends(get_trade_service)
):
    """
    Update an existing trade.
    
    Can be used to:
    - Close a trade (set exit_price, exit_date, status=CLOSED)
    - Add notes
    - Cancel a trade (status=CANCELLED)
    """
    trade = service.update_trade(trade_id, trade_data)
    if not trade:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with ID {trade_id} not found"
        )
    return TradeResponse.model_validate(trade)


@router.post("/{trade_id}/close", response_model=TradeResponse)
def close_trade(
    trade_id: int,
    exit_price: float = Query(..., gt=0, description="Exit price"),
    exit_date: Optional[datetime] = Query(None, description="Exit date (defaults to now)"),
    service: TradeService = Depends(get_trade_service)
):
    """
    Close an open trade.
    
    Automatically calculates P&L based on entry/exit prices.
    """
    trade = service.close_trade(trade_id, exit_price, exit_date)
    if not trade:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with ID {trade_id} not found or already closed"
        )
    return TradeResponse.model_validate(trade)


@router.delete("/{trade_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_trade(
    trade_id: int,
    service: TradeService = Depends(get_trade_service)
):
    """Delete a trade by ID."""
    success = service.delete_trade(trade_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade with ID {trade_id} not found"
        )


# ============== Metrics Endpoints ==============

@router.get("/metrics/portfolio", response_model=PortfolioMetrics)
def get_portfolio_metrics(
    initial_capital: float = Query(100000.0, gt=0, description="Initial capital in AUD"),
    service: TradeService = Depends(get_trade_service)
):
    """
    Get portfolio performance metrics.
    
    Includes:
    - Total value, cash balance, invested value
    - Total return, P&L
    - Win rate, trade statistics
    """
    return service.get_portfolio_metrics(initial_capital)


@router.get("/metrics/dashboard", response_model=DashboardSummary)
def get_dashboard_summary(
    initial_capital: float = Query(100000.0, gt=0, description="Initial capital in AUD"),
    service: TradeService = Depends(get_trade_service)
):
    """
    Get complete dashboard summary.
    
    Includes:
    - Portfolio metrics
    - Recent trades
    - Period P&L (today, week, month)
    - Open positions count
    """
    return service.get_dashboard_summary(initial_capital)


@router.get("/metrics/by-ticker", response_model=list[TradeStatsResponse])
def get_stats_by_ticker(
    service: TradeService = Depends(get_trade_service)
):
    """
    Get performance statistics grouped by ticker.
    
    Useful for identifying best/worst performing assets.
    """
    stats = service.get_stats_by_ticker()
    return [
        TradeStatsResponse(
            ticker=s['ticker'],
            total_trades=s['total_trades'],
            total_pnl=s['total_pnl'],
            avg_pnl=s['avg_pnl'],
            win_rate=0  # Would need additional calculation
        )
        for s in stats
    ]


# ============== Utility Endpoints ==============

@router.get("/utils/generate-id")
def generate_trade_id(
    prefix: str = Query("TRD", description="Trade ID prefix"),
    service: TradeService = Depends(get_trade_service)
):
    """Generate a unique trade ID."""
    return {"trade_id": service.generate_trade_id(prefix)}

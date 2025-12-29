"""
Universes Router
================
Endpoints for stock universe management.

Provides:
- List all available universes
- Get universe details and tickers
- Support for SPX500, NASDAQ100, ASX200, Russell 2000, ETFs
"""

import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent path for strategy imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategy.stock_universe import (
    UNIVERSE_REGISTRY,
    get_universe_tickers,
    get_universe_info,
    list_universes
)

router = APIRouter(prefix="/api/universes", tags=["universes"])


class UniverseInfo(BaseModel):
    """Information about a stock universe."""
    key: str
    name: str
    description: str
    region: str
    asset_class: str
    ticker_count: int


class UniverseDetail(UniverseInfo):
    """Detailed universe info including tickers."""
    tickers: List[str]


class UniverseListResponse(BaseModel):
    """Response for listing all universes."""
    universes: List[UniverseInfo]
    count: int


class TickerListResponse(BaseModel):
    """Response for getting universe tickers."""
    universe: str
    tickers: List[str]
    count: int


@router.get("/", response_model=UniverseListResponse)
async def get_universes():
    """
    List all available stock universes.
    
    Returns universe metadata including name, description, region, and ticker count.
    """
    universes = list_universes()
    return UniverseListResponse(
        universes=[UniverseInfo(**u) for u in universes],
        count=len(universes)
    )


@router.get("/{universe_key}", response_model=UniverseDetail)
async def get_universe(universe_key: str):
    """
    Get detailed information about a specific universe.
    
    Includes metadata and full ticker list.
    """
    try:
        info = get_universe_info(universe_key)
        tickers = get_universe_tickers(universe_key)
        return UniverseDetail(
            **info,
            tickers=tickers
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )


@router.get("/{universe_key}/tickers", response_model=TickerListResponse)
async def get_universe_ticker_list(universe_key: str):
    """
    Get just the ticker list for a universe.
    
    Lighter endpoint when only tickers are needed.
    """
    try:
        tickers = get_universe_tickers(universe_key)
        return TickerListResponse(
            universe=universe_key,
            tickers=tickers,
            count=len(tickers)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )


@router.get("/regions/{region}")
async def get_universes_by_region(region: str) -> Dict[str, Any]:
    """
    Get universes filtered by region.
    
    Valid regions: US, AU, GLOBAL
    """
    region_upper = region.upper()
    universes = [
        u for u in list_universes() 
        if u["region"] == region_upper
    ]
    
    if not universes:
        available_regions = list(set(u["region"] for u in list_universes()))
        raise HTTPException(
            status_code=404,
            detail=f"No universes found for region '{region}'. Available: {available_regions}"
        )
    
    return {
        "region": region_upper,
        "universes": universes,
        "count": len(universes)
    }

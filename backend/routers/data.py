"""
Data Router
===========
Endpoints for data management: refresh, status, universe.

Data Source Routing:
- Tiingo: US Stocks, US ETFs, Mutual Funds, Chinese A-Shares, Gold
- yFinance: ASX Stocks, ASX ETFs, VIX, BTC-USD
"""

import os
import sys
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Add parent path for strategy imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/data", tags=["data"])


class RefreshRequest(BaseModel):
    """Request model for data refresh."""
    tickers: Optional[List[str]] = None
    force: bool = False  # Force refresh even if cache is fresh


class RefreshResponse(BaseModel):
    """Response model for data refresh."""
    status: str
    message: str
    tickers_refreshed: int = 0
    timestamp: str
    cache_status: Optional[Dict[str, Any]] = None


class DataStatus(BaseModel):
    """Response model for data status."""
    tiingo_status: str
    yfinance_status: str
    cache_size_mb: float
    cache_files: int
    last_refresh: Optional[str]
    us_tickers_cached: int
    asx_tickers_cached: int
    oldest_cache: Optional[str]
    newest_cache: Optional[str]


def get_data_source(ticker: str) -> str:
    """
    Determine which data source to use for a ticker.
    
    Tiingo covers:
    - US Stocks (NYSE, NASDAQ, etc.)
    - US ETFs (SPY, QQQ, GLD, etc.)
    - Mutual Funds
    - Chinese A-Shares
    
    yFinance covers:
    - ASX Stocks (.AX suffix)
    - ASX ETFs (.AX suffix)
    - VIX Index (^VIX)
    - Bitcoin (BTC-USD, BTC-AUD)
    """
    # ASX tickers
    if ticker.endswith('.AX'):
        return 'yfinance'
    
    # Special indices and crypto
    if ticker in ['^VIX', '^GSPC', '^DJI', '^IXIC', 'BTC-USD', 'BTC-AUD', 'ETH-USD']:
        return 'yfinance'
    
    # Everything else → Tiingo (US stocks, ETFs, Gold)
    return 'tiingo'


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    cache_dir = Path(__file__).parent.parent.parent / "cache"
    
    stats = {
        "cache_dir": str(cache_dir),
        "total_files": 0,
        "total_size_mb": 0.0,
        "oldest_cache": None,
        "newest_cache": None,
        "files": []
    }
    
    if not cache_dir.exists():
        return stats
    
    cache_files = list(cache_dir.glob("*.parquet"))
    stats["total_files"] = len(cache_files)
    
    if cache_files:
        total_bytes = sum(f.stat().st_size for f in cache_files)
        stats["total_size_mb"] = round(total_bytes / 1024 / 1024, 2)
        
        mtimes = [f.stat().st_mtime for f in cache_files]
        stats["oldest_cache"] = datetime.fromtimestamp(min(mtimes)).isoformat()
        stats["newest_cache"] = datetime.fromtimestamp(max(mtimes)).isoformat()
        
        # Get file details
        for f in cache_files[:10]:  # Limit to first 10
            stats["files"].append({
                "name": f.name,
                "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
    
    return stats


@router.get("/status", response_model=DataStatus)
async def get_data_status():
    """
    Get status of data sources and cache.
    
    Returns:
    - Tiingo API status
    - yFinance status
    - Cache statistics
    """
    from ..config import settings
    
    cache_stats = get_cache_stats()
    
    # Check Tiingo API key
    tiingo_status = "configured" if settings.TIINGO_API_KEY else "not_configured"
    
    # Count cached tickers (rough estimate)
    us_cached = 0
    asx_cached = 0
    
    cache_dir = Path(__file__).parent.parent.parent / "cache"
    if cache_dir.exists():
        for f in cache_dir.glob("*.parquet"):
            if "tiingo" in f.name.lower() or "us" in f.name.lower():
                us_cached += 1
            elif "ax" in f.name.lower() or "asx" in f.name.lower():
                asx_cached += 1
            else:
                us_cached += 1  # Default to US
    
    return DataStatus(
        tiingo_status=tiingo_status,
        yfinance_status="available",
        cache_size_mb=cache_stats["total_size_mb"],
        cache_files=cache_stats["total_files"],
        last_refresh=cache_stats["newest_cache"],
        us_tickers_cached=us_cached,
        asx_tickers_cached=asx_cached,
        oldest_cache=cache_stats["oldest_cache"],
        newest_cache=cache_stats["newest_cache"]
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_data(
    request: RefreshRequest,
    background_tasks: BackgroundTasks
):
    """
    Refresh market data from Tiingo/yFinance.
    
    - If tickers provided: Refresh only those tickers
    - If no tickers: Refresh entire universe
    - force=true: Bypass cache freshness check
    
    Data is fetched in background to avoid timeout.
    """
    from ..config import settings
    
    try:
        # Import data loader
        from strategy.fast_data_loader import FastDataLoader
        
        loader = FastDataLoader(
            use_tiingo_fallback=True,
            tiingo_api_token=settings.TIINGO_API_KEY,
            tiingo_is_premium=settings.TIINGO_IS_PREMIUM
        )
        
        if request.tickers:
            tickers = request.tickers
        else:
            # Get full universe
            from strategy.stock_universe import get_screener_universe
            tickers = get_screener_universe()
        
        # Split by data source
        tiingo_tickers = [t for t in tickers if get_data_source(t) == 'tiingo']
        yfinance_tickers = [t for t in tickers if get_data_source(t) == 'yfinance']
        
        # Background refresh
        def do_refresh():
            try:
                if tiingo_tickers:
                    print(f"Refreshing {len(tiingo_tickers)} tickers from Tiingo...")
                    loader.fetch_prices_fast(tiingo_tickers, use_cache=not request.force)
                
                if yfinance_tickers:
                    print(f"Refreshing {len(yfinance_tickers)} tickers from yFinance...")
                    loader.fetch_prices_fast(yfinance_tickers, use_cache=not request.force)
                    
                print("Data refresh complete!")
            except Exception as e:
                print(f"Data refresh error: {e}")
        
        background_tasks.add_task(do_refresh)
        
        return RefreshResponse(
            status="started",
            message=f"Refreshing {len(tiingo_tickers)} US tickers (Tiingo) and {len(yfinance_tickers)} ASX tickers (yFinance)",
            tickers_refreshed=len(tickers),
            timestamp=datetime.now().isoformat(),
            cache_status=get_cache_stats()
        )
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to import data loader: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data refresh failed: {str(e)}"
        )


@router.get("/universe")
async def get_universe(
    include_sp500: bool = True,
    include_nasdaq100: bool = True,
    include_asx200: bool = True,
    include_etfs: bool = True
) -> Dict[str, Any]:
    """
    Get available ticker universe.
    
    Parameters:
    - include_sp500: Include S&P 500 constituents
    - include_nasdaq100: Include NASDAQ 100 constituents
    - include_asx200: Include ASX 200 constituents
    - include_etfs: Include core ETFs
    
    Returns grouped tickers with data source info.
    """
    try:
        from strategy.stock_universe import (
            get_screener_universe,
            get_sp500_tickers,
            get_nasdaq100_tickers,
            get_asx200_tickers,
            get_core_etfs,
            get_etf_info
        )
        
        universe = get_screener_universe(
            include_sp500=include_sp500,
            include_nasdaq100=include_nasdaq100,
            include_asx200=include_asx200,
            include_etfs=include_etfs
        )
        
        # Group by data source
        tiingo_tickers = [t for t in universe if get_data_source(t) == 'tiingo']
        yfinance_tickers = [t for t in universe if get_data_source(t) == 'yfinance']
        
        return {
            "total_tickers": len(universe),
            "by_source": {
                "tiingo": {
                    "count": len(tiingo_tickers),
                    "tickers": tiingo_tickers[:50],  # First 50
                    "coverage": "US Stocks, US ETFs, Mutual Funds, Gold"
                },
                "yfinance": {
                    "count": len(yfinance_tickers),
                    "tickers": yfinance_tickers[:50],  # First 50
                    "coverage": "ASX Stocks, ASX ETFs, VIX, BTC"
                }
            },
            "etfs": get_etf_info(),
            "indices": {
                "sp500": len(get_sp500_tickers()) if include_sp500 else 0,
                "nasdaq100": len(get_nasdaq100_tickers()) if include_nasdaq100 else 0,
                "asx200": len(get_asx200_tickers()) if include_asx200 else 0
            }
        }
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load stock universe: {str(e)}"
        )


@router.get("/source/{ticker}")
async def get_ticker_source(ticker: str) -> Dict[str, str]:
    """
    Get the data source for a specific ticker.
    
    Returns:
    - source: 'tiingo' or 'yfinance'
    - coverage: Description of what the source covers
    """
    source = get_data_source(ticker)
    
    coverage = {
        "tiingo": "US Stocks (NYSE, NASDAQ), US ETFs, Mutual Funds, Chinese A-Shares, Gold",
        "yfinance": "ASX Stocks, ASX ETFs, VIX Index, Cryptocurrencies (BTC, ETH)"
    }
    
    return {
        "ticker": ticker,
        "source": source,
        "coverage": coverage.get(source, "Unknown")
    }

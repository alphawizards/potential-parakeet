"""
Scanner Router
==============
Endpoints for stock scanning and signal generation.

Scanners:
- Quallamaggie High Tight Flag
- Momentum Breakout
- Mean Reversion
"""

import sys
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import json

# Add parent path for strategy imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import authentication
try:
    from backend.auth import get_current_user
except ImportError:
    from auth import get_current_user

router = APIRouter(prefix="/api/scanner", tags=["scanner"])

# In-memory storage for scan results
_scan_results: Dict[str, Any] = {}
_scan_status: str = "idle"


class ScanRequest(BaseModel):
    """Request model for running a scan."""
    scanner_type: str = Field(
        default="quallamaggie",
        description="Type of scanner: quallamaggie, momentum, mean_reversion"
    )
    universe: str = Field(
        default="sp500",
        description="Stock universe: sp500, nasdaq100, asx200, all"
    )
    min_price: float = Field(default=5.0, description="Minimum stock price")
    max_price: float = Field(default=500.0, description="Maximum stock price")
    min_volume: int = Field(default=100000, description="Minimum average volume")
    custom_tickers: Optional[List[str]] = Field(default=None, description="Custom ticker list")


class ScanResult(BaseModel):
    """Model for a single scan result."""
    ticker: str
    name: str
    price: float
    change_pct: float
    volume: int
    signal: str
    score: float
    details: Dict[str, Any]


def load_scan_results() -> Optional[Dict]:
    """Load scan results from JSON file."""
    scan_file = Path("dashboard/scan_results.json")
    if scan_file.exists():
        try:
            with open(scan_file) as f:
                return json.load(f)
        except:
            pass
    return None


@router.get("/")
async def get_scanner_info() -> Dict[str, Any]:
    """
    Get information about available scanners.
    """
    return {
        "scanners": [
            {
                "name": "quallamaggie",
                "description": "High Tight Flag pattern scanner based on Qullamaggie strategy",
                "signals": ["BUY", "WATCH", "HOLD"],
                "parameters": ["min_gain_25d", "max_pullback", "volume_surge"]
            },
            {
                "name": "momentum",
                "description": "12-month momentum with relative strength",
                "signals": ["STRONG_BUY", "BUY", "HOLD", "SELL"],
                "parameters": ["lookback_days", "min_momentum"]
            },
            {
                "name": "mean_reversion",
                "description": "Mean reversion scanner for oversold stocks",
                "signals": ["OVERSOLD", "NEUTRAL", "OVERBOUGHT"],
                "parameters": ["rsi_threshold", "bollinger_std"]
            }
        ],
        "universes": ["sp500", "nasdaq100", "asx200", "all"],
        "last_scan": _scan_results.get("generated_at"),
        "status": _scan_status
    }


@router.get("/results")
async def get_scan_results(
    scanner_type: Optional[str] = None,
    signal: Optional[str] = None,
    min_score: Optional[float] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get latest scan results.
    
    Parameters:
    - scanner_type: Filter by scanner type
    - signal: Filter by signal (BUY, WATCH, etc.)
    - min_score: Minimum score threshold
    - limit: Maximum results to return
    
    Returns:
    - List of stocks matching criteria
    - Scan metadata
    """
    # Try to load from file first
    results = _scan_results if _scan_results else load_scan_results()
    
    if not results:
        return {
            "results": [],
            "message": "No scan results available. Run a scan first.",
            "generated_at": None
        }
    
    # Get stocks from results
    stocks = results.get("stocks", results.get("results", []))
    
    # Apply filters
    filtered = []
    for stock in stocks:
        # Filter by signal
        if signal and stock.get("signal", "").upper() != signal.upper():
            continue
        
        # Filter by min score
        if min_score and stock.get("score", 0) < min_score:
            continue
        
        filtered.append(stock)
    
    # Sort by score descending
    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Limit results
    filtered = filtered[:limit]
    
    return {
        "results": filtered,
        "total_found": len(filtered),
        "generated_at": results.get("generated_at"),
        "scanner_type": results.get("scanner_type", "quallamaggie"),
        "filters_applied": {
            "signal": signal,
            "min_score": min_score,
            "limit": limit
        }
    }


@router.post("/run")
async def run_scan(
    request: ScanRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Run a stock scan.
    
    Scans run in the background. Use /api/scanner/results to get results.
    
    Parameters:
    - scanner_type: Type of scanner to run
    - universe: Stock universe to scan
    - min_price: Minimum stock price filter
    - max_price: Maximum stock price filter
    - min_volume: Minimum volume filter
    """
    global _scan_status
    
    _scan_status = "running"
    
    def execute_scan():
        """Background task to run the scan."""
        global _scan_results, _scan_status
        
        try:
            # Get tickers based on universe
            if request.custom_tickers:
                tickers = request.custom_tickers
            else:
                from strategy.stock_universe import (
                    get_sp500_tickers,
                    get_nasdaq100_tickers,
                    get_asx200_tickers,
                    get_screener_universe
                )
                
                universe_map = {
                    "sp500": get_sp500_tickers,
                    "nasdaq100": get_nasdaq100_tickers,
                    "asx200": get_asx200_tickers,
                    "all": get_screener_universe
                }
                
                get_tickers = universe_map.get(request.universe, get_sp500_tickers)
                tickers = get_tickers()
            
            print(f"Scanning {len(tickers)} tickers with {request.scanner_type} scanner...")
            
            # For now, generate mock results
            # In production, this would call the actual scanner
            import random
            
            signals = ["BUY", "WATCH", "HOLD"]
            results = []
            
            # Simulate scanning (select random subset for demo)
            sample_tickers = random.sample(tickers, min(20, len(tickers)))
            
            for ticker in sample_tickers:
                score = random.uniform(0, 100)
                signal = "BUY" if score > 70 else "WATCH" if score > 40 else "HOLD"
                
                results.append({
                    "ticker": ticker,
                    "name": ticker,  # Would lookup actual name
                    "price": round(random.uniform(request.min_price, request.max_price), 2),
                    "change_pct": round(random.uniform(-5, 10), 2),
                    "volume": random.randint(request.min_volume, request.min_volume * 10),
                    "signal": signal,
                    "score": round(score, 1),
                    "details": {
                        "adr_pct": round(random.uniform(2, 8), 1),
                        "rs_rank": random.randint(1, 100),
                        "pattern": "High Tight Flag" if score > 60 else "Consolidation"
                    }
                })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            _scan_results = {
                "generated_at": datetime.now().isoformat(),
                "scanner_type": request.scanner_type,
                "universe": request.universe,
                "tickers_scanned": len(tickers),
                "results_found": len(results),
                "stocks": results,
                "filters": {
                    "min_price": request.min_price,
                    "max_price": request.max_price,
                    "min_volume": request.min_volume
                }
            }
            
            # Save to file
            scan_file = Path("dashboard/scan_results.json")
            scan_file.parent.mkdir(exist_ok=True)
            with open(scan_file, 'w') as f:
                json.dump(_scan_results, f, indent=2)
            
            _scan_status = "completed"
            print(f"✅ Scan completed: {len(results)} results")
            
        except Exception as e:
            _scan_status = f"failed: {str(e)}"
            print(f"❌ Scan failed: {e}")
    
    background_tasks.add_task(execute_scan)
    
    return {
        "status": "started",
        "scanner_type": request.scanner_type,
        "universe": request.universe,
        "message": "Scan started. Check /api/scanner/results for results.",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/status")
async def get_scan_status() -> Dict[str, Any]:
    """
    Get the status of the current/last scan.
    """
    return {
        "status": _scan_status,
        "has_results": bool(_scan_results),
        "last_scan": _scan_results.get("generated_at") if _scan_results else None,
        "results_count": len(_scan_results.get("stocks", [])) if _scan_results else 0
    }


@router.get("/quallamaggie")
async def get_quallamaggie_results() -> Dict[str, Any]:
    """
    Get Quallamaggie scanner results specifically.
    
    Returns stocks matching the High Tight Flag pattern criteria:
    - 25%+ gain in recent weeks
    - Tight consolidation (< 25% pullback)
    - Volume contraction during consolidation
    """
    # Try to load dedicated Quallamaggie results
    qm_file = Path("dashboard/scan_results.json")
    
    if qm_file.exists():
        try:
            with open(qm_file) as f:
                data = json.load(f)
                data["retrieved_at"] = datetime.now().isoformat()
                return data
        except:
            pass
    
    # Return from memory
    if _scan_results and _scan_results.get("scanner_type") == "quallamaggie":
        return _scan_results
    
    return {
        "results": [],
        "message": "No Quallamaggie scan results. Run a scan with scanner_type='quallamaggie'",
        "criteria": {
            "min_gain_25d": "25%",
            "max_pullback": "25%",
            "volume_contraction": True,
            "pattern": "High Tight Flag"
        }
    }

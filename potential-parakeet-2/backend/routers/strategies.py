"""
Strategies Router
=================
Endpoints for strategy management and backtest execution.

Strategies:
- Quant 1.0: Momentum, HRP, Dual Momentum, Inverse Volatility
- Quant 2.0: Regime Detection, Stat Arb, Residual Momentum, Meta-Labeling
- OLMAR: Online Learning Mean Reversion
"""

import sys
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import traceback
import logging

logger = logging.getLogger(__name__)

# Add parent path for strategy imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/strategies", tags=["strategies"])

# In-memory storage for backtest results
_backtest_results: Dict[str, Any] = {}
_backtest_status: Dict[str, str] = {}


class BacktestRequest(BaseModel):
    """Request model for backtest execution."""
    strategy_name: str = Field(..., description="Name of the strategy to run")
    start_date: str = Field(default="2020-01-01", description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD), defaults to today")
    initial_capital: float = Field(default=100000.0, description="Initial capital in AUD")
    tickers: Optional[List[str]] = Field(default=None, description="Custom ticker list (overrides universe)")
    universe: str = Field(default="SPX500", description="Stock universe: SPX500, NASDAQ100, ASX200, RUSSELL2000, etc.")
    optimization_method: str = Field(default="HRP", description="HRP, MVO, InverseVol, or EqualWeight")


class BacktestResponse(BaseModel):
    """Response model for backtest execution."""
    backtest_id: str
    status: str
    strategy_name: str
    message: str
    timestamp: str


class StrategyInfo(BaseModel):
    """Information about a strategy."""
    name: str
    category: str
    description: str
    parameters: Dict[str, Any]
    status: str


# Strategy catalog
STRATEGY_CATALOG = {
    # Quant 1.0 Strategies
    "Momentum": {
        "category": "Quant 1.0",
        "description": "12-month momentum with 1-month skip. Long top performers, avoid recent losers.",
        "parameters": {"lookback": 252, "skip": 21, "top_n": 10},
        "status": "active"
    },
    "Dual_Momentum": {
        "category": "Quant 1.0",
        "description": "Combines absolute and relative momentum. Risk-on vs risk-off allocation.",
        "parameters": {"abs_lookback": 252, "rel_lookback": 126},
        "status": "active"
    },
    "HRP": {
        "category": "Quant 1.0",
        "description": "Hierarchical Risk Parity. Machine learning-based diversification.",
        "parameters": {"lookback": 252, "linkage": "ward"},
        "status": "active"
    },
    "InverseVolatility": {
        "category": "Quant 1.0",
        "description": "Weight inversely proportional to volatility. Lower vol = higher weight.",
        "parameters": {"lookback": 63},
        "status": "active"
    },
    
    # Quant 2.0 Strategies
    "Regime_Detection": {
        "category": "Quant 2.0",
        "description": "HMM-based market regime detection. Bull/Bear/Sideways classification.",
        "parameters": {"n_states": 3, "lookback": 504},
        "status": "active"
    },
    "Stat_Arb": {
        "category": "Quant 2.0",
        "description": "Statistical arbitrage with Kalman filter dynamic hedge ratios.",
        "parameters": {"zscore_entry": 2.0, "zscore_exit": 0.5},
        "status": "active"
    },
    "Residual_Momentum": {
        "category": "Quant 2.0",
        "description": "Fama-French residual momentum. Alpha after factor exposure.",
        "parameters": {"ff_factors": 5, "lookback": 252},
        "status": "active"
    },
    "Meta_Labeling": {
        "category": "Quant 2.0",
        "description": "ML filter for trade signals. Random Forest binary classifier.",
        "parameters": {"n_estimators": 100, "min_confidence": 0.6},
        "status": "active"
    },
    "Short_Vol": {
        "category": "Quant 2.0",
        "description": "Volatility risk premium harvesting. Short VIX in contango.",
        "parameters": {"vix_threshold": 20, "term_structure": "contango"},
        "status": "active"
    },
    "NCO": {
        "category": "Quant 2.0",
        "description": "Nested Cluster Optimization. Hierarchical portfolio construction.",
        "parameters": {"max_clusters": 10, "min_cluster_size": 3},
        "status": "active"
    },
    
    # OLMAR Strategy
    "OLMAR": {
        "category": "Mean Reversion",
        "description": "Online Learning for Portfolio Selection. Mean reversion with moving average.",
        "parameters": {"window": 5, "epsilon": 10},
        "status": "active"
    },
    
    # Quallamaggie Style
    "Quallamaggie": {
        "category": "Breakout",
        "description": "High tight flag pattern scanner. Momentum breakout strategy.",
        "parameters": {"adr_min": 3.0, "volume_surge": 2.0},
        "status": "active"
    }
}


@router.get("/", response_model=List[StrategyInfo])
async def list_strategies():
    """
    List all available strategies.
    
    Returns strategy name, category, description, and parameters.
    """
    strategies = []
    for name, info in STRATEGY_CATALOG.items():
        strategies.append(StrategyInfo(
            name=name,
            category=info["category"],
            description=info["description"],
            parameters=info["parameters"],
            status=info["status"]
        ))
    
    return strategies


@router.get("/{strategy_name}")
async def get_strategy_details(strategy_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific strategy.
    
    Includes:
    - Strategy metadata
    - Latest backtest results (if available)
    - Performance metrics
    """
    if strategy_name not in STRATEGY_CATALOG:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_name}' not found. Available: {list(STRATEGY_CATALOG.keys())}"
        )
    
    info = STRATEGY_CATALOG[strategy_name]
    
    # Check for cached results
    results = None
    if strategy_name in _backtest_results:
        results = _backtest_results[strategy_name]
    else:
        # Try to load from file
        results_file = Path(f"reports/{strategy_name}_results.json")
        if results_file.exists():
            try:
                with open(results_file) as f:
                    results = json.load(f)
            except:
                pass
    
    return {
        "name": strategy_name,
        "category": info["category"],
        "description": info["description"],
        "parameters": info["parameters"],
        "status": info["status"],
        "last_results": results,
        "has_results": results is not None
    }


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks
):
    """
    Execute a backtest for a strategy.
    
    The backtest runs in the background. Use /api/strategies/{name}/status
    to check progress and /api/strategies/{name}/results for results.
    
    Parameters:
    - strategy_name: Name of strategy to run
    - start_date: Backtest start date
    - end_date: Backtest end date (defaults to today)
    - initial_capital: Starting capital in AUD
    - tickers: Custom ticker list (optional)
    - optimization_method: Portfolio optimization method
    """
    if request.strategy_name not in STRATEGY_CATALOG:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{request.strategy_name}' not found"
        )
    
    # Generate backtest ID
    backtest_id = f"{request.strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set status to running
    _backtest_status[request.strategy_name] = "running"
    
    def execute_backtest():
        """Background task to run the backtest."""
        try:
            from strategy.pipeline.pipeline import TradingPipeline, PipelineConfig
            
            # Configure pipeline
            config = PipelineConfig(
                tickers=request.tickers,
                start_date=request.start_date,
                end_date=request.end_date,
                initial_capital=request.initial_capital
            )
            
            pipeline = TradingPipeline(config)
            
            # Run the strategy
            result = pipeline.run(
                strategy_name=request.strategy_name,
                optimization_method=request.optimization_method
            )
            
            # Save results
            results_data = {
                "backtest_id": backtest_id,
                "strategy_name": request.strategy_name,
                "completed_at": datetime.now().isoformat(),
                "config": {
                    "start_date": request.start_date,
                    "end_date": request.end_date or datetime.now().strftime("%Y-%m-%d"),
                    "initial_capital": request.initial_capital,
                    "optimization_method": request.optimization_method
                },
                "metrics": result.report.metrics.to_dict() if hasattr(result.report.metrics, 'to_dict') else {},
                "final_value": result.final_value,
                "weights": result.allocation.weights.to_dict() if hasattr(result.allocation.weights, 'to_dict') else {},
                "execution_time": result.execution_time_seconds
            }
            
            _backtest_results[request.strategy_name] = results_data
            _backtest_status[request.strategy_name] = "completed"
            
            # Save to file
            results_path = Path(f"reports/{request.strategy_name}_results.json")
            results_path.parent.mkdir(exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            print(f"✅ Backtest completed: {request.strategy_name}")
            
        except ImportError as e:
            error_msg = f"Required module not available: {e}"
            _backtest_status[request.strategy_name] = f"failed: {error_msg}"
            _backtest_results[request.strategy_name] = {"error": error_msg}
            logger.error(f"Backtest {request.strategy_name}: ImportError - {e}")
        except ValueError as e:
            error_msg = f"Invalid parameter: {e}"
            _backtest_status[request.strategy_name] = f"failed: {error_msg}"
            _backtest_results[request.strategy_name] = {"error": error_msg}
            logger.error(f"Backtest {request.strategy_name}: ValueError - {e}")
        except RuntimeError as e:
            error_msg = f"Strategy execution error: {e}"
            _backtest_status[request.strategy_name] = f"failed: {error_msg}"
            _backtest_results[request.strategy_name] = {"error": error_msg}
            logger.error(f"Backtest {request.strategy_name}: RuntimeError - {e}")
        except Exception as e:
            # Last resort catch - log full traceback but don't expose to client
            _backtest_status[request.strategy_name] = "failed: Internal error"
            _backtest_results[request.strategy_name] = {"error": "Internal server error"}
            logger.exception(f"Backtest {request.strategy_name}: Unhandled exception")
    
    # Add to background tasks
    background_tasks.add_task(execute_backtest)
    
    return BacktestResponse(
        backtest_id=backtest_id,
        status="started",
        strategy_name=request.strategy_name,
        message=f"Backtest started. Check /api/strategies/{request.strategy_name}/status for progress.",
        timestamp=datetime.now().isoformat()
    )


@router.get("/{strategy_name}/status")
async def get_backtest_status(strategy_name: str) -> Dict[str, Any]:
    """
    Get the status of a running backtest.
    
    Returns:
    - status: 'running', 'completed', 'failed', or 'not_started'
    - message: Additional details
    """
    status = _backtest_status.get(strategy_name, "not_started")
    
    return {
        "strategy_name": strategy_name,
        "status": status,
        "has_results": strategy_name in _backtest_results,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/{strategy_name}/results")
async def get_backtest_results(strategy_name: str) -> Dict[str, Any]:
    """
    Get the results of a completed backtest.
    
    Returns:
    - Backtest configuration
    - Performance metrics (CAGR, Sharpe, Drawdown, etc.)
    - Portfolio weights
    - Execution time
    """
    if strategy_name not in _backtest_results:
        # Try to load from file
        results_file = Path(f"reports/{strategy_name}_results.json")
        if results_file.exists():
            try:
                with open(results_file) as f:
                    return json.load(f)
            except:
                pass
        
        raise HTTPException(
            status_code=404,
            detail=f"No results found for strategy '{strategy_name}'. Run a backtest first."
        )
    
    return _backtest_results[strategy_name]


@router.get("/compare/all")
async def compare_all_strategies() -> Dict[str, Any]:
    """
    Compare performance of all strategies that have been run.
    
    Returns a comparison table with key metrics.
    """
    comparison = []
    
    for strategy_name in STRATEGY_CATALOG.keys():
        results = _backtest_results.get(strategy_name)
        
        if not results and Path(f"reports/{strategy_name}_results.json").exists():
            try:
                with open(f"reports/{strategy_name}_results.json") as f:
                    results = json.load(f)
            except:
                continue
        
        if results and "metrics" in results:
            metrics = results["metrics"]
            comparison.append({
                "strategy": strategy_name,
                "category": STRATEGY_CATALOG[strategy_name]["category"],
                "final_value": results.get("final_value", 0),
                "total_return": metrics.get("total_return", "N/A"),
                "cagr": metrics.get("cagr", "N/A"),
                "sharpe_ratio": metrics.get("sharpe_ratio", "N/A"),
                "max_drawdown": metrics.get("max_drawdown", "N/A"),
                "win_rate": metrics.get("win_rate", "N/A")
            })
    
    return {
        "comparison": comparison,
        "generated_at": datetime.now().isoformat(),
        "strategies_compared": len(comparison)
    }

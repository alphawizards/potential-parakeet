"""
Dashboard API
==============
FastAPI backend for the trading dashboard.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

# Import pipeline
from strategy.pipeline.pipeline import TradingPipeline, PipelineConfig, run_daily_scan
from strategy.pipeline.reporting_layer import ReportingManager


# Create FastAPI app
app = FastAPI(
    title="Trading Dashboard API",
    description="API for the Quantitative Trading Pipeline Dashboard",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global pipeline instance
pipeline: Optional[TradingPipeline] = None
last_scan_time: Optional[datetime] = None
cached_results: Dict[str, Any] = {}


# ============== Models ==============

class ScanRequest(BaseModel):
    strategies: Optional[List[str]] = None
    start_date: Optional[str] = "2020-01-01"
    

class StrategyMetrics(BaseModel):
    strategy_name: str
    final_value: float
    total_return: str
    cagr: str
    volatility: str
    sharpe_ratio: str
    sortino_ratio: str
    max_drawdown: str
    calmar_ratio: str
    win_rate: str
    rolling_sharpe_30d: str
    rolling_sharpe_90d: str


class DashboardData(BaseModel):
    last_updated: str
    strategies: List[StrategyMetrics]
    comparison_table: List[Dict[str, str]]
    equity_curves: Dict[str, List[float]]


# ============== Endpoints ==============

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Trading Dashboard API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/dashboard")
async def get_dashboard_data() -> Dict[str, Any]:
    """
    Get dashboard data including all strategy metrics.
    """
    global cached_results
    
    if not cached_results:
        # Try to load from file
        results_file = Path("reports/pipeline_results.json")
        if results_file.exists():
            with open(results_file) as f:
                cached_results = json.load(f)
    
    if not cached_results:
        return {
            "last_updated": None,
            "strategies": [],
            "comparison_table": [],
            "equity_curves": {},
            "message": "No scan results available. Run a scan first."
        }
    
    return {
        "last_updated": cached_results.get('generated_at'),
        "strategies": list(cached_results.get('strategies', {}).values()),
        "comparison_table": _format_comparison_table(cached_results),
        "config": cached_results.get('config', {})
    }


@app.post("/api/scan")
async def run_scan(
    request: ScanRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Run a strategy scan (can be run in background).
    """
    global pipeline, last_scan_time, cached_results
    
    # Initialize pipeline
    config = PipelineConfig(
        start_date=request.start_date
    )
    pipeline = TradingPipeline(config)
    
    try:
        if request.strategies:
            # Run specific strategies
            for strategy in request.strategies:
                pipeline.run(strategy)
        else:
            # Run all strategies
            pipeline.run_all_strategies()
        
        # Save results
        pipeline.save_results()
        cached_results = pipeline.export_results_json()
        last_scan_time = datetime.now()
        
        return {
            "status": "success",
            "message": f"Scan completed for {len(pipeline._results)} strategies",
            "timestamp": last_scan_time.isoformat(),
            "strategies": list(pipeline._results.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies")
async def get_strategies() -> Dict[str, Any]:
    """Get list of available strategies."""
    if pipeline is None:
        # Create temporary pipeline to get strategies
        temp_pipeline = TradingPipeline()
        strategies = temp_pipeline.signal_manager.list_strategies()
    else:
        strategies = pipeline.signal_manager.list_strategies()
    
    return {
        "strategies": strategies
    }


@app.get("/api/strategy/{strategy_name}")
async def get_strategy_details(strategy_name: str) -> Dict[str, Any]:
    """Get detailed metrics for a specific strategy."""
    global cached_results
    
    if not cached_results or strategy_name not in cached_results.get('strategies', {}):
        raise HTTPException(status_code=404, detail=f"Strategy {strategy_name} not found")
    
    return cached_results['strategies'][strategy_name]


@app.get("/api/report/{strategy_name}")
async def get_strategy_report(strategy_name: str):
    """Get HTML report for a strategy."""
    report_path = Path(f"reports/{strategy_name}_report.html")
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(report_path, media_type="text/html")


@app.get("/api/comparison")
async def get_comparison() -> Dict[str, Any]:
    """Get strategy comparison table."""
    global cached_results
    
    if not cached_results:
        return {"comparison": []}
    
    return {
        "comparison": _format_comparison_table(cached_results)
    }


@app.get("/api/rolling/{strategy_name}")
async def get_rolling_metrics(
    strategy_name: str,
    window: int = 30
) -> Dict[str, Any]:
    """Get rolling metrics for a strategy."""
    global pipeline
    
    if pipeline is None or strategy_name not in pipeline._results:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    result = pipeline._results[strategy_name]
    returns = result.portfolio_returns
    
    # Calculate rolling metrics
    reporting = ReportingManager()
    rolling = reporting.calculate_rolling_metrics(returns, windows=[window])
    
    return {
        "strategy_name": strategy_name,
        "window": window,
        "rolling_metrics": rolling.tail(20).to_dict()
    }


# ============== Helper Functions ==============

def _format_comparison_table(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Format results into comparison table."""
    comparison = []
    
    for name, data in results.get('strategies', {}).items():
        metrics = data.get('metrics', {})
        comparison.append({
            'Strategy': name,
            'Final Value': f"${data.get('final_value', 0):,.0f}",
            **metrics
        })
    
    return comparison


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

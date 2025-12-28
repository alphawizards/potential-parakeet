"""
Central Configuration for Strategy Package
==========================================

This module contains all configuration settings for the strategy package.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os

@dataclass
class AssetConfig:
    """Asset configuration."""
    ticker: str
    asset_type: str  # 'stock', 'etf', 'index', 'fx'
    currency: str
    exchange: str

@dataclass
class BacktestConfig:
    """Backtest configuration."""
    INITIAL_CAPITAL_AUD: float = 100000.0
    RISK_FREE_RATE: float = 0.04
    TRADING_DAYS_PER_YEAR: int = 252

class Config:
    """Global configuration."""
    RISK_FREE_RATE = 0.04

    # API Keys
    TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

    # Paths
    CACHE_DIR = "cache"
    DATA_DIR = "data"

    # Data Sources
    USE_TIINGO = True
    USE_YFINANCE = True
    USE_NORGATE = False

CONFIG = Config()
BACKTEST_CONFIG = BacktestConfig()

def is_us_ticker(ticker: str) -> bool:
    """Check if ticker is a US ticker."""
    return not ticker.endswith('.AX')

def get_us_tickers() -> List[str]:
    """Get list of US tickers."""
    return []

def get_asx_tickers() -> List[str]:
    """Get list of ASX tickers."""
    return []

def get_fx_cost(currency: str) -> float:
    """Get FX cost for currency."""
    if currency == 'AUD':
        return 0.0
    return 0.005  # 50 bps

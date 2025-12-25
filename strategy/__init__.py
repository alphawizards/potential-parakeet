"""
Quantitative Global Investing Strategy
=======================================

A comprehensive quantitative investing framework for Australian retail investors.

Modules:
- config: Strategy configuration and parameters
- data_loader: Data fetching and AUD normalization
- signals: Momentum and technical signal generation
- optimizer: Portfolio optimization using Riskfolio-Lib
- backtest: High-performance backtesting with vectorbt

Usage:
    from strategy import QuantStrategy
    
    strategy = QuantStrategy(portfolio_value=100000)
    recommendations = strategy.run_full_pipeline()
"""

from .config import CONFIG, BACKTEST_CONFIG
from .data_loader import DataLoader

# Optional imports - require pandas_ta which needs Python >= 3.12
try:
    from .signals import MomentumSignals, TechnicalSignals, CompositeSignal
except ImportError:
    MomentumSignals = None
    TechnicalSignals = None
    CompositeSignal = None

from .optimizer import PortfolioOptimizer, CostAwareOptimizer
from .backtest import PortfolioBacktester, VectorBTBacktester
from .main import QuantStrategy

__version__ = "1.0.0"
__author__ = "Quantitative Strategy Team"

__all__ = [
    'CONFIG',
    'BACKTEST_CONFIG',
    'DataLoader',
    'MomentumSignals',
    'TechnicalSignals',
    'CompositeSignal',
    'PortfolioOptimizer',
    'CostAwareOptimizer',
    'PortfolioBacktester',
    'VectorBTBacktester',
    'QuantStrategy',
]

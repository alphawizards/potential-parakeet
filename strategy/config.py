"""
Configuration for Quantitative Global Investing Strategy
=========================================================
All parameters centralized for easy tuning and backtesting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class AssetConfig:
    """Configuration for a single asset."""
    ticker: str
    name: str
    asset_class: str
    currency: str  # 'USD' or 'AUD'
    asx_equivalent: Optional[str] = None
    is_defensive: bool = False


@dataclass
class StrategyConfig:
    """Main strategy configuration."""
    
    # ========== UNIVERSE ==========
    # US ETFs (via Stake.com)
    US_ETFS: List[AssetConfig] = field(default_factory=lambda: [
        AssetConfig("SPY", "S&P 500", "US_EQUITY", "USD", asx_equivalent="IVV.AX"),
        AssetConfig("QQQ", "Nasdaq 100", "US_TECH", "USD", asx_equivalent="NDQ.AX"),
        AssetConfig("VEA", "Developed ex-US", "INTL_EQUITY", "USD", asx_equivalent="VGS.AX"),
        AssetConfig("VWO", "Emerging Markets", "EM_EQUITY", "USD", asx_equivalent="VGE.AX"),
        AssetConfig("TLT", "US Long Treasuries", "US_BONDS", "USD", is_defensive=True),
        AssetConfig("IEF", "US Intermediate Treasuries", "US_BONDS", "USD", is_defensive=True),
        AssetConfig("GLD", "Gold", "COMMODITIES", "USD", asx_equivalent="GOLD.AX"),
        AssetConfig("DBC", "Commodities Broad", "COMMODITIES", "USD"),
        AssetConfig("VNQ", "US REITs", "REAL_ESTATE", "USD"),
    ])
    
    # ASX ETFs (no FX friction)
    ASX_ETFS: List[AssetConfig] = field(default_factory=lambda: [
        AssetConfig("VAS.AX", "Australian Shares", "AU_EQUITY", "AUD"),
        AssetConfig("IVV.AX", "S&P 500 (AUD)", "US_EQUITY", "AUD"),
        AssetConfig("VGS.AX", "Int'l Shares (AUD)", "INTL_EQUITY", "AUD"),
        AssetConfig("VGE.AX", "Emerging Markets (AUD)", "EM_EQUITY", "AUD"),
        AssetConfig("VAF.AX", "Australian Bonds", "AU_BONDS", "AUD", is_defensive=True),
        AssetConfig("GOLD.AX", "Gold (AUD)", "COMMODITIES", "AUD"),
        AssetConfig("IHVV.AX", "S&P 500 Hedged", "US_EQUITY_HEDGED", "AUD"),
    ])
    
    # Defensive assets for risk-off
    DEFENSIVE_TICKERS: List[str] = field(default_factory=lambda: [
        "TLT", "IEF", "VAF.AX", "BIL"  # BIL = T-Bills for risk-free proxy
    ])
    
    # ========== SCREENER UNIVERSE ==========
    # High-beta tickers for Quallamaggie-style momentum screening
    # These are typically high-growth, volatile stocks suitable for swing trading
    SCREENER_UNIVERSE: List[str] = field(default_factory=lambda: [
        # Mega-cap Tech Leaders
        "NVDA",   # NVIDIA - AI/GPU leader
        "AMD",    # AMD - Semiconductor
        "TSLA",   # Tesla - EV/Energy
        "META",   # Meta Platforms - Social/AI
        "GOOGL",  # Alphabet - Search/Cloud
        "AMZN",   # Amazon - E-commerce/Cloud
        "MSFT",   # Microsoft - Cloud/Enterprise
        "AAPL",   # Apple - Consumer Tech
        
        # High-Growth Tech
        "PLTR",   # Palantir - Data Analytics/AI
        "COIN",   # Coinbase - Crypto Exchange
        "MSTR",   # MicroStrategy - Bitcoin Treasury
        "NET",    # Cloudflare - Edge Computing
        "CRWD",   # CrowdStrike - Cybersecurity
        "SNOW",   # Snowflake - Cloud Data
        "DDOG",   # Datadog - Monitoring
        "ZS",     # Zscaler - Security
        
        # Semiconductor
        "AVGO",   # Broadcom - Chips
        "MU",     # Micron - Memory
        "MRVL",   # Marvell - Chips
        "AMAT",   # Applied Materials - Equipment
        "LRCX",   # Lam Research - Equipment
        "KLAC",   # KLA Corp - Equipment
        "ARM",    # Arm Holdings - IP
        
        # EV/Clean Energy
        "RIVN",   # Rivian - EV
        "LCID",   # Lucid - EV
        "ENPH",   # Enphase - Solar
        "FSLR",   # First Solar - Solar
        
        # Biotech (High Beta)
        "MRNA",   # Moderna - mRNA
        "VRTX",   # Vertex - Biotech
        
        # Fintech/Payments
        "XYZ",    # Block Inc (formerly SQ/Square)
        "AFRM",   # Affirm - BNPL
        "UPST",   # Upstart - AI Lending
        
        # E-commerce/Consumer
        "SHOP",   # Shopify
        "ABNB",   # Airbnb
        "UBER",   # Uber
        "DASH",   # DoorDash
    ])
    
    # ========== MOMENTUM PARAMETERS ==========
    LOOKBACK_DAYS: int = 252  # 12 months (trading days)
    LOOKBACK_DAYS_SHORT: int = 21  # 1 month for short-term momentum
    RISK_FREE_RATE: float = 0.04  # Annual risk-free rate (4%)
    
    # ========== PORTFOLIO CONSTRAINTS ==========
    MIN_WEIGHT: float = 0.05  # 5% minimum position
    MAX_WEIGHT: float = 0.25  # 25% maximum position
    MAX_SECTOR_WEIGHT: float = 0.40  # 40% max in any sector
    
    # ========== COST PARAMETERS (Stake.com) ==========
    FX_FEE_BPS: float = 70  # 70 basis points per FX conversion
    ASX_BROKERAGE_AUD: float = 3.0  # $3 per ASX trade
    TAX_RATE_MARGINAL: float = 0.37  # 37% marginal tax rate
    CGT_DISCOUNT: float = 0.50  # 50% CGT discount if held > 12 months
    
    # ========== RISK MANAGEMENT ==========
    MAX_DRAWDOWN_REDUCE: float = 0.15  # Reduce risk at 15% drawdown
    MAX_DRAWDOWN_EXIT: float = 0.25  # Exit to defensive at 25% drawdown
    VOLATILITY_TARGET: float = 0.12  # Target 12% annual volatility
    
    # ========== DATA PARAMETERS ==========
    START_DATE: str = "2010-01-01"
    END_DATE: str = datetime.now().strftime("%Y-%m-%d")
    FX_TICKER: str = "AUDUSD=X"  # AUD per USD
    RISK_FREE_TICKER: str = "^IRX"  # 13-week T-Bill rate
    
    # ========== REBALANCING ==========
    REBALANCE_FREQUENCY: str = "monthly"  # 'daily', 'weekly', 'monthly', 'quarterly'
    SIGNAL_FREQUENCY: str = "monthly"  # How often to check signals
    MIN_TRADE_SIZE_AUD: float = 100.0  # Minimum trade size
    
    # ========== Z-SCORE FILTER (Data Quality) ==========
    ZSCORE_THRESHOLD: float = 4.0  # Reject returns with |z| > 4
    MIN_DATA_POINTS: int = 252  # Minimum 1 year of data required


@dataclass 
class BacktestConfig:
    """Backtest-specific configuration."""
    INITIAL_CAPITAL_AUD: float = 100_000.0
    BENCHMARK_TICKER: str = "IVV.AX"  # S&P 500 in AUD as benchmark
    INCLUDE_DIVIDENDS: bool = True
    SLIPPAGE_BPS: float = 5  # 5 bps slippage assumption


# Global config instance
CONFIG = StrategyConfig()
BACKTEST_CONFIG = BacktestConfig()


def get_all_tickers() -> List[str]:
    """Get all tickers in the universe."""
    us_tickers = [a.ticker for a in CONFIG.US_ETFS]
    asx_tickers = [a.ticker for a in CONFIG.ASX_ETFS]
    return us_tickers + asx_tickers


def get_us_tickers() -> List[str]:
    """Get US tickers only."""
    return [a.ticker for a in CONFIG.US_ETFS]


def get_asx_tickers() -> List[str]:
    """Get ASX tickers only."""
    return [a.ticker for a in CONFIG.ASX_ETFS]


def is_us_ticker(ticker: str) -> bool:
    """Check if ticker is US-domiciled."""
    return ticker in get_us_tickers()


def get_fx_cost(ticker: str, trade_value_aud: float) -> float:
    """Calculate FX cost for a trade."""
    if is_us_ticker(ticker):
        return trade_value_aud * (CONFIG.FX_FEE_BPS / 10000) * 2  # Round trip
    else:
        return CONFIG.ASX_BROKERAGE_AUD  # Flat fee for ASX


def get_screener_universe() -> List[str]:
    """Get the screener universe tickers for Quallamaggie screening."""
    return CONFIG.SCREENER_UNIVERSE

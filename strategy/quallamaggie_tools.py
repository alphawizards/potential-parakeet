"""
Quallamaggie Trading Tools
===========================
Comprehensive implementation of Kristjan Kullamägi's momentum breakout strategy tools.

Components:
1. Momentum Screener - Find top 7% stocks by 1/3/6-month momentum
2. RS Line Indicator - Relative strength vs benchmark
3. Watchlist Manager - Track and manage breakout candidates
4. Position Sizing Calculator - Risk-based position sizing
5. Breakout Alerts - Identify breakout candidates

Author: CIO & Data Analyst
Date: 2024-12-25
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import yfinance as yf
import json
import os
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class QuallamaggieToolsConfig:
    """Configuration for Quallamaggie trading tools."""
    
    # Momentum screening
    TOP_MOMENTUM_PCT: float = 0.07  # Top 7%
    MOMENTUM_1M_DAYS: int = 21
    MOMENTUM_3M_DAYS: int = 63
    MOMENTUM_6M_DAYS: int = 126
    
    # RS Line
    RS_BENCHMARK: str = "SPY"
    RS_LOOKBACK: int = 252  # 1 year
    
    # Pattern detection
    CONSOLIDATION_MIN_DAYS: int = 10
    CONSOLIDATION_MAX_DAYS: int = 40
    VOLATILITY_CONTRACTION_RATIO: float = 0.5
    BREAKOUT_VOLUME_MULT: float = 1.5
    
    # Position sizing
    DEFAULT_RISK_PCT: float = 0.005  # 0.5% risk per trade
    MAX_POSITION_PCT: float = 0.20  # 20% max position
    MIN_POSITION_SIZE: float = 1000.0  # Minimum $1000
    
    # Data
    DATA_START: str = "2023-01-01"
    WATCHLIST_FILE: str = "quallamaggie_watchlist.json"
    
    # Universe - Large & Mid Cap US Stocks
    UNIVERSE: List[str] = field(default_factory=lambda: [
        # Tech Giants
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'CRM',
        # Tech Growth
        'AMD', 'NFLX', 'ADBE', 'NOW', 'SNOW', 'PLTR', 'NET', 'DDOG', 'ZS', 'CRWD',
        'MDB', 'PANW', 'FTNT', 'WDAY', 'TEAM', 'VEEV', 'SPLK', 'OKTA', 'TTD', 'COIN',
        # Semiconductors
        'INTC', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'ON', 'SWKS',
        # Financials
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
        'V', 'MA', 'AXP', 'PYPL', 'SQ', 'BLK', 'SCHW', 'CME', 'ICE', 'SPGI',
        # Healthcare
        'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
        'AMGN', 'GILD', 'ISRG', 'VRTX', 'REGN', 'MRNA', 'BIIB', 'ZTS', 'HCA', 'ELV',
        # Consumer
        'HD', 'LOW', 'TGT', 'COST', 'WMT', 'MCD', 'SBUX', 'NKE', 'LULU', 'TJX',
        'DG', 'ROST', 'YUM', 'CMG', 'DHI', 'LEN', 'PHM', 'ORLY', 'AZO', 'ULTA',
        # Industrials
        'CAT', 'DE', 'HON', 'UNP', 'UPS', 'FDX', 'GE', 'MMM', 'LMT', 'RTX',
        'BA', 'GD', 'NOC', 'ITW', 'EMR', 'ROK', 'ETN', 'PH', 'IR', 'FAST',
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
        # Communications
        'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'NXPI', 'QRVO', 'WBD', 'PARA',
        # ETFs (for reference)
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLV', 'ARKK', 'SOXX'
    ])


CONFIG = QuallamaggieToolsConfig()


# ============================================================================
# 1. MOMENTUM SCREENER
# ============================================================================

class MomentumScreener:
    """
    Screen for top momentum stocks using 1/3/6-month returns.
    
    Identifies stocks in the top 7% by momentum across multiple timeframes.
    """
    
    def __init__(self, config: QuallamaggieToolsConfig = None):
        self.config = config or CONFIG
        self._prices_cache = None
        self._cache_date = None
    
    def fetch_prices(self, tickers: List[str] = None, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch historical prices with caching."""
        tickers = tickers or self.config.UNIVERSE
        today = datetime.now().date()
        
        # Use cache if available and fresh
        if not force_refresh and self._prices_cache is not None and self._cache_date == today:
            return self._prices_cache
        
        print(f"Fetching prices for {len(tickers)} tickers...")
        
        try:
            data = yf.download(
                tickers,
                start=self.config.DATA_START,
                end=datetime.now().strftime("%Y-%m-%d"),
                progress=False,
                threads=False,
                auto_adjust=True
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data[['Close']] if 'Close' in data.columns else data
                if len(tickers) == 1:
                    prices.columns = tickers[:1]
            
            # Filter for stocks with sufficient data
            min_data_pct = 0.8
            valid_cols = prices.columns[prices.notna().sum() / len(prices) >= min_data_pct]
            prices = prices[valid_cols].dropna(how='all')
            
            self._prices_cache = prices
            self._cache_date = today
            
            print(f"Loaded {len(prices.columns)} tickers")
            return prices
            
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return pd.DataFrame()
    
    def calculate_momentum(self, prices: pd.DataFrame, lookback: int) -> pd.Series:
        """Calculate momentum (returns) over lookback period."""
        if len(prices) < lookback:
            return pd.Series(dtype=float)
        
        latest = prices.iloc[-1]
        past = prices.iloc[-lookback] if lookback < len(prices) else prices.iloc[0]
        
        momentum = (latest / past - 1) * 100  # Percentage return
        return momentum.dropna().sort_values(ascending=False)
    
    def calculate_all_momentum(self, prices: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate 1M, 3M, 6M momentum for all stocks."""
        if prices is None:
            prices = self.fetch_prices()
        
        if prices.empty:
            return pd.DataFrame()
        
        mom_1m = self.calculate_momentum(prices, self.config.MOMENTUM_1M_DAYS)
        mom_3m = self.calculate_momentum(prices, self.config.MOMENTUM_3M_DAYS)
        mom_6m = self.calculate_momentum(prices, self.config.MOMENTUM_6M_DAYS)
        
        # Combine into DataFrame
        momentum_df = pd.DataFrame({
            'Momentum_1M': mom_1m,
            'Momentum_3M': mom_3m,
            'Momentum_6M': mom_6m
        })
        
        # Calculate composite score (weighted average)
        momentum_df['Composite'] = (
            momentum_df['Momentum_1M'] * 0.2 +
            momentum_df['Momentum_3M'] * 0.3 +
            momentum_df['Momentum_6M'] * 0.5
        )
        
        # Add rank columns
        for col in ['Momentum_1M', 'Momentum_3M', 'Momentum_6M', 'Composite']:
            momentum_df[f'{col}_Rank'] = momentum_df[col].rank(pct=True, ascending=True)
        
        return momentum_df.round(2)
    
    def get_top_momentum_stocks(
        self,
        prices: pd.DataFrame = None,
        top_pct: float = None,
        timeframe: str = '6M'
    ) -> pd.DataFrame:
        """
        Get top momentum stocks (default: top 7%).
        
        Args:
            prices: Price DataFrame (optional, will fetch if not provided)
            top_pct: Top percentage to include (default: 7%)
            timeframe: '1M', '3M', '6M', or 'Composite'
            
        Returns:
            DataFrame with top momentum stocks
        """
        top_pct = top_pct or self.config.TOP_MOMENTUM_PCT
        
        momentum_df = self.calculate_all_momentum(prices)
        if momentum_df.empty:
            return pd.DataFrame()
        
        # Map timeframe to column
        col_map = {
            '1M': 'Momentum_1M',
            '3M': 'Momentum_3M',
            '6M': 'Momentum_6M',
            'Composite': 'Composite'
        }
        sort_col = col_map.get(timeframe, 'Momentum_6M')
        rank_col = f'{sort_col}_Rank'
        
        # Filter for top performers
        threshold = 1 - top_pct
        top_stocks = momentum_df[momentum_df[rank_col] >= threshold].copy()
        top_stocks = top_stocks.sort_values(sort_col, ascending=False)
        
        print(f"\n📈 Top {top_pct*100:.0f}% Momentum Stocks ({timeframe}):")
        print(f"   Found {len(top_stocks)} stocks")
        
        return top_stocks
    
    def screen(self, timeframe: str = '6M') -> pd.DataFrame:
        """Main screening method - returns top momentum stocks."""
        return self.get_top_momentum_stocks(timeframe=timeframe)


# ============================================================================
# 2. RELATIVE STRENGTH (RS) LINE INDICATOR
# ============================================================================

class RSLineIndicator:
    """
    Calculate Relative Strength Line vs benchmark (typically SPY).
    
    RS Line = Stock Price / Benchmark Price
    RS Line at new highs = bullish sign
    """
    
    def __init__(self, config: QuallamaggieToolsConfig = None):
        self.config = config or CONFIG
    
    def calculate_rs_line(
        self,
        prices: pd.DataFrame,
        benchmark: str = None
    ) -> pd.DataFrame:
        """
        Calculate RS Line for all stocks vs benchmark.
        
        Args:
            prices: DataFrame with stock prices
            benchmark: Benchmark ticker (default: SPY)
            
        Returns:
            DataFrame with RS Line values (normalized to 100 at start)
        """
        benchmark = benchmark or self.config.RS_BENCHMARK
        
        if benchmark not in prices.columns:
            print(f"Warning: Benchmark {benchmark} not found in prices")
            return pd.DataFrame()
        
        benchmark_prices = prices[benchmark]
        
        # Calculate RS Line for each stock
        rs_lines = prices.div(benchmark_prices, axis=0)
        
        # Normalize to 100 at the start
        rs_lines = rs_lines / rs_lines.iloc[0] * 100
        
        return rs_lines
    
    def get_rs_metrics(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Get RS metrics for each stock:
        - Current RS value
        - RS 52-week high
        - RS distance from high (%)
        - RS is at new high (True/False)
        - RS trend (slope over last 20 days)
        """
        rs_lines = self.calculate_rs_line(prices)
        if rs_lines.empty:
            return pd.DataFrame()
        
        metrics = []
        
        for ticker in rs_lines.columns:
            if ticker == self.config.RS_BENCHMARK:
                continue
                
            rs = rs_lines[ticker].dropna()
            if len(rs) < 20:
                continue
            
            current_rs = rs.iloc[-1]
            rs_52w_high = rs.iloc[-252:].max() if len(rs) >= 252 else rs.max()
            rs_52w_low = rs.iloc[-252:].min() if len(rs) >= 252 else rs.min()
            
            # Distance from 52w high
            distance_from_high = (current_rs / rs_52w_high - 1) * 100
            
            # Is RS at new high (within 1%)?
            is_new_high = distance_from_high >= -1.0
            
            # RS trend (simple slope)
            rs_20d = rs.iloc[-20:]
            if len(rs_20d) >= 20:
                x = np.arange(len(rs_20d))
                slope = np.polyfit(x, rs_20d.values, 1)[0]
                rs_trend = "Up" if slope > 0.1 else ("Down" if slope < -0.1 else "Flat")
            else:
                rs_trend = "N/A"
            
            metrics.append({
                'Ticker': ticker,
                'RS_Current': round(current_rs, 2),
                'RS_52W_High': round(rs_52w_high, 2),
                'RS_Dist_From_High': round(distance_from_high, 1),
                'RS_At_New_High': is_new_high,
                'RS_Trend': rs_trend
            })
        
        return pd.DataFrame(metrics).set_index('Ticker')
    
    def get_rs_leaders(self, prices: pd.DataFrame, max_dist: float = 5.0) -> pd.DataFrame:
        """
        Get stocks with RS Line at or near new highs.
        
        Args:
            prices: DataFrame with stock prices
            max_dist: Maximum distance from 52w high (default: 5%)
            
        Returns:
            DataFrame with RS leaders
        """
        metrics = self.get_rs_metrics(prices)
        if metrics.empty:
            return pd.DataFrame()
        
        # Filter for RS leaders (near 52w high)
        leaders = metrics[metrics['RS_Dist_From_High'] >= -max_dist].copy()
        leaders = leaders.sort_values('RS_Dist_From_High', ascending=False)
        
        print(f"\n🏆 RS Leaders (within {max_dist}% of 52W high): {len(leaders)} stocks")
        
        return leaders


# ============================================================================
# 3. WATCHLIST MANAGER
# ============================================================================

@dataclass
class WatchlistEntry:
    """A stock on the watchlist."""
    ticker: str
    added_date: str
    entry_price_target: float
    stop_loss: float
    notes: str = ""
    pattern: str = ""  # HTF, VCP, EP, etc.
    priority: int = 1  # 1=highest
    status: str = "watching"  # watching, triggered, closed


class WatchlistManager:
    """
    Manage a watchlist of breakout candidates.
    
    Supports:
    - Adding/removing stocks
    - Updating entry/stop levels
    - Tracking patterns
    - Saving/loading to JSON
    """
    
    def __init__(self, config: QuallamaggieToolsConfig = None):
        self.config = config or CONFIG
        self.watchlist: Dict[str, WatchlistEntry] = {}
        self.watchlist_file = os.path.join(
            os.path.dirname(__file__),
            self.config.WATCHLIST_FILE
        )
        self.load()
    
    def add(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        pattern: str = "",
        notes: str = "",
        priority: int = 1
    ) -> bool:
        """Add a stock to the watchlist."""
        ticker = ticker.upper()
        
        entry = WatchlistEntry(
            ticker=ticker,
            added_date=datetime.now().strftime("%Y-%m-%d"),
            entry_price_target=entry_price,
            stop_loss=stop_loss,
            notes=notes,
            pattern=pattern,
            priority=priority,
            status="watching"
        )
        
        self.watchlist[ticker] = entry
        self.save()
        print(f"✅ Added {ticker} to watchlist (Entry: ${entry_price:.2f}, Stop: ${stop_loss:.2f})")
        return True
    
    def remove(self, ticker: str) -> bool:
        """Remove a stock from the watchlist."""
        ticker = ticker.upper()
        if ticker in self.watchlist:
            del self.watchlist[ticker]
            self.save()
            print(f"🗑️ Removed {ticker} from watchlist")
            return True
        return False
    
    def update(self, ticker: str, **kwargs) -> bool:
        """Update a watchlist entry."""
        ticker = ticker.upper()
        if ticker not in self.watchlist:
            print(f"❌ {ticker} not in watchlist")
            return False
        
        entry = self.watchlist[ticker]
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        
        self.save()
        print(f"📝 Updated {ticker}")
        return True
    
    def get(self, ticker: str) -> Optional[WatchlistEntry]:
        """Get a watchlist entry."""
        return self.watchlist.get(ticker.upper())
    
    def list_all(self, status: str = None) -> pd.DataFrame:
        """List all watchlist entries as DataFrame."""
        entries = []
        for ticker, entry in self.watchlist.items():
            if status and entry.status != status:
                continue
            entries.append({
                'Ticker': entry.ticker,
                'Added': entry.added_date,
                'Entry': f"${entry.entry_price_target:.2f}",
                'Stop': f"${entry.stop_loss:.2f}",
                'Risk%': f"{(1 - entry.stop_loss/entry.entry_price_target)*100:.1f}%",
                'Pattern': entry.pattern,
                'Priority': entry.priority,
                'Status': entry.status,
                'Notes': entry.notes[:30] + "..." if len(entry.notes) > 30 else entry.notes
            })
        
        if not entries:
            print("📋 Watchlist is empty")
            return pd.DataFrame()
        
        df = pd.DataFrame(entries)
        df = df.sort_values(['Priority', 'Added'], ascending=[True, False])
        return df
    
    def check_triggers(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Check which watchlist stocks have triggered (price > entry)."""
        triggered = []
        
        for ticker, entry in self.watchlist.items():
            if entry.status != "watching":
                continue
                
            if ticker not in prices.columns:
                continue
            
            current_price = prices[ticker].iloc[-1]
            if pd.isna(current_price):
                continue
            
            if current_price >= entry.entry_price_target:
                triggered.append({
                    'Ticker': ticker,
                    'Current': f"${current_price:.2f}",
                    'Entry Target': f"${entry.entry_price_target:.2f}",
                    'Stop': f"${entry.stop_loss:.2f}",
                    'Pattern': entry.pattern
                })
                # Update status
                self.watchlist[ticker].status = "triggered"
        
        self.save()
        
        if triggered:
            print(f"\n🚨 TRIGGERED: {len(triggered)} stocks!")
            return pd.DataFrame(triggered)
        return pd.DataFrame()
    
    def save(self):
        """Save watchlist to JSON file."""
        data = {}
        for ticker, entry in self.watchlist.items():
            data[ticker] = {
                'ticker': entry.ticker,
                'added_date': entry.added_date,
                'entry_price_target': entry.entry_price_target,
                'stop_loss': entry.stop_loss,
                'notes': entry.notes,
                'pattern': entry.pattern,
                'priority': entry.priority,
                'status': entry.status
            }
        
        try:
            with open(self.watchlist_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving watchlist: {e}")
    
    def load(self):
        """Load watchlist from JSON file."""
        if not os.path.exists(self.watchlist_file):
            return
        
        try:
            with open(self.watchlist_file, 'r') as f:
                data = json.load(f)
            
            for ticker, entry_data in data.items():
                self.watchlist[ticker] = WatchlistEntry(**entry_data)
            
            print(f"📋 Loaded {len(self.watchlist)} watchlist entries")
        except Exception as e:
            print(f"Error loading watchlist: {e}")


# ============================================================================
# 4. POSITION SIZING CALCULATOR
# ============================================================================

class PositionSizer:
    """
    Calculate position sizes based on Quallamaggie's risk management rules.
    
    Rules:
    - Risk 0.25% - 1% of account per trade
    - Position size based on distance to stop loss
    - Never more than 20-25% of account in one position
    """
    
    def __init__(self, config: QuallamaggieToolsConfig = None):
        self.config = config or CONFIG
    
    def calculate_position(
        self,
        account_value: float,
        entry_price: float,
        stop_loss: float,
        risk_pct: float = None
    ) -> Dict[str, Any]:
        """
        Calculate position size based on risk.
        
        Args:
            account_value: Total account value in $
            entry_price: Planned entry price
            stop_loss: Stop loss price
            risk_pct: Risk per trade as decimal (default: 0.5%)
            
        Returns:
            Dict with position details
        """
        risk_pct = risk_pct or self.config.DEFAULT_RISK_PCT
        
        # Calculate risk per share
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return {"error": "Stop loss must be below entry price"}
        
        risk_pct_per_share = (risk_per_share / entry_price) * 100
        
        # Calculate max dollar risk
        max_risk_dollars = account_value * risk_pct
        
        # Calculate shares based on risk
        shares = int(max_risk_dollars / risk_per_share)
        
        # Calculate position value
        position_value = shares * entry_price
        position_pct = position_value / account_value
        
        # Check max position constraint
        max_position_value = account_value * self.config.MAX_POSITION_PCT
        if position_value > max_position_value:
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
            position_pct = position_value / account_value
            constraint_hit = "max_position"
        elif position_value < self.config.MIN_POSITION_SIZE:
            return {"error": f"Position too small (${position_value:.0f} < ${self.config.MIN_POSITION_SIZE:.0f})"}
        else:
            constraint_hit = None
        
        # Calculate actual risk with adjusted position
        actual_risk = shares * risk_per_share
        actual_risk_pct = actual_risk / account_value
        
        # R-multiples for different exit scenarios
        target_1r = entry_price + risk_per_share
        target_2r = entry_price + (2 * risk_per_share)
        target_3r = entry_price + (3 * risk_per_share)
        target_5r = entry_price + (5 * risk_per_share)
        
        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "shares": shares,
            "position_value": round(position_value, 2),
            "position_pct": round(position_pct * 100, 2),
            "risk_per_share": round(risk_per_share, 2),
            "risk_per_share_pct": round(risk_pct_per_share, 2),
            "total_risk": round(actual_risk, 2),
            "total_risk_pct": round(actual_risk_pct * 100, 3),
            "constraint_hit": constraint_hit,
            "target_1r": round(target_1r, 2),
            "target_2r": round(target_2r, 2),
            "target_3r": round(target_3r, 2),
            "target_5r": round(target_5r, 2)
        }
    
    def calculate_from_atr(
        self,
        account_value: float,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 1.0,
        risk_pct: float = None
    ) -> Dict[str, Any]:
        """
        Calculate position size using ATR-based stop.
        
        Args:
            account_value: Total account value
            entry_price: Entry price
            atr: Average True Range value
            atr_multiplier: ATR multiplier for stop (default: 1.0)
            risk_pct: Risk per trade
            
        Returns:
            Dict with position details
        """
        stop_loss = entry_price - (atr * atr_multiplier)
        return self.calculate_position(account_value, entry_price, stop_loss, risk_pct)
    
    def print_position(self, result: Dict[str, Any]):
        """Pretty print position sizing result."""
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print("\n" + "=" * 50)
        print("POSITION SIZING CALCULATOR")
        print("=" * 50)
        print(f"Entry Price:     ${result['entry_price']:.2f}")
        print(f"Stop Loss:       ${result['stop_loss']:.2f}")
        print(f"Risk/Share:      ${result['risk_per_share']:.2f} ({result['risk_per_share_pct']:.1f}%)")
        print("-" * 50)
        print(f"Shares:          {result['shares']}")
        print(f"Position Value:  ${result['position_value']:,.2f}")
        print(f"Position Size:   {result['position_pct']:.1f}% of account")
        print(f"Total Risk:      ${result['total_risk']:,.2f} ({result['total_risk_pct']:.2f}%)")
        if result['constraint_hit']:
            print(f"⚠️  Constraint: {result['constraint_hit']}")
        print("-" * 50)
        print("Profit Targets:")
        print(f"  1R:  ${result['target_1r']:.2f} (+{result['risk_per_share_pct']:.1f}%)")
        print(f"  2R:  ${result['target_2r']:.2f} (+{result['risk_per_share_pct']*2:.1f}%)")
        print(f"  3R:  ${result['target_3r']:.2f} (+{result['risk_per_share_pct']*3:.1f}%)")
        print(f"  5R:  ${result['target_5r']:.2f} (+{result['risk_per_share_pct']*5:.1f}%)")
        print("=" * 50)


# ============================================================================
# 5. BREAKOUT ALERT SCANNER
# ============================================================================

class BreakoutScanner:
    """
    Scan for breakout candidates based on Quallamaggie patterns.
    
    Patterns detected:
    - VCP (Volatility Contraction Pattern)
    - High Tight Flag
    - Episodic Pivot (gap up on volume)
    - Consolidation near highs
    """
    
    def __init__(self, config: QuallamaggieToolsConfig = None):
        self.config = config or CONFIG
    
    def scan_for_breakouts(self, prices: pd.DataFrame, volume: pd.DataFrame = None) -> pd.DataFrame:
        """
        Scan for breakout candidates.
        
        Returns DataFrame with:
        - Ticker
        - Pattern type
        - Current price
        - Breakout level
        - Support level
        - Score (quality rating)
        """
        alerts = []
        
        for ticker in prices.columns:
            if ticker in ['SPY', 'QQQ', 'IWM', 'DIA']:  # Skip benchmarks
                continue
            
            price = prices[ticker].dropna()
            if len(price) < 60:
                continue
            
            vol = None
            if volume is not None and ticker in volume.columns:
                vol = volume[ticker].dropna()
            
            # Check for patterns
            pattern_result = self._detect_pattern(price, vol)
            
            if pattern_result['pattern'] != 'None':
                alerts.append({
                    'Ticker': ticker,
                    'Pattern': pattern_result['pattern'],
                    'Current': round(price.iloc[-1], 2),
                    'Breakout': round(pattern_result['breakout_level'], 2),
                    'Support': round(pattern_result['support_level'], 2),
                    'Score': pattern_result['score'],
                    'Notes': pattern_result['notes']
                })
        
        if not alerts:
            print("🔍 No breakout candidates found")
            return pd.DataFrame()
        
        df = pd.DataFrame(alerts)
        df = df.sort_values('Score', ascending=False)
        
        print(f"\n🚀 Found {len(alerts)} Breakout Candidates:")
        return df
    
    def _detect_pattern(self, price: pd.Series, volume: pd.Series = None) -> Dict:
        """Detect which pattern (if any) the stock is forming."""
        
        result = {
            'pattern': 'None',
            'breakout_level': 0,
            'support_level': 0,
            'score': 0,
            'notes': ''
        }
        
        current = price.iloc[-1]
        high_20d = price.iloc[-20:].max()
        low_20d = price.iloc[-20:].min()
        high_50d = price.iloc[-50:].max()
        low_50d = price.iloc[-50:].min()
        
        # Calculate recent volatility
        returns = price.pct_change().dropna()
        vol_recent = returns.iloc[-10:].std()
        vol_prior = returns.iloc[-30:-10].std() if len(returns) >= 30 else returns.std()
        
        # 1. Check for Consolidation near Highs (within 5% of 20-day high)
        dist_from_high = (current / high_20d - 1) * 100
        
        if dist_from_high >= -5:
            # Check for volatility contraction (VCP-like)
            if vol_prior > 0 and vol_recent < vol_prior * 0.6:
                result['pattern'] = 'VCP'
                result['breakout_level'] = high_20d
                result['support_level'] = low_20d
                result['score'] = 8
                result['notes'] = f"Vol contracted {(1-vol_recent/vol_prior)*100:.0f}%"
            else:
                result['pattern'] = 'Tight Range'
                result['breakout_level'] = high_20d
                result['support_level'] = low_20d
                result['score'] = 6
                result['notes'] = f"{dist_from_high:.1f}% from 20D high"
        
        # 2. Check for High Tight Flag (big run + tight consolidation)
        if len(price) >= 60:
            return_60d = (high_50d / price.iloc[-60] - 1) * 100
            consolidation_range = (high_20d / low_20d - 1) * 100
            
            if return_60d >= 50 and consolidation_range <= 20:
                result['pattern'] = 'High Tight Flag'
                result['breakout_level'] = high_20d
                result['support_level'] = low_20d
                result['score'] = 9
                result['notes'] = f"+{return_60d:.0f}% in 60d, {consolidation_range:.0f}% range"
        
        # 3. Check for Episodic Pivot (gap up on volume)
        if volume is not None and len(volume) >= 50:
            avg_vol = volume.iloc[-50:].mean()
            today_vol = volume.iloc[-1]
            yesterday_price = price.iloc[-2] if len(price) >= 2 else price.iloc[-1]
            gap_pct = (current / yesterday_price - 1) * 100
            
            if gap_pct >= 5 and today_vol >= avg_vol * 2:
                result['pattern'] = 'Episodic Pivot'
                result['breakout_level'] = current
                result['support_level'] = yesterday_price
                result['score'] = 8
                result['notes'] = f"Gap +{gap_pct:.1f}%, Vol {today_vol/avg_vol:.1f}x"
        
        return result
    
    def get_daily_alerts(self, prices: pd.DataFrame, volume: pd.DataFrame = None) -> str:
        """Generate daily alert summary as formatted string."""
        alerts_df = self.scan_for_breakouts(prices, volume)
        
        if alerts_df.empty:
            return "No breakout alerts today."
        
        summary = []
        summary.append("=" * 60)
        summary.append("🚀 DAILY BREAKOUT ALERTS")
        summary.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        summary.append("=" * 60)
        
        for _, row in alerts_df.iterrows():
            summary.append(f"\n{row['Ticker']} - {row['Pattern']} (Score: {row['Score']}/10)")
            summary.append(f"  Current: ${row['Current']}")
            summary.append(f"  Breakout: ${row['Breakout']}")
            summary.append(f"  Support: ${row['Support']}")
            summary.append(f"  Notes: {row['Notes']}")
        
        summary.append("\n" + "=" * 60)
        
        return "\n".join(summary)


# ============================================================================
# MAIN: INTEGRATED SCANNER
# ============================================================================

class QuallamaggieTradingSystem:
    """
    Integrated trading system combining all Quallamaggie tools.
    """
    
    def __init__(self, config: QuallamaggieToolsConfig = None):
        self.config = config or CONFIG
        self.screener = MomentumScreener(config)
        self.rs_indicator = RSLineIndicator(config)
        self.watchlist = WatchlistManager(config)
        self.position_sizer = PositionSizer(config)
        self.breakout_scanner = BreakoutScanner(config)
        
        self.prices = None
        self.volume = None
    
    def load_data(self, force_refresh: bool = False):
        """Load price and volume data."""
        print("Loading market data...")
        self.prices = self.screener.fetch_prices(force_refresh=force_refresh)
        
        # Fetch volume separately
        try:
            vol_data = yf.download(
                self.config.UNIVERSE,
                start=self.config.DATA_START,
                period="1y",
                progress=False,
                threads=False
            )
            if isinstance(vol_data.columns, pd.MultiIndex):
                self.volume = vol_data['Volume']
            else:
                self.volume = vol_data[['Volume']] if 'Volume' in vol_data.columns else None
        except Exception as e:
            print(f"Warning: Could not fetch volume data: {e}")
            self.volume = None
    
    def run_daily_scan(self) -> Dict[str, pd.DataFrame]:
        """
        Run complete daily scan.
        
        Returns dict with:
        - momentum: Top momentum stocks
        - rs_leaders: Stocks with RS at new highs
        - breakouts: Breakout candidates
        - watchlist_triggers: Triggered watchlist stocks
        """
        print("\n" + "=" * 60)
        print("QUALLAMAGGIE DAILY SCAN")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)
        
        if self.prices is None:
            self.load_data()
        
        results = {}
        
        # 1. Momentum Screen
        print("\n📈 MOMENTUM SCREENING...")
        results['momentum'] = self.screener.screen(timeframe='6M')
        
        # 2. RS Leaders
        print("\n🏆 RS LINE ANALYSIS...")
        results['rs_leaders'] = self.rs_indicator.get_rs_leaders(self.prices)
        
        # 3. Breakout Candidates
        print("\n🚀 BREAKOUT SCANNER...")
        results['breakouts'] = self.breakout_scanner.scan_for_breakouts(
            self.prices, self.volume
        )
        
        # 4. Watchlist Triggers
        print("\n📋 WATCHLIST CHECK...")
        results['watchlist_triggers'] = self.watchlist.check_triggers(self.prices)
        
        # Summary
        print("\n" + "=" * 60)
        print("SCAN SUMMARY")
        print("=" * 60)
        print(f"Top Momentum Stocks: {len(results['momentum'])}")
        print(f"RS Leaders: {len(results['rs_leaders'])}")
        print(f"Breakout Candidates: {len(results['breakouts'])}")
        print(f"Watchlist Triggers: {len(results['watchlist_triggers'])}")
        
        return results
    
    def calculate_position(
        self,
        account_value: float,
        ticker: str,
        entry_price: float,
        stop_loss: float
    ):
        """Calculate and display position sizing."""
        result = self.position_sizer.calculate_position(
            account_value, entry_price, stop_loss
        )
        self.position_sizer.print_position(result)
        return result


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point for running the trading system."""
    
    # Initialize system
    system = QuallamaggieTradingSystem()
    
    # Run daily scan
    results = system.run_daily_scan()
    
    # Example: Show top momentum stocks
    if not results['momentum'].empty:
        print("\n📊 TOP MOMENTUM STOCKS (6M):")
        print(results['momentum'].head(10).to_string())
    
    # Example: Show breakout candidates
    if not results['breakouts'].empty:
        print("\n🚀 BREAKOUT CANDIDATES:")
        print(results['breakouts'].head(10).to_string())
    
    # Example: Position sizing
    print("\n💰 EXAMPLE POSITION SIZING:")
    system.calculate_position(
        account_value=100000,
        ticker="NVDA",
        entry_price=140.00,
        stop_loss=135.00
    )
    
    return system


if __name__ == "__main__":
    main()

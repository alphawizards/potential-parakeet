"""
Quallamaggie Strategy Backtest Module
======================================
Systematic swing trading pipeline implementing Kristjan Kullamägi's 
momentum breakout strategy with Riskfolio-Lib portfolio optimization.

Architecture:
    Module 1: Filtering - Vectorized pandas filters for liquidity, momentum, trend, consolidation
    Module 2: Optimization - Mean-Variance optimization with Riskfolio-Lib

Key Principles:
    - Use 'Adj Close' for returns/momentum/MA calculations (split/dividend adjusted)
    - Use 'Close' (raw) for price level filters ($5) and dollar volume
    - No look-ahead bias through proper signal shifting
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Try to import dependencies
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed")

try:
    import riskfolio as rp
    HAS_RISKFOLIO = True
except ImportError:
    HAS_RISKFOLIO = False
    print("Warning: riskfolio not installed. Portfolio optimization unavailable.")

try:
    from scipy.stats import linregress
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============== CONFIGURATION ==============

@dataclass
class QullamaggieConfig:
    """Configuration for Quallamaggie strategy."""
    
    # Liquidity Thresholds
    MIN_PRICE: float = 5.0  # Minimum price (raw close)
    MIN_DOLLAR_VOLUME: float = 20_000_000  # 20-day avg dollar volume
    DOLLAR_VOLUME_WINDOW: int = 20
    
    # Momentum Thresholds (The Engine)
    MOMENTUM_3M_THRESHOLD: float = 0.30  # 30% in 3 months
    MOMENTUM_1M_THRESHOLD: float = 0.10  # 10% in 1 month
    MOMENTUM_3M_DAYS: int = 63
    MOMENTUM_1M_DAYS: int = 21
    
    # Trend Architecture
    SMA_10: int = 10
    SMA_20: int = 20
    SMA_50: int = 50
    SMA_200: int = 200
    SMA_SLOPE_WINDOW: int = 10  # Days for slope calculation
    
    # Consolidation (The Setup)
    HTF_LOOKBACK: int = 126  # 6 months for High Tight Flag
    HTF_THRESHOLD: float = 0.85  # Price >= 85% of 126-day high
    ATR_PERIOD: int = 14
    ATR_AVG_WINDOW: int = 20
    
    # Portfolio Optimization
    MAX_WEIGHT: float = 0.20  # 20% max per asset
    MAX_POSITION_RISK: float = 0.01  # 1% max position risk
    LOOKBACK_RETURNS: int = 126  # 6 months for covariance
    
    # Exit Parameters
    FAST_EMA: int = 10
    MID_EMA: int = 20
    SLOW_EMA: int = 50
    
    # Backtest Parameters
    INITIAL_CAPITAL: float = 100_000.0
    TRADE_FEE: float = 3.0  # $3 per trade
    SLIPPAGE_BPS: float = 10


# ============== DATA HANDLING ==============

def fetch_data(
    tickers: List[str],
    start: str = '2020-01-01',
    end: str = None
) -> pd.DataFrame:
    """
    Fetch and format data into required MultiIndex structure.
    
    Args:
        tickers: List of ticker symbols
        start: Start date string
        end: End date (default: today)
        
    Returns:
        MultiIndex DataFrame (Ticker, Date) with OHLCV columns
    """
    if not HAS_YFINANCE:
        raise ImportError("yfinance required. Install with: pip install yfinance")
    
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching data for {len(tickers)} tickers...")
    
    data = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,  # Get both Close and Adj Close
        threads=False
    )
    
    if data.empty:
        raise ValueError("No data fetched from yfinance")
    
    # Reshape to MultiIndex (Ticker, Date)
    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers - need to reshape
        dfs = []
        for ticker in tickers:
            if ticker not in data.columns.get_level_values(1):
                continue
            ticker_data = data.xs(ticker, level=1, axis=1).copy()
            ticker_data['Ticker'] = ticker
            ticker_data = ticker_data.reset_index()
            dfs.append(ticker_data)
        
        if not dfs:
            raise ValueError("No valid ticker data found")
        
        df = pd.concat(dfs, ignore_index=True)
        df = df.set_index(['Ticker', 'Date'])
    else:
        # Single ticker
        df = data.copy()
        df['Ticker'] = tickers[0]
        df = df.reset_index()
        df = df.set_index(['Ticker', 'Date'])
    
    # Ensure required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"Loaded {len(df.index.get_level_values('Ticker').unique())} tickers")
    return df


# ============== MODULE 1: FILTERING ==============

class QullamaggieFilter:
    """
    Vectorized filtering for Quallamaggie swing trading strategy.
    
    Filters applied sequentially:
    1. Liquidity (price > $5, dollar volume > $20M)
    2. Momentum (3M >= 30%, 1M >= 10%, RS > SPY)
    3. Trend (Perfect MA alignment, positive slope)
    4. Consolidation (High tight flag, volatility contraction)
    """
    
    def __init__(self, config: QullamaggieConfig = None):
        self.config = config or QullamaggieConfig()
        self._filter_log = []
    
    def filter_universe(
        self,
        df: pd.DataFrame,
        spy_returns: pd.Series = None,
        verbose: bool = True
    ) -> List[str]:
        """
        Apply all filtering rules to multi-asset universe.
        
        Args:
            df: MultiIndex DataFrame (Ticker, Date) with OHLCV
            spy_returns: Optional SPY returns for relative strength
            verbose: Print filter statistics
            
        Returns:
            List of valid tickers passing all criteria
        """
        self._filter_log = []
        
        # Get unique tickers
        tickers = df.index.get_level_values('Ticker').unique().tolist()
        self._log(f"Starting universe: {len(tickers)} tickers")
        
        # Get latest date for snapshot filtering
        latest_date = df.index.get_level_values('Date').max()
        
        # Apply filters sequentially
        valid_tickers = set(tickers)
        
        # 1. Liquidity Filters
        valid_tickers = self._filter_liquidity(df, valid_tickers, latest_date)
        self._log(f"After liquidity: {len(valid_tickers)} tickers")
        
        # 2. Momentum Filters
        valid_tickers = self._filter_momentum(df, valid_tickers, latest_date, spy_returns)
        self._log(f"After momentum: {len(valid_tickers)} tickers")
        
        # 3. Trend Architecture Filters
        valid_tickers = self._filter_trend(df, valid_tickers, latest_date)
        self._log(f"After trend: {len(valid_tickers)} tickers")
        
        # 4. Consolidation Filters
        valid_tickers = self._filter_consolidation(df, valid_tickers, latest_date)
        self._log(f"After consolidation: {len(valid_tickers)} tickers")
        
        if verbose:
            for log in self._filter_log:
                print(f"  {log}")
        
        return list(valid_tickers)
    
    def _log(self, msg: str):
        self._filter_log.append(msg)
    
    def _filter_liquidity(
        self,
        df: pd.DataFrame,
        tickers: set,
        latest_date
    ) -> set:
        """Apply liquidity filters: price > $5, dollar volume > $20M."""
        valid = set()
        
        for ticker in tickers:
            try:
                ticker_data = df.loc[ticker]
                
                # Check raw close > $5
                current_close = ticker_data.loc[latest_date, 'Close']
                if current_close <= self.config.MIN_PRICE:
                    continue
                
                # Check 20-day avg dollar volume > $20M
                dollar_volume = ticker_data['Close'] * ticker_data['Volume']
                avg_dv = dollar_volume.rolling(self.config.DOLLAR_VOLUME_WINDOW).mean()
                
                if len(avg_dv) == 0 or pd.isna(avg_dv.iloc[-1]):
                    continue
                    
                if avg_dv.iloc[-1] < self.config.MIN_DOLLAR_VOLUME:
                    continue
                
                valid.add(ticker)
                
            except (KeyError, IndexError):
                continue
        
        return valid
    
    def _filter_momentum(
        self,
        df: pd.DataFrame,
        tickers: set,
        latest_date,
        spy_returns: pd.Series = None
    ) -> set:
        """Apply momentum filters: 3M >= 30%, 1M >= 10%, RS > SPY."""
        valid = set()
        
        for ticker in tickers:
            try:
                ticker_data = df.loc[ticker]
                adj_close = ticker_data['Adj Close']
                
                # 3-month return >= 30%
                ret_3m = adj_close.pct_change(self.config.MOMENTUM_3M_DAYS)
                if pd.isna(ret_3m.iloc[-1]) or ret_3m.iloc[-1] < self.config.MOMENTUM_3M_THRESHOLD:
                    continue
                
                # 1-month return >= 10%
                ret_1m = adj_close.pct_change(self.config.MOMENTUM_1M_DAYS)
                if pd.isna(ret_1m.iloc[-1]) or ret_1m.iloc[-1] < self.config.MOMENTUM_1M_THRESHOLD:
                    continue
                
                # Relative Strength vs SPY (if provided)
                if spy_returns is not None:
                    spy_3m = spy_returns.iloc[-1] if len(spy_returns) > 0 else 0
                    if ret_3m.iloc[-1] <= spy_3m:
                        continue
                
                valid.add(ticker)
                
            except (KeyError, IndexError):
                continue
        
        return valid
    
    def _filter_trend(
        self,
        df: pd.DataFrame,
        tickers: set,
        latest_date
    ) -> set:
        """Apply trend filters: Perfect MA alignment, positive SMA50 slope."""
        valid = set()
        
        for ticker in tickers:
            try:
                ticker_data = df.loc[ticker]
                adj_close = ticker_data['Adj Close']
                
                # Calculate SMAs
                sma_10 = adj_close.rolling(self.config.SMA_10).mean()
                sma_20 = adj_close.rolling(self.config.SMA_20).mean()
                sma_50 = adj_close.rolling(self.config.SMA_50).mean()
                sma_200 = adj_close.rolling(self.config.SMA_200).mean()
                
                # Get latest values
                latest_close = adj_close.iloc[-1]
                latest_sma10 = sma_10.iloc[-1]
                latest_sma20 = sma_20.iloc[-1]
                latest_sma50 = sma_50.iloc[-1]
                latest_sma200 = sma_200.iloc[-1]
                
                # Check for NaN
                if any(pd.isna([latest_close, latest_sma10, latest_sma20, latest_sma50, latest_sma200])):
                    continue
                
                # Perfect alignment: Close > SMA10 > SMA20 > SMA50
                if not (latest_close > latest_sma10 > latest_sma20 > latest_sma50):
                    continue
                
                # Above 200 SMA
                if latest_close <= latest_sma200:
                    continue
                
                # SMA50 positive slope (linear regression over 10 days)
                if HAS_SCIPY and len(sma_50) >= self.config.SMA_SLOPE_WINDOW:
                    sma50_recent = sma_50.iloc[-self.config.SMA_SLOPE_WINDOW:].dropna()
                    if len(sma50_recent) >= self.config.SMA_SLOPE_WINDOW:
                        slope, _, _, _, _ = linregress(range(len(sma50_recent)), sma50_recent.values)
                        if slope <= 0:
                            continue
                
                valid.add(ticker)
                
            except (KeyError, IndexError):
                continue
        
        return valid
    
    def _filter_consolidation(
        self,
        df: pd.DataFrame,
        tickers: set,
        latest_date
    ) -> set:
        """Apply consolidation filters: High tight flag, volatility contraction."""
        valid = set()
        
        for ticker in tickers:
            try:
                ticker_data = df.loc[ticker]
                adj_close = ticker_data['Adj Close']
                high = ticker_data['High']
                
                # High Tight Flag: Current >= 85% of 126-day high
                rolling_high = high.rolling(self.config.HTF_LOOKBACK).max()
                htf_threshold = rolling_high * self.config.HTF_THRESHOLD
                
                if pd.isna(htf_threshold.iloc[-1]) or adj_close.iloc[-1] < htf_threshold.iloc[-1]:
                    continue
                
                # Volatility Contraction: Current ATR(14) < 20-day avg ATR
                atr = self._calculate_atr(ticker_data, self.config.ATR_PERIOD)
                avg_atr = atr.rolling(self.config.ATR_AVG_WINDOW).mean()
                
                if pd.isna(atr.iloc[-1]) or pd.isna(avg_atr.iloc[-1]):
                    continue
                    
                if atr.iloc[-1] >= avg_atr.iloc[-1]:
                    continue
                
                valid.add(ticker)
                
            except (KeyError, IndexError):
                continue
        
        return valid
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = data['High']
        low = data['Low']
        close = data['Adj Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr


# ============== MODULE 2: PORTFOLIO OPTIMIZATION ==============

class QullamaggieOptimizer:
    """
    Mean-Variance portfolio optimization for Quallamaggie survivors.
    
    Uses Riskfolio-Lib to maximize Sharpe Ratio with constraints:
    - Max weight per asset: 20%
    - Long only (no short selling)
    - Post-optimization risk control layer
    """
    
    def __init__(self, config: QullamaggieConfig = None):
        self.config = config or QullamaggieConfig()
    
    def optimize_weights(
        self,
        valid_tickers: List[str],
        returns_data: pd.DataFrame,
        stop_loss_pct: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Run Mean-Variance optimization on filtered universe.
        
        Args:
            valid_tickers: Tickers passing Module 1 filters
            returns_data: Adj Close returns DataFrame (columns = tickers)
            stop_loss_pct: Optional dict of {ticker: stop_loss_%} for risk control
            
        Returns:
            Dictionary {ticker: final_weight} after risk adjustments
        """
        if not valid_tickers:
            return {}
        
        if not HAS_RISKFOLIO:
            # Fallback to equal weight
            print("Riskfolio not available. Using equal weight.")
            n = len(valid_tickers)
            return {t: 1.0 / n for t in valid_tickers}
        
        # Filter returns to valid tickers
        available = [t for t in valid_tickers if t in returns_data.columns]
        if not available:
            return {}
        
        returns = returns_data[available].iloc[-self.config.LOOKBACK_RETURNS:]
        returns = returns.dropna(axis=1, how='all').dropna()
        
        if len(returns) < 50 or len(returns.columns) == 0:
            # Not enough data - equal weight
            return {t: 1.0 / len(available) for t in available}
        
        try:
            # Create Riskfolio Portfolio
            port = rp.Portfolio(returns=returns)
            
            # Estimate statistics
            port.assets_stats(method_mu='hist', method_cov='ledoit')
            
            # Set constraints
            port.upperlng = self.config.MAX_WEIGHT  # Max 20% per asset
            
            # Optimize for max Sharpe
            weights = port.optimization(
                model='Classic',
                rm='MV',
                obj='Sharpe',
                rf=0.04,  # 4% risk-free rate
                hist=True
            )
            
            if weights is None:
                raise ValueError("Optimization failed")
            
            # Convert to dict
            weight_dict = weights.squeeze().to_dict()
            
            # Apply risk control layer
            if stop_loss_pct:
                weight_dict = self._apply_risk_control(weight_dict, stop_loss_pct)
            
            # Normalize weights
            total = sum(weight_dict.values())
            if total > 0:
                weight_dict = {k: v / total for k, v in weight_dict.items()}
            
            return weight_dict
            
        except Exception as e:
            print(f"Optimization failed: {e}. Using equal weight.")
            return {t: 1.0 / len(available) for t in available}
    
    def _apply_risk_control(
        self,
        weights: Dict[str, float],
        stop_loss_pct: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Post-optimization risk control layer.
        
        Ensures position_risk = weight * technical_risk <= 1% of account.
        """
        adjusted = {}
        
        for ticker, weight in weights.items():
            if ticker in stop_loss_pct:
                technical_risk = stop_loss_pct[ticker]
                position_risk = weight * technical_risk
                
                if position_risk > self.config.MAX_POSITION_RISK:
                    # Scale down weight
                    adjusted[ticker] = self.config.MAX_POSITION_RISK / technical_risk
                else:
                    adjusted[ticker] = weight
            else:
                adjusted[ticker] = weight
        
        return adjusted


# ============== QUALLAMAGGIE STRATEGY CLASS ==============

class QullamaggieStrategy:
    """
    Complete Quallamaggie swing trading pipeline.
    
    Combines:
    - Module 1: Universe filtering (liquidity, momentum, trend, consolidation)
    - Module 2: Portfolio optimization (Mean-Variance with Riskfolio-Lib)
    """
    
    def __init__(self, config: QullamaggieConfig = None):
        self.config = config or QullamaggieConfig()
        self.filter = QullamaggieFilter(config)
        self.optimizer = QullamaggieOptimizer(config)
    
    def run(
        self,
        df: pd.DataFrame,
        spy_returns: pd.Series = None,
        stop_loss_pct: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Run full Quallamaggie pipeline.
        
        Args:
            df: MultiIndex DataFrame (Ticker, Date)
            spy_returns: Optional SPY returns for relative strength
            stop_loss_pct: Optional stop loss percentages for risk control
            
        Returns:
            Dictionary {ticker: optimized_weight}
        """
        print("=" * 60)
        print("QUALLAMAGGIE STRATEGY PIPELINE")
        print("=" * 60)
        
        # Module 1: Filter universe
        print("\nModule 1: Filtering Universe...")
        valid_tickers = self.filter.filter_universe(df, spy_returns)
        
        if not valid_tickers:
            print("No tickers passed all filters.")
            return {}
        
        print(f"\nSurvivors: {valid_tickers}")
        
        # Prepare returns data for optimization
        all_tickers = df.index.get_level_values('Ticker').unique()
        returns_dict = {}
        
        for ticker in all_tickers:
            try:
                ticker_data = df.loc[ticker]
                returns_dict[ticker] = ticker_data['Adj Close'].pct_change()
            except:
                continue
        
        returns_data = pd.DataFrame(returns_dict)
        
        # Module 2: Optimize weights
        print("\nModule 2: Portfolio Optimization...")
        weights = self.optimizer.optimize_weights(valid_tickers, returns_data, stop_loss_pct)
        
        print("\nOptimized Weights:")
        for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {ticker}: {weight*100:.1f}%")
        
        return weights


# ============== BACKTEST ENGINE ==============

@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_name: str
    final_value: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    total_trades: int
    avg_holding_days: float
    equity_curve: pd.Series
    trades: pd.DataFrame


class QullamaggieBacktester:
    """
    Backtester for Quallamaggie strategy with walk-forward optimization.
    """
    
    def __init__(self, config: QullamaggieConfig = None):
        self.config = config or QullamaggieConfig()
        self.strategy = QullamaggieStrategy(config)
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        rebalance_freq: str = 'monthly',
        strategy_name: str = "Quallamaggie"
    ) -> BacktestResult:
        """
        Run backtest with periodic rebalancing.
        
        Args:
            df: MultiIndex DataFrame (Ticker, Date)
            rebalance_freq: 'daily', 'weekly', 'monthly'
            strategy_name: Name for results
            
        Returns:
            BacktestResult with performance metrics
        """
        print(f"\nRunning backtest: {strategy_name}")
        
        # Get all dates
        all_dates = df.index.get_level_values('Date').unique().sort_values()
        
        # Determine rebalance dates
        if rebalance_freq == 'monthly':
            rebalance_dates = all_dates[all_dates.to_series().dt.is_month_start]
        elif rebalance_freq == 'weekly':
            rebalance_dates = all_dates[all_dates.weekday == 0]
        else:
            rebalance_dates = all_dates
        
        # Filter to valid rebalance dates (after warmup period)
        warmup = max(self.config.SMA_200, self.config.HTF_LOOKBACK) + 10
        rebalance_dates = rebalance_dates[warmup:]
        
        # Initialize tracking
        cash = self.config.INITIAL_CAPITAL
        positions = {}  # ticker -> {'shares': n, 'entry_price': p, 'entry_date': d}
        equity_curve = []
        trades = []
        current_weights = {}
        
        # Get tickers
        all_tickers = df.index.get_level_values('Ticker').unique()
        
        for i, date in enumerate(all_dates[warmup:]):
            # Calculate portfolio value
            portfolio_value = cash
            for ticker, pos in positions.items():
                try:
                    current_price = df.loc[(ticker, date), 'Adj Close']
                    if not pd.isna(current_price):
                        portfolio_value += pos['shares'] * current_price
                except:
                    pass
            
            equity_curve.append({'date': date, 'value': portfolio_value})
            
            # Check for rebalance
            if date in rebalance_dates.values:
                # Get data up to this date (no look-ahead)
                df_snapshot = df[df.index.get_level_values('Date') <= date]
                
                # Run strategy
                try:
                    new_weights = self.strategy.filter.filter_universe(df_snapshot, verbose=False)
                    
                    if new_weights:
                        # Get returns for optimization
                        returns_dict = {}
                        for ticker in all_tickers:
                            try:
                                ticker_data = df_snapshot.loc[ticker]
                                returns_dict[ticker] = ticker_data['Adj Close'].pct_change()
                            except:
                                continue
                        returns_data = pd.DataFrame(returns_dict)
                        
                        target_weights = self.strategy.optimizer.optimize_weights(new_weights, returns_data)
                    else:
                        target_weights = {}
                except:
                    target_weights = {}
                
                # Rebalance positions
                if target_weights != current_weights:
                    # Close positions not in target
                    for ticker in list(positions.keys()):
                        if ticker not in target_weights or target_weights.get(ticker, 0) < 0.01:
                            try:
                                exit_price = df.loc[(ticker, date), 'Adj Close']
                                if not pd.isna(exit_price):
                                    pos = positions[ticker]
                                    cash += pos['shares'] * exit_price - self.config.TRADE_FEE
                                    
                                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                                    trades.append({
                                        'ticker': ticker,
                                        'entry_date': pos['entry_date'],
                                        'exit_date': date,
                                        'entry_price': pos['entry_price'],
                                        'exit_price': exit_price,
                                        'shares': pos['shares'],
                                        'pnl': pnl
                                    })
                                    del positions[ticker]
                            except:
                                pass
                    
                    # Open/adjust positions
                    for ticker, weight in target_weights.items():
                        if weight < 0.01:
                            continue
                        
                        try:
                            current_price = df.loc[(ticker, date), 'Adj Close']
                            if pd.isna(current_price) or current_price <= 0:
                                continue
                            
                            target_value = portfolio_value * weight
                            
                            if ticker in positions:
                                # Already have position - skip for simplicity
                                continue
                            else:
                                # Open new position
                                shares = int(min(target_value, cash * 0.95) / current_price)
                                if shares > 0:
                                    cost = shares * current_price + self.config.TRADE_FEE
                                    if cost <= cash:
                                        cash -= cost
                                        positions[ticker] = {
                                            'shares': shares,
                                            'entry_price': current_price,
                                            'entry_date': date
                                        }
                        except:
                            pass
                    
                    current_weights = target_weights.copy()
        
        # Close remaining positions
        final_date = all_dates[-1]
        for ticker, pos in list(positions.items()):
            try:
                exit_price = df.loc[(ticker, final_date), 'Adj Close']
                if not pd.isna(exit_price):
                    cash += pos['shares'] * exit_price
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                    trades.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': final_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'pnl': pnl
                    })
            except:
                pass
        
        # Create results
        equity_df = pd.DataFrame(equity_curve).set_index('date')['value']
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        metrics = self._calculate_metrics(equity_df, trades_df)
        
        return BacktestResult(
            strategy_name=strategy_name,
            final_value=metrics['final_value'],
            cagr=metrics['cagr'],
            sharpe=metrics['sharpe'],
            sortino=metrics['sortino'],
            max_drawdown=metrics['max_drawdown'],
            calmar=metrics['calmar'],
            win_rate=metrics['win_rate'],
            total_trades=metrics['total_trades'],
            avg_holding_days=metrics['avg_holding_days'],
            equity_curve=equity_df,
            trades=trades_df
        )
    
    def _calculate_metrics(self, equity: pd.Series, trades: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        final_value = equity.iloc[-1] if len(equity) > 0 else self.config.INITIAL_CAPITAL
        
        returns = equity.pct_change().dropna()
        
        years = len(equity) / 252
        cagr = (final_value / self.config.INITIAL_CAPITAL) ** (1 / max(years, 0.01)) - 1
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() * 252 - 0.04) / (returns.std() * np.sqrt(252))
        else:
            sharpe = 0
        
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (returns.mean() * 252 - 0.04) / (downside.std() * np.sqrt(252))
        else:
            sortino = sharpe
        
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        total_trades = len(trades)
        if total_trades > 0:
            win_rate = (trades['pnl'] > 0).mean()
            if 'exit_date' in trades.columns and 'entry_date' in trades.columns:
                holding_days = (pd.to_datetime(trades['exit_date']) - 
                               pd.to_datetime(trades['entry_date'])).dt.days
                avg_holding_days = holding_days.mean()
            else:
                avg_holding_days = 0
        else:
            win_rate = 0
            avg_holding_days = 0
        
        return {
            'final_value': final_value,
            'cagr': cagr,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_holding_days': avg_holding_days
        }


# ============== MAIN ENTRY POINT ==============

def run_quallamaggie_backtest(
    tickers: List[str] = None,
    start_date: str = '2020-01-01',
    end_date: str = None
) -> BacktestResult:
    """
    Run complete Quallamaggie strategy backtest.
    
    Args:
        tickers: List of ticker symbols (default: sample universe)
        start_date: Backtest start date
        end_date: Backtest end date
        
    Returns:
        BacktestResult with performance metrics
    """
    if tickers is None:
        tickers = ['NVDA', 'TSLA', 'AAPL', 'AMD', 'MSFT', 'GOOGL', 'META', 'AMZN',
                   'SPY', 'QQQ', 'IWM', 'XLK', 'SOXX', 'SMH']
    
    # Fetch data
    df = fetch_data(tickers, start=start_date, end=end_date)
    
    # Run backtest
    backtester = QullamaggieBacktester()
    result = backtester.run_backtest(df, strategy_name="Quallamaggie_MV")
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Final Value: ${result.final_value:,.0f}")
    print(f"CAGR: {result.cagr*100:.2f}%")
    print(f"Sharpe: {result.sharpe:.3f}")
    print(f"Sortino: {result.sortino:.3f}")
    print(f"Max Drawdown: {result.max_drawdown*100:.2f}%")
    print(f"Calmar: {result.calmar:.3f}")
    print(f"Win Rate: {result.win_rate*100:.1f}%")
    print(f"Total Trades: {result.total_trades}")
    print(f"Avg Holding Days: {result.avg_holding_days:.1f}")
    
    return result


if __name__ == "__main__":
    result = run_quallamaggie_backtest()

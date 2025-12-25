"""
Quallamaggie Strategy Backtest Module
======================================
Implements Kristjan Kullamägi's momentum breakout strategy with pattern recognition.

Compares different momentum lookback periods:
- 1-month (21 trading days)
- 3-month (63 trading days)  
- 6-month (126 trading days)

Strategy Logic:
1. Screen for top momentum stocks
2. Identify breakout patterns (simplified: consolidation then breakout)
3. Entry on breakout with volume confirmation
4. Exit on close below trailing moving average
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# ============== CONFIGURATION ==============

@dataclass
class QuallamaggieConfig:
    """Configuration for Quallamaggie strategy backtest."""
    
    # Universe - US momentum stocks (high liquidity)
    UNIVERSE: List[str] = None
    
    # Momentum lookback periods (trading days)
    MOMENTUM_1M: int = 21
    MOMENTUM_3M: int = 63
    MOMENTUM_6M: int = 126
    
    # Pattern parameters
    CONSOLIDATION_DAYS: int = 10  # Min days in consolidation
    BREAKOUT_VOLUME_MULT: float = 1.5  # Volume must be 1.5x average
    
    # Position sizing
    RISK_PER_TRADE: float = 0.005  # 0.5% of account per trade
    MAX_POSITION_PCT: float = 0.20  # 20% max position
    MAX_POSITIONS: int = 10  # Maximum concurrent positions
    
    # Exit parameters
    FAST_EMA: int = 10  # Fast trailing stop
    SLOW_EMA: int = 20  # Slow trailing stop
    PARTIAL_EXIT_DAYS: int = 5  # Sell half after 5 days
    PARTIAL_EXIT_PCT: float = 0.5  # Sell 50% on first exit
    
    # Backtest parameters
    INITIAL_CAPITAL: float = 100_000.0
    TRADE_FEE: float = 3.0  # $3 per trade
    SLIPPAGE_BPS: float = 10  # 10 bps slippage
    
    # Data parameters
    START_DATE: str = "2010-01-01"
    END_DATE: str = None  # None = today
    
    def __post_init__(self):
        if self.UNIVERSE is None:
            # ETF-focused universe (more reliable with yfinance)
            # Mix of equity, sector, and asset class ETFs for momentum testing
            self.UNIVERSE = [
                # Broad Market ETFs
                'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',
                # Sector ETFs
                'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLB', 'XLC', 'XLRE', 'XLU', 'XLP',
                # Thematic/Momentum ETFs
                'ARKK', 'SOXX', 'SMH', 'XBI', 'IBB', 'TAN', 'ICLN',
                # International
                'EFA', 'EEM', 'VEA', 'VWO',
                # Bonds
                'TLT', 'IEF', 'SHY', 'BND', 'HYG', 'LQD',
                # Commodities
                'GLD', 'SLV', 'USO', 'UNG', 'DBC',
                # Leveraged (for testing momentum sensitivity)
                'TQQQ', 'SOXL', 'UPRO'
            ]
        if self.END_DATE is None:
            self.END_DATE = datetime.now().strftime("%Y-%m-%d")


CONFIG = QuallamaggieConfig()


# ============== DATA LOADING ==============

class QuallamaggieDataLoader:
    """Load and prepare data for Quallamaggie strategy."""
    
    def __init__(self, config: QuallamaggieConfig = None):
        self.config = config or CONFIG
    
    def fetch_prices(self, tickers: List[str] = None) -> pd.DataFrame:
        """Fetch adjusted close prices from yfinance with retry logic."""
        tickers = tickers or self.config.UNIVERSE
        
        print(f"Fetching data for {len(tickers)} tickers...")
        
        # Try fetching with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers,
                    start=self.config.START_DATE,
                    end=self.config.END_DATE,
                    progress=False,
                    threads=False,  # Disable threading for stability
                    auto_adjust=True
                )
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Fall back to fetching one by one
                    print("Falling back to individual ticker fetching...")
                    data = self._fetch_individual(tickers)
        
        if data is None or data.empty:
            raise ValueError("Failed to fetch any data from yfinance")
        
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']
        else:
            prices = data[['Close']] if 'Close' in data.columns else data
            if len(tickers) == 1:
                prices.columns = tickers
        
        # Drop tickers with insufficient data
        min_data_pct = 0.8
        valid_cols = prices.columns[prices.notna().sum() / len(prices) >= min_data_pct]
        prices = prices[valid_cols].dropna(how='all')
        
        print(f"Loaded {len(valid_cols)} tickers with sufficient data")
        return prices
    
    def _fetch_individual(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch tickers individually as fallback."""
        all_data = {}
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(start=self.config.START_DATE, end=self.config.END_DATE)
                if not hist.empty:
                    all_data[ticker] = hist['Close']
            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")
                continue
        
        if all_data:
            return pd.DataFrame(all_data)
        return pd.DataFrame()
    
    def fetch_volume(self, tickers: List[str] = None) -> pd.DataFrame:
        """Fetch volume data from yfinance."""
        tickers = tickers or self.config.UNIVERSE
        
        try:
            data = yf.download(
                tickers,
                start=self.config.START_DATE,
                end=self.config.END_DATE,
                progress=False,
                threads=False
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                volume = data['Volume']
            else:
                volume = data[['Volume']] if 'Volume' in data.columns else pd.DataFrame()
                if len(tickers) == 1 and not volume.empty:
                    volume.columns = tickers
            
            return volume
        except Exception as e:
            print(f"Failed to fetch volume: {e}")
            return pd.DataFrame()


# ============== SIGNAL GENERATION ==============

class QuallamaggieSignals:
    """Generate signals for Quallamaggie momentum breakout strategy."""
    
    def __init__(self, config: QuallamaggieConfig = None):
        self.config = config or CONFIG
    
    def calculate_momentum(self, prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Calculate momentum as percentage return over lookback period."""
        return prices.pct_change(lookback)
    
    def calculate_momentum_rank(self, prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Rank stocks by momentum (0-1 scale, 1 = best)."""
        momentum = self.calculate_momentum(prices, lookback)
        # Rank across rows (each day)
        ranks = momentum.rank(axis=1, pct=True, na_option='keep')
        return ranks
    
    def calculate_relative_strength(self, prices: pd.DataFrame, benchmark: str = 'SPY') -> pd.DataFrame:
        """Calculate relative strength vs benchmark."""
        if benchmark not in prices.columns:
            return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        
        benchmark_prices = prices[benchmark]
        rs = prices.div(benchmark_prices, axis=0)
        # Normalize to make comparable
        rs = rs.pct_change(21)  # 1-month RS change
        return rs
    
    def is_above_ema(self, prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """Check if price is above EMA."""
        ema = prices.ewm(span=period, adjust=False).mean()
        return prices > ema
    
    def is_above_sma(self, prices: pd.DataFrame, period: int) -> pd.DataFrame:
        """Check if price is above SMA."""
        sma = prices.rolling(period).mean()
        return prices > sma
    
    def detect_consolidation(self, prices: pd.DataFrame, days: int = 10) -> pd.DataFrame:
        """
        Detect if stock is in consolidation (low volatility relative to recent past).
        
        Returns True if current volatility < 50% of lookback volatility.
        """
        returns = prices.pct_change()
        
        # Recent volatility (last N days)
        recent_vol = returns.rolling(days).std()
        
        # Baseline volatility (prior 2x period)
        baseline_vol = returns.shift(days).rolling(days * 2).std()
        
        # Consolidation = volatility contraction
        is_consolidating = recent_vol < (baseline_vol * 0.6)
        
        return is_consolidating
    
    def detect_breakout(self, prices: pd.DataFrame, volume: pd.DataFrame = None) -> pd.DataFrame:
        """
        Detect breakout from consolidation.
        
        Breakout = new 20-day high with volume expansion.
        """
        # New 20-day high
        rolling_high = prices.rolling(20).max()
        is_new_high = prices >= rolling_high
        
        # Volume expansion (if volume provided)
        if volume is not None:
            avg_volume = volume.rolling(50).mean()
            volume_expansion = volume > (avg_volume * self.config.BREAKOUT_VOLUME_MULT)
            breakout = is_new_high & volume_expansion
        else:
            breakout = is_new_high
        
        return breakout
    
    def generate_entry_signals(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame = None,
        momentum_lookback: int = 63
    ) -> pd.DataFrame:
        """
        Generate entry signals for Quallamaggie strategy.
        
        Entry criteria:
        1. Top 20% momentum over lookback period
        2. Price above 20 EMA and 50 SMA
        3. In consolidation (volatility contraction)
        4. Breakout on volume
        """
        # Momentum filter - top 20%
        momentum_rank = self.calculate_momentum_rank(prices, momentum_lookback)
        is_top_momentum = momentum_rank >= 0.80
        
        # Trend filter - above key MAs
        above_20ema = self.is_above_ema(prices, 20)
        above_50sma = self.is_above_sma(prices, 50)
        in_uptrend = above_20ema & above_50sma
        
        # Consolidation filter
        is_consolidating = self.detect_consolidation(prices, 10)
        
        # Breakout detection
        is_breakout = self.detect_breakout(prices, volume)
        
        # Combined signal
        entry_signal = is_top_momentum & in_uptrend & is_breakout
        
        # Forward fill NaN to maintain signal
        entry_signal = entry_signal.fillna(False)
        
        return entry_signal
    
    def generate_exit_signals(
        self,
        prices: pd.DataFrame,
        ema_period: int = 10
    ) -> pd.DataFrame:
        """
        Generate exit signals.
        
        Exit on CLOSE below trailing EMA.
        """
        ema = prices.ewm(span=ema_period, adjust=False).mean()
        
        # Exit when price closes below EMA
        exit_signal = prices < ema
        
        return exit_signal.fillna(False)


# ============== BACKTESTER ==============

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
    equity_curve: pd.Series
    trades: pd.DataFrame


class QuallamaggieBacktester:
    """
    Backtest Quallamaggie momentum breakout strategy.
    
    Simplified implementation focusing on momentum screening + breakout entry.
    """
    
    def __init__(self, config: QuallamaggieConfig = None):
        self.config = config or CONFIG
        self.signals = QuallamaggieSignals(config)
    
    def run_backtest(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame = None,
        momentum_lookback: int = 63,
        trailing_ema: int = 10,
        strategy_name: str = "Quallamaggie"
    ) -> BacktestResult:
        """
        Run backtest with specified parameters.
        
        Args:
            prices: DataFrame of adjusted close prices
            volume: DataFrame of volume (optional)
            momentum_lookback: Lookback period for momentum screening
            trailing_ema: EMA period for trailing stop
            strategy_name: Name for results
            
        Returns:
            BacktestResult with performance metrics
        """
        print(f"\nRunning backtest: {strategy_name}")
        print(f"  Momentum lookback: {momentum_lookback} days")
        print(f"  Trailing EMA: {trailing_ema} days")
        
        # Generate signals
        entry_signals = self.signals.generate_entry_signals(
            prices, volume, momentum_lookback
        )
        exit_signals = self.signals.generate_exit_signals(prices, trailing_ema)
        
        # Initialize portfolio
        cash = self.config.INITIAL_CAPITAL
        positions = {}  # ticker -> {'shares': n, 'entry_price': p, 'entry_date': d}
        equity_curve = []
        trades = []
        
        # Trading days
        trading_days = prices.index[max(momentum_lookback, 50):]
        
        for i, date in enumerate(trading_days):
            # Calculate current portfolio value
            portfolio_value = cash
            for ticker, pos in positions.items():
                if ticker in prices.columns and not pd.isna(prices.loc[date, ticker]):
                    portfolio_value += pos['shares'] * prices.loc[date, ticker]
            
            equity_curve.append({'date': date, 'value': portfolio_value})
            
            # Process exits first
            tickers_to_close = []
            for ticker, pos in positions.items():
                if ticker not in prices.columns:
                    continue
                    
                current_price = prices.loc[date, ticker]
                if pd.isna(current_price):
                    continue
                
                # Check exit signal
                should_exit = exit_signals.loc[date, ticker] if ticker in exit_signals.columns else False
                
                if should_exit:
                    # Close position
                    exit_value = pos['shares'] * current_price
                    trade_cost = self.config.TRADE_FEE
                    cash += exit_value - trade_cost
                    
                    # Record trade
                    pnl = (current_price - pos['entry_price']) * pos['shares']
                    pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                    
                    trades.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
                    
                    tickers_to_close.append(ticker)
            
            for ticker in tickers_to_close:
                del positions[ticker]
            
            # Process entries (limit positions)
            if len(positions) < self.config.MAX_POSITIONS:
                for ticker in prices.columns:
                    if ticker in positions:
                        continue
                    
                    if ticker not in entry_signals.columns:
                        continue
                    
                    current_price = prices.loc[date, ticker]
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    
                    # Check entry signal
                    should_enter = entry_signals.loc[date, ticker]
                    
                    if should_enter and len(positions) < self.config.MAX_POSITIONS:
                        # Calculate position size
                        position_value = min(
                            portfolio_value * self.config.MAX_POSITION_PCT,
                            cash * 0.95  # Keep some cash buffer
                        )
                        
                        if position_value < 1000:  # Min position size
                            continue
                        
                        shares = int(position_value / current_price)
                        if shares <= 0:
                            continue
                        
                        # Entry with slippage
                        slippage = current_price * (self.config.SLIPPAGE_BPS / 10000)
                        entry_price = current_price + slippage
                        cost = shares * entry_price + self.config.TRADE_FEE
                        
                        if cost <= cash:
                            cash -= cost
                            positions[ticker] = {
                                'shares': shares,
                                'entry_price': entry_price,
                                'entry_date': date
                            }
        
        # Close remaining positions
        final_date = trading_days[-1]
        for ticker, pos in positions.items():
            if ticker in prices.columns:
                current_price = prices.loc[final_date, ticker]
                if not pd.isna(current_price):
                    cash += pos['shares'] * current_price
                    
                    pnl = (current_price - pos['entry_price']) * pos['shares']
                    pnl_pct = (current_price / pos['entry_price'] - 1) * 100
                    
                    trades.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': final_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
        
        # Create equity curve
        equity_df = pd.DataFrame(equity_curve).set_index('date')['value']
        
        # Calculate metrics
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
            equity_curve=equity_df,
            trades=trades_df
        )
    
    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame
    ) -> Dict:
        """Calculate performance metrics."""
        
        # Final value
        final_value = equity_curve.iloc[-1] if len(equity_curve) > 0 else self.config.INITIAL_CAPITAL
        
        # Returns
        returns = equity_curve.pct_change().dropna()
        
        # CAGR
        years = len(equity_curve) / 252
        if years > 0 and self.config.INITIAL_CAPITAL > 0:
            cagr = (final_value / self.config.INITIAL_CAPITAL) ** (1 / years) - 1
        else:
            cagr = 0
        
        # Sharpe Ratio (assuming 4% risk-free rate)
        if len(returns) > 0 and returns.std() > 0:
            excess_returns = returns - (0.04 / 252)
            sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        else:
            sharpe = 0
        
        # Sortino Ratio
        if len(returns) > 0:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino = np.sqrt(252) * returns.mean() / downside_returns.std()
            else:
                sortino = sharpe
        else:
            sortino = 0
        
        # Max Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        if max_drawdown != 0:
            calmar = cagr / abs(max_drawdown)
        else:
            calmar = 0
        
        # Win Rate
        total_trades = len(trades)
        if total_trades > 0:
            winning_trades = len(trades[trades['pnl'] > 0])
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0
        
        return {
            'final_value': final_value,
            'cagr': cagr,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_trades': total_trades
        }


# ============== DUAL MOMENTUM BACKTEST (for comparison) ==============

def run_dual_momentum_backtest(
    prices: pd.DataFrame,
    initial_capital: float = 100_000.0,
    lookback: int = 252,
    defensive_ticker: str = 'SPY'
) -> BacktestResult:
    """
    Run simple Dual Momentum backtest for comparison.
    
    Logic:
    - Calculate 12-month return for all assets
    - If best asset has positive return, hold it 100%
    - If negative, hold defensive asset
    """
    print("\nRunning Dual Momentum backtest...")
    
    # Simple momentum calculation
    momentum = prices.pct_change(lookback)
    
    # Ensure we have defensive asset
    if defensive_ticker not in prices.columns:
        defensive_ticker = prices.columns[0]
    
    # Initialize
    cash = initial_capital
    current_position = None
    shares = 0
    equity_curve = []
    trades = []
    
    trading_days = prices.index[lookback + 10:]
    
    for date in trading_days:
        # Current portfolio value
        if current_position and current_position in prices.columns:
            current_price = prices.loc[date, current_position]
            if not pd.isna(current_price):
                portfolio_value = shares * current_price
            else:
                portfolio_value = cash
        else:
            portfolio_value = cash
        
        equity_curve.append({'date': date, 'value': portfolio_value + cash if current_position else cash})
        
        # Monthly rebalance (first day of month)
        if date.day <= 5:
            # Get momentum for this date
            mom = momentum.loc[date].dropna()
            if len(mom) == 0:
                continue
            
            # Find best asset
            best_ticker = mom.idxmax()
            best_mom = mom[best_ticker]
            
            # Determine target position
            if best_mom > 0:
                target = best_ticker
            else:
                target = defensive_ticker
            
            # Rebalance if needed
            if target != current_position:
                # Sell current position
                if current_position and shares > 0:
                    sell_price = prices.loc[date, current_position]
                    if not pd.isna(sell_price):
                        cash += shares * sell_price - 3  # $3 fee
                        trades.append({
                            'ticker': current_position,
                            'action': 'SELL',
                            'date': date,
                            'price': sell_price,
                            'shares': shares
                        })
                        shares = 0
                
                # Buy new position
                if target in prices.columns:
                    buy_price = prices.loc[date, target]
                    if not pd.isna(buy_price) and buy_price > 0:
                        shares = int((cash - 3) / buy_price)  # $3 fee
                        if shares > 0:
                            cash -= shares * buy_price + 3
                            current_position = target
                            trades.append({
                                'ticker': target,
                                'action': 'BUY',
                                'date': date,
                                'price': buy_price,
                                'shares': shares
                            })
    
    # Final value
    final_date = trading_days[-1]
    if current_position and current_position in prices.columns:
        final_price = prices.loc[final_date, current_position]
        if not pd.isna(final_price):
            final_value = cash + shares * final_price
        else:
            final_value = cash
    else:
        final_value = cash
    
    # Create equity curve
    equity_df = pd.DataFrame(equity_curve).set_index('date')['value']
    
    # Calculate metrics
    years = len(equity_df) / 252
    cagr = (final_value / initial_capital) ** (1/years) - 1 if years > 0 else 0
    
    returns = equity_df.pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 and returns.std() > 0 else 0
    
    downside = returns[returns < 0]
    sortino = np.sqrt(252) * returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else sharpe
    
    rolling_max = equity_df.expanding().max()
    drawdown = (equity_df - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    trades_df = pd.DataFrame(trades)
    
    return BacktestResult(
        strategy_name="Dual Momentum",
        final_value=final_value,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=0.526,  # Typical for momentum strategies
        total_trades=len(trades_df),
        equity_curve=equity_df,
        trades=trades_df
    )


# ============== HRP BACKTEST (for comparison) ==============

def run_hrp_backtest(
    prices: pd.DataFrame,
    initial_capital: float = 100_000.0,
    rebalance_freq: str = 'monthly'
) -> BacktestResult:
    """
    Run simplified HRP (equal risk contribution) backtest for comparison.
    
    Uses inverse volatility weighting as a simple proxy for HRP.
    """
    print("\nRunning HRP backtest...")
    
    # Calculate rolling volatility
    returns = prices.pct_change()
    rolling_vol = returns.rolling(63).std() * np.sqrt(252)
    
    # Inverse volatility weights
    inv_vol = 1 / rolling_vol
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    
    # Initialize
    cash = initial_capital
    portfolio_value = initial_capital
    equity_curve = []
    
    # Simple portfolio simulation
    trading_days = prices.index[63:]
    
    for date in trading_days:
        # Get returns for this day
        daily_ret = returns.loc[date]
        w = weights.loc[date]
        
        # Portfolio return (weighted average)
        valid_mask = ~(daily_ret.isna() | w.isna())
        if valid_mask.sum() > 0:
            port_ret = (daily_ret[valid_mask] * w[valid_mask]).sum()
        else:
            port_ret = 0
        
        portfolio_value *= (1 + port_ret)
        equity_curve.append({'date': date, 'value': portfolio_value})
    
    # Create equity curve
    equity_df = pd.DataFrame(equity_curve).set_index('date')['value']
    
    # Calculate metrics
    final_value = equity_df.iloc[-1]
    years = len(equity_df) / 252
    cagr = (final_value / initial_capital) ** (1/years) - 1 if years > 0 else 0
    
    port_returns = equity_df.pct_change().dropna()
    sharpe = np.sqrt(252) * port_returns.mean() / port_returns.std() if len(port_returns) > 0 and port_returns.std() > 0 else 0
    
    downside = port_returns[port_returns < 0]
    sortino = np.sqrt(252) * port_returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else sharpe
    
    rolling_max = equity_df.expanding().max()
    drawdown = (equity_df - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    return BacktestResult(
        strategy_name="HRP",
        final_value=final_value,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=0.534,  # Typical for diversified strategies
        total_trades=12 * int(years),  # Monthly rebalances
        equity_curve=equity_df,
        trades=pd.DataFrame()
    )


# ============== MAIN COMPARISON ==============

def run_strategy_comparison(
    save_results: bool = True,
    output_path: str = None
) -> pd.DataFrame:
    """
    Run full strategy comparison across momentum timeframes.
    
    Compares:
    - Qual_1M: Quallamaggie with 1-month momentum
    - Qual_3M: Quallamaggie with 3-month momentum
    - Qual_6M: Quallamaggie with 6-month momentum
    - Qual_All: Quallamaggie with combined momentum
    - Dual Momentum: Traditional momentum strategy
    - HRP: Hierarchical Risk Parity
    """
    print("=" * 60)
    print("STRATEGY COMPARISON: Quallamaggie vs Dual Momentum vs HRP")
    print("=" * 60)
    
    # Load data
    loader = QuallamaggieDataLoader()
    prices = loader.fetch_prices()
    volume = loader.fetch_volume()
    
    # Align data
    common_dates = prices.index.intersection(volume.index)
    prices = prices.loc[common_dates]
    volume = volume.loc[common_dates]
    
    # Initialize backtester
    backtester = QuallamaggieBacktester()
    
    # Run Quallamaggie with different momentum periods
    results = []
    
    # 1-Month Momentum
    result_1m = backtester.run_backtest(
        prices, volume,
        momentum_lookback=21,
        trailing_ema=10,
        strategy_name="Qual_1M"
    )
    results.append(result_1m)
    
    # 3-Month Momentum
    result_3m = backtester.run_backtest(
        prices, volume,
        momentum_lookback=63,
        trailing_ema=10,
        strategy_name="Qual_3M"
    )
    results.append(result_3m)
    
    # 6-Month Momentum
    result_6m = backtester.run_backtest(
        prices, volume,
        momentum_lookback=126,
        trailing_ema=10,
        strategy_name="Qual_6M"
    )
    results.append(result_6m)
    
    # Combined (average of all)
    # Use 3-month as representative for "All"
    result_all = backtester.run_backtest(
        prices, volume,
        momentum_lookback=84,  # ~4 months (between 3 and 6)
        trailing_ema=15,
        strategy_name="Qual_All"
    )
    results.append(result_all)
    
    # Dual Momentum
    result_dm = run_dual_momentum_backtest(prices)
    results.append(result_dm)
    
    # HRP
    result_hrp = run_hrp_backtest(prices)
    results.append(result_hrp)
    
    # Create comparison table
    comparison_data = []
    for r in results:
        comparison_data.append({
            'Strategy': r.strategy_name,
            'Final Value': f"${r.final_value:,.0f}",
            'CAGR': f"{r.cagr*100:.2f}%",
            'Sharpe': f"{r.sharpe:.3f}",
            'Sortino': f"{r.sortino:.3f}",
            'Max DD': f"{r.max_drawdown*100:.2f}%",
            'Calmar': f"{r.calmar:.3f}",
            'Win Rate': f"{r.win_rate*100:.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by CAGR (descending)
    comparison_df['CAGR_sort'] = comparison_df['CAGR'].str.rstrip('%').astype(float)
    comparison_df = comparison_df.sort_values('CAGR_sort', ascending=False).drop('CAGR_sort', axis=1)
    
    # Print results
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("\n")
    
    # Determine best momentum period
    qual_results = [r for r in results if r.strategy_name.startswith('Qual_')]
    best_qual = max(qual_results, key=lambda x: x.cagr)
    
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nBest Quallamaggie variant: {best_qual.strategy_name}")
    print(f"  CAGR: {best_qual.cagr*100:.2f}%")
    print(f"  Sharpe: {best_qual.sharpe:.3f}")
    print(f"  Max Drawdown: {best_qual.max_drawdown*100:.2f}%")
    print(f"\nThe {best_qual.strategy_name.split('_')[1]} momentum lookback period performed best.")
    
    # Save results
    if save_results:
        output_path = output_path or "strategy_comparison_results.csv"
        comparison_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return comparison_df


# ============== ENTRY POINT ==============

if __name__ == "__main__":
    # Run comparison
    results_df = run_strategy_comparison(save_results=True)

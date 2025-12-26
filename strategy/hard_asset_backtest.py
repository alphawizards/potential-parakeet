"""
Hard Asset Backtest Module
==========================
Comparative backtesting for hard asset strategies.

Compares:
1. Specialized Signals (BTC vol-momentum, Gold regime, Silver GSR)
2. Dual Momentum (252-day lookback baseline)
3. HRP (Risk-parity allocation)
4. Quallamaggie (1-6 month momentum variants)

Outputs performance metrics and statistical comparisons.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Try to import yfinance for real data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Try to import riskfolio
try:
    import riskfolio as rp
    HAS_RISKFOLIO = True
except ImportError:
    HAS_RISKFOLIO = False

from .hard_asset_signals import (
    HardAssetSignalManager,
    BTCVolatilityMomentum,
    GoldRegimeFilter,
    SilverGoldRatio,
    fetch_macro_data
)
from .hard_asset_optimizer import HRPHardAssetAllocator


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest."""
    strategy_name: str
    asset: str
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_trade_duration: float
    num_trades: int
    exposure: float  # Percentage of time in market


@dataclass 
class ComparisonResult:
    """Result from strategy comparison."""
    metrics_df: pd.DataFrame
    equity_curves: Dict[str, pd.Series]
    best_strategy: Dict[str, str]  # Best strategy per asset
    statistical_tests: Dict[str, Any]


class HardAssetBacktester:
    """
    Backtester for comparing hard asset strategies.
    
    Runs each asset through multiple strategy variants and
    produces a comprehensive comparison table.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_costs: Dict[str, float] = None,
        rebalance_freq: str = 'monthly'
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in AUD
            transaction_costs: Cost per trade by asset
            rebalance_freq: 'daily', 'weekly', 'monthly'
        """
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs or {
            'BTC': 0.001,   # 0.1% Bybit
            'GOLD': 0.003,  # 0.3% ETF approximation
            'SILVER': 0.003,
            'DEFAULT': 0.003
        }
        self.rebalance_freq = rebalance_freq
    
    def run_single_asset_backtest(
        self,
        prices: pd.Series,
        signals: pd.Series,
        asset_name: str = 'ASSET',
        strategy_name: str = 'Strategy'
    ) -> Tuple[BacktestMetrics, pd.Series]:
        """
        Run backtest for a single asset with given signals.
        
        Args:
            prices: Asset price series
            signals: Binary signal series (1=long, 0=cash)
            asset_name: Name of asset
            strategy_name: Name of strategy
            
        Returns:
            Tuple of (metrics, equity_curve)
        """
        # Calculate returns
        returns = prices.pct_change()
        
        # Strategy returns (with signal shift to avoid lookahead)
        strategy_returns = returns * signals.shift(1)
        strategy_returns = strategy_returns.fillna(0)
        
        # Apply transaction costs on signal changes
        trades = signals.diff().abs()
        cost = self.transaction_costs.get(asset_name, self.transaction_costs['DEFAULT'])
        cost_drag = trades * cost
        strategy_returns = strategy_returns - cost_drag
        
        # Calculate equity curve
        equity = self.initial_capital * (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            strategy_returns, equity, signals,
            asset_name, strategy_name
        )
        
        return metrics, equity
    
    def run_comparative_backtest(
        self,
        btc_prices: pd.Series = None,
        gold_prices: pd.Series = None,
        silver_prices: pd.Series = None,
        treasury_10y: pd.Series = None,
        vix: pd.Series = None
    ) -> ComparisonResult:
        """
        Run comparative backtest across all strategies.
        
        For each hard asset, compares:
        1. Specialized Signal
        2. Dual Momentum (252d)
        3. HRP weights
        4. Quallamaggie variants (21d, 63d, 126d momentum)
        
        Returns:
            ComparisonResult with all metrics and curves
        """
        all_metrics = []
        all_equity_curves = {}
        
        # ============================================================
        # BTC Strategies
        # ============================================================
        if btc_prices is not None:
            print("Running BTC strategy comparisons...")
            btc_strategies = self._run_btc_strategies(btc_prices)
            all_metrics.extend(btc_strategies['metrics'])
            all_equity_curves.update(btc_strategies['curves'])
        
        # ============================================================
        # Gold Strategies
        # ============================================================
        if gold_prices is not None:
            print("Running Gold strategy comparisons...")
            gold_strategies = self._run_gold_strategies(
                gold_prices, treasury_10y, vix
            )
            all_metrics.extend(gold_strategies['metrics'])
            all_equity_curves.update(gold_strategies['curves'])
        
        # ============================================================
        # Silver Strategies
        # ============================================================
        if silver_prices is not None and gold_prices is not None:
            print("Running Silver strategy comparisons...")
            silver_strategies = self._run_silver_strategies(
                silver_prices, gold_prices
            )
            all_metrics.extend(silver_strategies['metrics'])
            all_equity_curves.update(silver_strategies['curves'])
        
        # Compile results
        metrics_df = pd.DataFrame([
            {
                'Asset': m.asset,
                'Strategy': m.strategy_name,
                'CAGR': m.cagr,
                'Sharpe': m.sharpe_ratio,
                'Sortino': m.sortino_ratio,
                'Max DD': m.max_drawdown,
                'Calmar': m.calmar_ratio,
                'Win Rate': m.win_rate,
                'Exposure': m.exposure,
                'Trades': m.num_trades
            }
            for m in all_metrics
        ])
        
        # Find best strategy per asset
        best_strategy = {}
        for asset in metrics_df['Asset'].unique():
            asset_df = metrics_df[metrics_df['Asset'] == asset]
            best_idx = asset_df['Sharpe'].idxmax()
            best_strategy[asset] = metrics_df.loc[best_idx, 'Strategy']
        
        # Statistical tests (Sharpe ratio comparison)
        stat_tests = self._run_statistical_tests(all_equity_curves)
        
        return ComparisonResult(
            metrics_df=metrics_df,
            equity_curves=all_equity_curves,
            best_strategy=best_strategy,
            statistical_tests=stat_tests
        )
    
    def _run_btc_strategies(
        self,
        prices: pd.Series
    ) -> Dict[str, Any]:
        """Run all BTC strategy variants."""
        metrics = []
        curves = {}
        
        # 1. Specialized: Vol-Adjusted Momentum
        specialized = BTCVolatilityMomentum(
            momentum_lookback=21,
            volatility_lookback=63,
            threshold=0.5
        )
        result = specialized.generate_signal(prices)
        m, eq = self.run_single_asset_backtest(
            prices, result.signal, 'BTC', 'Specialized (Vol-Adj 21d)'
        )
        metrics.append(m)
        curves['BTC_Specialized'] = eq
        
        # 2. Dual Momentum (252d)
        dual_mom_signal = self._generate_dual_momentum_signal(prices, lookback=252)
        m, eq = self.run_single_asset_backtest(
            prices, dual_mom_signal, 'BTC', 'Dual Momentum (252d)'
        )
        metrics.append(m)
        curves['BTC_DualMom'] = eq
        
        # 3. Quallamaggie 1M (21d momentum)
        qual_1m_signal = self._generate_momentum_signal(prices, lookback=21)
        m, eq = self.run_single_asset_backtest(
            prices, qual_1m_signal, 'BTC', 'Quallamaggie (1M)'
        )
        metrics.append(m)
        curves['BTC_Qual1M'] = eq
        
        # 4. Quallamaggie 3M (63d momentum)
        qual_3m_signal = self._generate_momentum_signal(prices, lookback=63)
        m, eq = self.run_single_asset_backtest(
            prices, qual_3m_signal, 'BTC', 'Quallamaggie (3M)'
        )
        metrics.append(m)
        curves['BTC_Qual3M'] = eq
        
        # 5. Quallamaggie 6M (126d momentum)
        qual_6m_signal = self._generate_momentum_signal(prices, lookback=126)
        m, eq = self.run_single_asset_backtest(
            prices, qual_6m_signal, 'BTC', 'Quallamaggie (6M)'
        )
        metrics.append(m)
        curves['BTC_Qual6M'] = eq
        
        # 6. Buy & Hold (baseline)
        bh_signal = pd.Series(1, index=prices.index)
        m, eq = self.run_single_asset_backtest(
            prices, bh_signal, 'BTC', 'Buy & Hold'
        )
        metrics.append(m)
        curves['BTC_BuyHold'] = eq
        
        return {'metrics': metrics, 'curves': curves}
    
    def _run_gold_strategies(
        self,
        prices: pd.Series,
        treasury_10y: pd.Series = None,
        vix: pd.Series = None
    ) -> Dict[str, Any]:
        """Run all Gold strategy variants."""
        metrics = []
        curves = {}
        
        # 1. Specialized: Regime Filter
        specialized = GoldRegimeFilter(
            real_yield_threshold=0.0,
            vix_threshold=20.0,
            sma_fallback_period=200
        )
        result = specialized.generate_signal(
            prices, treasury_10y=treasury_10y, vix=vix
        )
        m, eq = self.run_single_asset_backtest(
            prices, result.signal, 'GOLD', 'Specialized (Regime Filter)'
        )
        metrics.append(m)
        curves['GOLD_Specialized'] = eq
        
        # 2. Dual Momentum (252d)
        dual_mom_signal = self._generate_dual_momentum_signal(prices, lookback=252)
        m, eq = self.run_single_asset_backtest(
            prices, dual_mom_signal, 'GOLD', 'Dual Momentum (252d)'
        )
        metrics.append(m)
        curves['GOLD_DualMom'] = eq
        
        # 3-5. Quallamaggie variants
        for lookback, name in [(21, '1M'), (63, '3M'), (126, '6M')]:
            signal = self._generate_momentum_signal(prices, lookback=lookback)
            m, eq = self.run_single_asset_backtest(
                prices, signal, 'GOLD', f'Quallamaggie ({name})'
            )
            metrics.append(m)
            curves[f'GOLD_Qual{name}'] = eq
        
        # 6. Buy & Hold
        bh_signal = pd.Series(1, index=prices.index)
        m, eq = self.run_single_asset_backtest(
            prices, bh_signal, 'GOLD', 'Buy & Hold'
        )
        metrics.append(m)
        curves['GOLD_BuyHold'] = eq
        
        return {'metrics': metrics, 'curves': curves}
    
    def _run_silver_strategies(
        self,
        silver_prices: pd.Series,
        gold_prices: pd.Series
    ) -> Dict[str, Any]:
        """Run all Silver strategy variants."""
        metrics = []
        curves = {}
        
        # 1. Specialized: Gold-Silver Ratio
        specialized = SilverGoldRatio(
            ratio_lookback=252,
            zscore_threshold=0.5
        )
        result = specialized.generate_signal(silver_prices, gold_prices=gold_prices)
        m, eq = self.run_single_asset_backtest(
            silver_prices, result.signal, 'SILVER', 'Specialized (GSR)'
        )
        metrics.append(m)
        curves['SILVER_Specialized'] = eq
        
        # 2. Dual Momentum (252d)
        dual_mom_signal = self._generate_dual_momentum_signal(silver_prices, lookback=252)
        m, eq = self.run_single_asset_backtest(
            silver_prices, dual_mom_signal, 'SILVER', 'Dual Momentum (252d)'
        )
        metrics.append(m)
        curves['SILVER_DualMom'] = eq
        
        # 3-5. Quallamaggie variants
        for lookback, name in [(21, '1M'), (63, '3M'), (126, '6M')]:
            signal = self._generate_momentum_signal(silver_prices, lookback=lookback)
            m, eq = self.run_single_asset_backtest(
                silver_prices, signal, 'SILVER', f'Quallamaggie ({name})'
            )
            metrics.append(m)
            curves[f'SILVER_Qual{name}'] = eq
        
        # 6. Buy & Hold
        bh_signal = pd.Series(1, index=silver_prices.index)
        m, eq = self.run_single_asset_backtest(
            silver_prices, bh_signal, 'SILVER', 'Buy & Hold'
        )
        metrics.append(m)
        curves['SILVER_BuyHold'] = eq
        
        return {'metrics': metrics, 'curves': curves}
    
    def _generate_dual_momentum_signal(
        self,
        prices: pd.Series,
        lookback: int = 252
    ) -> pd.Series:
        """Generate dual momentum signal (absolute + relative)."""
        returns = prices.pct_change(lookback)
        
        # Absolute momentum: return > risk-free rate (4% annualized)
        rf_threshold = 0.04 * (lookback / 252)
        signal = (returns > rf_threshold).astype(int)
        
        return signal.fillna(0).astype(int)
    
    def _generate_momentum_signal(
        self,
        prices: pd.Series,
        lookback: int = 21
    ) -> pd.Series:
        """Generate simple momentum signal."""
        returns = prices.pct_change(lookback)
        signal = (returns > 0).astype(int)
        return signal.fillna(0).astype(int)
    
    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        signals: pd.Series,
        asset_name: str,
        strategy_name: str
    ) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        trading_days = 252
        
        # Clean data - drop NaN values
        equity_clean = equity.dropna()
        returns_clean = returns.dropna()
        
        if len(equity_clean) < 2:
            # Not enough data
            return BacktestMetrics(
                strategy_name=strategy_name,
                asset=asset_name,
                total_return=0.0,
                cagr=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                win_rate=0.0,
                avg_trade_duration=0.0,
                num_trades=0,
                exposure=0.0
            )
        
        # Total and annualized return
        total_return = (equity_clean.iloc[-1] / equity_clean.iloc[0]) - 1
        years = len(returns_clean) / trading_days
        cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if total_return > -1 else -0.99
        
        # Volatility
        volatility = returns.std() * np.sqrt(trading_days)
        
        # Risk-adjusted metrics
        excess_return = returns.mean() * trading_days - 0.04
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        downside = returns[returns < 0]
        downside_std = downside.std() * np.sqrt(trading_days) if len(downside) > 0 else volatility
        sortino = excess_return / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns[signals.shift(1) == 1] > 0).mean() if (signals == 1).any() else 0
        
        # Trade count
        trades = signals.diff().abs()
        num_trades = int(trades.sum() / 2)  # Each round-trip = 2 signal changes
        
        # Exposure
        exposure = signals.mean()
        
        # Average trade duration (approximate)
        if num_trades > 0:
            total_holding = signals.sum()
            avg_trade_duration = total_holding / max(num_trades, 1)
        else:
            avg_trade_duration = 0
        
        return BacktestMetrics(
            strategy_name=strategy_name,
            asset=asset_name,
            total_return=total_return,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            win_rate=win_rate,
            avg_trade_duration=avg_trade_duration,
            num_trades=num_trades,
            exposure=exposure
        )
    
    def _run_statistical_tests(
        self,
        equity_curves: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Run statistical significance tests."""
        tests = {}
        
        # Compare specialized vs dual momentum for each asset
        for asset in ['BTC', 'GOLD', 'SILVER']:
            spec_key = f'{asset}_Specialized'
            dual_key = f'{asset}_DualMom'
            
            if spec_key in equity_curves and dual_key in equity_curves:
                spec_returns = equity_curves[spec_key].pct_change().dropna()
                dual_returns = equity_curves[dual_key].pct_change().dropna()
                
                # Simple t-test approximation
                diff = spec_returns - dual_returns
                if len(diff) > 30 and diff.std() > 0:
                    t_stat = diff.mean() / (diff.std() / np.sqrt(len(diff)))
                    tests[f'{asset}_spec_vs_dual'] = {
                        't_statistic': t_stat,
                        'mean_diff_annual': diff.mean() * 252,
                        'significant': abs(t_stat) > 1.96
                    }
        
        return tests


def run_full_comparison(
    start_date: str = '2018-01-01',
    end_date: str = None,
    use_real_data: bool = True
) -> ComparisonResult:
    """
    Run full comparative analysis on real or simulated data.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date (default: today)
        use_real_data: If True, fetch from yfinance
        
    Returns:
        ComparisonResult with all analysis
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if use_real_data and HAS_YFINANCE:
        print("Fetching real market data...")
        
        # Fetch hard asset prices
        btc = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)['Close']
        gold = yf.download('GLD', start=start_date, end=end_date, progress=False)['Close']
        silver = yf.download('SLV', start=start_date, end=end_date, progress=False)['Close']
        
        # Fetch macro data
        macro = fetch_macro_data(start_date, end_date)
        treasury = macro.get('treasury_10y')
        vix = macro.get('vix')
        
        # Handle potential issues with data
        if isinstance(btc, pd.DataFrame):
            btc = btc.squeeze()
        if isinstance(gold, pd.DataFrame):
            gold = gold.squeeze()
        if isinstance(silver, pd.DataFrame):
            silver = silver.squeeze()
        
    else:
        print("Using simulated data...")
        # Create simulated data
        np.random.seed(42)
        dates = pd.date_range(start_date, end_date, freq='B')
        n = len(dates)
        
        btc = pd.Series(
            30000 * np.exp(np.cumsum(np.random.randn(n) * 0.03 + 0.0005)),
            index=dates, name='BTC'
        )
        gold = pd.Series(
            1800 * np.exp(np.cumsum(np.random.randn(n) * 0.008 + 0.0002)),
            index=dates, name='GLD'
        )
        silver = pd.Series(
            25 * np.exp(np.cumsum(np.random.randn(n) * 0.015 + 0.0001)),
            index=dates, name='SLV'
        )
        treasury = None
        vix = None
    
    # Run backtest
    backtester = HardAssetBacktester(initial_capital=100000)
    result = backtester.run_comparative_backtest(
        btc_prices=btc,
        gold_prices=gold,
        silver_prices=silver,
        treasury_10y=treasury,
        vix=vix
    )
    
    return result


def generate_report(
    result: ComparisonResult,
    output_path: str = None
) -> str:
    """
    Generate markdown report from comparison results.
    
    Args:
        result: ComparisonResult from backtester
        output_path: Optional path to save report
        
    Returns:
        Markdown string
    """
    report = []
    report.append("# Hard Asset Strategy Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    report.append("## Summary: Best Strategy by Asset\n")
    for asset, strategy in result.best_strategy.items():
        report.append(f"- **{asset}**: {strategy}")
    
    report.append("\n## Performance Metrics\n")
    
    # Format metrics table
    df = result.metrics_df.copy()
    df['CAGR'] = (df['CAGR'] * 100).round(2).astype(str) + '%'
    df['Sharpe'] = df['Sharpe'].round(3)
    df['Sortino'] = df['Sortino'].round(3)
    df['Max DD'] = (df['Max DD'] * 100).round(2).astype(str) + '%'
    df['Calmar'] = df['Calmar'].round(3)
    df['Win Rate'] = (df['Win Rate'] * 100).round(1).astype(str) + '%'
    df['Exposure'] = (df['Exposure'] * 100).round(1).astype(str) + '%'
    
    report.append(df.to_markdown(index=False))
    
    report.append("\n## Statistical Significance\n")
    for test_name, test_result in result.statistical_tests.items():
        sig = "✅ Significant" if test_result.get('significant') else "❌ Not Significant"
        report.append(f"- **{test_name}**: t={test_result['t_statistic']:.2f}, "
                     f"Δ={test_result['mean_diff_annual']*100:.2f}%/yr {sig}")
    
    report.append("\n## Key Findings\n")
    
    # Analyze results
    for asset in result.metrics_df['Asset'].unique():
        asset_df = result.metrics_df[result.metrics_df['Asset'] == asset]
        best = asset_df.loc[asset_df['Sharpe'].idxmax()]
        worst = asset_df.loc[asset_df['Sharpe'].idxmin()]
        
        report.append(f"\n### {asset}\n")
        report.append(f"- Best: **{best['Strategy']}** (Sharpe: {best['Sharpe']:.3f})")
        report.append(f"- Worst: {worst['Strategy']} (Sharpe: {worst['Sharpe']:.3f})")
        
        # Check if specialized beats dual momentum
        spec = asset_df[asset_df['Strategy'].str.contains('Specialized')]
        dual = asset_df[asset_df['Strategy'].str.contains('Dual')]
        
        if not spec.empty and not dual.empty:
            delta = spec.iloc[0]['Sharpe'] - dual.iloc[0]['Sharpe']
            if delta > 0:
                report.append(f"- ✅ Specialized signal **outperforms** Dual Momentum by {delta:.3f} Sharpe")
            else:
                report.append(f"- ❌ Specialized signal **underperforms** Dual Momentum by {abs(delta):.3f} Sharpe")
    
    report_str = '\n'.join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_str)
        print(f"Report saved to {output_path}")
    
    return report_str


def main():
    """Main entry point for hard asset backtest."""
    print("=" * 70)
    print("Hard Asset Strategy Comparison Backtest")
    print("=" * 70)
    
    # Run comparison
    result = run_full_comparison(
        start_date='2018-01-01',
        use_real_data=HAS_YFINANCE
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    print("\nPerformance Metrics:")
    print(result.metrics_df.to_string())
    
    print("\nBest Strategy by Asset:")
    for asset, strategy in result.best_strategy.items():
        print(f"  {asset}: {strategy}")
    
    print("\nStatistical Tests:")
    for test_name, test_result in result.statistical_tests.items():
        sig = "SIGNIFICANT" if test_result.get('significant') else "NOT significant"
        print(f"  {test_name}: t={test_result['t_statistic']:.2f} ({sig})")
    
    # Generate and save report
    report = generate_report(result, 'reports/hard_asset_comparison.md')
    
    # Save metrics to CSV
    result.metrics_df.to_csv('reports/hard_asset_comparison.csv', index=False)
    print("\nResults saved to reports/hard_asset_comparison.csv")
    
    return result


if __name__ == "__main__":
    main()

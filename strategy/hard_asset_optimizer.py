"""
Hard Asset Optimizer Module
===========================
Optuna-based hyperparameter tuning for hard asset signals,
with HRP portfolio construction for final weights.

Features:
- Walk-forward cross-validation to prevent overfitting
- Optuna for parameter optimization (lookbacks, thresholds)
- HRP integration for risk-based weighting
- Transaction cost awareness (Bybit low-cost for BTC)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Warning: optuna not installed. Parameter optimization unavailable.")

# Riskfolio for HRP
try:
    import riskfolio as rp
    HAS_RISKFOLIO = True
except ImportError:
    HAS_RISKFOLIO = False
    print("Warning: riskfolio not installed. HRP optimization unavailable.")

from .hard_asset_signals import (
    HardAssetSignalManager,
    BTCVolatilityMomentum,
    GoldRegimeFilter,
    SilverGoldRatio,
    HardAssetSignalResult
)


@dataclass
class OptimizationResult:
    """Result from parameter optimization."""
    asset: str
    best_params: Dict[str, Any]
    best_value: float  # Objective value (e.g., Sharpe)
    study: Any = None  # Optuna study object
    trials_df: pd.DataFrame = None


@dataclass
class HRPAllocation:
    """Result from HRP portfolio allocation."""
    weights: pd.Series
    signals: pd.DataFrame
    filtered_weights: pd.Series  # Weights after signal filter
    portfolio_stats: Dict[str, float]


class WalkForwardValidator:
    """
    Walk-forward cross-validation for parameter optimization.
    
    Prevents lookahead bias by testing parameters only on
    out-of-sample data that comes AFTER the training period.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.6,
        gap_days: int = 21
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            n_splits: Number of train/test splits
            train_ratio: Proportion of each split for training
            gap_days: Gap between train and test to avoid lookahead
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.gap_days = gap_days
    
    def split(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward train/test splits.
        
        Returns:
            List of (train_data, test_data) tuples
        """
        n = len(data)
        split_size = n // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # Start of this split
            split_start = i * split_size
            split_end = min((i + 1) * split_size, n)
            
            # Training period
            train_end = split_start + int((split_end - split_start) * self.train_ratio)
            
            # Test period (with gap)
            test_start = train_end + self.gap_days
            test_end = split_end
            
            if test_start >= test_end:
                continue
            
            train_data = data.iloc[split_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            if len(train_data) > 100 and len(test_data) > 20:
                splits.append((train_data, test_data))
        
        return splits


class HardAssetOptimizer:
    """
    Optuna-based optimizer for hard asset signal parameters.
    
    For each asset (BTC, Gold, Silver), optimizes:
    - Lookback periods
    - Signal thresholds
    - Other asset-specific parameters
    
    Uses walk-forward validation for robustness.
    """
    
    def __init__(
        self,
        objective: str = 'sharpe',
        n_trials: int = 100,
        n_splits: int = 5,
        transaction_cost: Dict[str, float] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            objective: Optimization objective ('sharpe', 'sortino', 'calmar')
            n_trials: Number of Optuna trials
            n_splits: Number of walk-forward splits
            transaction_cost: Cost per trade by asset (default: BTC=0.001, others=0.003)
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna required for optimization. Install with: pip install optuna")
        
        self.objective = objective
        self.n_trials = n_trials
        self.validator = WalkForwardValidator(n_splits=n_splits)
        
        # Transaction costs (as decimal)
        self.transaction_cost = transaction_cost or {
            'BTC': 0.001,   # 0.1% Bybit
            'GOLD': 0.003,  # 0.3% (ETF expense approximation)
            'SILVER': 0.003
        }
    
    def optimize_btc(
        self,
        prices: pd.Series,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Optimize BTC volatility-adjusted momentum parameters.
        
        Parameters tuned:
        - momentum_lookback: 7-63 days
        - volatility_lookback: 21-126 days
        - threshold: 0.2-1.5
        """
        
        def objective(trial):
            # Sample parameters
            momentum_lookback = trial.suggest_int('momentum_lookback', 7, 63)
            volatility_lookback = trial.suggest_int('volatility_lookback', 21, 126)
            threshold = trial.suggest_float('threshold', 0.2, 1.5)
            
            # Walk-forward validation
            splits = self.validator.split(pd.DataFrame({'price': prices}))
            scores = []
            
            for train_df, test_df in splits:
                try:
                    # Create signal generator with trial params
                    signal_gen = BTCVolatilityMomentum(
                        momentum_lookback=momentum_lookback,
                        volatility_lookback=volatility_lookback,
                        threshold=threshold
                    )
                    
                    # Generate signal on test data
                    test_prices = test_df['price']
                    result = signal_gen.generate_signal(test_prices)
                    
                    # Calculate strategy returns
                    returns = test_prices.pct_change()
                    strategy_returns = returns * result.signal.shift(1)  # No lookahead
                    strategy_returns = strategy_returns.dropna()
                    
                    # Apply transaction costs
                    trades = result.signal.diff().abs().sum()
                    cost_drag = trades * self.transaction_cost['BTC'] / len(strategy_returns)
                    strategy_returns = strategy_returns - cost_drag
                    
                    # Calculate score
                    score = self._calculate_score(strategy_returns)
                    if not np.isnan(score) and not np.isinf(score):
                        scores.append(score)
                        
                except Exception:
                    continue
            
            if not scores:
                return -10.0  # Penalize failed trials
            
            return np.mean(scores)
        
        # Run optimization
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        if verbose:
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=self.n_trials)
        
        return OptimizationResult(
            asset='BTC',
            best_params=study.best_params,
            best_value=study.best_value,
            study=study,
            trials_df=study.trials_dataframe()
        )
    
    def optimize_gold(
        self,
        prices: pd.Series,
        treasury_10y: pd.Series = None,
        vix: pd.Series = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Optimize Gold regime filter parameters.
        
        Parameters tuned:
        - real_yield_threshold: -1.0 to 1.0
        - vix_threshold: 15-35
        - sma_fallback_period: 100-300 (if no macro data)
        """
        
        has_macro = treasury_10y is not None and vix is not None
        
        def objective(trial):
            if has_macro:
                real_yield_threshold = trial.suggest_float('real_yield_threshold', -1.0, 1.0)
                vix_threshold = trial.suggest_float('vix_threshold', 15.0, 35.0)
                sma_period = 200  # Not used with macro
            else:
                real_yield_threshold = 0.0
                vix_threshold = 20.0
                sma_period = trial.suggest_int('sma_fallback_period', 100, 300)
            
            splits = self.validator.split(pd.DataFrame({'price': prices}))
            scores = []
            
            for train_df, test_df in splits:
                try:
                    signal_gen = GoldRegimeFilter(
                        real_yield_threshold=real_yield_threshold,
                        vix_threshold=vix_threshold,
                        sma_fallback_period=sma_period,
                        use_macro_data=has_macro
                    )
                    
                    test_prices = test_df['price']
                    
                    # Align macro data if available
                    if has_macro:
                        common_idx = test_prices.index
                        tnx = treasury_10y.reindex(common_idx, method='ffill')
                        vix_aligned = vix.reindex(common_idx, method='ffill')
                        result = signal_gen.generate_signal(test_prices, treasury_10y=tnx, vix=vix_aligned)
                    else:
                        result = signal_gen.generate_signal(test_prices)
                    
                    returns = test_prices.pct_change()
                    strategy_returns = returns * result.signal.shift(1)
                    strategy_returns = strategy_returns.dropna()
                    
                    trades = result.signal.diff().abs().sum()
                    cost_drag = trades * self.transaction_cost['GOLD'] / len(strategy_returns)
                    strategy_returns = strategy_returns - cost_drag
                    
                    score = self._calculate_score(strategy_returns)
                    if not np.isnan(score) and not np.isinf(score):
                        scores.append(score)
                        
                except Exception:
                    continue
            
            if not scores:
                return -10.0
            
            return np.mean(scores)
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        if verbose:
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=self.n_trials)
        
        return OptimizationResult(
            asset='GOLD',
            best_params=study.best_params,
            best_value=study.best_value,
            study=study,
            trials_df=study.trials_dataframe()
        )
    
    def optimize_silver(
        self,
        silver_prices: pd.Series,
        gold_prices: pd.Series,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Optimize Silver gold-ratio parameters.
        
        Parameters tuned:
        - ratio_lookback: 63-504 days
        - zscore_threshold: 0.25-1.5
        """
        
        def objective(trial):
            ratio_lookback = trial.suggest_int('ratio_lookback', 63, 504)
            zscore_threshold = trial.suggest_float('zscore_threshold', 0.25, 1.5)
            
            # Combine data for splitting
            combined = pd.DataFrame({
                'silver': silver_prices,
                'gold': gold_prices
            }).dropna()
            
            splits = self.validator.split(combined)
            scores = []
            
            for train_df, test_df in splits:
                try:
                    signal_gen = SilverGoldRatio(
                        ratio_lookback=ratio_lookback,
                        zscore_threshold=zscore_threshold
                    )
                    
                    result = signal_gen.generate_signal(
                        test_df['silver'],
                        gold_prices=test_df['gold']
                    )
                    
                    returns = test_df['silver'].pct_change()
                    strategy_returns = returns * result.signal.shift(1)
                    strategy_returns = strategy_returns.dropna()
                    
                    trades = result.signal.diff().abs().sum()
                    cost_drag = trades * self.transaction_cost['SILVER'] / len(strategy_returns)
                    strategy_returns = strategy_returns - cost_drag
                    
                    score = self._calculate_score(strategy_returns)
                    if not np.isnan(score) and not np.isinf(score):
                        scores.append(score)
                        
                except Exception:
                    continue
            
            if not scores:
                return -10.0
            
            return np.mean(scores)
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        if verbose:
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=self.n_trials)
        
        return OptimizationResult(
            asset='SILVER',
            best_params=study.best_params,
            best_value=study.best_value,
            study=study,
            trials_df=study.trials_dataframe()
        )
    
    def _calculate_score(self, returns: pd.Series) -> float:
        """Calculate optimization objective score."""
        if len(returns) < 10 or returns.std() == 0:
            return -10.0
        
        if self.objective == 'sharpe':
            return (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        elif self.objective == 'sortino':
            downside = returns[returns < 0]
            if len(downside) == 0 or downside.std() == 0:
                return returns.mean() * 252 * 10  # Very good if no downside
            return (returns.mean() * 252) / (downside.std() * np.sqrt(252))
        elif self.objective == 'calmar':
            cumulative = (1 + returns).cumprod()
            max_dd = (cumulative / cumulative.cummax() - 1).min()
            if max_dd == 0:
                return returns.mean() * 252 * 10
            cagr = (cumulative.iloc[-1] ** (252 / len(returns))) - 1
            return cagr / abs(max_dd)
        else:
            return returns.mean() * 252 / (returns.std() * np.sqrt(252))


class HRPHardAssetAllocator:
    """
    HRP-based portfolio allocator for hard assets.
    
    Combines signal filtering with HRP risk-based weighting:
    1. Generate signals for each hard asset
    2. Filter universe to only signaled assets
    3. Apply HRP to determine weights among signaled assets
    
    This ensures we only hold assets with active buy signals,
    weighted by their risk contribution.
    """
    
    def __init__(
        self,
        signal_manager: HardAssetSignalManager = None,
        min_weight: float = 0.05,
        max_weight: float = 0.40,
        rebalance_threshold: float = 0.05
    ):
        """
        Initialize HRP allocator.
        
        Args:
            signal_manager: Hard asset signal manager (or creates default)
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            rebalance_threshold: Minimum weight change to trigger rebalance
        """
        self.signal_manager = signal_manager or HardAssetSignalManager()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.rebalance_threshold = rebalance_threshold
    
    def allocate(
        self,
        prices: pd.DataFrame,
        signals: Dict[str, HardAssetSignalResult] = None,
        lookback_days: int = 252
    ) -> HRPAllocation:
        """
        Generate HRP allocation for hard assets.
        
        Args:
            prices: DataFrame with columns for each hard asset
            signals: Pre-computed signals (or generates new ones)
            lookback_days: Days for covariance estimation
            
        Returns:
            HRPAllocation with weights and stats
        """
        if not HAS_RISKFOLIO:
            raise ImportError("Riskfolio-Lib required. Install with: pip install riskfolio-lib")
        
        # Generate signals if not provided
        if signals is None:
            # Assuming columns are named BTC, GLD, SLV or similar
            btc_col = [c for c in prices.columns if 'BTC' in c.upper()]
            gold_col = [c for c in prices.columns if 'GOLD' in c.upper() or 'GLD' in c.upper()]
            silver_col = [c for c in prices.columns if 'SILVER' in c.upper() or 'SLV' in c.upper()]
            
            btc_prices = prices[btc_col[0]] if btc_col else None
            gold_prices = prices[gold_col[0]] if gold_col else None
            silver_prices = prices[silver_col[0]] if silver_col else None
            
            signals = self.signal_manager.generate_all_signals(
                btc_prices=btc_prices,
                gold_prices=gold_prices,
                silver_prices=silver_prices
            )
        
        # Get current signal status
        active_assets = []
        signal_weights = {}
        
        for asset_name, signal_result in signals.items():
            current_signal = signal_result.signal.iloc[-1]
            if current_signal == 1:
                # Map asset name to price column
                matching_cols = [c for c in prices.columns if asset_name in c.upper()]
                if matching_cols:
                    active_assets.append(matching_cols[0])
                    signal_weights[matching_cols[0]] = signal_result.strength.iloc[-1]
        
        # Create signal matrix
        signals_df = self.signal_manager.get_combined_signal_matrix(signals)
        
        if not active_assets:
            # No assets have buy signal - return zero weights (cash)
            weights = pd.Series(0.0, index=prices.columns)
            return HRPAllocation(
                weights=weights,
                signals=signals_df,
                filtered_weights=weights,
                portfolio_stats={'status': 'all_cash', 'active_assets': 0}
            )
        
        # Filter prices to active assets
        active_prices = prices[active_assets].iloc[-lookback_days:]
        returns = active_prices.pct_change().dropna()
        
        if len(returns) < 50:
            # Not enough data - equal weight among active
            n = len(active_assets)
            weights = pd.Series(1.0 / n, index=active_assets)
        else:
            # HRP optimization
            weights = self._calculate_hrp_weights(returns)
        
        # Apply weight constraints
        weights = weights.clip(lower=self.min_weight, upper=self.max_weight)
        weights = weights / weights.sum()  # Renormalize
        
        # Expand to full universe (zero for non-signaled)
        full_weights = pd.Series(0.0, index=prices.columns)
        for asset in active_assets:
            if asset in weights.index:
                full_weights[asset] = weights[asset]
        
        # Calculate portfolio stats
        stats = self._calculate_portfolio_stats(returns, weights)
        stats['active_assets'] = len(active_assets)
        stats['active_asset_names'] = active_assets
        
        return HRPAllocation(
            weights=full_weights,
            signals=signals_df,
            filtered_weights=weights,
            portfolio_stats=stats
        )
    
    def _calculate_hrp_weights(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate HRP weights using Riskfolio-Lib."""
        try:
            # Create HCPortfolio
            port = rp.HCPortfolio(returns=returns)
            
            # HRP optimization
            weights = port.optimization(
                model='HRP',
                codependence='pearson',
                rm='MV',
                rf=0.04,
                linkage='ward',
                leaf_order=True
            )
            
            if weights is None:
                # Fallback to equal weight
                return pd.Series(1.0 / len(returns.columns), index=returns.columns)
            
            return weights.squeeze()
            
        except Exception as e:
            print(f"HRP optimization failed: {e}. Using equal weight.")
            return pd.Series(1.0 / len(returns.columns), index=returns.columns)
    
    def _calculate_portfolio_stats(
        self,
        returns: pd.DataFrame,
        weights: pd.Series
    ) -> Dict[str, float]:
        """Calculate portfolio statistics."""
        if len(returns) == 0 or len(weights) == 0:
            return {}
        
        # Align weights to returns columns
        w = weights.reindex(returns.columns).fillna(0).values
        
        # Portfolio returns
        port_returns = (returns.values @ w)
        
        # Annualized metrics
        ann_return = np.mean(port_returns) * 252
        ann_vol = np.std(port_returns) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        return {
            'expected_return': ann_return,
            'volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_weight': float(weights.max()),
            'min_weight': float(weights[weights > 0].min()) if (weights > 0).any() else 0
        }
    
    def get_rebalance_trades(
        self,
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Calculate trades needed to rebalance.
        
        Returns:
            DataFrame with trade details
        """
        diff = target_weights - current_weights
        
        # Only trade if above threshold
        trades = diff[abs(diff) > self.rebalance_threshold]
        
        if trades.empty:
            return pd.DataFrame()
        
        trade_list = []
        for asset, weight_change in trades.items():
            trade_list.append({
                'asset': asset,
                'action': 'BUY' if weight_change > 0 else 'SELL',
                'weight_change': abs(weight_change),
                'dollar_amount': abs(weight_change) * portfolio_value
            })
        
        return pd.DataFrame(trade_list)


def demo():
    """Demonstrate hard asset optimization and HRP allocation."""
    print("=" * 60)
    print("Hard Asset Optimizer Demo")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-01', freq='B')
    n = len(dates)
    
    prices = pd.DataFrame({
        'BTC-USD': 30000 * np.exp(np.cumsum(np.random.randn(n) * 0.03 + 0.0005)),
        'GLD': 1800 * np.exp(np.cumsum(np.random.randn(n) * 0.008 + 0.0002)),
        'SLV': 25 * np.exp(np.cumsum(np.random.randn(n) * 0.015 + 0.0001)),
    }, index=dates)
    
    # Test HRP Allocator (without full Optuna - just demonstrate)
    print("\n" + "=" * 60)
    print("HRP Hard Asset Allocation")
    print("=" * 60)
    
    allocator = HRPHardAssetAllocator()
    
    # Generate allocation
    allocation = allocator.allocate(prices)
    
    print("\nTarget Weights:")
    print(allocation.weights.round(4))
    
    print("\nFiltered Weights (active assets only):")
    print(allocation.filtered_weights.round(4))
    
    print("\nPortfolio Stats:")
    for key, value in allocation.portfolio_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Show trades needed
    current = pd.Series(0.33, index=prices.columns)  # Equal starting
    trades = allocator.get_rebalance_trades(current, allocation.weights, 100000)
    
    if not trades.empty:
        print("\nTrades Needed ($100k portfolio):")
        print(trades.to_string())


if __name__ == "__main__":
    demo()

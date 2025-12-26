"""
Hard Asset Signals Module
=========================
Specialized signal generators for Bitcoin, Gold, and Silver.

Implements asset-specific regime filters based on academic research:
- BTC: Volatility-adjusted momentum (Liu & Tsyvinski 2021)
- Gold: Real yields regime filter (Erb & Harvey 2013)
- Silver: Gold-Silver ratio mean reversion

These signals are designed to be used with HRP portfolio construction.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Try to import yfinance for macro data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Macro data unavailable.")


@dataclass
class HardAssetSignalResult:
    """Result from hard asset signal generation."""
    asset: str
    signal: pd.Series  # 1 = buy, 0 = no position
    strength: pd.Series  # signal strength 0-1
    regime: pd.Series  # regime state (string labels)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseHardAssetSignal(ABC):
    """
    Abstract base class for hard asset signal generators.
    
    Each hard asset (BTC, Gold, Silver) gets its own specialized signal
    based on academic research showing asset-specific factors.
    """
    
    @property
    @abstractmethod
    def asset_name(self) -> str:
        """Asset ticker/name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Strategy description."""
        pass
    
    @abstractmethod
    def generate_signal(
        self,
        prices: pd.Series,
        **kwargs
    ) -> HardAssetSignalResult:
        """
        Generate trading signal for this hard asset.
        
        Args:
            prices: Price series for this asset
            **kwargs: Additional data (macro indicators, etc.)
            
        Returns:
            HardAssetSignalResult with signal and metadata
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return strategy parameters for logging."""
        return {}


class BTCVolatilityMomentum(BaseHardAssetSignal):
    """
    Bitcoin Volatility-Adjusted Momentum Signal.
    
    Based on Liu & Tsyvinski (2021) finding that crypto momentum
    is strongest at shorter lookbacks (7-30 days) when adjusted
    for the higher volatility of the asset.
    
    Signal Logic:
        1. Calculate 21-day returns
        2. Normalize by rolling 63-day volatility
        3. Buy when vol-adjusted momentum > threshold
    
    Rationale: BTC's high volatility makes raw momentum noisy.
    Volatility adjustment creates a more stable signal.
    
    Cost Consideration: Bybit < 0.1% fees allow higher turnover.
    """
    
    def __init__(
        self,
        momentum_lookback: int = 21,
        volatility_lookback: int = 63,
        threshold: float = 0.5,
        min_holding_days: int = 5
    ):
        """
        Initialize BTC momentum signal.
        
        Args:
            momentum_lookback: Days for momentum calculation (7-63)
            volatility_lookback: Days for volatility calculation
            threshold: Vol-adjusted momentum threshold
            min_holding_days: Minimum days before signal change
        """
        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
        self.threshold = threshold
        self.min_holding_days = min_holding_days
    
    @property
    def asset_name(self) -> str:
        return "BTC"
    
    @property
    def description(self) -> str:
        return f"BTC {self.momentum_lookback}D Vol-Adjusted Momentum"
    
    def generate_signal(
        self,
        prices: pd.Series,
        **kwargs
    ) -> HardAssetSignalResult:
        """Generate BTC volatility-adjusted momentum signal."""
        
        # Calculate momentum
        returns = prices.pct_change()
        momentum = prices.pct_change(self.momentum_lookback)
        
        # Calculate rolling volatility (annualized)
        volatility = returns.rolling(self.volatility_lookback).std() * np.sqrt(252)
        
        # Vol-adjusted momentum (like a rolling Sharpe)
        vol_adj_momentum = momentum / volatility
        
        # Generate raw signal
        raw_signal = (vol_adj_momentum > self.threshold).astype(int)
        
        # Apply minimum holding period to reduce whipsaws
        signal = self._apply_holding_period(raw_signal)
        
        # Calculate signal strength (normalized vol-adj momentum)
        # Cap at 2 std devs to avoid extreme values
        strength = vol_adj_momentum.clip(lower=0)
        strength = (strength / strength.rolling(252).quantile(0.95)).clip(upper=1.0)
        strength = strength.fillna(0)
        
        # Define regime labels
        regime = pd.Series('NEUTRAL', index=prices.index)
        regime[vol_adj_momentum > self.threshold * 1.5] = 'STRONG_MOMENTUM'
        regime[vol_adj_momentum > self.threshold] = 'MOMENTUM'
        regime[vol_adj_momentum < -self.threshold] = 'DOWNTREND'
        
        return HardAssetSignalResult(
            asset=self.asset_name,
            signal=signal,
            strength=strength,
            regime=regime,
            metadata={
                'momentum_lookback': self.momentum_lookback,
                'volatility_lookback': self.volatility_lookback,
                'threshold': self.threshold,
                'current_vol_adj_momentum': float(vol_adj_momentum.iloc[-1]) if not pd.isna(vol_adj_momentum.iloc[-1]) else 0.0
            }
        )
    
    def _apply_holding_period(self, signal: pd.Series) -> pd.Series:
        """Apply minimum holding period to reduce signal churn."""
        result = signal.copy()
        last_change_idx = 0
        last_value = signal.iloc[0]
        
        for i in range(1, len(signal)):
            if signal.iloc[i] != last_value:
                if i - last_change_idx >= self.min_holding_days:
                    last_value = signal.iloc[i]
                    last_change_idx = i
                else:
                    result.iloc[i] = last_value
            else:
                result.iloc[i] = last_value
        
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'momentum_lookback': self.momentum_lookback,
            'volatility_lookback': self.volatility_lookback,
            'threshold': self.threshold,
            'min_holding_days': self.min_holding_days
        }


class GoldRegimeFilter(BaseHardAssetSignal):
    """
    Gold Regime Filter Signal.
    
    Based on Erb & Harvey (2013) "The Golden Dilemma" finding that
    Gold performs best in specific regimes:
    - Negative real interest rates
    - High market uncertainty (VIX > 20)
    - Dollar weakness
    
    Signal Logic:
        1. Calculate real yield (10Y - inflation expectations)
        2. Check VIX level
        3. Buy Gold when: Real Yield < 0 OR VIX > 20
    
    Fallback: If macro data unavailable, use 200-day SMA trend filter.
    """
    
    def __init__(
        self,
        real_yield_threshold: float = 0.0,
        vix_threshold: float = 20.0,
        sma_fallback_period: int = 200,
        use_macro_data: bool = True
    ):
        """
        Initialize Gold regime filter.
        
        Args:
            real_yield_threshold: Real yield level below which Gold is favored
            vix_threshold: VIX level above which Gold is favored
            sma_fallback_period: SMA period for fallback trend filter
            use_macro_data: Whether to try using macro data
        """
        self.real_yield_threshold = real_yield_threshold
        self.vix_threshold = vix_threshold
        self.sma_fallback_period = sma_fallback_period
        self.use_macro_data = use_macro_data
    
    @property
    def asset_name(self) -> str:
        return "GOLD"
    
    @property
    def description(self) -> str:
        return "Gold Real Yields + VIX Regime Filter"
    
    def generate_signal(
        self,
        prices: pd.Series,
        treasury_10y: pd.Series = None,
        inflation_expectations: pd.Series = None,
        vix: pd.Series = None,
        **kwargs
    ) -> HardAssetSignalResult:
        """
        Generate Gold regime filter signal.
        
        Args:
            prices: Gold prices (e.g., GLD ETF)
            treasury_10y: 10-Year Treasury yield (optional)
            inflation_expectations: Breakeven inflation (optional)
            vix: VIX index (optional)
        """
        
        use_fallback = not self.use_macro_data
        
        # Try to use macro data if available
        if self.use_macro_data and treasury_10y is not None and vix is not None:
            # Align data
            common_idx = prices.index.intersection(treasury_10y.index).intersection(vix.index)
            
            if len(common_idx) > 100:
                treasury = treasury_10y.reindex(common_idx)
                vix_aligned = vix.reindex(common_idx)
                
                # Calculate real yield
                if inflation_expectations is not None:
                    inflation = inflation_expectations.reindex(common_idx)
                    real_yield = treasury - inflation
                else:
                    # Use trailing 12-month gold return as inflation proxy
                    real_yield = treasury - 2.5  # Assume 2.5% average inflation
                
                # Generate regime signal
                low_real_yield = real_yield < self.real_yield_threshold
                high_vix = vix_aligned > self.vix_threshold
                
                signal = (low_real_yield | high_vix).astype(int)
                signal = signal.reindex(prices.index, method='ffill').fillna(0).astype(int)
                
                # Regime labels
                regime = pd.Series('NEUTRAL', index=prices.index)
                regime_aligned = pd.Series('NEUTRAL', index=common_idx)
                regime_aligned[low_real_yield & high_vix] = 'CRISIS'
                regime_aligned[low_real_yield & ~high_vix] = 'LOW_REAL_YIELD'
                regime_aligned[~low_real_yield & high_vix] = 'HIGH_UNCERTAINTY'
                regime = regime_aligned.reindex(prices.index, method='ffill').fillna('NEUTRAL')
                
                # Strength based on how far into regime territory
                strength = pd.Series(0.0, index=prices.index)
                strength_calc = pd.Series(0.0, index=common_idx)
                
                # Normalize contributions
                yield_strength = (self.real_yield_threshold - real_yield).clip(lower=0) / 2.0
                vix_strength = ((vix_aligned - self.vix_threshold) / 20).clip(lower=0, upper=1)
                strength_calc = (yield_strength.clip(upper=1) + vix_strength) / 2
                
                strength = strength_calc.reindex(prices.index, method='ffill').fillna(0)
                
                return HardAssetSignalResult(
                    asset=self.asset_name,
                    signal=signal,
                    strength=strength,
                    regime=regime,
                    metadata={
                        'using_macro_data': True,
                        'real_yield_threshold': self.real_yield_threshold,
                        'vix_threshold': self.vix_threshold,
                        'current_real_yield': float(real_yield.iloc[-1]) if len(real_yield) > 0 else None,
                        'current_vix': float(vix_aligned.iloc[-1]) if len(vix_aligned) > 0 else None
                    }
                )
            else:
                use_fallback = True
        else:
            use_fallback = True
        
        # Fallback: SMA trend filter
        if use_fallback:
            return self._generate_sma_fallback(prices)
    
    def _generate_sma_fallback(self, prices: pd.Series) -> HardAssetSignalResult:
        """Generate signal using SMA trend filter as fallback."""
        sma = prices.rolling(self.sma_fallback_period).mean()
        
        # Signal: Price above SMA
        signal = (prices > sma).astype(int)
        
        # Strength: Distance from SMA
        distance = (prices - sma) / sma
        strength = distance.clip(lower=0, upper=0.2) / 0.2  # Normalize to 0-1
        
        # Regime based on trend
        regime = pd.Series('NEUTRAL', index=prices.index)
        regime[prices > sma * 1.05] = 'UPTREND'
        regime[prices < sma * 0.95] = 'DOWNTREND'
        
        return HardAssetSignalResult(
            asset=self.asset_name,
            signal=signal,
            strength=strength.fillna(0),
            regime=regime,
            metadata={
                'using_macro_data': False,
                'sma_period': self.sma_fallback_period,
                'fallback_reason': 'Macro data unavailable'
            }
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'real_yield_threshold': self.real_yield_threshold,
            'vix_threshold': self.vix_threshold,
            'sma_fallback_period': self.sma_fallback_period
        }


class SilverGoldRatio(BaseHardAssetSignal):
    """
    Silver Gold-Ratio Mean Reversion Signal.
    
    The Gold-Silver Ratio (GSR) has historically mean-reverted.
    When GSR is high (Silver cheap relative to Gold), Silver tends
    to outperform.
    
    Signal Logic:
        1. Calculate Gold-Silver Ratio
        2. Calculate rolling mean and std dev
        3. Buy Silver when: GSR > mean + 0.5 * std
           (Silver is historically cheap)
    
    Historical Context:
        - Long-term average GSR: ~60
        - Range: 30-120
        - High GSR = Silver undervalued
    """
    
    def __init__(
        self,
        ratio_lookback: int = 252,
        zscore_threshold: float = 0.5,
        use_absolute_threshold: bool = False,
        absolute_threshold: float = 80.0
    ):
        """
        Initialize Silver GSR signal.
        
        Args:
            ratio_lookback: Days for rolling mean/std calculation
            zscore_threshold: Z-score threshold for signal (0.5 = half std)
            use_absolute_threshold: If True, use absolute GSR threshold
            absolute_threshold: Absolute GSR level to trigger signal
        """
        self.ratio_lookback = ratio_lookback
        self.zscore_threshold = zscore_threshold
        self.use_absolute_threshold = use_absolute_threshold
        self.absolute_threshold = absolute_threshold
    
    @property
    def asset_name(self) -> str:
        return "SILVER"
    
    @property
    def description(self) -> str:
        return "Silver Gold-Ratio Mean Reversion"
    
    def generate_signal(
        self,
        prices: pd.Series,
        gold_prices: pd.Series = None,
        **kwargs
    ) -> HardAssetSignalResult:
        """
        Generate Silver GSR signal.
        
        Args:
            prices: Silver prices (e.g., SLV ETF)
            gold_prices: Gold prices (e.g., GLD ETF) - required
        """
        
        if gold_prices is None:
            # Fallback to simple momentum
            return self._generate_momentum_fallback(prices)
        
        # Calculate Gold-Silver Ratio
        # Align indices
        common_idx = prices.index.intersection(gold_prices.index)
        silver = prices.reindex(common_idx)
        gold = gold_prices.reindex(common_idx)
        
        gsr = gold / silver
        
        # Calculate rolling statistics
        gsr_mean = gsr.rolling(self.ratio_lookback).mean()
        gsr_std = gsr.rolling(self.ratio_lookback).std()
        
        # Z-score
        gsr_zscore = (gsr - gsr_mean) / gsr_std
        
        if self.use_absolute_threshold:
            # Use absolute GSR threshold
            signal = (gsr > self.absolute_threshold).astype(int)
        else:
            # Use z-score threshold
            signal = (gsr_zscore > self.zscore_threshold).astype(int)
        
        # Reindex to full price index
        signal = signal.reindex(prices.index, method='ffill').fillna(0).astype(int)
        
        # Strength based on how far ratio is from mean
        strength = gsr_zscore.clip(lower=0, upper=2) / 2
        strength = strength.reindex(prices.index, method='ffill').fillna(0)
        
        # Regime labels
        regime = pd.Series('NEUTRAL', index=prices.index)
        regime_calc = pd.Series('NEUTRAL', index=common_idx)
        regime_calc[gsr_zscore > 1.0] = 'SILVER_VERY_CHEAP'
        regime_calc[(gsr_zscore > self.zscore_threshold) & (gsr_zscore <= 1.0)] = 'SILVER_CHEAP'
        regime_calc[gsr_zscore < -self.zscore_threshold] = 'SILVER_EXPENSIVE'
        regime = regime_calc.reindex(prices.index, method='ffill').fillna('NEUTRAL')
        
        return HardAssetSignalResult(
            asset=self.asset_name,
            signal=signal,
            strength=strength,
            regime=regime,
            metadata={
                'ratio_lookback': self.ratio_lookback,
                'zscore_threshold': self.zscore_threshold,
                'current_gsr': float(gsr.iloc[-1]) if len(gsr) > 0 else None,
                'current_gsr_zscore': float(gsr_zscore.iloc[-1]) if len(gsr_zscore) > 0 and not pd.isna(gsr_zscore.iloc[-1]) else None,
                'gsr_mean': float(gsr_mean.iloc[-1]) if len(gsr_mean) > 0 and not pd.isna(gsr_mean.iloc[-1]) else None
            }
        )
    
    def _generate_momentum_fallback(self, prices: pd.Series) -> HardAssetSignalResult:
        """Fallback to simple momentum if gold prices unavailable."""
        momentum = prices.pct_change(63)  # 3-month momentum
        signal = (momentum > 0).astype(int)
        strength = (momentum.clip(lower=0, upper=0.3) / 0.3).fillna(0)
        
        regime = pd.Series('NEUTRAL', index=prices.index)
        regime[momentum > 0.1] = 'UPTREND'
        regime[momentum < -0.1] = 'DOWNTREND'
        
        return HardAssetSignalResult(
            asset=self.asset_name,
            signal=signal,
            strength=strength,
            regime=regime,
            metadata={
                'fallback': True,
                'reason': 'Gold prices unavailable for GSR calculation'
            }
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'ratio_lookback': self.ratio_lookback,
            'zscore_threshold': self.zscore_threshold,
            'use_absolute_threshold': self.use_absolute_threshold,
            'absolute_threshold': self.absolute_threshold
        }


class HardAssetSignalManager:
    """
    Central manager for all hard asset signals.
    
    Coordinates signal generation across BTC, Gold, and Silver,
    and provides unified interface for the portfolio construction layer.
    """
    
    def __init__(self):
        """Initialize with default signal configurations."""
        self.signals: Dict[str, BaseHardAssetSignal] = {}
        self._results: Dict[str, HardAssetSignalResult] = {}
        
        # Register default signals
        self.register_signal(BTCVolatilityMomentum())
        self.register_signal(GoldRegimeFilter())
        self.register_signal(SilverGoldRatio())
    
    def register_signal(self, signal: BaseHardAssetSignal):
        """Register a hard asset signal generator."""
        self.signals[signal.asset_name] = signal
        print(f"   ✓ Registered hard asset signal: {signal.asset_name} - {signal.description}")
    
    def update_signal_params(self, asset: str, **params):
        """
        Update parameters for a specific signal.
        
        Useful for Optuna optimization.
        """
        if asset not in self.signals:
            raise ValueError(f"Unknown asset: {asset}")
        
        signal = self.signals[asset]
        for key, value in params.items():
            if hasattr(signal, key):
                setattr(signal, key, value)
    
    def generate_all_signals(
        self,
        btc_prices: pd.Series = None,
        gold_prices: pd.Series = None,
        silver_prices: pd.Series = None,
        treasury_10y: pd.Series = None,
        vix: pd.Series = None,
        **kwargs
    ) -> Dict[str, HardAssetSignalResult]:
        """
        Generate signals for all registered hard assets.
        
        Args:
            btc_prices: Bitcoin price series
            gold_prices: Gold price series (GLD)
            silver_prices: Silver price series (SLV)
            treasury_10y: 10Y Treasury yield
            vix: VIX index
            
        Returns:
            Dict mapping asset name to SignalResult
        """
        results = {}
        
        # BTC Signal
        if btc_prices is not None and 'BTC' in self.signals:
            try:
                result = self.signals['BTC'].generate_signal(btc_prices)
                results['BTC'] = result
                self._results['BTC'] = result
            except Exception as e:
                print(f"   ⚠️ Error generating BTC signal: {e}")
        
        # Gold Signal
        if gold_prices is not None and 'GOLD' in self.signals:
            try:
                result = self.signals['GOLD'].generate_signal(
                    gold_prices,
                    treasury_10y=treasury_10y,
                    vix=vix
                )
                results['GOLD'] = result
                self._results['GOLD'] = result
            except Exception as e:
                print(f"   ⚠️ Error generating GOLD signal: {e}")
        
        # Silver Signal
        if silver_prices is not None and 'SILVER' in self.signals:
            try:
                result = self.signals['SILVER'].generate_signal(
                    silver_prices,
                    gold_prices=gold_prices
                )
                results['SILVER'] = result
                self._results['SILVER'] = result
            except Exception as e:
                print(f"   ⚠️ Error generating SILVER signal: {e}")
        
        return results
    
    def get_combined_signal_matrix(
        self,
        results: Dict[str, HardAssetSignalResult] = None
    ) -> pd.DataFrame:
        """
        Combine signals into a matrix for portfolio construction.
        
        Returns:
            DataFrame with columns for each asset, values 0/1
        """
        if results is None:
            results = self._results
        
        if not results:
            raise ValueError("No signals generated. Call generate_all_signals first.")
        
        # Find common date range
        all_signals = {name: r.signal for name, r in results.items()}
        
        # Create combined DataFrame
        signals_df = pd.DataFrame(all_signals)
        
        return signals_df
    
    def get_signal_summary(self) -> pd.DataFrame:
        """Get summary of current signal states."""
        if not self._results:
            return pd.DataFrame()
        
        summary = []
        for asset, result in self._results.items():
            summary.append({
                'asset': asset,
                'current_signal': int(result.signal.iloc[-1]) if len(result.signal) > 0 else None,
                'current_strength': float(result.strength.iloc[-1]) if len(result.strength) > 0 else None,
                'current_regime': result.regime.iloc[-1] if len(result.regime) > 0 else None,
                'strategy': self.signals[asset].description
            })
        
        return pd.DataFrame(summary)


def fetch_macro_data(
    start_date: str = '2015-01-01',
    end_date: str = None
) -> Dict[str, pd.Series]:
    """
    Fetch macro data for regime filters.
    
    Returns:
        Dict with 'treasury_10y', 'vix', 'tips_5y' series
    """
    if not HAS_YFINANCE:
        print("yfinance not available. Cannot fetch macro data.")
        return {}
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = {}
    
    try:
        # 10-Year Treasury
        tnx = yf.download('^TNX', start=start_date, end=end_date, progress=False)
        if not tnx.empty:
            data['treasury_10y'] = tnx['Close'].squeeze()
    except Exception as e:
        print(f"Error fetching Treasury data: {e}")
    
    try:
        # VIX
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if not vix.empty:
            data['vix'] = vix['Close'].squeeze()
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
    
    return data


def demo():
    """Demonstrate hard asset signal generation."""
    print("=" * 60)
    print("Hard Asset Signals Demo")
    print("=" * 60)
    
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-01', freq='B')
    n = len(dates)
    
    # Simulated prices
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
    
    # Initialize manager
    manager = HardAssetSignalManager()
    
    # Generate signals (without macro data for demo)
    results = manager.generate_all_signals(
        btc_prices=btc,
        gold_prices=gold,
        silver_prices=silver
    )
    
    print("\n" + "=" * 60)
    print("Signal Summary")
    print("=" * 60)
    print(manager.get_signal_summary().to_string())
    
    # Show combined signals
    signals_matrix = manager.get_combined_signal_matrix()
    print("\n" + "=" * 60)
    print("Last 10 Days Signal Matrix")
    print("=" * 60)
    print(signals_matrix.tail(10))
    
    # BTC specific analysis
    print("\n" + "=" * 60)
    print("BTC Signal Analysis")
    print("=" * 60)
    btc_result = results['BTC']
    print(f"Signal on days: {btc_result.signal.sum()} / {len(btc_result.signal)}")
    print(f"Regime distribution:")
    print(btc_result.regime.value_counts())


if __name__ == "__main__":
    demo()

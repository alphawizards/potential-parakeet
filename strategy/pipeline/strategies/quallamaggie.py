"""
Quallamaggie Strategy Plugin
=============================
Implementation of Kristjan Kullamägi's momentum breakout strategy.

Strategy Components:
- 1M/3M/6M momentum screening (top performers)
- RS Line relative to SPY
- Pattern detection (VCP, HTF, consolidation)
- Breakout on volume
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..signal_layer import BaseStrategy, SignalResult


@dataclass
class QuallamaggieConfig:
    """Configuration for Quallamaggie strategy."""
    
    # Momentum settings
    momentum_lookback: int = 126  # 6 months default
    top_momentum_pct: float = 0.10  # Top 10%
    
    # Moving averages
    ema_10: int = 10
    ema_20: int = 20
    sma_50: int = 50
    sma_200: int = 200
    
    # Pattern detection
    consolidation_min_days: int = 10
    consolidation_max_range: float = 0.15  # 15% range
    breakout_volume_mult: float = 1.5  # 1.5x average
    
    # RS Line
    rs_benchmark: str = "SPY"
    
    # Position
    max_positions: int = 10


class QuallamaggieStrategy(BaseStrategy):
    """
    Quallamaggie Momentum Breakout Strategy.
    
    This strategy identifies stocks with:
    1. Strong momentum (top performers over lookback period)
    2. RS Line at or near new highs vs benchmark
    3. In consolidation/base pattern
    4. Breaking out on above-average volume
    """
    
    def __init__(self, config: QuallamaggieConfig = None):
        self.config = config or QuallamaggieConfig()
    
    @property
    def name(self) -> str:
        days = self.config.momentum_lookback
        months = days // 21
        return f"Quallamaggie_{months}M"
    
    @property
    def description(self) -> str:
        return f"Quallamaggie momentum breakout ({self.config.momentum_lookback}d lookback)"
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame = None,
        **kwargs
    ) -> SignalResult:
        """
        Generate Quallamaggie trading signals.
        
        Signal logic:
        1. Calculate momentum over lookback period
        2. Filter for top performers (momentum filter)
        3. Check if in uptrend (above key MAs)
        4. Detect consolidation/base pattern
        5. Generate buy signal on breakout with volume
        """
        lookback = self.config.momentum_lookback
        
        # Initialize signal DataFrames
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        strength = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        # Need enough history
        if len(prices) < lookback + 50:
            return SignalResult(
                strategy_name=self.name,
                signals=signals,
                strength=strength,
                metadata={'error': 'Insufficient data'}
            )
        
        # Calculate components
        momentum = self._calculate_momentum(prices, lookback)
        ema_10 = prices.ewm(span=self.config.ema_10).mean()
        ema_20 = prices.ewm(span=self.config.ema_20).mean()
        sma_50 = prices.rolling(self.config.sma_50).mean()
        sma_200 = prices.rolling(self.config.sma_200).mean()
        
        # RS Line (relative to benchmark)
        rs_line = self._calculate_rs_line(prices)
        
        # Volume analysis
        if volume is not None:
            avg_volume = volume.rolling(50).mean()
            relative_volume = volume / avg_volume
        else:
            relative_volume = None
        
        # Generate signals for each day
        for i in range(lookback, len(prices)):
            date = prices.index[i]
            
            # Get momentum ranking for this date
            mom_row = momentum.iloc[i] if i < len(momentum) else None
            if mom_row is None or mom_row.isna().all():
                continue
            
            # Filter for top momentum stocks
            threshold = mom_row.quantile(1 - self.config.top_momentum_pct)
            top_momentum_stocks = mom_row[mom_row >= threshold].index
            
            for ticker in top_momentum_stocks:
                if ticker not in prices.columns:
                    continue
                
                # Check trend alignment
                price = prices.loc[date, ticker]
                if pd.isna(price):
                    continue
                
                _ema10 = ema_10.loc[date, ticker] if ticker in ema_10.columns else np.nan
                _ema20 = ema_20.loc[date, ticker] if ticker in ema_20.columns else np.nan
                _sma50 = sma_50.loc[date, ticker] if ticker in sma_50.columns else np.nan
                _sma200 = sma_200.loc[date, ticker] if ticker in sma_200.columns else np.nan
                
                # Trend filter: price above key MAs
                in_uptrend = (
                    price > _ema10 and 
                    price > _ema20 and 
                    _ema10 > _ema20 and
                    (pd.isna(_sma50) or price > _sma50)
                )
                
                if not in_uptrend:
                    continue
                
                # Check RS Line (near highs)
                rs_good = self._check_rs_strength(rs_line, date, ticker)
                
                # Check for consolidation pattern
                recent_prices = prices[ticker].iloc[max(0, i-20):i+1]
                in_consolidation = self._detect_consolidation(recent_prices)
                
                # Volume confirmation
                vol_confirmed = True
                if relative_volume is not None:
                    rel_vol = relative_volume.loc[date, ticker] if ticker in relative_volume.columns else 1.0
                    vol_confirmed = rel_vol >= self.config.breakout_volume_mult
                
                # Generate signal
                if in_uptrend and (in_consolidation or rs_good):
                    signal_strength = self._calculate_signal_strength(
                        momentum=mom_row[ticker] if ticker in mom_row.index else 0,
                        rs_good=rs_good,
                        vol_confirmed=vol_confirmed
                    )
                    
                    signals.loc[date, ticker] = 1
                    strength.loc[date, ticker] = signal_strength
        
        return SignalResult(
            strategy_name=self.name,
            signals=signals,
            strength=strength,
            metadata={
                'lookback': lookback,
                'top_pct': self.config.top_momentum_pct,
                'max_positions': self.config.max_positions
            }
        )
    
    def _calculate_momentum(
        self,
        prices: pd.DataFrame,
        lookback: int
    ) -> pd.DataFrame:
        """Calculate momentum (percentage return over lookback)."""
        return prices.pct_change(lookback) * 100
    
    def _calculate_rs_line(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate RS Line vs benchmark."""
        benchmark = self.config.rs_benchmark
        
        if benchmark not in prices.columns:
            # Return normalized prices as proxy
            return prices / prices.iloc[0] * 100
        
        benchmark_prices = prices[benchmark]
        rs_lines = prices.div(benchmark_prices, axis=0)
        rs_lines = rs_lines / rs_lines.iloc[0] * 100
        
        return rs_lines
    
    def _check_rs_strength(
        self,
        rs_line: pd.DataFrame,
        date,
        ticker: str
    ) -> bool:
        """Check if RS Line is near new highs."""
        if ticker not in rs_line.columns:
            return False
        
        rs = rs_line[ticker]
        idx = rs_line.index.get_loc(date)
        
        if idx < 50:
            return True  # Not enough data, assume good
        
        current_rs = rs.iloc[idx]
        rs_50d_high = rs.iloc[max(0, idx-50):idx+1].max()
        
        # RS within 5% of 50-day high
        return current_rs >= rs_50d_high * 0.95
    
    def _detect_consolidation(self, prices: pd.Series) -> bool:
        """Detect if stock is in consolidation pattern."""
        if len(prices) < self.config.consolidation_min_days:
            return False
        
        price_range = (prices.max() - prices.min()) / prices.min()
        return price_range <= self.config.consolidation_max_range
    
    def _calculate_signal_strength(
        self,
        momentum: float,
        rs_good: bool,
        vol_confirmed: bool
    ) -> float:
        """Calculate overall signal strength (0-1)."""
        base_strength = 0.5
        
        # Momentum contribution
        if momentum > 50:
            base_strength += 0.2
        elif momentum > 25:
            base_strength += 0.1
        
        # RS Line contribution
        if rs_good:
            base_strength += 0.15
        
        # Volume contribution
        if vol_confirmed:
            base_strength += 0.15
        
        return min(1.0, base_strength)
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'momentum_lookback': self.config.momentum_lookback,
            'top_momentum_pct': self.config.top_momentum_pct,
            'max_positions': self.config.max_positions
        }


# Factory functions for different Quallamaggie variants
def create_quallamaggie_1m() -> QuallamaggieStrategy:
    """Create 1-month Quallamaggie strategy."""
    return QuallamaggieStrategy(QuallamaggieConfig(momentum_lookback=21))


def create_quallamaggie_3m() -> QuallamaggieStrategy:
    """Create 3-month Quallamaggie strategy."""
    return QuallamaggieStrategy(QuallamaggieConfig(momentum_lookback=63))


def create_quallamaggie_6m() -> QuallamaggieStrategy:
    """Create 6-month Quallamaggie strategy."""
    return QuallamaggieStrategy(QuallamaggieConfig(momentum_lookback=126))

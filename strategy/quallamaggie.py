"""
Quallamaggie Screener Module
============================
Implements the Quallamaggie (Kristjan Kullamägi) screening methodology for
identifying high-momentum stocks with strong technical setups.

The screener acts as a funnel to filter a broad universe of stocks down to
a shortlist based on strict technical criteria:
1. Trend Template (Price > SMAs in proper alignment)
2. Proximity to 52-week High (within 25%)
3. Distance from 52-week Low (> 30% above)
4. Average Daily Range (ADR > 4%)
5. Positive Momentum (1m, 3m, 6m returns > 0)

Reference: Kristjan Kullamägi's trading methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pandas_ta as ta
from dataclasses import dataclass

from .config import CONFIG


@dataclass
class ScreenerResult:
    """Container for screener results for a single ticker."""
    ticker: str
    passes_screen: bool
    price: float
    volume_millions: float
    adr_percent: float
    dist_from_high_pct: float
    dist_from_low_pct: float
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    trend_template: bool
    sma_10: float
    sma_20: float
    sma_50: float
    sma_200: float
    high_52w: float
    low_52w: float
    reason: str = ""


class QuallamaggieScreener:
    """
    Quallamaggie-style stock screener for identifying high-momentum setups.
    
    The screener applies multiple technical filters to identify stocks that:
    - Are in a strong uptrend (price > aligned SMAs)
    - Are near their 52-week highs (within 25%)
    - Have pulled back from lows significantly (> 30% above)
    - Show high average daily range (ADR > 4%)
    - Have positive momentum across multiple timeframes
    
    Attributes:
        ohlc_data: Dict mapping ticker -> DataFrame with OHLC + Volume
        min_adr: Minimum ADR percentage threshold (default: 4.0)
        max_dist_from_high: Maximum distance from 52-week high (default: 25%)
        min_dist_from_low: Minimum distance from 52-week low (default: 30%)
    """
    
    def __init__(
        self,
        ohlc_data: Dict[str, pd.DataFrame],
        min_adr: float = 4.0,
        max_dist_from_high: float = 25.0,
        min_dist_from_low: float = 30.0
    ):
        """
        Initialize the Quallamaggie Screener.
        
        Args:
            ohlc_data: Dictionary mapping ticker symbols to DataFrames
                       containing OHLC and Volume data. Each DataFrame should
                       have columns: ['Open', 'High', 'Low', 'Close', 'Volume']
            min_adr: Minimum Average Daily Range percentage (default: 4.0%)
            max_dist_from_high: Max % below 52-week high (default: 25%)
            min_dist_from_low: Min % above 52-week low (default: 30%)
        """
        self.ohlc_data = ohlc_data
        self.min_adr = min_adr
        self.max_dist_from_high = max_dist_from_high
        self.min_dist_from_low = min_dist_from_low
        
        # Validate input data
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Validate that OHLC data has required columns."""
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        
        for ticker, df in self.ohlc_data.items():
            if df.empty:
                raise ValueError(f"Empty DataFrame for ticker: {ticker}")
            
            # Check for required columns (case-insensitive)
            df_cols = {col.title() for col in df.columns}
            missing = required_cols - df_cols
            if missing:
                raise ValueError(
                    f"Missing columns for {ticker}: {missing}. "
                    f"Available: {list(df.columns)}"
                )
    
    def get_trend_template(self, series: pd.Series) -> bool:
        """
        Check if price satisfies the Quallamaggie Trend Template.
        
        The trend template requires:
        1. Price > SMA(10) > SMA(20) > SMA(50) (proper alignment)
        2. Price > SMA(200) (long-term uptrend)
        3. Price within 25% of 52-week High
        4. Price > 30% above 52-week Low
        
        Args:
            series: Price series (typically Close prices)
            
        Returns:
            bool: True if all trend template conditions are met
        """
        if len(series) < 252:  # Need at least 1 year of data
            return False
        
        # Calculate SMAs using pandas-ta
        sma_10 = ta.sma(series, length=10)
        sma_20 = ta.sma(series, length=20)
        sma_50 = ta.sma(series, length=50)
        sma_200 = ta.sma(series, length=200)
        
        if sma_10 is None or sma_200 is None:
            return False
            
        # Get latest values
        price = series.iloc[-1]
        sma_10_val = sma_10.iloc[-1]
        sma_20_val = sma_20.iloc[-1]
        sma_50_val = sma_50.iloc[-1]
        sma_200_val = sma_200.iloc[-1]
        
        # Check for NaN values
        if any(pd.isna([price, sma_10_val, sma_20_val, sma_50_val, sma_200_val])):
            return False
        
        # Condition 1: Price > SMA(10) > SMA(20) > SMA(50) (proper alignment)
        sma_alignment = (
            price > sma_10_val > sma_20_val > sma_50_val
        )
        
        # Condition 2: Price > SMA(200) (long-term uptrend)
        above_200sma = price > sma_200_val
        
        # Condition 3 & 4: Check 52-week high/low positioning
        high_52w = series.rolling(window=252).max().iloc[-1]
        low_52w = series.rolling(window=252).min().iloc[-1]
        
        # Within 25% of 52-week high
        dist_from_high_pct = ((high_52w - price) / high_52w) * 100
        near_high = dist_from_high_pct <= self.max_dist_from_high
        
        # More than 30% above 52-week low
        dist_from_low_pct = ((price - low_52w) / low_52w) * 100
        above_low = dist_from_low_pct >= self.min_dist_from_low
        
        return sma_alignment and above_200sma and near_high and above_low
    
    def calculate_adr(self, ticker: str, window: int = 20) -> float:
        """
        Calculate Average Daily Range (ADR) percentage.
        
        ADR% = Average((High - Low) / Low) * 100
        
        This measures typical daily volatility as a percentage.
        High ADR stocks are more suitable for swing trading.
        
        Args:
            ticker: Ticker symbol
            window: Rolling window for average (default: 20 days)
            
        Returns:
            float: ADR as a percentage (e.g., 4.5 for 4.5%)
        """
        df = self.ohlc_data.get(ticker)
        if df is None or df.empty:
            return 0.0
        
        # Normalize column names
        df = df.copy()
        df.columns = [col.title() for col in df.columns]
        
        high = df['High']
        low = df['Low']
        
        # Calculate daily range as percentage of low
        daily_range_pct = ((high - low) / low) * 100
        
        # Calculate rolling average
        adr = daily_range_pct.rolling(window=window).mean().iloc[-1]
        
        return float(adr) if not pd.isna(adr) else 0.0
    
    def calculate_adr_from_volatility(
        self, 
        ticker: str, 
        window: int = 20
    ) -> float:
        """
        Approximate ADR from Close price volatility.
        
        Use this fallback when High/Low data is unreliable.
        ADR ≈ Daily Volatility * sqrt(2/pi) * 2 (approximate range from std)
        
        Args:
            ticker: Ticker symbol
            window: Rolling window for average (default: 20 days)
            
        Returns:
            float: Approximate ADR as a percentage
        """
        df = self.ohlc_data.get(ticker)
        if df is None or df.empty:
            return 0.0
        
        df = df.copy()
        df.columns = [col.title() for col in df.columns]
        
        close = df['Close']
        returns = close.pct_change().dropna()
        
        # Rolling volatility (daily)
        daily_vol = returns.rolling(window=window).std().iloc[-1]
        
        # Convert to approximate range (range ≈ 2.5 * std for normal dist)
        # Multiply by 100 to get percentage
        adr_approx = daily_vol * 2.5 * 100
        
        return float(adr_approx) if not pd.isna(adr_approx) else 0.0
    
    def calculate_momentum(
        self, 
        ticker: str
    ) -> Tuple[float, float, float]:
        """
        Calculate momentum returns over multiple timeframes.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Tuple of (1-month, 3-month, 6-month) returns as percentages
        """
        df = self.ohlc_data.get(ticker)
        if df is None or df.empty:
            return (0.0, 0.0, 0.0)
        
        df = df.copy()
        df.columns = [col.title() for col in df.columns]
        
        close = df['Close']
        
        if len(close) < 126:  # Need at least 6 months
            return (0.0, 0.0, 0.0)
        
        current_price = close.iloc[-1]
        
        # 1-month return (21 trading days)
        if len(close) >= 21:
            price_1m = close.iloc[-21]
            mom_1m = ((current_price - price_1m) / price_1m) * 100
        else:
            mom_1m = 0.0
        
        # 3-month return (63 trading days)
        if len(close) >= 63:
            price_3m = close.iloc[-63]
            mom_3m = ((current_price - price_3m) / price_3m) * 100
        else:
            mom_3m = 0.0
        
        # 6-month return (126 trading days)
        if len(close) >= 126:
            price_6m = close.iloc[-126]
            mom_6m = ((current_price - price_6m) / price_6m) * 100
        else:
            mom_6m = 0.0
        
        return (float(mom_1m), float(mom_3m), float(mom_6m))
    
    def _get_sma_values(self, series: pd.Series) -> Dict[str, float]:
        """
        Calculate all SMA values for reporting.
        
        Args:
            series: Price series
            
        Returns:
            Dict with SMA values
        """
        sma_10 = ta.sma(series, length=10)
        sma_20 = ta.sma(series, length=20)
        sma_50 = ta.sma(series, length=50)
        sma_200 = ta.sma(series, length=200)
        
        return {
            'sma_10': float(sma_10.iloc[-1]) if sma_10 is not None else 0.0,
            'sma_20': float(sma_20.iloc[-1]) if sma_20 is not None else 0.0,
            'sma_50': float(sma_50.iloc[-1]) if sma_50 is not None else 0.0,
            'sma_200': float(sma_200.iloc[-1]) if sma_200 is not None else 0.0,
        }
    
    def _get_52w_stats(self, series: pd.Series) -> Dict[str, float]:
        """
        Calculate 52-week high/low statistics.
        
        Args:
            series: Price series
            
        Returns:
            Dict with 52-week stats
        """
        if len(series) < 252:
            return {
                'high_52w': series.max(),
                'low_52w': series.min(),
                'dist_from_high_pct': 0.0,
                'dist_from_low_pct': 0.0,
            }
        
        price = series.iloc[-1]
        high_52w = series.rolling(window=252).max().iloc[-1]
        low_52w = series.rolling(window=252).min().iloc[-1]
        
        dist_from_high_pct = ((high_52w - price) / high_52w) * 100
        dist_from_low_pct = ((price - low_52w) / low_52w) * 100
        
        return {
            'high_52w': float(high_52w),
            'low_52w': float(low_52w),
            'dist_from_high_pct': float(dist_from_high_pct),
            'dist_from_low_pct': float(dist_from_low_pct),
        }
    
    def screen_ticker(self, ticker: str) -> ScreenerResult:
        """
        Screen a single ticker against all criteria.
        
        Args:
            ticker: Ticker symbol to screen
            
        Returns:
            ScreenerResult with detailed screening information
        """
        df = self.ohlc_data.get(ticker)
        
        # Check for None, empty, or all-NaN data
        if df is None or df.empty or df['Close'].isna().all():
            return ScreenerResult(
                ticker=ticker,
                passes_screen=False,
                price=0.0,
                volume_millions=0.0,
                adr_percent=0.0,
                dist_from_high_pct=0.0,
                dist_from_low_pct=0.0,
                momentum_1m=0.0,
                momentum_3m=0.0,
                momentum_6m=0.0,
                trend_template=False,
                sma_10=0.0,
                sma_20=0.0,
                sma_50=0.0,
                sma_200=0.0,
                high_52w=0.0,
                low_52w=0.0,
                reason="No data available"
            )
        
        # Normalize column names
        df = df.copy()
        df.columns = [col.title() for col in df.columns]
        
        close = df['Close']
        volume = df['Volume']
        price = float(close.iloc[-1])
        
        # Calculate average dollar volume (in millions)
        avg_volume = volume.rolling(window=20).mean().iloc[-1]
        volume_millions = (avg_volume * price) / 1_000_000
        
        # Calculate ADR (prefer OHLC method, fallback to volatility)
        adr = self.calculate_adr(ticker)
        if adr == 0.0:
            adr = self.calculate_adr_from_volatility(ticker)
        
        # Calculate momentum
        mom_1m, mom_3m, mom_6m = self.calculate_momentum(ticker)
        
        # Get SMA values
        sma_values = self._get_sma_values(close)
        
        # Get 52-week stats
        stats_52w = self._get_52w_stats(close)
        
        # Check trend template
        trend_ok = self.get_trend_template(close)
        
        # Determine pass/fail with reason
        reasons = []
        
        if not trend_ok:
            reasons.append("Failed trend template")
        
        if adr < self.min_adr:
            reasons.append(f"ADR {adr:.1f}% < {self.min_adr}%")
        
        if mom_1m <= 0:
            reasons.append(f"1M momentum {mom_1m:.1f}% <= 0")
        
        if mom_3m <= 0:
            reasons.append(f"3M momentum {mom_3m:.1f}% <= 0")
        
        if mom_6m <= 0:
            reasons.append(f"6M momentum {mom_6m:.1f}% <= 0")
        
        passes = len(reasons) == 0
        reason_str = "; ".join(reasons) if reasons else "All criteria passed"
        
        return ScreenerResult(
            ticker=ticker,
            passes_screen=passes,
            price=price,
            volume_millions=float(volume_millions) if not pd.isna(volume_millions) else 0.0,
            adr_percent=adr,
            dist_from_high_pct=stats_52w['dist_from_high_pct'],
            dist_from_low_pct=stats_52w['dist_from_low_pct'],
            momentum_1m=mom_1m,
            momentum_3m=mom_3m,
            momentum_6m=mom_6m,
            trend_template=trend_ok,
            sma_10=sma_values['sma_10'],
            sma_20=sma_values['sma_20'],
            sma_50=sma_values['sma_50'],
            sma_200=sma_values['sma_200'],
            high_52w=stats_52w['high_52w'],
            low_52w=stats_52w['low_52w'],
            reason=reason_str
        )
    
    def screen_universe(self) -> List[ScreenerResult]:
        """
        Screen all tickers in the universe.
        
        Iterates through all tickers in ohlc_data and returns those that
        pass all screening criteria:
        - Trend Template (SMA alignment, 52-week positioning)
        - ADR > min_adr (default 4%)
        - Positive momentum (1m, 3m, 6m > 0)
        
        Returns:
            List of ScreenerResult objects for tickers passing all filters,
            sorted by ADR descending (highest volatility first)
        """
        results = []
        
        print(f"\nScreening {len(self.ohlc_data)} tickers...")
        print("-" * 60)
        
        for ticker in self.ohlc_data.keys():
            result = self.screen_ticker(ticker)
            results.append(result)
            
            status = "✓ PASS" if result.passes_screen else "✗ FAIL"
            print(f"  {ticker}: {status} - {result.reason}")
        
        # Filter to passing tickers and sort by ADR descending
        passing = [r for r in results if r.passes_screen]
        passing.sort(key=lambda x: x.adr_percent, reverse=True)
        
        print("-" * 60)
        print(f"Passed: {len(passing)} / {len(results)} tickers")
        
        return passing
    
    def get_shortlist_dataframe(self) -> pd.DataFrame:
        """
        Run screener and return results as a formatted DataFrame.
        
        Returns:
            pd.DataFrame with columns:
            - Ticker
            - Price
            - ADR%
            - Volume($M)
            - 1M%
            - 3M%
            - 6M%
            - Dist from High%
            - Trend
        """
        results = self.screen_universe()
        
        if not results:
            return pd.DataFrame()
        
        data = []
        for r in results:
            data.append({
                'Ticker': r.ticker,
                'Price': f"${r.price:.2f}",
                'ADR%': f"{r.adr_percent:.1f}%",
                'Volume($M)': f"${r.volume_millions:.1f}M",
                '1M%': f"{r.momentum_1m:+.1f}%",
                '3M%': f"{r.momentum_3m:+.1f}%",
                '6M%': f"{r.momentum_6m:+.1f}%",
                'Dist from High%': f"{r.dist_from_high_pct:.1f}%",
                'Trend': "✓" if r.trend_template else "✗"
            })
        
        return pd.DataFrame(data)
    
    def get_detailed_report(self, ticker: str) -> str:
        """
        Generate a detailed screening report for a single ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Formatted string report
        """
        result = self.screen_ticker(ticker)
        
        report = f"""
{'=' * 50}
QUALLAMAGGIE SCREENER REPORT: {ticker}
{'=' * 50}

OVERALL: {'✓ PASS' if result.passes_screen else '✗ FAIL'}
Reason: {result.reason}

PRICE DATA:
  Current Price:     ${result.price:.2f}
  52-Week High:      ${result.high_52w:.2f} ({result.dist_from_high_pct:+.1f}%)
  52-Week Low:       ${result.low_52w:.2f} ({result.dist_from_low_pct:+.1f}%)
  Avg Daily Volume:  ${result.volume_millions:.1f}M

TREND TEMPLATE: {'✓ PASS' if result.trend_template else '✗ FAIL'}
  SMA(10):   ${result.sma_10:.2f}
  SMA(20):   ${result.sma_20:.2f}
  SMA(50):   ${result.sma_50:.2f}
  SMA(200):  ${result.sma_200:.2f}
  Alignment: Price > SMA10 > SMA20 > SMA50 {'✓' if result.trend_template else '✗'}

VOLATILITY:
  ADR%: {result.adr_percent:.2f}% (min: {self.min_adr}%)
  
MOMENTUM:
  1-Month:  {result.momentum_1m:+.1f}%
  3-Month:  {result.momentum_3m:+.1f}%
  6-Month:  {result.momentum_6m:+.1f}%

{'=' * 50}
"""
        return report


def demo():
    """Demonstrate screener functionality with sample data."""
    print("=" * 60)
    print("Quallamaggie Screener Demo")
    print("=" * 60)
    
    # Create sample OHLC data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-01', freq='B')
    
    # Simulated uptrending stock (should pass)
    base_uptrend = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.015 + 0.001))
    uptrend_df = pd.DataFrame({
        'Open': base_uptrend * (1 - np.random.uniform(0, 0.02, len(dates))),
        'High': base_uptrend * (1 + np.random.uniform(0.02, 0.05, len(dates))),
        'Low': base_uptrend * (1 - np.random.uniform(0.02, 0.05, len(dates))),
        'Close': base_uptrend,
        'Volume': np.random.randint(1_000_000, 10_000_000, len(dates))
    }, index=dates)
    
    # Simulated downtrending stock (should fail)
    base_downtrend = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.015 - 0.0008))
    downtrend_df = pd.DataFrame({
        'Open': base_downtrend * (1 - np.random.uniform(0, 0.02, len(dates))),
        'High': base_downtrend * (1 + np.random.uniform(0.01, 0.03, len(dates))),
        'Low': base_downtrend * (1 - np.random.uniform(0.01, 0.03, len(dates))),
        'Close': base_downtrend,
        'Volume': np.random.randint(500_000, 5_000_000, len(dates))
    }, index=dates)
    
    ohlc_data = {
        'UPTREND': uptrend_df,
        'DOWNTREND': downtrend_df,
    }
    
    # Run screener
    screener = QuallamaggieScreener(ohlc_data, min_adr=3.0)
    
    # Get shortlist
    shortlist = screener.get_shortlist_dataframe()
    
    print("\n" + "=" * 60)
    print("SHORTLIST")
    print("=" * 60)
    if not shortlist.empty:
        print(shortlist.to_string(index=False))
    else:
        print("No tickers passed all screening criteria.")
    
    # Print detailed report for uptrend stock
    print(screener.get_detailed_report('UPTREND'))


if __name__ == "__main__":
    demo()

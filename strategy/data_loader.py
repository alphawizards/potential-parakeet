"""
Data Loader Module
==================
Fetches price data from yfinance and normalizes US assets to AUD.

CRITICAL: All analysis must be performed on AUD-converted data to capture
true volatility for an Australian investor.

Formula: P_AUD = P_USD * (1 / FX_USD_AUD)
Where FX_USD_AUD = AUDUSD=X from yfinance (how many USD per 1 AUD)
So: P_AUD = P_USD / FX_USD_AUD (since we need AUD value of USD asset)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from scipy import stats

from .config import CONFIG, get_us_tickers, get_asx_tickers, is_us_ticker

warnings.filterwarnings('ignore')


# Type alias for OHLC data dictionary
OHLCDict = Dict[str, pd.DataFrame]


class DataLoader:
    """
    Loads and normalizes financial data for Australian investors.
    
    Key Features:
    - Fetches data from yfinance
    - Converts all US assets to AUD
    - Applies Z-score filtering for data quality
    - Handles missing data gracefully
    """
    
    def __init__(self, start_date: str = None, end_date: str = None):
        """
        Initialize DataLoader.
        
        Args:
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
        """
        self.start_date = start_date or CONFIG.START_DATE
        self.end_date = end_date or CONFIG.END_DATE
        self._fx_data: Optional[pd.Series] = None
        self._raw_prices: Optional[pd.DataFrame] = None
        self._aud_prices: Optional[pd.DataFrame] = None
        
    def fetch_fx_rate(self) -> pd.Series:
        """
        Fetch AUD/USD exchange rate.
        
        Returns:
            pd.Series: Daily FX rate (USD per AUD)
            
        Note: yfinance AUDUSD=X returns "how many USD per 1 AUD"
        To convert USD price to AUD: P_AUD = P_USD / FX
        """
        if self._fx_data is not None:
            return self._fx_data
            
        print(f"Fetching FX data: {CONFIG.FX_TICKER}")
        fx = yf.download(
            CONFIG.FX_TICKER,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        
        if fx.empty:
            raise ValueError(f"No FX data available for {CONFIG.FX_TICKER}")
        
        # Handle multi-level columns from yfinance
        if isinstance(fx.columns, pd.MultiIndex):
            fx_series = fx['Close'][CONFIG.FX_TICKER] if CONFIG.FX_TICKER in fx['Close'].columns else fx['Close'].iloc[:, 0]
        else:
            fx_series = fx['Close']
        
        # Forward fill missing values (weekends, holidays)
        fx_series = fx_series.ffill().bfill()
        
        self._fx_data = fx_series
        print(f"FX data loaded: {len(fx_series)} days, range {fx_series.min():.4f} - {fx_series.max():.4f}")
        
        return fx_series
    
    def fetch_prices(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch adjusted close prices for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            pd.DataFrame: Adjusted close prices (columns = tickers)
        """
        print(f"Fetching price data for {len(tickers)} tickers...")
        
        # Download all tickers at once (more efficient)
        data = yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True  # Use adjusted prices
        )
        
        if data.empty:
            raise ValueError("No price data available")
        
        # Extract Close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Close']
        else:
            # Single ticker case
            prices = data[['Close']]
            prices.columns = tickers
        
        # Forward fill missing values
        prices = prices.ffill()
        
        # Report data quality
        for ticker in prices.columns:
            valid_pct = (prices[ticker].notna().sum() / len(prices)) * 100
            print(f"  {ticker}: {valid_pct:.1f}% valid data")
        
        return prices
    
    def fetch_ohlc(
        self, 
        tickers: List[str],
        convert_to_aud: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLC (Open, High, Low, Close) + Volume data for multiple tickers.
        
        Unlike fetch_prices() which returns only Close prices, this method
        returns complete OHLC data needed for technical analysis and screening.
        
        Args:
            tickers: List of ticker symbols
            convert_to_aud: If True, convert USD prices to AUD (affects OHLC).
                           Volume is NOT converted. Default: False (raw USD).
                           
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping ticker -> DataFrame
                with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                
        Note:
            For screening purposes, running on raw USD data is often preferred
            since we're comparing relative metrics (ADR%, momentum) rather than
            absolute AUD values. Set convert_to_aud=True only if you need
            AUD-denominated prices for position sizing or P&L calculations.
            
        Example:
            >>> loader = DataLoader()
            >>> ohlc = loader.fetch_ohlc(['NVDA', 'AMD', 'TSLA'])
            >>> ohlc['NVDA'].head()
                          Open    High     Low   Close    Volume
            Date
            2024-01-02  495.22  497.82  473.40  476.19  49891200
            ...
        """
        print(f"Fetching OHLC data for {len(tickers)} tickers...")
        
        # Download all tickers at once
        data = yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True,  # Use adjusted prices
            group_by='ticker' if len(tickers) > 1 else 'column'
        )
        
        if data.empty:
            raise ValueError("No OHLC data available")
        
        # Fetch FX rate if conversion needed
        fx_rate = None
        if convert_to_aud:
            fx_rate = self.fetch_fx_rate()
        
        # Build dictionary of DataFrames
        ohlc_dict: Dict[str, pd.DataFrame] = {}
        
        if len(tickers) == 1:
            # Single ticker case - data is not grouped
            ticker = tickers[0]
            df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.ffill()
            
            # Convert to AUD if requested and ticker is US
            if convert_to_aud and is_us_ticker(ticker) and fx_rate is not None:
                df = self._convert_ohlc_to_aud(df, fx_rate)
            
            ohlc_dict[ticker] = df
            valid_pct = (df['Close'].notna().sum() / len(df)) * 100
            print(f"  {ticker}: {valid_pct:.1f}% valid data")
            
        else:
            # Multiple tickers - data is grouped by ticker
            for ticker in tickers:
                try:
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker][['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                        ticker_data = ticker_data.ffill()
                        
                        # Convert to AUD if requested and ticker is US
                        if convert_to_aud and is_us_ticker(ticker) and fx_rate is not None:
                            ticker_data = self._convert_ohlc_to_aud(ticker_data, fx_rate)
                        
                        ohlc_dict[ticker] = ticker_data
                        valid_pct = (ticker_data['Close'].notna().sum() / len(ticker_data)) * 100
                        print(f"  {ticker}: {valid_pct:.1f}% valid data")
                    else:
                        print(f"  {ticker}: No data available")
                except Exception as e:
                    print(f"  {ticker}: Error - {str(e)}")
        
        print(f"\nLoaded OHLC data for {len(ohlc_dict)} tickers")
        
        return ohlc_dict
    
    def _convert_ohlc_to_aud(
        self, 
        ohlc_df: pd.DataFrame, 
        fx_rate: pd.Series
    ) -> pd.DataFrame:
        """
        Convert OHLC prices from USD to AUD.
        
        Formula: P_AUD = P_USD / FX_USD_AUD
        
        Volume is NOT converted (kept in shares, not dollar value).
        
        Args:
            ohlc_df: DataFrame with OHLC columns in USD
            fx_rate: Series of FX rates (USD per AUD)
            
        Returns:
            pd.DataFrame: OHLC with prices in AUD, volume unchanged
        """
        # Align indices
        common_idx = ohlc_df.index.intersection(fx_rate.index)
        
        if len(common_idx) == 0:
            print("Warning: No overlapping dates for FX conversion, returning USD data")
            return ohlc_df
        
        ohlc_aligned = ohlc_df.loc[common_idx].copy()
        fx_aligned = fx_rate.loc[common_idx]
        
        # Convert OHLC prices (not Volume)
        for col in ['Open', 'High', 'Low', 'Close']:
            ohlc_aligned[col] = ohlc_aligned[col] / fx_aligned
        
        return ohlc_aligned
    
    def convert_to_aud(self, usd_prices: pd.DataFrame, fx_rate: pd.Series) -> pd.DataFrame:
        """
        Convert USD prices to AUD.
        
        Formula: P_AUD = P_USD / FX_USD_AUD
        
        Args:
            usd_prices: DataFrame of USD prices
            fx_rate: Series of FX rates (USD per AUD)
            
        Returns:
            pd.DataFrame: Prices in AUD
        """
        # Align indices
        common_idx = usd_prices.index.intersection(fx_rate.index)
        
        if len(common_idx) == 0:
            raise ValueError("No overlapping dates between prices and FX data")
        
        usd_aligned = usd_prices.loc[common_idx]
        fx_aligned = fx_rate.loc[common_idx]
        
        # Convert: P_AUD = P_USD / FX (since FX = USD per AUD)
        aud_prices = usd_aligned.div(fx_aligned, axis=0)
        
        print(f"Converted {len(usd_aligned.columns)} US assets to AUD")
        
        return aud_prices
    
    def apply_zscore_filter(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Z-score filter to remove outlier returns.
        
        Outliers (|z| > threshold) are replaced with NaN, then forward-filled.
        This prevents extreme data errors from affecting analysis.
        
        Args:
            returns: DataFrame of returns
            
        Returns:
            pd.DataFrame: Filtered returns
        """
        threshold = CONFIG.ZSCORE_THRESHOLD
        
        # Calculate z-scores for each column
        zscores = returns.apply(lambda x: stats.zscore(x.dropna()), axis=0)
        
        # Create mask for outliers
        outlier_mask = np.abs(zscores) > threshold
        
        # Count outliers
        outlier_count = outlier_mask.sum().sum()
        total_count = returns.notna().sum().sum()
        
        if outlier_count > 0:
            print(f"Z-score filter: Removed {outlier_count} outliers ({outlier_count/total_count*100:.2f}%)")
        
        # Replace outliers with NaN, then forward fill
        filtered = returns.where(~outlier_mask)
        filtered = filtered.ffill()
        
        return filtered
    
    def load_full_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load complete dataset with AUD normalization.
        
        Returns:
            Tuple of:
            - prices_aud: All prices in AUD
            - returns_aud: Daily returns in AUD (filtered)
        """
        # Get all tickers
        us_tickers = get_us_tickers()
        asx_tickers = get_asx_tickers()
        
        # Fetch FX rate
        fx_rate = self.fetch_fx_rate()
        
        # Fetch US prices and convert to AUD
        if us_tickers:
            us_prices = self.fetch_prices(us_tickers)
            us_prices_aud = self.convert_to_aud(us_prices, fx_rate)
        else:
            us_prices_aud = pd.DataFrame()
        
        # Fetch ASX prices (already in AUD)
        if asx_tickers:
            asx_prices = self.fetch_prices(asx_tickers)
        else:
            asx_prices = pd.DataFrame()
        
        # Combine all prices
        if not us_prices_aud.empty and not asx_prices.empty:
            # Align to common dates
            common_idx = us_prices_aud.index.intersection(asx_prices.index)
            prices_aud = pd.concat([
                us_prices_aud.loc[common_idx],
                asx_prices.loc[common_idx]
            ], axis=1)
        elif not us_prices_aud.empty:
            prices_aud = us_prices_aud
        else:
            prices_aud = asx_prices
        
        # Calculate returns
        returns_aud = prices_aud.pct_change().dropna()
        
        # Apply Z-score filter
        returns_filtered = self.apply_zscore_filter(returns_aud)
        
        # Store for later use
        self._aud_prices = prices_aud
        
        print(f"\nDataset Summary:")
        print(f"  Date range: {prices_aud.index[0]} to {prices_aud.index[-1]}")
        print(f"  Trading days: {len(prices_aud)}")
        print(f"  Assets: {len(prices_aud.columns)}")
        
        return prices_aud, returns_filtered
    
    def load_selective_dataset(self, tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load dataset for specific tickers only.
        
        Args:
            tickers: List of tickers to load
            
        Returns:
            Tuple of (prices_aud, returns_aud)
        """
        us_tickers = [t for t in tickers if is_us_ticker(t)]
        asx_tickers = [t for t in tickers if not is_us_ticker(t)]
        
        fx_rate = self.fetch_fx_rate()
        
        all_prices_aud = pd.DataFrame()
        
        if us_tickers:
            us_prices = self.fetch_prices(us_tickers)
            us_prices_aud = self.convert_to_aud(us_prices, fx_rate)
            all_prices_aud = us_prices_aud
        
        if asx_tickers:
            asx_prices = self.fetch_prices(asx_tickers)
            if not all_prices_aud.empty:
                common_idx = all_prices_aud.index.intersection(asx_prices.index)
                all_prices_aud = pd.concat([
                    all_prices_aud.loc[common_idx],
                    asx_prices.loc[common_idx]
                ], axis=1)
            else:
                all_prices_aud = asx_prices
        
        returns_aud = all_prices_aud.pct_change().dropna()
        returns_filtered = self.apply_zscore_filter(returns_aud)
        
        return all_prices_aud, returns_filtered
    
    def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate (annualized).
        
        Returns:
            float: Annualized risk-free rate
        """
        try:
            # Try to fetch 13-week T-Bill rate
            tbill = yf.download(
                CONFIG.RISK_FREE_TICKER,
                period="5d",
                progress=False
            )
            if not tbill.empty:
                # Convert from percentage to decimal
                if isinstance(tbill.columns, pd.MultiIndex):
                    rate = tbill['Close'].iloc[-1].values[0] / 100
                else:
                    rate = tbill['Close'].iloc[-1] / 100
                return float(rate)
        except Exception:
            pass
        
        # Fallback to config
        return CONFIG.RISK_FREE_RATE
    
    def calculate_currency_contribution(self, 
                                        us_returns_usd: pd.DataFrame,
                                        us_returns_aud: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate how much of AUD returns come from currency vs asset.
        
        Useful for understanding currency risk contribution.
        
        Args:
            us_returns_usd: Returns in USD
            us_returns_aud: Returns in AUD
            
        Returns:
            pd.DataFrame: Currency contribution to returns
        """
        # r_aud ≈ r_usd + r_fx (for small returns)
        # currency_contribution = r_aud - r_usd
        currency_contribution = us_returns_aud - us_returns_usd
        
        return currency_contribution


def demo():
    """Demonstrate data loading functionality."""
    print("=" * 60)
    print("Data Loader Demo")
    print("=" * 60)
    
    # Initialize loader with recent data
    loader = DataLoader(
        start_date="2020-01-01",
        end_date="2024-12-01"
    )
    
    # Load a subset for demo
    demo_tickers = ["SPY", "QQQ", "TLT", "GLD", "IVV.AX", "VAS.AX"]
    
    prices, returns = loader.load_selective_dataset(demo_tickers)
    
    print("\n" + "=" * 60)
    print("Sample Data (Last 5 Days)")
    print("=" * 60)
    print(prices.tail())
    
    print("\n" + "=" * 60)
    print("Return Statistics (Annualized)")
    print("=" * 60)
    
    annual_returns = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_returns / annual_vol
    
    stats_df = pd.DataFrame({
        'Ann. Return': annual_returns,
        'Ann. Volatility': annual_vol,
        'Sharpe Ratio': sharpe
    })
    print(stats_df.round(4))
    
    print("\n" + "=" * 60)
    print("Risk-Free Rate")
    print("=" * 60)
    rf = loader.get_risk_free_rate()
    print(f"Current risk-free rate: {rf*100:.2f}%")


if __name__ == "__main__":
    demo()

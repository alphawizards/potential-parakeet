"""
Unified ASX Data Loader
=======================
Hybrid data ingestion layer that prioritizes Norgate Data for Australian assets
and falls back to yfinance when Norgate is unavailable.

Data Standards:
- Frequency: Daily
- Fields: Adjusted Close (Total Return) and Volume
- Stock Convention: System uses 'BHP.AX', Norgate uses 'BHP'
- Index Convention: System uses '^AXJO', Norgate uses '$XJO'

Usage:
    from strategy.unified_asx_loader import UnifiedASXLoader
    
    loader = UnifiedASXLoader()
    prices_df, volume_df = loader.fetch_data(
        tickers=['BHP.AX', 'CBA.AX', '^AXJO'],
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
import warnings
import logging

warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


# Try to import Norgate Data
try:
    import norgatedata
    from norgatedata import StockPriceAdjustmentType, PaddingType
    HAS_NORGATE = True
    logger.info("Norgate Data library loaded successfully")
except ImportError:
    HAS_NORGATE = False
    logger.warning("Norgate Data not available. Install with: pip install norgatedata")

# yfinance is always available as fallback
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.error("yfinance not available! Install with: pip install yfinance")


# ============================================================================
# INDEX MAPPINGS
# ============================================================================

# yFinance -> Norgate index symbol mapping
INDEX_YFINANCE_TO_NORGATE: Dict[str, str] = {
    '^AXJO': '$XJO',      # S&P/ASX 200
    '^AORD': '$XAO',      # All Ordinaries
    '^ATLI': '$XTL',      # ASX 20
    '^AXMD': '$XMD',      # S&P/ASX Mid Cap 50
    '^AXSM': '$XSO',      # S&P/ASX Small Ordinaries
    '^AXKO': '$XKO',      # S&P/ASX 300
    '^AXMJ': '$XMJ',      # S&P/ASX 300 Metals & Mining
    '^AXEJ': '$XEJ',      # S&P/ASX 200 Energy
    '^AXFJ': '$XFJ',      # S&P/ASX 200 Financials
    '^AXHJ': '$XHJ',      # S&P/ASX 200 Health Care
    '^AXIJ': '$XIJ',      # S&P/ASX 200 Industrials
    '^AXDJ': '$XDJ',      # S&P/ASX 200 Consumer Discretionary
    '^AXSJ': '$XSJ',      # S&P/ASX 200 Consumer Staples
    '^AXUJ': '$XUJ',      # S&P/ASX 200 Utilities
    '^AXJR': '$XJR',      # S&P/ASX 200 A-REIT
    '^AXNJ': '$XNJ',      # S&P/ASX 200 IT
}

# Reverse mapping (Norgate -> yFinance)
INDEX_NORGATE_TO_YFINANCE: Dict[str, str] = {v: k for k, v in INDEX_YFINANCE_TO_NORGATE.items()}


# ============================================================================
# UNIFIED ASX LOADER CLASS
# ============================================================================

class UnifiedASXLoader:
    """
    Unified data loader for ASX stocks and indices.
    
    Prioritizes Norgate Data for high-quality adjusted prices,
    with automatic fallback to yfinance when Norgate is unavailable.
    
    Features:
    - Automatic ticker format conversion (BHP.AX <-> BHP)
    - Index symbol mapping (^AXJO <-> $XJO)
    - Total Return adjusted prices (accounts for dividends/splits)
    - Graceful fallback with detailed logging
    - Source tracking for data provenance
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the UnifiedASXLoader.
        
        Args:
            verbose: If True, print detailed status messages
        """
        self.verbose = verbose
        self._norgate_available: Optional[bool] = None
        self._source_report: Dict[str, str] = {}  # ticker -> 'Norgate' or 'yFinance'
        
    def _log(self, message: str, level: str = 'info'):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            elif level == 'error':
                logger.error(message)
    
    # ========================================================================
    # STEP 1: CONNECTION HEALTH CHECK
    # ========================================================================
    
    def _check_norgate_connection(self) -> bool:
        """
        Check if Norgate Data is available and functional.
        
        Attempts to fetch a test symbol ($XJO - S&P/ASX 200 index)
        to verify the connection is working.
        
        Returns:
            bool: True if Norgate is available and working
        """
        if not HAS_NORGATE:
            self._log("Norgate library not installed", 'warning')
            return False
        
        try:
            # Try to fetch a simple test (S&P/ASX 200 index)
            test_data = norgatedata.price_timeseries(
                '$XJO',
                stock_price_adjustment_setting=StockPriceAdjustmentType.TOTALRETURN,
                padding_setting=PaddingType.NONE,
                limit=5  # Just fetch last 5 days for speed
            )
            
            if test_data is not None and len(test_data) > 0:
                self._log("✓ Norgate Data connection verified")
                return True
            else:
                self._log("Norgate returned empty data for test symbol", 'warning')
                return False
                
        except Exception as e:
            self._log(f"Norgate connection check failed: {e}", 'warning')
            return False
    
    @property
    def norgate_available(self) -> bool:
        """Check and cache Norgate availability."""
        if self._norgate_available is None:
            self._norgate_available = self._check_norgate_connection()
        return self._norgate_available
    
    # ========================================================================
    # STEP 2: TICKER CLASSIFICATION & MAPPING
    # ========================================================================
    
    def _classify_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Classify tickers into stocks and indices.
        
        Args:
            tickers: List of tickers in system format (e.g., 'BHP.AX', '^AXJO')
            
        Returns:
            Tuple of (stock_tickers, index_tickers)
        """
        stocks = []
        indices = []
        
        for ticker in tickers:
            if ticker.startswith('^') or ticker.startswith('$'):
                indices.append(ticker)
            else:
                stocks.append(ticker)
        
        return stocks, indices
    
    def _to_norgate_stock(self, ticker: str) -> str:
        """
        Convert system stock ticker to Norgate format.
        
        Example: 'BHP.AX' -> 'BHP'
        """
        if ticker.endswith('.AX'):
            return ticker[:-3]  # Remove '.AX' suffix
        return ticker
    
    def _from_norgate_stock(self, ticker: str) -> str:
        """
        Convert Norgate stock ticker to system format.
        
        Example: 'BHP' -> 'BHP.AX'
        """
        if not ticker.endswith('.AX') and not ticker.startswith('$'):
            return f"{ticker}.AX"
        return ticker
    
    def _to_norgate_index(self, ticker: str) -> str:
        """
        Convert yFinance index symbol to Norgate format.
        
        Example: '^AXJO' -> '$XJO'
        """
        return INDEX_YFINANCE_TO_NORGATE.get(ticker, ticker)
    
    def _from_norgate_index(self, ticker: str) -> str:
        """
        Convert Norgate index symbol to yFinance format.
        
        Example: '$XJO' -> '^AXJO'
        """
        return INDEX_NORGATE_TO_YFINANCE.get(ticker, ticker)
    
    # ========================================================================
    # STEP 3: NORGATE FETCH (PRIMARY)
    # ========================================================================
    
    def _fetch_norgate(self, 
                       tickers: List[str],
                       start_date: str,
                       end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
        """
        Fetch data from Norgate Data.
        
        Uses TOTALRETURN adjustment for accurate adjusted close prices
        that account for dividends and splits.
        
        Args:
            tickers: List of tickers in system format
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Tuple of (prices_df, volume_df, successful_tickers)
        """
        if not self.norgate_available:
            return pd.DataFrame(), pd.DataFrame(), set()
        
        stocks, indices = self._classify_tickers(tickers)
        
        prices_data = {}
        volume_data = {}
        successful = set()
        
        # Fetch stocks
        for ticker in stocks:
            norgate_ticker = self._to_norgate_stock(ticker)
            try:
                df = norgatedata.price_timeseries(
                    norgate_ticker,
                    stock_price_adjustment_setting=StockPriceAdjustmentType.TOTALRETURN,
                    padding_setting=PaddingType.NONE,
                    start_date=start_date,
                    end_date=end_date,
                    format='pandas-dataframe'
                )
                
                if df is not None and len(df) > 0:
                    # Extract Close (Total Return Adjusted) and Volume
                    system_ticker = self._from_norgate_stock(norgate_ticker)
                    prices_data[system_ticker] = df['Close']
                    if 'Volume' in df.columns:
                        volume_data[system_ticker] = df['Volume']
                    successful.add(ticker)
                    self._source_report[system_ticker] = 'Norgate'
                    
            except Exception as e:
                self._log(f"Norgate fetch failed for {ticker}: {e}", 'warning')
        
        # Fetch indices
        for ticker in indices:
            norgate_ticker = self._to_norgate_index(ticker)
            try:
                df = norgatedata.price_timeseries(
                    norgate_ticker,
                    stock_price_adjustment_setting=StockPriceAdjustmentType.TOTALRETURN,
                    padding_setting=PaddingType.NONE,
                    start_date=start_date,
                    end_date=end_date,
                    format='pandas-dataframe'
                )
                
                if df is not None and len(df) > 0:
                    # Indices use original yfinance symbol as key
                    prices_data[ticker] = df['Close']
                    if 'Volume' in df.columns:
                        volume_data[ticker] = df['Volume']
                    successful.add(ticker)
                    self._source_report[ticker] = 'Norgate'
                    
            except Exception as e:
                self._log(f"Norgate fetch failed for index {ticker}: {e}", 'warning')
        
        # Convert to DataFrames
        prices_df = pd.DataFrame(prices_data)
        volume_df = pd.DataFrame(volume_data)
        
        if not prices_df.empty:
            prices_df.index = pd.to_datetime(prices_df.index)
            prices_df = prices_df.sort_index()
        if not volume_df.empty:
            volume_df.index = pd.to_datetime(volume_df.index)
            volume_df = volume_df.sort_index()
        
        self._log(f"Norgate: Successfully fetched {len(successful)}/{len(tickers)} tickers")
        
        return prices_df, volume_df, successful
    
    # ========================================================================
    # STEP 4: YFINANCE FALLBACK (SECONDARY)
    # ========================================================================
    
    def _fetch_yfinance(self,
                        tickers: List[str],
                        start_date: str,
                        end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch data from yfinance as fallback.
        
        Uses auto_adjust=True to get adjusted close prices.
        
        Args:
            tickers: List of tickers in system format
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Tuple of (prices_df, volume_df)
        """
        if not HAS_YFINANCE:
            self._log("yfinance not available!", 'error')
            return pd.DataFrame(), pd.DataFrame()
        
        if not tickers:
            return pd.DataFrame(), pd.DataFrame()
        
        self._log(f"Falling back to yFinance for: {tickers}")
        
        try:
            # Download all tickers at once
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,  # Ensures 'Close' is Adjusted Close
                threads=True
            )
            
            if data.empty:
                self._log("yfinance returned no data", 'warning')
                return pd.DataFrame(), pd.DataFrame()
            
            # Extract Close and Volume
            if isinstance(data.columns, pd.MultiIndex):
                # Multiple tickers
                prices_df = data['Close']
                volume_df = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else pd.DataFrame()
            else:
                # Single ticker
                prices_df = data[['Close']]
                prices_df.columns = tickers
                if 'Volume' in data.columns:
                    volume_df = data[['Volume']]
                    volume_df.columns = tickers
                else:
                    volume_df = pd.DataFrame()
            
            # Mark sources
            for ticker in prices_df.columns:
                if ticker not in self._source_report:
                    self._source_report[ticker] = 'yFinance'
            
            self._log(f"yFinance: Successfully fetched {len(prices_df.columns)} tickers")
            
            return prices_df, volume_df
            
        except Exception as e:
            self._log(f"yfinance fetch failed: {e}", 'error')
            return pd.DataFrame(), pd.DataFrame()
    
    # ========================================================================
    # STEP 5: MAIN FETCH METHOD (MERGE & ALIGN)
    # ========================================================================
    
    def fetch_data(self,
                   tickers: List[str],
                   start_date: str,
                   end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch price and volume data for ASX tickers.
        
        Prioritizes Norgate Data, automatically falls back to yfinance
        for any tickers that fail to fetch from Norgate.
        
        Args:
            tickers: List of tickers (e.g., ['BHP.AX', 'CBA.AX', '^AXJO'])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            Tuple of:
            - prices_df: DataFrame of Adjusted Close prices (columns=tickers)
            - volume_df: DataFrame of daily volumes (columns=tickers)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Reset source report for this fetch
        self._source_report = {}
        
        self._log(f"\n{'='*60}")
        self._log(f"UnifiedASXLoader: Fetching {len(tickers)} tickers")
        self._log(f"Date Range: {start_date} to {end_date}")
        self._log(f"{'='*60}")
        
        # Try Norgate first
        norgate_prices, norgate_volume, norgate_success = self._fetch_norgate(
            tickers, start_date, end_date
        )
        
        # Identify failed tickers
        failed_tickers = [t for t in tickers if t not in norgate_success]
        
        # Fetch failed tickers from yfinance
        if failed_tickers:
            self._log(f"\nAttempting yFinance fallback for {len(failed_tickers)} tickers")
            yf_prices, yf_volume = self._fetch_yfinance(
                failed_tickers, start_date, end_date
            )
        else:
            yf_prices, yf_volume = pd.DataFrame(), pd.DataFrame()
        
        # Merge dataframes
        prices_df = self._merge_dataframes(norgate_prices, yf_prices)
        volume_df = self._merge_dataframes(norgate_volume, yf_volume)
        
        # Ensure consistent DatetimeIndex
        if not prices_df.empty:
            prices_df.index = pd.to_datetime(prices_df.index)
            prices_df = prices_df.sort_index()
            # Forward fill missing values (weekends, holidays)
            prices_df = prices_df.ffill()
        
        if not volume_df.empty:
            volume_df.index = pd.to_datetime(volume_df.index)
            volume_df = volume_df.sort_index()
            # Fill volume with 0 for missing days
            volume_df = volume_df.fillna(0)
        
        # Print summary
        self._print_summary(prices_df, volume_df)
        
        return prices_df, volume_df
    
    def _merge_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge two DataFrames, preferring df1 for overlapping columns.
        
        Args:
            df1: Primary DataFrame (Norgate data)
            df2: Secondary DataFrame (yFinance data)
            
        Returns:
            Merged DataFrame
        """
        if df1.empty and df2.empty:
            return pd.DataFrame()
        elif df1.empty:
            return df2
        elif df2.empty:
            return df1
        
        # Get overlapping date range
        all_dates = df1.index.union(df2.index).sort_values()
        
        # Reindex both to common dates
        df1_aligned = df1.reindex(all_dates)
        df2_aligned = df2.reindex(all_dates)
        
        # Combine, preferring df1
        combined = df1_aligned.combine_first(df2_aligned)
        
        return combined
    
    def _print_summary(self, prices_df: pd.DataFrame, volume_df: pd.DataFrame):
        """Print a summary of the fetch operation."""
        if prices_df.empty:
            self._log("No data fetched!", 'error')
            return
        
        self._log(f"\n{'='*60}")
        self._log("FETCH SUMMARY")
        self._log(f"{'='*60}")
        self._log(f"Date Range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
        self._log(f"Trading Days: {len(prices_df)}")
        self._log(f"Tickers Fetched: {len(prices_df.columns)}")
        
        # Source breakdown
        norgate_count = sum(1 for v in self._source_report.values() if v == 'Norgate')
        yf_count = sum(1 for v in self._source_report.values() if v == 'yFinance')
        total = norgate_count + yf_count
        
        if total > 0:
            self._log("\nData Sources:")
            self._log(f"  Norgate: {norgate_count} tickers ({norgate_count/total*100:.1f}%)")
            self._log(f"  yFinance: {yf_count} tickers ({yf_count/total*100:.1f}%)")
        
        self._log(f"{'='*60}\n")
    
    def get_source_report(self) -> Dict[str, str]:
        """
        Get the data source for each ticker from the last fetch.
        
        Returns:
            Dict mapping ticker -> 'Norgate' or 'yFinance'
        """
        return self._source_report.copy()
    
    def get_source_statistics(self) -> Dict[str, any]:
        """
        Get statistics about data sources from the last fetch.
        
        Returns:
            Dict with source statistics
        """
        norgate_tickers = [k for k, v in self._source_report.items() if v == 'Norgate']
        yfinance_tickers = [k for k, v in self._source_report.items() if v == 'yFinance']
        total = len(self._source_report)
        
        return {
            'total_tickers': total,
            'norgate_count': len(norgate_tickers),
            'norgate_pct': len(norgate_tickers) / total * 100 if total > 0 else 0,
            'norgate_tickers': norgate_tickers,
            'yfinance_count': len(yfinance_tickers),
            'yfinance_pct': len(yfinance_tickers) / total * 100 if total > 0 else 0,
            'yfinance_tickers': yfinance_tickers,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def fetch_asx_data(tickers: List[str],
                   start_date: str,
                   end_date: str = None,
                   verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to fetch ASX data.
    
    Args:
        tickers: List of ASX tickers (e.g., ['BHP.AX', 'CBA.AX'])
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        verbose: Print status messages
        
    Returns:
        Tuple of (prices_df, volume_df)
    """
    loader = UnifiedASXLoader(verbose=verbose)
    return loader.fetch_data(tickers, start_date, end_date)


def demo():
    """Demonstrate the UnifiedASXLoader."""
    print("=" * 60)
    print("UnifiedASXLoader Demo")
    print("=" * 60)
    
    # Sample tickers (mix of stocks and indices)
    demo_tickers = [
        'BHP.AX',   # BHP Group (mining)
        'CBA.AX',   # Commonwealth Bank
        'CSL.AX',   # CSL Limited (biotech)
        'NAB.AX',   # National Australia Bank
        'WBC.AX',   # Westpac
        '^AXJO',    # S&P/ASX 200 Index
        '^AORD',    # All Ordinaries Index
    ]
    
    loader = UnifiedASXLoader(verbose=True)
    
    prices, volume = loader.fetch_data(
        tickers=demo_tickers,
        start_date='2024-01-01'
    )
    
    if not prices.empty:
        print("\n" + "=" * 60)
        print("Sample Price Data (Last 5 Days)")
        print("=" * 60)
        print(prices.tail())
        
        print("\n" + "=" * 60)
        print("Sample Volume Data (Last 5 Days)")
        print("=" * 60)
        print(volume.tail())
        
        # Source breakdown
        stats = loader.get_source_statistics()
        print("\n" + "=" * 60)
        print("Source Report")
        print("=" * 60)
        print(f"Norgate: {stats['norgate_count']} ({stats['norgate_pct']:.1f}%)")
        print(f"yFinance: {stats['yfinance_count']} ({stats['yfinance_pct']:.1f}%)")
    else:
        print("No data fetched!")


if __name__ == "__main__":
    demo()

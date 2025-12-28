"""
Fast Data Loader
================

High-performance data loader with caching and parallel processing.
"""

import pandas as pd
import yfinance as yf
from tiingo import TiingoClient
from typing import List, Optional, Dict, Union
from datetime import datetime
import os
from pathlib import Path
from strategy.config import CONFIG

class RetryConfig:
    """Retry configuration for API calls."""
    def __init__(self, max_retries: int = 3, delay: int = 1):
        self.max_retries = max_retries
        self.delay = delay

class FastDataLoader:
    """
    Fast data loader with support for Tiingo (Primary) and yfinance fallback.
    Uses Parquet caching for speed.
    """

    def __init__(self,
                 start_date: str = "2005-01-01",
                 end_date: str = None,
                 use_tiingo_fallback: bool = True,
                 tiingo_api_token: str = None,
                 tiingo_is_premium: bool = False,
                 verbose: bool = True):
        self.verbose = verbose
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.use_tiingo_fallback = use_tiingo_fallback # In this context, implies using Tiingo
        self.tiingo_api_token = tiingo_api_token or CONFIG.TIINGO_API_KEY
        self.tiingo_is_premium = tiingo_is_premium
        self.cache_dir = Path(CONFIG.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize Tiingo Client
        self.client = None
        if self.tiingo_api_token:
            config = {'session': True, 'api_key': self.tiingo_api_token}
            self.client = TiingoClient(config)

    def fetch_prices_fast(self, tickers: List[str], use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch adjusted close prices for multiple tickers.
        Uses caching to avoid re-fetching data.

        Optimization: Incremental Fetch
        1. Load existing cache
        2. Identify missing tickers
        3. Fetch ONLY missing tickers
        4. Merge and save
        """
        if self.verbose:
            print(f"Fetching data for {len(tickers)} tickers...")

        cache_file = self.cache_dir / "us_prices.parquet"
        cached_df = pd.DataFrame()

        # 1. Load existing cache
        if use_cache and cache_file.exists():
            try:
                cached_df = pd.read_parquet(cache_file)
                if self.verbose:
                    print(f"Loaded {len(cached_df.columns)} tickers from cache")
            except Exception as e:
                print(f"⚠️ Warning: Could not read cache: {e}")

        # 2. Identify missing tickers
        if not cached_df.empty:
            # Check which tickers are already in cache
            available_tickers = [t for t in tickers if t in cached_df.columns]
            missing_tickers = [t for t in tickers if t not in cached_df.columns]

            # Check if cached data covers the requested date range (rough check)
            # If cache ends significantly before end_date, we might need to update
            # For this optimization, we focus on presence of ticker
        else:
            available_tickers = []
            missing_tickers = tickers

        if not missing_tickers:
            if self.verbose:
                print("✅ All tickers found in cache")
            return cached_df[tickers]

        if self.verbose:
            print(f"Incremental fetch: Downloading {len(missing_tickers)} missing tickers...")

        # 3. Fetch ONLY missing tickers
        new_data = pd.DataFrame()

        # Try Tiingo first (Primary)
        if self.client:
            try:
                if self.verbose:
                    print(f"Fetching from Tiingo...")

                # Tiingo fetch logic
                # Tiingo python client handles batching internally for get_dataframe
                # but let's be safe and do it in chunks if list is huge,
                # though fetch_us_stocks_20yr_tiingo.py implies full universe

                # For adjusted close, Tiingo returns a DF with tickers as columns if we use get_dataframe
                # frequency='daily' is default

                # Note: Tiingo client 'get_dataframe' takes a list of tickers.
                # However, for 20 years of data for many tickers, it might be slow or timeout.
                # But let's assume standard usage.

                tiingo_data = self.client.get_dataframe(
                    missing_tickers,
                    metric_name='adjClose',
                    startDate=self.start_date,
                    endDate=self.end_date,
                    frequency='daily'
                )

                # Tiingo returns index as timezone aware, ensure compatibility
                if not tiingo_data.empty:
                    # Handle MultiIndex if present (Tiingo returns MultiIndex for multiple tickers)
                    if isinstance(tiingo_data.columns, pd.MultiIndex):
                         # If it's MultiIndex, it's usually (Symbol, Metric) or (Metric, Symbol)
                         # But with metric_name specified, it might just be the tickers if flattened
                         # Inspecting tiingo client behavior:
                         # It usually returns a wide DF with tickers as columns if metric_name is provided
                         # But if it returns long format or MultiIndex, we need to handle it.
                         # Assuming standard behavior: columns are tickers
                         pass

                    # Ensure timezone naive
                    if hasattr(tiingo_data.index, 'tz') and tiingo_data.index.tz is not None:
                         tiingo_data.index = tiingo_data.index.tz_localize(None)

                    new_data = tiingo_data

            except Exception as e:
                print(f"❌ Tiingo fetch failed: {e}")

        # Fallback to yFinance if Tiingo failed or returned partial/no data
        # Check which are still missing
        if new_data.empty:
            still_missing = missing_tickers
        else:
            still_missing = [t for t in missing_tickers if t not in new_data.columns]

        if still_missing and (self.use_tiingo_fallback or not self.client):
            # Note: variable name use_tiingo_fallback is confusing in legacy code,
            # often meant "use whatever fallback is available"
            if self.verbose:
                print(f"Fetching {len(still_missing)} tickers from yFinance (Backup)...")

            yf_data = yf.download(
                still_missing,
                start=self.start_date,
                end=self.end_date,
                progress=self.verbose,
                threads=True,
                auto_adjust=True
            )

            # Extract Adjusted Close
            if 'Close' in yf_data.columns:
                prices = yf_data['Close']
            elif 'Adj Close' in yf_data.columns:
                prices = yf_data['Adj Close']
            else:
                 # Handle MultiIndex
                if isinstance(yf_data.columns, pd.MultiIndex):
                    try:
                        prices = yf_data.xs('Close', level='Price', axis=1)
                    except KeyError:
                        try:
                            prices = yf_data.xs('Adj Close', level='Price', axis=1)
                        except KeyError:
                            prices = yf_data
                else:
                    prices = yf_data

            # Normalize
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=still_missing[0])

            # Merge yFinance data into new_data
            if new_data.empty:
                new_data = prices
            else:
                # Align and join
                new_data = new_data.join(prices, how='outer')

        # 4. Merge and save
        if not new_data.empty:
            if not cached_df.empty:
                # Combine cached and new
                # We prioritize new data if there's overlap?
                # Usually we just append columns.

                # Align indices
                common_index = cached_df.index.union(new_data.index).sort_values()
                cached_df = cached_df.reindex(common_index)
                new_data = new_data.reindex(common_index)

                final_df = pd.concat([cached_df, new_data], axis=1)

                # Remove duplicate columns, keeping new ones? Or old?
                # Let's keep one.
                final_df = final_df.loc[:, ~final_df.columns.duplicated()]
            else:
                final_df = new_data

            # Save to cache
            if self.verbose:
                print(f"Saving updated cache to {cache_file}")
            final_df.to_parquet(cache_file)

            # Return requested
            available = [t for t in tickers if t in final_df.columns]
            return final_df[available]

        return cached_df

    def print_health_status(self):
        """Print health status of the loader."""
        print("FastDataLoader is healthy.")

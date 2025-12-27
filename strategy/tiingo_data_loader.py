"""
Tiingo Data Loader
==================
Institutional-grade market data from Tiingo API.

Features:
- Adjusted close prices with corporate actions tracked
- 30+ years of historical data
- Premium: Unlimited tickers, no rate limits
"""

import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List
import time


class TiingoDataLoader:
    """
    Tiingo API data loader with premium support.
    
    Premium benefits:
    - Unlimited tickers
    - Higher rate limits
    - Real-time IEX data
    """
    
    def __init__(self, api_token: str, is_premium: bool = False):
        """
        Initialize Tiingo data loader.
        
        Args:
            api_token: Tiingo API token
            is_premium: True if using premium plan (unlimited tickers)
        """
        self.api_token = api_token
        self.is_premium = is_premium
        self.base_url = "https://api.tiingo.com/tiingo/daily"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {api_token}'
        }
        
        # Rate limits
        self.rate_limit_per_hour = 50000 if not is_premium else 1000000  # Premium gets higher limit
        self.request_count = 0
        self.last_reset = time.time()
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        if not self.is_premium:  # Premium has much higher limits
            current_time = time.time()
            if current_time - self.last_reset >= 3600:  # 1 hour
                self.request_count = 0
                self.last_reset = current_time
            
            if self.request_count >= self.rate_limit_per_hour:
                sleep_time = 3600 - (current_time - self.last_reset)
                if sleep_time > 0:
                    print(f"⚠️  Rate limit reached, sleeping {sleep_time/60:.1f} minutes")
                    time.sleep(sleep_time)
                self.request_count = 0
                self.last_reset = time.time()
            
            self.request_count += 1
    
    def fetch_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch daily adjusted close prices from Tiingo.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with date index and adjusted close column
        """
        self._rate_limit()
        
        url = f"{self.base_url}/{ticker}/prices"
        
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'format': 'json'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index = df.index.tz_localize(None)  # Remove timezone
            
            # Return adjusted close (Tiingo's 'adjClose' field)
            return df[['adjClose']].rename(columns={'adjClose': ticker})
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Tiingo error for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_batch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch multiple tickers (sequential requests).
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with date index and all tickers as columns
        """
        dfs = []
        
        for ticker in tickers:
            df = self.fetch_prices(ticker, start_date, end_date)
            if not df.empty:
                dfs.append(df)
            time.sleep(0.1)  # Small delay between requests
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine all tickers
        return pd.concat(dfs, axis=1)
    
    def get_metadata(self, ticker: str) -> dict:
        """
        Get ticker metadata (name, exchange, etc.).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with ticker metadata
        """
        url = f"{self.base_url}/{ticker}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Error fetching metadata for {ticker}: {e}")
            return {}


def get_tiingo_loader():
    """Get a Tiingo loader with API key from environment."""
    import os
    api_key = os.environ.get('TIINGO_API_KEY', '')
    is_premium = os.environ.get('TIINGO_IS_PREMIUM', 'true').lower() == 'true'
    
    if not api_key:
        raise ValueError("TIINGO_API_KEY environment variable not set")
    
    return TiingoDataLoader(api_token=api_key, is_premium=is_premium)


if __name__ == "__main__":
    # Test the Tiingo loader
    print("="*70)
    print("TIINGO DATA LOADER TEST")
    print("="*70)
    
    # Initialize loader from environment
    import os
    api_key = os.environ.get('TIINGO_API_KEY', 'a49dba6e5f76ba7414cc23aba45fa93f435ad2d5')
    is_premium = os.environ.get('TIINGO_IS_PREMIUM', 'true').lower() == 'true'
    
    loader = TiingoDataLoader(
        api_token=api_key,
        is_premium=is_premium
    )
    
    # Test single ticker
    print("\n📥 Fetching AAPL data...")
    aapl = loader.fetch_prices("AAPL", "2024-12-01", "2024-12-26")
    print(f"✅ AAPL: {len(aapl)} rows")
    print(aapl.head())
    
    # Test multiple tickers
    print("\n📥 Fetching batch (AAPL, MSFT, GOOGL)...")
    batch = loader.fetch_batch(["AAPL", "MSFT", "GOOGL"], "2024-12-01", "2024-12-26")
    print(f"✅ Batch: {len(batch)} rows, {len(batch.columns)} tickers")
    print(batch.head())
    
    # Test metadata
    print("\n📋 Fetching AAPL metadata...")
    meta = loader.get_metadata("AAPL")
    if meta:
        print(f"   Name: {meta.get('name')}")
        print(f"   Exchange: {meta.get('exchangeCode')}")
        print(f"   Start Date: {meta.get('startDate')}")
        print(f"   End Date: {meta.get('endDate')}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

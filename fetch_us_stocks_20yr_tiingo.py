"""
Fetch 20 Years of US Stock Data from Tiingo
============================================
Fetches adjusted close and volume data for US stocks only
from Tiingo API (2005-2025).
"""

from strategy.fast_data_loader import FastDataLoader
from strategy.stock_universe import get_screener_universe, get_us_etfs
from datetime import datetime
import time

print("="*80)
print("TIINGO 20-YEAR US STOCK DATA FETCH")
print("="*80)

# Get US-only tickers (exclude ASX)
print("\n📊 Loading US stock universe...")
all_universe = get_screener_universe()
us_etfs = get_us_etfs()

# Filter out ASX tickers (those ending with .AX)
us_stocks = [t for t in all_universe if not t.endswith('.AX')]
us_tickers = list(set(us_stocks + us_etfs))

print(f"✅ Total US tickers to fetch: {len(us_tickers)}")
print(f"   - US Stocks: {len(us_stocks)}")
print(f"   - US ETFs: {len(us_etfs)}")
print(f"\n⚠️  Excluded ASX tickers (will fetch separately with yFinance)")

# Initialize FastDataLoader with Tiingo PRIMARY
print("\n🔧 Initializing FastDataLoader...")
loader = FastDataLoader(
    start_date="2005-01-01",  # 20 years ago
    end_date="2025-12-26",
    use_tiingo_fallback=True,
    tiingo_api_token="a49dba6e5f76ba7414cc23aba45fa93f435ad2d5",
    tiingo_is_premium=True
)

# Fetch data
print(f"\n🚀 Starting 20-year historical data fetch...")
print(f"   Date range: 2005-01-01 to 2025-12-26 (20 years)")
print(f"   Expected Tiingo requests: {len(us_tickers)}")
print(f"   Expected time: ~{len(us_tickers) * 0.15 / 60:.1f} minutes")
print(f"   Quota usage: {len(us_tickers)/150000*100:.2f}% of daily limit")
print()

start_time = time.time()

# Fetch prices (use cache if available)
prices = loader.fetch_prices_fast(us_tickers)

end_time = time.time()
elapsed = end_time - start_time

# Results
print("\n" + "="*80)
print("FETCH RESULTS")
print("="*80)
print(f"✅ Success! Fetched {len(prices.columns)} out of {len(us_tickers)} tickers")
print(f"⏱️  Time elapsed: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
print(f"📊 Data shape: {prices.shape[0]} days × {prices.shape[1]} tickers")
print(f"📅 Date range: {prices.index.min().date()} to {prices.index.max().date()}")
print(f"📈 Years of data: {(prices.index.max() - prices.index.min()).days / 365.25:.1f} years")
print(f"🎯 Success rate: {len(prices.columns)/len(us_tickers)*100:.1f}%")

# Show sample tickers
print(f"\n📋 Sample tickers fetched:")
print(f"   {list(prices.columns[:10])}")

# Show sample data
print(f"\n📈 Sample data (last 5 days, first 5 tickers):")
print(prices.iloc[-5:, :5])

# Health status
print("\n" + "="*80)
print("LOADER HEALTH STATUS")
print("="*80)
loader.print_health_status()

# Tiingo verification
print("\n" + "="*80)
print("TIINGO REQUEST VERIFICATION")
print("="*80)
print(f"Expected Tiingo requests: {len(us_tickers)}")
print(f"Successful fetches: {len(prices.columns)}")
print(f"Failed fetches: {len(us_tickers) - len(prices.columns)}")
print(f"Daily quota used: {len(us_tickers)/150000*100:.2f}%")
print(f"Remaining quota: {150000 - len(us_tickers):,} requests")
print()
print("⚠️  IMPORTANT: Verify actual usage on Tiingo dashboard:")
print("   URL: https://www.tiingo.com/account/usage")
print()

# Cache info
from pathlib import Path
cache_dir = Path("cache")
if cache_dir.exists():
    cache_files = list(cache_dir.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in cache_files) / (1024**2)
    print(f"💾 Cache statistics:")
    print(f"   Files: {len(cache_files)}")
    print(f"   Total size: {total_size:.1f} MB")
    print(f"   Location: {cache_dir.absolute()}")

print()
print("="*80)
print("✅ 20-YEAR US STOCK DATA FETCH COMPLETE")
print("="*80)
print(f"\nYou now have {len(prices.columns)} US tickers with {prices.shape[0]:,} days of data!")
print(f"Ready for backtesting! 🚀")

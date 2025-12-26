"""
US Ticker Data Loader
======================
Fetches comprehensive US ticker lists from GitHub repository:
https://github.com/rreichel3/US-Stock-Symbols

Maintains JSON files for NYSE, NASDAQ, and AMEX exchanges.
"""

import json
import urllib.request
from typing import List, Dict


def get_us_tickers_from_github() -> Dict[str, List[str]]:
    """
    Fetch all US ticker symbols from GitHub repository.
    
    Returns:
        Dictionary with exchange names as keys and ticker lists as values
    """
    base_url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main"
    
    exchanges = {
        'NYSE': f"{base_url}/nyse/nyse_tickers.json",
        'NASDAQ': f"{base_url}/nasdaq/nasdaq_tickers.json",
        'AMEX': f"{base_url}/amex/amex_tickers.json"
    }
    
    tickers = {}
    
    for exchange_name, url in exchanges.items():
        try:
            response = urllib.request.urlopen(url)
            data = json.loads(response.read())
            tickers[exchange_name] = sorted(data)
            print(f"✅ {exchange_name}: {len(data):,} tickers")
        except Exception as e:
            print(f"❌ Error fetching {exchange_name}: {e}")
            tickers[exchange_name] = []
    
    return tickers


def get_all_us_tickers() -> List[str]:
    """
    Get all unique US tickers across all exchanges.
    
    Returns:
        Sorted list of unique ticker symbols
    """
    data = get_us_tickers_from_github()
    all_tickers = []
    
    for exchange, tickers in data.items():
        all_tickers.extend(tickers)
    
    # Return unique, sorted tickers
    unique_tickers = sorted(set(all_tickers))
    print(f"\n📊 Total Unique Tickers: {len(unique_tickers):,}")
    
    return unique_tickers


def filter_common_stocks(tickers: List[str]) -> List[str]:
    """
    Filter out preferred shares, warrants, and special securities.
    
    Keeps only common stocks (removes tickers with: ^, /, -, ., special suffixes)
    
    Args:
        tickers: List of all tickers
        
    Returns:
        Filtered list of common stock tickers only
    """
    common_stocks = []
    
    for ticker in tickers:
        # Skip if contains special characters indicating non-common stock
        if any(char in ticker for char in ['^', '/', '-']):
            continue
        
        # Skip warrants (W suffix) and units (U suffix)
        if ticker.endswith('W') or ticker.endswith('U'):
            # But allow genuine tickers ending in W/U
            if len(ticker) > 3:  # e.g., KULRW, ABVEW are likely warrants
                continue
        
        # Skip preferred shares (ends with digits or special letters)
        if ticker and ticker[-1].isdigit():
            continue
            
        common_stocks.append(ticker)
    
    return common_stocks


def get_liquid_tickers(min_length: int = 1, max_length: int = 5) -> List[str]:
    """
    Get only liquid common stocks (filter by ticker length as proxy).
    
    Args:
        min_length: Minimum ticker length (default 1)
        max_length: Maximum ticker length (default 5, excludes many ETFs/special securities)
        
    Returns:
        Filtered list of likely liquid stocks
    """
    all_tickers = get_all_us_tickers()
    common = filter_common_stocks(all_tickers)
    
    # Filter by length
    liquid = [t for t in common if min_length <= len(t) <= max_length]
    
    print(f"🔎 Filtered Common Stocks: {len(liquid):,} (from {len(all_tickers):,} total)")
    
    return liquid


if __name__ == "__main__":
    print("="*70)
    print("US TICKER DATA LOADER")
    print("="*70)
    print()
    
    # Get all tickers by exchange
    print("📥 Fetching from GitHub...")
    data = get_us_tickers_from_github()
    
    # Get all unique tickers
    print()
    all_us = get_all_us_tickers()
    
    # Get common stocks only
    print("\n🧹 Filtering common stocks...")
    common_stocks = filter_common_stocks(all_us)
    print(f"   Common Stocks: {len(common_stocks):,}")
    
    # Get liquid stocks (1-5 character tickers)
    print("\n🌊 Filtering for liquidity (1-5 char tickers)...")
    liquid = get_liquid_tickers()
    
    # Save to file
    output_file = "data/us_tickers_full.json"
    with open(output_file, 'w') as f:
        json.dump({
            'byExchange': data,
            'allUnique': all_us,
            'commonStocks': common_stocks,
            'liquidStocks': liquid,
            'stats': {
                'totalUnique': len(all_us),
                'commonStocks': len(common_stocks),
                'liquidStocks': len(liquid),
                'nyse': len(data['NYSE']),
                'nasdaq': len(data['NASDAQ']),
                'amex': len(data['AMEX'])
            }
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {output_file}")
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Tickers: {len(all_us):,}")
    print(f"Common Stocks: {len(common_stocks):,}")
    print(f"Liquid Stocks (1-5 char): {len(liquid):,}")
    print("="*70)

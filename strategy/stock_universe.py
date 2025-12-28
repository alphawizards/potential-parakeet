"""
Stock Universe Definitions
==========================

Defines the stock universes for the strategy package.
Includes S&P 500, NASDAQ 100, and ASX 200.
"""

from typing import List
import requests
import pandas as pd
from io import StringIO
import functools

@functools.lru_cache(maxsize=1)
def get_sp500_tickers() -> List[str]:
    """Get S&P 500 tickers from GitHub (cached)."""
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        sp500_df = pd.read_csv(StringIO(response.text))
        return sorted(list(set(sp500_df['Symbol'].str.replace('.', '-').tolist())))
    except Exception as e:
        print(f"⚠️ Failed to fetch S&P 500 from GitHub: {e}")
        # Fallback to a sample
        return ["AAPL", "MSFT", "AMZN", "GOOG", "TSLA", "SPY", "QQQ", "GLD", "TLT", "NVDA", "META", "BRK-B", "JPM", "V"]

def get_nasdaq100_tickers() -> List[str]:
    """Get NASDAQ 100 tickers (sample for now)."""
    return ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA", "AVGO", "COST", "PEP"]

def get_us_etfs() -> List[str]:
    """Get US ETFs."""
    return ["SPY", "QQQ", "GLD", "TLT", "IWM", "EEM", "VTI", "BND"]

def get_asx200_tickers() -> List[str]:
    """Get ASX 200 tickers (sample)."""
    return ["BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "WES.AX", "MQG.AX", "WOW.AX", "TLS.AX"]

def get_asx_etfs() -> List[str]:
    """Get ASX ETFs."""
    return ["VAS.AX", "VGS.AX", "STW.AX", "IOZ.AX", "IVV.AX"]

def get_core_etfs() -> List[str]:
    """Get core ETFs."""
    return get_us_etfs() + get_asx_etfs()

def get_screener_universe() -> List[str]:
    """Get the full screener universe."""
    return sorted(list(set(get_sp500_tickers() + get_us_etfs() + get_asx200_tickers())))

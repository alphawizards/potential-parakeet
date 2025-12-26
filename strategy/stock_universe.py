"""
Stock Universe Configuration for Live Screeners
================================================
Defines the universe of stocks and ETFs for live recommendations.
Includes S&P 500, NASDAQ 100, ASX 200, and core ETFs.

Usage:
    from strategy.stock_universe import get_screener_universe, get_core_etfs
    
    universe = get_screener_universe()
    print(f"Total tickers: {len(universe)}")
"""

import pandas as pd
import requests
from io import StringIO
from typing import List, Set, Dict
from functools import lru_cache
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Headers to avoid 403 Forbidden from Wikipedia
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


# ============================================================================
# CORE ETFs (21 Total)
# ============================================================================

US_ETFS = [
    {"ticker": "SPY", "name": "S&P 500", "category": "US Equity"},
    {"ticker": "QQQ", "name": "Nasdaq 100", "category": "US Tech"},
    {"ticker": "VEA", "name": "Developed ex-US", "category": "International"},
    {"ticker": "VWO", "name": "Emerging Markets", "category": "Emerging Markets"},
    {"ticker": "TLT", "name": "US Long Treasuries", "category": "Bonds"},
    {"ticker": "IEF", "name": "US Intermediate Treasuries", "category": "Bonds"},
    {"ticker": "GLD", "name": "Gold", "category": "Commodities"},
    {"ticker": "DBC", "name": "Commodities Broad", "category": "Commodities"},
    {"ticker": "VNQ", "name": "US REITs", "category": "Real Estate"},
]

ASX_ETFS = [
    # Original 7
    {"ticker": "VAS.AX", "name": "Australian Shares", "category": "AU Equity"},
    {"ticker": "IVV.AX", "name": "S&P 500 (AUD)", "category": "US Equity"},
    {"ticker": "VGS.AX", "name": "Int'l Shares (AUD)", "category": "International"},
    {"ticker": "VGE.AX", "name": "Emerging Markets (AUD)", "category": "Emerging Markets"},
    {"ticker": "VAF.AX", "name": "Australian Bonds", "category": "Bonds"},
    {"ticker": "GOLD.AX", "name": "Gold (AUD)", "category": "Commodities"},
    {"ticker": "IHVV.AX", "name": "S&P 500 Hedged", "category": "US Equity Hedged"},
    # Additional 5 requested
    {"ticker": "XMET.AX", "name": "Critical Metals", "category": "Commodities"},
    {"ticker": "NDQ.AX", "name": "Nasdaq 100 (AUD)", "category": "US Tech"},
    {"ticker": "ETPMAG.AX", "name": "Physical Silver", "category": "Commodities"},
    {"ticker": "RSSL.AX", "name": "Russell 2000 (AUD)", "category": "US Small Cap"},
    {"ticker": "QAU.AX", "name": "Quality Shares", "category": "AU Equity"},
]


def get_core_etfs() -> List[str]:
    """Get all 21 core ETFs."""
    us = [e["ticker"] for e in US_ETFS]
    asx = [e["ticker"] for e in ASX_ETFS]
    return us + asx


def get_us_etfs() -> List[str]:
    """Get US-listed ETFs only."""
    return [e["ticker"] for e in US_ETFS]


def get_asx_etfs() -> List[str]:
    """Get ASX-listed ETFs only."""
    return [e["ticker"] for e in ASX_ETFS]


def get_etf_info() -> List[Dict]:
    """Get detailed ETF information."""
    return US_ETFS + ASX_ETFS


def _fetch_wikipedia_table(url: str) -> List[pd.DataFrame]:
    """Fetch tables from Wikipedia with proper headers."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        return tables
    except Exception as e:
        raise Exception(f"Failed to fetch: {e}")


# ============================================================================
# S&P 500 CONSTITUENTS
# ============================================================================

@lru_cache(maxsize=1)
def get_sp500_tickers() -> List[str]:
    """
    Fetch S&P 500 constituents from Wikipedia.
    Results are cached for the session.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = _fetch_wikipedia_table(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Clean ticker symbols (some have . instead of -)
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"[Universe] Loaded {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"[Universe] Failed to fetch S&P 500: {e}")
        return _get_sp500_fallback()


def _get_sp500_fallback() -> List[str]:
    """Fallback list of major S&P 500 components."""
    return [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 'BRK-B', 'UNH',
        'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
        'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'WMT', 'MCD', 'CSCO', 'TMO', 'ACN',
        'ABT', 'CRM', 'DHR', 'ADBE', 'NEE', 'VZ', 'NKE', 'TXN', 'WFC', 'PM',
        'CMCSA', 'UPS', 'RTX', 'T', 'HON', 'INTC', 'QCOM', 'BA', 'SPGI', 'LOW'
    ]


# ============================================================================
# NASDAQ 100 CONSTITUENTS
# ============================================================================

NASDAQ_100_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'GOOG', 'AVGO', 'COST',
    'PEP', 'CSCO', 'ADBE', 'NFLX', 'AMD', 'CMCSA', 'TMUS', 'INTC', 'INTU', 'QCOM',
    'TXN', 'AMGN', 'ISRG', 'HON', 'AMAT', 'BKNG', 'SBUX', 'VRTX', 'LRCX', 'ADP',
    'GILD', 'MDLZ', 'ADI', 'REGN', 'PANW', 'PYPL', 'KLAC', 'SNPS', 'MU', 'CDNS',
    'MELI', 'CSX', 'MAR', 'ORLY', 'MNST', 'ASML', 'CRWD', 'NXPI', 'CTAS', 'PCAR',
    'LULU', 'MRVL', 'AEP', 'WDAY', 'KDP', 'PAYX', 'FTNT', 'CHTR', 'CPRT', 'ADSK',
    'EXC', 'ROST', 'KHC', 'MRNA', 'AZN', 'DXCM', 'ODFL', 'IDXX', 'FAST', 'CTSH',
    'EA', 'VRSK', 'XEL', 'GEHC', 'ON', 'CSGP', 'BKR', 'CCEP', 'ANSS', 'FANG',
    'BIIB', 'CEG', 'CDW', 'TTWO', 'DDOG', 'WBD', 'ZS', 'TEAM', 'GFS', 'ILMN',
    'ALGN', 'WBA', 'ENPH', 'SIRI', 'LCID', 'RIVN', 'JD', 'PDD', 'BIDU', 'ZM'
]


def get_nasdaq100_tickers() -> List[str]:
    """Get NASDAQ 100 constituents."""
    print(f"[Universe] Loaded {len(NASDAQ_100_TICKERS)} NASDAQ 100 tickers")
    return NASDAQ_100_TICKERS.copy()


# ============================================================================
# ASX 200 CONSTITUENTS
# ============================================================================

@lru_cache(maxsize=1)
def get_asx200_tickers() -> List[str]:
    """
    Fetch ASX 200 constituents from Wikipedia.
    Results are cached for the session.
    """
    try:
        url = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
        tables = _fetch_wikipedia_table(url)
        # Find the table with ticker codes
        for table in tables:
            if 'Code' in table.columns:
                tickers = table['Code'].tolist()
                # Add .AX suffix for yfinance
                tickers = [f"{t}.AX" for t in tickers if isinstance(t, str)]
                print(f"[Universe] Loaded {len(tickers)} ASX 200 tickers")
                return tickers
        # Fallback if Code column not found
        return _get_asx200_fallback()
    except Exception as e:
        print(f"[Universe] Failed to fetch ASX 200: {e}")
        return _get_asx200_fallback()


def _get_asx200_fallback() -> List[str]:
    """Fallback list of major ASX 200 components."""
    return [
        'BHP.AX', 'CBA.AX', 'CSL.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX', 'MQG.AX', 'WES.AX',
        'TLS.AX', 'WOW.AX', 'RIO.AX', 'FMG.AX', 'TCL.AX', 'WDS.AX', 'GMG.AX', 'ALL.AX',
        'REA.AX', 'COL.AX', 'QBE.AX', 'STO.AX', 'JHX.AX', 'SHL.AX', 'CPU.AX', 'NCM.AX',
        'QAN.AX', 'XRO.AX', 'SOL.AX', 'ORG.AX', 'AMC.AX', 'ASX.AX', 'IAG.AX', 'MIN.AX',
        'RHC.AX', 'APA.AX', 'TWE.AX', 'SGP.AX', 'SUN.AX', 'MPL.AX', 'AGL.AX', 'BXB.AX'
    ]


# ============================================================================
# COMBINED UNIVERSE
# ============================================================================

def get_screener_universe(
    include_sp500: bool = True,
    include_nasdaq100: bool = True,
    include_asx200: bool = True,
    include_etfs: bool = True
) -> List[str]:
    """
    Get the complete screener universe.
    
    Args:
        include_sp500: Include S&P 500 constituents
        include_nasdaq100: Include NASDAQ 100 constituents
        include_asx200: Include ASX 200 constituents
        include_etfs: Include 21 core ETFs
        
    Returns:
        List of unique ticker symbols
    """
    universe: Set[str] = set()
    
    if include_sp500:
        universe.update(get_sp500_tickers())
    
    if include_nasdaq100:
        universe.update(get_nasdaq100_tickers())
    
    if include_asx200:
        universe.update(get_asx200_tickers())
    
    if include_etfs:
        universe.update(get_core_etfs())
    
    # Sort: US tickers first, then ASX
    us_tickers = sorted([t for t in universe if not t.endswith('.AX')])
    asx_tickers = sorted([t for t in universe if t.endswith('.AX')])
    
    combined = us_tickers + asx_tickers
    print(f"[Universe] Total screener universe: {len(combined)} tickers")
    
    return combined


def get_universe_summary() -> Dict:
    """Get summary statistics of the screener universe."""
    sp500 = get_sp500_tickers()
    nasdaq = get_nasdaq100_tickers()
    asx200 = get_asx200_tickers()
    etfs = get_core_etfs()
    full = get_screener_universe()
    
    return {
        "sp500_count": len(sp500),
        "nasdaq100_count": len(nasdaq),
        "asx200_count": len(asx200),
        "etf_count": len(etfs),
        "total_unique": len(full),
        "us_tickers": len([t for t in full if not t.endswith('.AX')]),
        "asx_tickers": len([t for t in full if t.endswith('.AX')]),
        "etf_list": etfs,
    }


# ============================================================================
# DEMO / CLI
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Stock Universe Configuration")
    print("=" * 60)
    
    summary = get_universe_summary()
    
    print(f"\nIndex Coverage:")
    print(f"  S&P 500:    {summary['sp500_count']} tickers")
    print(f"  NASDAQ 100: {summary['nasdaq100_count']} tickers")
    print(f"  ASX 200:    {summary['asx200_count']} tickers")
    
    print(f"\nETFs ({summary['etf_count']} total):")
    for etf in get_etf_info():
        print(f"  {etf['ticker']:12} - {etf['name']:30} [{etf['category']}]")
    
    print(f"\nTotal Universe: {summary['total_unique']} unique tickers")
    print(f"  US:  {summary['us_tickers']}")
    print(f"  ASX: {summary['asx_tickers']}")

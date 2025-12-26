"""
Cache Health Monitor
====================
Provides health monitoring and data quality validation for FastDataLoader cache.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

# Cache directory
CACHE_DIR = Path("cache")


class CacheHealthMonitor:
    """Monitor health and quality of cached data."""
    
    def check_cache_health(self, detailed: bool = False) -> Dict:
        """
        Check health status of all cached data sources.
        
        Returns:
            Dictionary with health metrics and recommendations
        """
        health = {
            'overall_status': 'healthy',
            'score': 100,
            'us_stocks': {},
            'etfs': {},
            'vix': {},
            'gold': {},
            'btc': {},
            'recommendations': []
        }
        
        issues = 0
        warnings = 0
        
        # Check US Stocks
        stock_files = [
            "prices_80729c86b695.parquet",
            "historical_delisted_tickers_20yr.parquet",
            "nasdaq100_additional_tickers_21yr.parquet",
            "sp500_additional_tickers_21yr.parquet"
        ]
        
        stock_health = self._check_data_source(
            stock_files, 
            'US Stocks',
            expected_min_tickers=500,
            expected_min_days=5000
        )
        health['us_stocks'] = stock_health
        if stock_health['status'] == 'critical':
            issues += 1
        elif stock_health['status'] == 'degraded':
            warnings += 1
        
        # Check VIX
        vix_health = self._check_data_source(
            ["vix_yfinance.parquet"],
            'VIX Index',
            expected_min_days=5000
        )
        health['vix'] = vix_health
        if vix_health['status'] == 'critical':
            issues += 1
        elif vix_health['status'] == 'degraded':
            warnings += 1
        
       # Check Gold
        gold_health = self._check_data_source(
            ["gold_tiingo_20yr.parquet"],
            'Gold (GLD)',
            expected_min_days=5000
        )
        health['gold'] = gold_health
        if gold_health['status'] == 'critical':
            issues += 1
        elif gold_health['status'] == 'degraded':
            warnings += 1
        
        # Check BTC
        btc_health = self._check_data_source(
            ["btc_usd_yfinance.parquet"],
            'Bitcoin',
            expected_min_days=2000
        )
        health['btc'] = btc_health
        if btc_health['status'] == 'critical':
            issues += 1
        elif btc_health['status'] == 'degraded':
            warnings += 1
        
        # Check ETFs
        etf_files = [
            'vas_ax_yfinance.parquet',
            'a200_ax_yfinance.parquet',
            'ioz_ax_yfinance.parquet',
            'stw_ax_yfinance.parquet',
            'vgs_ax_yfinance.parquet',
            'vge_ax_yfinance.parquet'
        ]
        etf_health = self._check_data_source(
            etf_files,
            'ASX ETFs',
            expected_min_tickers=5,
            expected_min_days=1000
        )
        health['etfs'] = etf_health
        if etf_health['status'] == 'critical':
            issues += 1
        elif etf_health['status'] == 'degraded':
            warnings += 1
        
        # Calculate overall score
        if issues > 2:
            health['overall_status'] = 'critical'
            health['score'] = 30
        elif issues > 0:
            health['overall_status'] = 'degraded'
            health['score'] = 60
        elif warnings > 2:
            health['overall_status'] = 'degraded'
            health['score'] = 75
        elif warnings > 0:
            health['score'] = 85
        
        # Generate recommendations
        if issues > 0:
            health['recommendations'].append(
                "🚨 CRITICAL: Some data sources missing or outdated. Run data refresh."
            )
        if warnings > 0:
            health['recommendations'].append(
                "⚠️ WARNING: Some data sources need updating. Consider refreshing cache."
            )
        
        if stock_health.get('days_old', 0) > 7:
            health['recommendations'].append(
                f"Data is {stock_health['days_old']} days old. Refresh recommended."
            )
        
        return health
    
    def _check_data_source(self, files: List[str], name: str, 
                           expected_min_tickers: int = None,
                           expected_min_days: int = None) -> Dict:
        """Check health of specific data source."""
        
        result = {
            'name': name,
            'status': 'healthy',
            'files_found': 0,
            'files_expected': len(files),
            'tickers': 0,
            'days': 0,
            'date_range': None,
            'days_old': None,
            'issues': []
        }
        
        dfs = []
        for filename in files:
            filepath = CACHE_DIR / filename
            if filepath.exists():
                result['files_found'] += 1
                try:
                    df = pd.read_parquet(filepath)
                    dfs.append(df)
                except Exception as e:
                    result['issues'].append(f"Error reading {filename}: {str(e)[:50]}")
            else:
                result['issues'].append(f"Missing file: {filename}")
        
        if not dfs:
            result['status'] = 'critical'
            result['issues'].append(f"No data files found for {name}")
            return result
        
        # Combine data
        combined = pd.concat(dfs, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        result['tickers'] = len(combined.columns) if len(combined.shape) > 1 else 1
        result['days'] = len(combined)
        
        if len(combined) > 0:
            result['date_range'] = (
                combined.index.min().strftime('%Y-%m-%d'),
                combined.index.max().strftime('%Y-%m-%d')
            )
            result['days_old'] = (datetime.now().date() - combined.index.max().date()).days
        
        # Check against expectations
        if expected_min_tickers and result['tickers'] < expected_min_tickers:
            result['status'] = 'degraded'
            result['issues'].append(
                f"Only {result['tickers']} tickers (expected >={expected_min_tickers})"
            )
        
        if expected_min_days and result['days'] < expected_min_days:
            result['status'] = 'degraded'
            result['issues'].append(
                f"Only {result['days']} days of data (expected >={expected_min_days})"
            )
        
        # Check freshness
        if result['days_old'] is not None:
            if result['days_old'] > 7:
                result['status'] = 'degraded'
                result['issues'].append(f"Data is {result['days_old']} days old")
            elif result['days_old'] > 14:
                result['status'] = 'critical'
                result['issues'].append(f"Data critically outdated ({result['days_old']} days)")
        
        return result
    
    def validate_data_quality(self, df: pd.DataFrame, name: str = "dataset") -> Dict:
        """
        Validate data quality with comprehensive checks.
        
        Returns:
            Dictionary with validation results including score and issues
        """
        result = {
            'passed': True,
            'score': 100,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        if df.empty:
            result['passed'] = False
            result['score'] = 0
            result['issues'].append(f"{name} is empty")
            return result
        
        # Check for NaN values
        total_cells = df.shape[0] * df.shape[1]
        nan_cells = df.isna().sum().sum()
        nan_pct = (nan_cells / total_cells) * 100 if total_cells > 0 else 0
        
        result['stats']['total_cells'] = total_cells
        result['stats']['nan_cells'] = nan_cells
        result['stats']['nan_percentage'] = nan_pct
        
        if nan_pct > 50:
            result['passed'] = False
            result['score'] -= 50
            result['issues'].append(f"High NaN rate: {nan_pct:.1f}%")
        elif nan_pct > 20:
            result['score'] -= 20
            result['warnings'].append(f"Moderate NaN rate: {nan_pct:.1f}%")
        elif nan_pct > 5:
            result['score'] -= 5
            result['warnings'].append(f"Some NaN values: {nan_pct:.1f}%")
        
        # Check for infinite values
        if df.select_dtypes(include=[np.number]).apply(lambda x: np.isinf(x).any()).any():
            result['passed'] = False
            result['score'] -= 30
            result['issues'].append("Contains infinite values")
        
        # Check for duplicate indices
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            result['score'] -= 15
            result['warnings'].append(f"{dup_count} duplicate index entries")
        
        # Check for duplicate columns
        if df.columns.duplicated().any():
            dup_count = df.columns.duplicated().sum()
            result['score'] -= 10
            result['warnings'].append(f"{dup_count} duplicate column names")
        
        # Final score
        result['score'] = max(0, result['score'])
        if result['score'] < 50:
            result['passed'] = False
        
        return result
    
    def print_health_report(self):
        """Print comprehensive health report."""
        health = self.check_cache_health()
        
        print("\n" + "="*70)
        print("🏥 CACHE HEALTH REPORT")
        print("="*70)
        
        # Overall status
        status_icon = "✅" if health['overall_status'] == 'healthy' else "⚠️" if health['overall_status'] == 'degraded' else "❌"
        print(f"\n{status_icon} Overall Status: {health['overall_status'].upper()}")
        print(f"   Health Score: {health['score']}/100")
        
        # Data sources
        print("\n" + "-"*70)
        print("Data Sources:")
        print("-"*70)
        
        for source_key in ['us_stocks', 'vix', 'gold', 'btc', 'etfs']:
            source = health[source_key]
            if not source:
                continue
            
            status_icon = "✅" if source['status'] == 'healthy' else "⚠️" if source['status'] == 'degraded' else "❌"
            print(f"\n{status_icon} {source['name']}:")
            print(f"   Status: {source['status']}")
            print(f"   Files: {source['files_found']}/{source['files_expected']}")
            
            if source.get('tickers'):
                print(f"   Tickers: {source['tickers']}")
            
            print(f"   Days: {source['days']}")
            
            if source.get('date_range'):
                print(f"   Range: {source['date_range'][0]} to {source['date_range'][1]}")
                print(f"   Age: {source['days_old']} days old")
            
            if source.get('issues'):
                for issue in source['issues']:
                    print(f"   ⚠️  {issue}")
        
        # Recommendations
        if health['recommendations']:
            print("\n" + "-"*70)
            print("Recommendations:")
            print("-"*70)
            for i, rec in enumerate(health['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Test health monitoring
    monitor = CacheHealthMonitor()
    monitor.print_health_report()

"""
Pairs Trading Strategy
======================
Integrated pairs trading using DBSCAN clustering and Kalman filtering.

Combines:
1. ClusteringEngine for universe selection (similar assets)
2. KalmanHedgeRatio for dynamic hedging
3. Z-score based entry/exit signals
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

from .clustering import ClusteringEngine, ClusteringResult
from .kalman import KalmanHedgeRatio, KalmanResult


@dataclass
class PairPosition:
    """Active pair position."""
    ticker_y: str
    ticker_x: str
    cluster_id: int
    direction: int  # 1 = long spread, -1 = short spread
    entry_zscore: float
    entry_date: pd.Timestamp
    hedge_ratio: float


@dataclass
class PairsStrategyResult:
    """Result from pairs strategy scan."""
    active_pairs: List[Dict]
    all_pair_signals: pd.DataFrame
    cluster_result: ClusteringResult
    metadata: dict


class PairsStrategy:
    """
    Statistical Arbitrage Pairs Trading Strategy.
    
    Pipeline:
    1. Cluster stocks by factor loadings (DBSCAN)
    2. Identify candidate pairs within clusters
    3. Estimate dynamic hedge ratios (Kalman)
    4. Generate z-score based trading signals
    
    Attributes:
        clustering_engine: ClusteringEngine for universe selection
        kalman: KalmanHedgeRatio for hedge estimation
        entry_zscore: Z-score threshold to enter trades
        exit_zscore: Z-score threshold to exit trades
    """
    
    def __init__(
        self,
        n_components: int = 10,
        eps: float = 0.5,
        min_samples: int = 3,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        max_pairs_per_cluster: int = 5,
        min_half_life: int = 5,
        max_half_life: int = 60
    ):
        """
        Initialize Pairs Strategy.
        
        Args:
            n_components: PCA components for clustering
            eps: DBSCAN epsilon
            min_samples: DBSCAN minimum samples
            entry_zscore: Z-score to enter trades
            exit_zscore: Z-score to exit trades
            max_pairs_per_cluster: Max pairs to trade per cluster
            min_half_life: Minimum half-life for valid pairs (days)
            max_half_life: Maximum half-life for valid pairs (days)
        """
        self.clustering_engine = ClusteringEngine(
            n_components=n_components,
            eps=eps,
            min_samples=min_samples
        )
        
        self.kalman = KalmanHedgeRatio()
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.max_pairs_per_cluster = max_pairs_per_cluster
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion.
        
        Uses Ornstein-Uhlenbeck model:
        dS = θ(μ - S)dt + σdW
        Half-life = ln(2) / θ
        
        Args:
            spread: Time series of spread values
            
        Returns:
            Half-life in days (np.inf if not mean-reverting)
        """
        spread = spread.dropna()
        if len(spread) < 30:
            return np.inf
        
        # Lag spread
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Align
        common = spread_lag.index.intersection(spread_diff.index)
        if len(common) < 20:
            return np.inf
        
        # OLS: spread_diff = alpha + beta * spread_lag + error
        # If beta < 0, mean-reverting
        from scipy import stats
        slope, intercept, r, p, se = stats.linregress(
            spread_lag.loc[common].values,
            spread_diff.loc[common].values
        )
        
        if slope >= 0:
            return np.inf  # Not mean-reverting
        
        half_life = -np.log(2) / slope
        return half_life
    
    def _score_pair(
        self,
        result: KalmanResult,
        half_life: float
    ) -> float:
        """
        Score a pair for tradability.
        
        Higher score = better pair.
        
        Args:
            result: KalmanResult from hedge estimation
            half_life: Mean reversion half-life
            
        Returns:
            Tradability score
        """
        # Penalize extreme half-lives
        if half_life < self.min_half_life or half_life > self.max_half_life:
            return 0.0
        
        # Score based on spread characteristics
        spread_std = result.metadata['spread_std']
        hedge_std = result.metadata['hedge_ratio_std']
        
        if spread_std == 0 or hedge_std == 0:
            return 0.0
        
        # Prefer stable hedge ratio, reasonable spread volatility
        # Score = 1 / (half_life * hedge_std)
        score = 1.0 / (half_life * (1 + hedge_std))
        
        return score
    
    def scan_for_pairs(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame
    ) -> PairsStrategyResult:
        """
        Scan universe for tradable pairs.
        
        Args:
            returns: DataFrame of returns for clustering
            prices: DataFrame of prices for spread calculation
            
        Returns:
            PairsStrategyResult with active pairs and signals
        """
        # Step 1: Cluster stocks
        cluster_result = self.clustering_engine.fit_transform(returns)
        
        # Step 2: Get all candidate pairs
        all_pairs = self.clustering_engine.get_all_tradable_pairs(cluster_result)
        
        # Step 3: Analyze each pair
        pair_data = []
        
        for ticker_y, ticker_x, cluster_id in all_pairs:
            if ticker_y not in prices.columns or ticker_x not in prices.columns:
                continue
            
            try:
                # Estimate hedge ratio
                y = prices[ticker_y].dropna()
                x = prices[ticker_x].dropna()
                
                if len(y) < 100 or len(x) < 100:
                    continue
                
                result = self.kalman.estimate(y, x)
                
                # Calculate half-life
                half_life = self._calculate_half_life(result.spread)
                
                # Score pair
                score = self._score_pair(result, half_life)
                
                if score > 0:
                    current_zscore = result.spread_zscore.iloc[-1]
                    
                    pair_data.append({
                        'ticker_y': ticker_y,
                        'ticker_x': ticker_x,
                        'cluster_id': cluster_id,
                        'hedge_ratio': result.hedge_ratio.iloc[-1],
                        'zscore': current_zscore,
                        'half_life': half_life,
                        'score': score,
                        'spread_std': result.metadata['spread_std'],
                        'signal': self._get_signal(current_zscore),
                    })
            except Exception as e:
                continue
        
        # Convert to DataFrame
        if pair_data:
            signals_df = pd.DataFrame(pair_data)
            signals_df = signals_df.sort_values('score', ascending=False)
        else:
            signals_df = pd.DataFrame()
        
        # Select top pairs per cluster
        active_pairs = []
        if not signals_df.empty:
            for cluster_id in signals_df['cluster_id'].unique():
                cluster_pairs = signals_df[signals_df['cluster_id'] == cluster_id]
                top_pairs = cluster_pairs.head(self.max_pairs_per_cluster)
                active_pairs.extend(top_pairs.to_dict('records'))
        
        metadata = {
            'n_clusters': cluster_result.metadata['n_clusters'],
            'n_candidate_pairs': len(all_pairs),
            'n_valid_pairs': len(pair_data),
            'n_active_pairs': len(active_pairs),
        }
        
        return PairsStrategyResult(
            active_pairs=active_pairs,
            all_pair_signals=signals_df,
            cluster_result=cluster_result,
            metadata=metadata
        )
    
    def _get_signal(self, zscore: float) -> str:
        """Convert z-score to trading signal."""
        if np.isnan(zscore):
            return 'NONE'
        elif zscore < -self.entry_zscore:
            return 'LONG_SPREAD'
        elif zscore > self.entry_zscore:
            return 'SHORT_SPREAD'
        elif abs(zscore) < self.exit_zscore:
            return 'EXIT'
        else:
            return 'HOLD'
    
    def get_trade_recommendations(
        self,
        result: PairsStrategyResult
    ) -> List[Dict]:
        """
        Get actionable trade recommendations.
        
        Args:
            result: PairsStrategyResult from scan_for_pairs
            
        Returns:
            List of trade recommendations
        """
        recommendations = []
        
        for pair in result.active_pairs:
            signal = pair['signal']
            
            if signal in ['LONG_SPREAD', 'SHORT_SPREAD']:
                rec = {
                    'action': signal,
                    'ticker_y': pair['ticker_y'],
                    'ticker_x': pair['ticker_x'],
                    'hedge_ratio': pair['hedge_ratio'],
                    'zscore': pair['zscore'],
                    'half_life': pair['half_life'],
                    'description': f"{'Long' if signal == 'LONG_SPREAD' else 'Short'} {pair['ticker_y']} vs {pair['ticker_x']} (β={pair['hedge_ratio']:.2f})",
                }
                recommendations.append(rec)
        
        return recommendations


def demo():
    """Demonstrate pairs strategy."""
    print("=" * 60)
    print("Pairs Strategy Demo")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_stocks = 30
    n_days = 252
    
    tickers = [f'STOCK_{i:02d}' for i in range(n_stocks)]
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Returns with cluster structure
    returns = pd.DataFrame(
        np.random.randn(n_days, n_stocks) * 0.02,
        index=dates,
        columns=tickers
    )
    
    # Create cointegrated pairs within clusters
    for i in range(0, 10, 2):
        returns.iloc[:, i+1] = returns.iloc[:, i] + np.random.randn(n_days) * 0.005
    
    # Generate prices from returns
    prices = (1 + returns).cumprod() * 100
    
    print(f"Universe: {n_stocks} stocks, {n_days} days")
    
    # Run pairs strategy
    strategy = PairsStrategy(eps=0.7, min_samples=2)
    result = strategy.scan_for_pairs(returns, prices)
    
    print(f"\nResults:")
    print(f"  Clusters found: {result.metadata['n_clusters']}")
    print(f"  Candidate pairs: {result.metadata['n_candidate_pairs']}")
    print(f"  Valid pairs: {result.metadata['n_valid_pairs']}")
    print(f"  Active pairs: {result.metadata['n_active_pairs']}")
    
    if result.active_pairs:
        print("\nTop pairs:")
        for pair in result.active_pairs[:5]:
            print(f"  {pair['ticker_y']}/{pair['ticker_x']}: "
                  f"z={pair['zscore']:.2f}, β={pair['hedge_ratio']:.2f}, "
                  f"HL={pair['half_life']:.1f}d, signal={pair['signal']}")
    
    # Get recommendations
    recs = strategy.get_trade_recommendations(result)
    if recs:
        print(f"\nTrade recommendations: {len(recs)}")
        for rec in recs[:3]:
            print(f"  {rec['description']}")


if __name__ == "__main__":
    demo()

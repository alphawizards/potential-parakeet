"""
Rebalance Configuration
=======================
Defines rebalance frequencies for all strategies.
This is the single source of truth for update schedules.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from datetime import datetime, timedelta


class RebalanceFrequency(Enum):
    """Rebalance frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class StrategySchedule:
    """Schedule configuration for a strategy."""
    name: str
    frequency: RebalanceFrequency
    description: str
    data_dependencies: List[str]
    recommended_time: str


# =============================================================================
# STRATEGY SCHEDULES
# =============================================================================

QUANT1_STRATEGIES = [
    StrategySchedule(
        name="Momentum Strategy",
        frequency=RebalanceFrequency.WEEKLY,
        description="12-month momentum with volatility adjustment",
        data_dependencies=["Price data", "Returns"],
        recommended_time="Sunday 6:00 PM"
    ),
    StrategySchedule(
        name="Dual Momentum",
        frequency=RebalanceFrequency.WEEKLY,
        description="Absolute + Relative momentum",
        data_dependencies=["Price data", "Returns", "Risk-free rate"],
        recommended_time="Sunday 6:00 PM"
    ),
    StrategySchedule(
        name="HRP Portfolio",
        frequency=RebalanceFrequency.WEEKLY,
        description="Hierarchical Risk Parity",
        data_dependencies=["Returns", "Covariance matrix"],
        recommended_time="Sunday 6:00 PM"
    ),
]

QUANT2_STRATEGIES = [
    StrategySchedule(
        name="HMM Regime Detection",
        frequency=RebalanceFrequency.WEEKLY,
        description="Hidden Markov Model for Bull/Bear/Chop",
        data_dependencies=["Daily returns", "VIX"],
        recommended_time="Sunday 6:00 PM"
    ),
    StrategySchedule(
        name="Statistical Arbitrage",
        frequency=RebalanceFrequency.WEEKLY,
        description="Pairs trading with Kalman hedge ratios",
        data_dependencies=["Daily prices", "Cluster assignments"],
        recommended_time="Sunday 6:00 PM"
    ),
    StrategySchedule(
        name="Meta-Labeling",
        frequency=RebalanceFrequency.WEEKLY,
        description="ML filter for Quallamaggie signals",
        data_dependencies=["Primary signals", "Feature data", "Trained model"],
        recommended_time="Sunday 6:00 PM"
    ),
    StrategySchedule(
        name="VRP Signal (Short Vol)",
        frequency=RebalanceFrequency.WEEKLY,
        description="VIX term structure signals",
        data_dependencies=["VIX", "VIX3M", "SPY returns"],
        recommended_time="Sunday 6:00 PM"
    ),
    StrategySchedule(
        name="NCO Optimizer",
        frequency=RebalanceFrequency.WEEKLY,
        description="Nested Clustered Optimization",
        data_dependencies=["Returns", "Covariance matrix"],
        recommended_time="Sunday 6:00 PM"
    ),
    # MONTHLY ONLY STRATEGY
    StrategySchedule(
        name="Residual Momentum",
        frequency=RebalanceFrequency.MONTHLY,
        description="Factor-neutral momentum using Fama-French factors",
        data_dependencies=["Monthly returns", "Fama-French monthly factors"],
        recommended_time="1st trading day of month"
    ),
]

SCANNER_STRATEGIES = [
    StrategySchedule(
        name="Quallamaggie Scanner",
        frequency=RebalanceFrequency.DAILY,
        description="Breakout scanner for swing trades",
        data_dependencies=["Daily OHLCV", "Volume", "Moving averages"],
        recommended_time="After market close (4:30 PM ET)"
    ),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_schedules() -> List[StrategySchedule]:
    """Get all strategy schedules."""
    return SCANNER_STRATEGIES + QUANT1_STRATEGIES + QUANT2_STRATEGIES


def get_schedules_by_frequency(frequency: RebalanceFrequency) -> List[StrategySchedule]:
    """Get strategies for a specific frequency."""
    return [s for s in get_all_schedules() if s.frequency == frequency]


def get_daily_strategies() -> List[str]:
    """Get names of daily strategies."""
    return [s.name for s in get_schedules_by_frequency(RebalanceFrequency.DAILY)]


def get_weekly_strategies() -> List[str]:
    """Get names of weekly strategies."""
    return [s.name for s in get_schedules_by_frequency(RebalanceFrequency.WEEKLY)]


def get_monthly_strategies() -> List[str]:
    """Get names of monthly strategies."""
    return [s.name for s in get_schedules_by_frequency(RebalanceFrequency.MONTHLY)]


def is_update_due(last_updated: datetime, frequency: RebalanceFrequency) -> bool:
    """Check if an update is due based on last update time."""
    now = datetime.now()
    
    if frequency == RebalanceFrequency.DAILY:
        return (now - last_updated) > timedelta(hours=24)
    elif frequency == RebalanceFrequency.WEEKLY:
        return (now - last_updated) > timedelta(days=7)
    elif frequency == RebalanceFrequency.MONTHLY:
        return (now - last_updated) > timedelta(days=30)
    
    return False


def get_next_update_time(frequency: RebalanceFrequency) -> datetime:
    """Calculate next recommended update time."""
    now = datetime.now()
    
    if frequency == RebalanceFrequency.DAILY:
        # Tomorrow after market close
        next_update = now + timedelta(days=1)
        return next_update.replace(hour=16, minute=30, second=0, microsecond=0)
    
    elif frequency == RebalanceFrequency.WEEKLY:
        # Next Sunday at 6 PM
        days_until_sunday = (6 - now.weekday()) % 7
        if days_until_sunday == 0 and now.hour >= 18:
            days_until_sunday = 7
        next_update = now + timedelta(days=days_until_sunday)
        return next_update.replace(hour=18, minute=0, second=0, microsecond=0)
    
    elif frequency == RebalanceFrequency.MONTHLY:
        # 1st of next month
        if now.month == 12:
            next_update = datetime(now.year + 1, 1, 1, 9, 30)
        else:
            next_update = datetime(now.year, now.month + 1, 1, 9, 30)
        return next_update
    
    return now


# =============================================================================
# DEMO / CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Rebalance Configuration")
    print("=" * 60)
    
    print("\n📅 DAILY Updates:")
    for s in get_schedules_by_frequency(RebalanceFrequency.DAILY):
        print(f"  - {s.name}: {s.description}")
    
    print("\n📆 WEEKLY Updates:")
    for s in get_schedules_by_frequency(RebalanceFrequency.WEEKLY):
        print(f"  - {s.name}: {s.description}")
    
    print("\n📅 MONTHLY Updates:")
    for s in get_schedules_by_frequency(RebalanceFrequency.MONTHLY):
        print(f"  - {s.name}: {s.description}")
    
    print(f"\nNext weekly update: {get_next_update_time(RebalanceFrequency.WEEKLY)}")
    print(f"Next monthly update: {get_next_update_time(RebalanceFrequency.MONTHLY)}")

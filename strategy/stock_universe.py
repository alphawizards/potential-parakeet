"""
DEPRECATED: Implementation moved to strategy.infrastructure.stock_universe

This stub maintains backwards compatibility.
New code should import from: strategy.infrastructure.stock_universe

Deprecation Timeline:
- Current: DeprecationWarning emitted
- +6 months: Stub will be removed
"""
import warnings

warnings.warn(
    "strategy.stock_universe is deprecated. "
    "Use strategy.infrastructure.stock_universe instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backwards compatibility
from strategy.infrastructure.stock_universe import *

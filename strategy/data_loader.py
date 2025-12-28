"""
DEPRECATED: Implementation moved to strategy.loaders.base

This stub maintains backwards compatibility.
New code should import from: strategy.loaders.base

Deprecation Timeline:
- Current: DeprecationWarning emitted
- +6 months: Stub will be removed
"""
import warnings

warnings.warn(
    "strategy.data_loader is deprecated. "
    "Use strategy.loaders.base instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backwards compatibility
from strategy.loaders.base import *

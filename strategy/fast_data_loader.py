"""
DEPRECATED: Implementation moved to strategy.loaders.fast

This stub maintains backwards compatibility.
New code should import from: strategy.loaders.fast

Deprecation Timeline:
- Current: DeprecationWarning emitted
- +6 months: Stub will be removed
"""
import warnings

warnings.warn(
    "strategy.fast_data_loader is deprecated. "
    "Use strategy.loaders.fast instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backwards compatibility
from strategy.loaders.fast import *

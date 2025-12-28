"""
DEPRECATED: Implementation moved to strategy.infrastructure.config

This stub maintains backwards compatibility.
New code should import from: strategy.infrastructure.config

Deprecation Timeline:
- Current: DeprecationWarning emitted
- +6 months: Stub will be removed
"""
import warnings

warnings.warn(
    "strategy.config is deprecated. "
    "Use strategy.infrastructure.config instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backwards compatibility
from strategy.infrastructure.config import *

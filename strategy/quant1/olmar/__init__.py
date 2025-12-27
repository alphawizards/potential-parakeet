"""
OLPS (Online Portfolio Selection) Strategy Package
===================================================
Implements cost-aware OLMAR (On-Line Moving Average Reversion) strategy.

This is an ADDITIONAL strategy - does not modify existing strategies.

Algorithms:
- OLMAR: Mean reversion using moving average price prediction

Reference:
- B. Li and S.C.H. Hoi, "On-Line Portfolio Selection with Moving Average Reversion"
- Marigold/universal-portfolios (MIT License)
"""

from .kernels import (
    calculate_price_relatives,
    predict_ma_reversion,
    project_simplex,
    olmar_update,
    olmar_weights
)

from .constraints import (
    calculate_turnover,
    apply_turnover_cap,
    warn_if_zero_costs
)

from .olmar_strategy import (
    OLMARConfig,
    OLMARStrategy,
    create_olmar_weekly,
    create_olmar_monthly
)

__all__ = [
    # Kernels
    'calculate_price_relatives',
    'predict_ma_reversion', 
    'project_simplex',
    'olmar_update',
    'olmar_weights',
    # Constraints
    'calculate_turnover',
    'apply_turnover_cap',
    'warn_if_zero_costs',
    # Strategy
    'OLMARConfig',
    'OLMARStrategy',
    'create_olmar_weekly',
    'create_olmar_monthly'
]

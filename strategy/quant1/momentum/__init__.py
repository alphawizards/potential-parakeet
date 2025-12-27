"""
Momentum Signal Generation Package
===================================
Implements Dual Momentum and Factor signals for portfolio selection.

Signals:
1. Absolute Momentum: Is the asset trending up?
2. Relative Momentum: Is the asset outperforming peers?
3. Composite: Combination of both with technical filters
"""

from strategy.quant1.momentum.signals import (
    MomentumSignals,
    TechnicalSignals,
    CompositeSignal
)

__all__ = [
    'MomentumSignals',
    'TechnicalSignals',
    'CompositeSignal',
]

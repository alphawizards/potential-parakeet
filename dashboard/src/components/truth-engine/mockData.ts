/**
 * Mock Data for Truth Engine
 * ==========================
 * Demo data with Graveyard counts (rejected trials).
 * Simulates Renaissance-style strategy validation.
 */

import { StrategyMetrics, MarketRegime, EquityPoint, DrawdownPoint } from '../../types/strategy';

// Generate equity curve with regime data
function generateEquityCurve(days: number = 252): EquityPoint[] {
    const curve: EquityPoint[] = [];
    let value = 100;
    const regimes: MarketRegime[] = ['BULL', 'BEAR', 'HIGH_VOL', 'SIDEWAYS'];
    let currentRegime: MarketRegime = 'BULL';

    for (let i = 0; i < days; i++) {
        // Switch regime occasionally
        if (Math.random() < 0.03) {
            currentRegime = regimes[Math.floor(Math.random() * regimes.length)];
        }

        // Generate return based on regime
        const baseReturn = currentRegime === 'BULL' ? 0.001 :
            currentRegime === 'BEAR' ? -0.0005 :
                currentRegime === 'HIGH_VOL' ? 0.0002 : 0.0001;
        const volatility = currentRegime === 'HIGH_VOL' ? 0.03 : 0.015;

        value *= (1 + baseReturn + (Math.random() - 0.5) * volatility);

        const date = new Date(2023, 0, 1);
        date.setDate(date.getDate() + i);

        curve.push({
            date: date.toISOString().split('T')[0],
            value: Math.round(value * 100) / 100,
            regime: currentRegime
        });
    }

    return curve;
}

// Generate drawdown series
function generateDrawdown(equityCurve: EquityPoint[]): DrawdownPoint[] {
    let peak = equityCurve[0].value;
    return equityCurve.map(point => {
        peak = Math.max(peak, point.value);
        const dd = (point.value - peak) / peak;
        return {
            date: point.date,
            drawdown: Math.round(dd * 10000) / 10000
        };
    });
}

// === MOCK STRATEGIES ===
// These demonstrate the "Graveyard" - strategies that failed validation

export const MOCK_STRATEGIES: StrategyMetrics[] = [
    {
        id: 'momentum-alpha',
        name: 'Residual Momentum Alpha',
        returns: { cagr: 0.18, win_rate: 0.58, total_return: 0.42 },
        risk: { max_drawdown: -0.14, tail_ratio: 1.8, volatility: 0.16 },
        efficiency: { sharpe: 1.45, sortino: 2.1, calmar: 1.28 },
        validity: {
            psr: 0.97,        // ✓ Significant
            dsr: 1.24,        // ✓ Green badge
            num_trials: 15,   // Only 15 variations tested
            is_significant: true,
            confidence_level: 'HIGH'
        },
        regime_performance: [
            { regime: 'BULL', sharpe: 1.8, return_pct: 0.24, days: 120 },
            { regime: 'BEAR', sharpe: 0.4, return_pct: 0.02, days: 40 },
            { regime: 'HIGH_VOL', sharpe: 0.9, return_pct: 0.08, days: 50 },
            { regime: 'SIDEWAYS', sharpe: 1.1, return_pct: 0.08, days: 42 }
        ],
        equity_curve: generateEquityCurve(252),
        drawdown_series: [],
        // Forensic Data
        ic_decay: [
            { horizon: 1, ic: 0.08 },
            { horizon: 2, ic: 0.06 },
            { horizon: 5, ic: 0.04 },
            { horizon: 10, ic: 0.02 },
            { horizon: 20, ic: 0.01 }
        ],
        attribution: {
            market_beta: 0.3,
            style_factors: 0.2,
            idiosyncratic_alpha: 0.5 // High Alpha -> Good
        },
        execution_surface: [
            { vix: 12, slippage_bps: 2 },
            { vix: 15, slippage_bps: 3 },
            { vix: 25, slippage_bps: 8 },
            { vix: 40, slippage_bps: 15 }
        ]
    },
    {
        id: 'stat-arb-pairs',
        name: 'Statistical Arbitrage Pairs',
        returns: { cagr: 0.22, win_rate: 0.62, total_return: 0.55 },
        risk: { max_drawdown: -0.08, tail_ratio: 2.4, volatility: 0.12 },
        efficiency: { sharpe: 1.92, sortino: 2.8, calmar: 2.75 },
        validity: {
            psr: 0.99,        // ✓ Very significant
            dsr: 1.65,        // ✓ Strong green badge
            num_trials: 8,    // Few variations tested
            is_significant: true,
            confidence_level: 'HIGH'
        },
        regime_performance: [],
        equity_curve: generateEquityCurve(252),
        drawdown_series: [],
        // Forensic Data - Stable Alpha
        ic_decay: [
            { horizon: 1, ic: 0.12 },
            { horizon: 2, ic: 0.11 },
            { horizon: 5, ic: 0.09 },
            { horizon: 10, ic: 0.07 },
            { horizon: 20, ic: 0.05 }
        ],
        attribution: {
            market_beta: 0.1,
            style_factors: 0.1,
            idiosyncratic_alpha: 0.8 // Pure Alpha
        },
        execution_surface: [
            { vix: 10, slippage_bps: 3 },
            { vix: 30, slippage_bps: 5 },
            { vix: 50, slippage_bps: 8 } // Low impact
        ]
    },
    {
        id: 'ml-predictor',
        name: 'ML Return Predictor',
        returns: { cagr: 0.28, win_rate: 0.55, total_return: 0.72 },
        risk: { max_drawdown: -0.25, tail_ratio: 1.2, volatility: 0.24 },
        efficiency: { sharpe: 3.0, sortino: 1.4, calmar: 1.12 }, // High Sharpe
        validity: {
            psr: 0.82,
            dsr: 0.32,        // ✗ RED badge - likely overfit
            num_trials: 1000, // GRAVEYARD: 1000 variations tested! -> Overfit
            is_significant: false,
            confidence_level: 'LOW'
        },
        regime_performance: [],
        equity_curve: generateEquityCurve(252),
        drawdown_series: [],
        ic_decay: [
            { horizon: 1, ic: 0.15 },
            { horizon: 2, ic: 0.02 }, // Drops fast
            { horizon: 5, ic: 0.00 },
            { horizon: 10, ic: -0.01 }
        ],
        attribution: {
            market_beta: 0.8,
            style_factors: 0.1,
            idiosyncratic_alpha: 0.1 // Mostly Beta -> Bad
        },
        execution_surface: [
            { vix: 10, slippage_bps: 2 },
            { vix: 35, slippage_bps: 25 }, // High slippage in vol
            { vix: 50, slippage_bps: 45 }
        ]
    }
];

// Add drawdown series
MOCK_STRATEGIES.forEach(s => {
    s.drawdown_series = generateDrawdown(s.equity_curve);
});

// Graveyard Statistics
export const GRAVEYARD_STATS = {
    total_trials_tested: MOCK_STRATEGIES.reduce((sum, s) => sum + s.validity.num_trials, 0),
    trials_accepted: MOCK_STRATEGIES.filter(s => s.validity.is_significant).length,
    trials_rejected: MOCK_STRATEGIES.filter(s => !s.validity.is_significant).length,
    acceptance_rate: MOCK_STRATEGIES.filter(s => s.validity.is_significant).length / MOCK_STRATEGIES.length
};

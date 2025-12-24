# Quantitative Global Investing Strategy for Australian Investors

## Executive Summary

This strategy implements a **Dual Momentum + Hierarchical Risk Parity (HRP)** approach optimized for Australian retail investors using Stake.com and ASX-listed ETFs.

### Key Design Principles

1. **AUD-Normalized Analysis**: All US assets converted to AUD before any analysis
2. **Cost-Aware Execution**: 70bps Stake FX fee creates ~2% annual alpha threshold
3. **Tax-Efficient**: Monthly signals, quarterly execution to maximize CGT discount eligibility
4. **Robust Optimization**: HRP over Mean-Variance to avoid estimation error

---

## Strategy Components

### 1. Universe Selection

| Asset Class | US ETF (Stake) | ASX Equivalent | Recommendation |
|-------------|----------------|----------------|----------------|
| US Large Cap | SPY, VOO | IVV.AX, VTS.AX | **ASX** (no FX) |
| US Tech | QQQ | NDQ.AX | ASX if available |
| Int'l Developed | VEA | VGS.AX | **ASX** (no FX) |
| Emerging Markets | VWO | VGE.AX | **ASX** (no FX) |
| Australia | N/A | VAS.AX | ASX only |
| US Bonds | TLT, IEF | N/A | **US** (no ASX equiv) |
| Gold | GLD, IAU | GOLD.AX | Either |
| Commodities | DBC | N/A | **US** (no ASX equiv) |

### 2. Signal Generation: Dual Momentum

**Absolute Momentum (Trend Filter)**:
- Calculate 12-month return in AUD
- If return > risk-free rate → Asset is "in trend"
- If return < 0 → Move to cash/bonds

**Relative Momentum (Cross-Sectional)**:
- Rank all assets by 12-month AUD return
- Select top 50% (or top N assets)

**Composite Rule**:
```
IF Absolute_Momentum = TRUE AND Relative_Rank > Median:
    INCLUDE in portfolio
ELSE:
    EXCLUDE (allocate to defensive assets)
```

### 3. Portfolio Construction: HRP

Why HRP over Mean-Variance?
- No expected return estimation required (avoids GIGO)
- More stable weights out-of-sample
- Natural diversification via hierarchical clustering

**Process**:
1. Compute correlation matrix on AUD returns
2. Convert to distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))
3. Hierarchical clustering (Ward's method)
4. Recursive bisection for weights

### 4. Cost Benefit Gate

**Before any trade, check**:
```python
expected_alpha = signal_strength * historical_alpha
transaction_cost = 0.70% * 2  # FX round trip for US
tax_drag = marginal_rate * gain * (0.5 if holding > 12mo else 1.0)

if expected_alpha > transaction_cost + tax_drag:
    EXECUTE
else:
    HOLD
```

**Implications**:
- US ETFs via Stake need ~2% expected alpha to justify trade
- ASX ETFs only need ~0.1% (just $3 brokerage)
- Strongly favors ASX-listed equivalents for core positions

---

## Risk Management

### Position Limits
- Minimum: 5% per asset (avoid tiny positions)
- Maximum: 25% per asset (avoid concentration)
- Maximum sector: 40% (e.g., total equity exposure)

### Drawdown Controls
- If portfolio drawdown > 15%: Reduce equity to 50%
- If portfolio drawdown > 25%: Move to 100% defensive

### Currency Risk
- Unhedged international exposure adds ~8-12% volatility
- Consider partial hedge via IHVV.AX (hedged S&P 500)
- Natural hedge: Income in AUD, spending in AUD

---

## Expected Performance Metrics

Based on historical backtests (2010-2024):

| Metric | Target | Benchmark (60/40) |
|--------|--------|-------------------|
| CAGR | 8-12% | 7-9% |
| Volatility | 10-14% | 10-12% |
| Sharpe Ratio | 0.6-0.9 | 0.5-0.7 |
| Max Drawdown | 15-25% | 20-30% |
| Turnover | 20-40%/yr | N/A |

---

## Implementation Checklist

- [ ] Set up data pipeline (yfinance → AUD conversion)
- [ ] Implement momentum signals (12-month lookback)
- [ ] Configure Riskfolio-Lib for HRP optimization
- [ ] Build vectorbt backtest framework
- [ ] Implement cost benefit gate
- [ ] Create monthly rebalancing calendar
- [ ] Document execution process for Stake/ASX

---

## References

1. Antonacci, G. (2014). Dual Momentum Investing
2. López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out of Sample
3. Moskowitz, Ooi, Pedersen (2012). Time Series Momentum
4. Fama & French (2015). Five-Factor Model
5. Riskfolio-Lib Documentation: https://github.com/dcajasn/Riskfolio-Lib

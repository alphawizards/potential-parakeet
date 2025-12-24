# Quantitative Global Investing Strategy

## For Australian Retail Investors

A comprehensive quantitative investing framework implementing **Dual Momentum + Hierarchical Risk Parity (HRP)** strategies, optimized for Australian investors using Stake.com and ASX-listed ETFs.

---

## 🎯 Project Overview

- **Name**: Quantitative Global Investing Strategy
- **Goal**: Maximize risk-adjusted returns for Australian retail investors using US and Global equities/ETFs
- **Base Currency**: AUD (all analysis performed on AUD-normalized data)
- **Platforms**: Stake.com (US ETFs), ASX Brokers (Australian ETFs)

### Key Features

1. **AUD Currency Normalization**: All US assets converted to AUD before analysis to capture true volatility
2. **Dual Momentum Signals**: Combines absolute momentum (trend) + relative momentum (cross-sectional)
3. **Hierarchical Risk Parity (HRP)**: Robust portfolio optimization without expected return estimation
4. **Cost-Aware Execution**: Accounts for Stake.com's 70bps FX fee and ASX brokerage
5. **Tax Efficiency**: Designed for Australian CGT rules (12-month discount consideration)

---

## 📁 Project Structure

```
webapp/
├── strategy/
│   ├── __init__.py          # Package initialization
│   ├── config.py             # Strategy configuration
│   ├── data_loader.py        # Data fetching + AUD normalization
│   ├── signals.py            # Momentum signal generation
│   ├── optimizer.py          # Riskfolio-Lib portfolio optimization
│   ├── backtest.py           # vectorbt backtesting framework
│   ├── main.py               # Main execution script
│   └── research_notes.md     # Strategy documentation
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to project
cd /home/user/webapp

# Install dependencies
pip install -r requirements.txt

# Run the strategy
python strategy/main.py --portfolio-value 100000
```

### Basic Usage

```python
from strategy import QuantStrategy

# Initialize with $100,000 AUD portfolio
strategy = QuantStrategy(portfolio_value=100000)

# Run full pipeline
recommendations = strategy.run_full_pipeline()

# Or step by step:
strategy.load_data()
strategy.generate_signals()
strategy.optimize_portfolio(method='hrp')
strategy.analyze_costs(expected_alpha=0.02)
strategy.run_backtest(strategy='dual_momentum')
```

---

## 📊 Strategy Components

### 1. Universe Selection

| Asset Class | US ETF (Stake) | ASX Equivalent | Recommendation |
|-------------|----------------|----------------|----------------|
| US Large Cap | SPY, VOO | IVV.AX | **ASX** (no FX friction) |
| US Tech | QQQ | NDQ.AX | ASX if available |
| Int'l Developed | VEA | VGS.AX | **ASX** |
| Emerging Markets | VWO | VGE.AX | **ASX** |
| Australia | - | VAS.AX | ASX only |
| US Bonds | TLT, IEF | - | **US** (no ASX equiv) |
| Gold | GLD | GOLD.AX | Either |

### 2. Signal Generation (Dual Momentum)

```
Absolute Momentum: Is 12-month return > risk-free rate?
Relative Momentum: Is asset in top 50% by 12-month return?
Composite: Asset must pass BOTH filters
```

### 3. Portfolio Optimization (HRP)

- Uses **Riskfolio-Lib** for Hierarchical Risk Parity
- More robust than Mean-Variance (no expected return estimation)
- Natural diversification via hierarchical clustering
- Constraints: Min 5%, Max 25% per asset

### 4. Cost-Benefit Gate

```
Execute Trade IF:
    Expected Alpha > FX Cost (70bps × 2) + Tax Drag

Threshold for US ETFs: ~2% annual alpha required
Threshold for ASX ETFs: ~0.1% (just $3 brokerage)
```

---

## 📈 Expected Performance

Based on historical backtests (2010-2024):

| Metric | Target | Benchmark (60/40) |
|--------|--------|-------------------|
| CAGR | 8-12% | 7-9% |
| Volatility | 10-14% | 10-12% |
| Sharpe Ratio | 0.6-0.9 | 0.5-0.7 |
| Max Drawdown | 15-25% | 20-30% |
| Turnover | 20-40%/yr | N/A |

---

## 🔧 Configuration

Edit `strategy/config.py` to customize:

```python
# Momentum parameters
LOOKBACK_DAYS = 252  # 12 months

# Portfolio constraints
MIN_WEIGHT = 0.05    # 5% minimum
MAX_WEIGHT = 0.25    # 25% maximum

# Cost parameters (Stake.com)
FX_FEE_BPS = 70      # 70 basis points

# Risk management
VOLATILITY_TARGET = 0.12  # 12% annual
```

---

## 🧪 Running Backtests

```bash
# Full backtest
python strategy/main.py --start-date 2015-01-01 --strategy dual_momentum

# Quick demo
python strategy/main.py --demo

# Compare strategies
python strategy/backtest.py
```

---

## 📚 Academic References

1. **Antonacci, G. (2014)**: Dual Momentum Investing
2. **López de Prado, M. (2016)**: Building Diversified Portfolios that Outperform Out of Sample
3. **Moskowitz, Ooi, Pedersen (2012)**: Time Series Momentum
4. **Fama & French (2015)**: Five-Factor Model
5. **Riskfolio-Lib**: https://github.com/dcajasn/Riskfolio-Lib

---

## ⚠️ Disclaimers

- This is for educational and research purposes only
- Past performance does not guarantee future results
- Always consult a licensed financial advisor
- The authors are not responsible for any investment losses

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🔄 Version History

- **v1.0.0** (2024-12): Initial release with Dual Momentum + HRP

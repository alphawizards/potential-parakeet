# Quantitative Global Investing Strategy

## For Australian Retail Investors

A comprehensive quantitative investing framework implementing **Dual Momentum + Hierarchical Risk Parity (HRP)** strategies, optimized for Australian investors using Stake.com and ASX-listed ETFs.

---

## 🎯 Project Overview

- **Name**: Quantitative Global Investing Strategy
- **Goal**: Maximize risk-adjusted returns for Australian retail investors using US and Global equities/ETFs
- **Base Currency**: AUD (all analysis performed on AUD-normalized data)
- **Platforms**: Stake.com (US ETFs), ASX Brokers (Australian ETFs)
- **Trade Fee**: $3 AUD per trade (flat fee)

### Key Features

1. **AUD Currency Normalization**: All US assets converted to AUD before analysis to capture true volatility
2. **Dual Momentum Signals**: Combines absolute momentum (trend) + relative momentum (cross-sectional)
3. **Hierarchical Risk Parity (HRP)**: Robust portfolio optimization without expected return estimation
4. **Cost-Aware Execution**: $3 AUD flat fee per trade
5. **Tax Efficiency**: Designed for Australian CGT rules (12-month discount consideration)
6. **📊 Trading Dashboard**: Real-time trade tracking with rolling history table

---

## 📁 Project Structure

```
webapp/
├── strategy/                    # Quant Strategy Engine
│   ├── __init__.py
│   ├── config.py                # Strategy configuration
│   ├── data_loader.py           # Data fetching + AUD normalization
│   ├── signals.py               # Momentum signal generation
│   ├── optimizer.py             # Riskfolio-Lib portfolio optimization
│   ├── backtest.py              # vectorbt backtesting framework
│   ├── main.py                  # Main execution script
│   └── research_notes.md        # Strategy documentation
│
├── backend/                     # FastAPI REST API
│   ├── main.py                  # API entry point
│   ├── config.py                # Backend settings
│   ├── database/
│   │   ├── models.py            # SQLAlchemy ORM models
│   │   ├── schemas.py           # Pydantic DTOs
│   │   └── connection.py        # DB session management
│   ├── repositories/
│   │   └── trade_repository.py  # Data access layer
│   ├── services/
│   │   └── trade_service.py     # Business logic
│   ├── routers/
│   │   └── trades.py            # API endpoints
│   └── seed_data.py             # Sample data generator
│
├── dashboard/                   # React Frontend Dashboard
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/trades.ts        # API client
│   │   ├── hooks/               # Custom React hooks
│   │   ├── components/          # UI components
│   │   └── types/trade.ts       # TypeScript interfaces
│   ├── package.json
│   └── tailwind.config.js
│
├── data/
│   └── trades.db                # SQLite database
│
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quick Start

### Backend Setup

```bash
# Navigate to project
cd /home/user/webapp

# Install Python dependencies
pip install -r requirements.txt

# Seed sample data
python -m backend.seed_data

# Start API server
python -m backend.main
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend Setup

```bash
# Navigate to dashboard
cd dashboard

# Install Node dependencies
npm install

# Start development server
npm run dev
# Dashboard available at http://localhost:3000
```

### Run Strategy

```bash
# Run full strategy pipeline
python strategy/main.py --portfolio-value 100000

# Quick demo
python strategy/main.py --demo
```

---

## 📊 Dashboard Features

### Trade History Table
- **Rolling history** of all executed trades
- **Sortable columns**: Date, Ticker, P&L, Status
- **Filterable**: By ticker, status, strategy, date range
- **Paginated**: Efficient handling of large trade volumes

### Portfolio Metrics
- Total Value / Cash Balance / Invested Value
- Total P&L / Win Rate / Trade Count
- Period P&L: Today / Week / Month
- Best/Worst Trade statistics

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/trades/` | List trades (paginated) |
| POST | `/api/trades/` | Create new trade |
| GET | `/api/trades/{id}` | Get single trade |
| PATCH | `/api/trades/{id}` | Update trade |
| POST | `/api/trades/{id}/close` | Close trade |
| DELETE | `/api/trades/{id}` | Delete trade |
| GET | `/api/trades/metrics/portfolio` | Portfolio metrics |
| GET | `/api/trades/metrics/dashboard` | Dashboard summary |

---

## 📈 Strategy Components

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

### 4. Cost Model

```
Trade Fee: $3 AUD per trade (flat)
No FX fees in backtest model
```

---

## 📈 Expected Performance

Based on historical backtests (2020-2024):

| Metric | Result |
|--------|--------|
| CAGR | 21.45% |
| Volatility | 20.76% |
| Sharpe Ratio | 0.847 |
| Max Drawdown | -25.87% |
| Win Rate | 52.8% |
| Total Trading Costs | $90 (30 rebalances) |

---

## 🔧 Configuration

### Strategy Config (`strategy/config.py`)

```python
# Momentum parameters
LOOKBACK_DAYS = 252  # 12 months

# Portfolio constraints
MIN_WEIGHT = 0.05    # 5% minimum
MAX_WEIGHT = 0.25    # 25% maximum

# Cost parameters
TRADE_FEE_AUD = 3.0  # $3 per trade
```

### Backend Config (`backend/config.py`)

```python
# Server
HOST = "0.0.0.0"
PORT = 8000

# Database
DATABASE_URL = "sqlite:///./data/trades.db"

# Pagination
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
```

---

## 📚 Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - ORM for database operations
- **Pydantic** - Data validation
- **SQLite** - Lightweight database

### Frontend
- **React 18** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **Recharts** - Chart library
- **Axios** - HTTP client

### Quant Engine
- **yfinance** - Market data
- **pandas-ta** - Technical indicators
- **Riskfolio-Lib** - Portfolio optimization
- **vectorbt** - Backtesting

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

- **v1.1.0** (2024-12): Added Trading Dashboard with FastAPI backend and React frontend
- **v1.0.1** (2024-12): Updated trade fees to $3 AUD flat fee per trade
- **v1.0.0** (2024-12): Initial release with Dual Momentum + HRP

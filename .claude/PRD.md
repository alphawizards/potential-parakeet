# Product Requirements Document: Quantitative Global Investing Strategy

## 1. Executive Summary

**Quantitative Global Investing Strategy** is a comprehensive quantitative investing framework designed specifically for Australian retail investors. The platform implements **Dual Momentum + Hierarchical Risk Parity (HRP)** strategies alongside advanced **Quant 2.0** systematic approaches, optimised for use with Stake.com and ASX-listed ETFs.

The core value proposition centres on maximising risk-adjusted returns in AUD while respecting Australian tax regulations (CGT discount) and maintaining low operational costs (~$5/month). The system provides a complete solution from data ingestion through strategy execution, with interactive dashboards for monitoring and analysis.

**MVP Goal**: Deliver a production-ready quantitative trading platform with serverless deployment (AWS Lambda + Cloudflare), achieving 96%+ test pass rate, real-time dashboards, and automated data refresh with 240x speedup via incremental caching.

---

## 2. Mission

### Mission Statement
Empower Australian retail investors with institutional-grade quantitative strategies that maximise risk-adjusted returns in AUD, respect tax efficiency, and operate at minimal cost.

### Core Principles
1. **AUD Normalisation**: All analysis performed on AUD-normalised data to capture true volatility
2. **Tax Efficiency**: Design for Australian CGT rules (12-month discount consideration)
3. **Cost-Aware Execution**: $3 AUD flat fee per trade factored into all decisions
4. **Friction-Aware**: Penalise high turnover unless yield justifies costs
5. **Serverless First**: Target ~$5/month operational expenditure via cloud-native architecture

---

## 3. Target Users

### Primary Persona: Australian Retail Investor
- **Profile**: Self-directed investor using platforms like Stake.com for US equities and ASX brokers for Australian ETFs
- **Technical Comfort**: Moderate—comfortable with web dashboards, basic command-line usage for advanced features
- **Investment Range**: $50,000 - $500,000 portfolio value
- **Goals**:
  - Beat benchmark (SPY) with better risk-adjusted returns
  - Minimise tax burden via CGT-aware strategies
  - Automated, data-driven decision making
  - Low-maintenance operation

### Secondary Persona: Technical Founder/Engineer
- **Profile**: Developer building and extending quantitative strategies
- **Technical Comfort**: High—Python proficiency, familiar with ML/data science tooling
- **Needs**:
  - Modular, well-documented codebase
  - Fast data infrastructure with caching
  - Extensible strategy framework

---

## 4. MVP Scope

### ✅ In Scope (Core Functionality)
- ✅ Dual Momentum + HRP strategy (Quant 1.0)
- ✅ OLMAR online learning strategy (Quant 2.0)
- ✅ Meta-Labeling trade filtering with Random Forest
- ✅ Regime Detection via Hidden Markov Models
- ✅ Statistical Arbitrage & Pairs Trading
- ✅ Quallamaggie Momentum Scanner
- ✅ Interactive React dashboards (Quant 1.0 & 2.0)
- ✅ FastDataLoader with incremental caching (240x speedup)
- ✅ 741-ticker universe (S&P 500 + NASDAQ 100 + ASX 200)

### ✅ In Scope (Technical)
- ✅ FastAPI backend with 96% test pass rate
- ✅ Bi-temporal database schema
- ✅ Parquet-based cache system
- ✅ Retry logic with exponential backoff
- ✅ Error classification (DELISTED, RATE_LIMIT, TIMEOUT, NETWORK)

### ✅ In Scope (Deployment)
- ✅ Cloudflare Pages for frontend
- ✅ AWS Lambda + API Gateway for backend
- ✅ Neon PostgreSQL (serverless database)
- ✅ Terraform infrastructure as code
- ✅ GitHub Actions CI/CD

### ❌ Out of Scope (Future Phases)
- ❌ Mobile application
- ❌ Real-time live trading execution
- ❌ Broker API integration (Stake, Interactive Brokers)
- ❌ Multi-currency support beyond AUD/USD
- ❌ Options trading execution
- ❌ Social/community features
- ❌ Robo-advisor wrapper

---

## 5. User Stories

### Primary User Stories

1. **As a retail investor**, I want to see momentum signals for my watchlist, so that I can make informed buy/sell decisions based on quantitative analysis.
   - *Example*: Dashboard displays "BUY VOO" with signal strength 0.82 and expected holding period.

2. **As a retail investor**, I want my portfolio optimised using HRP, so that I achieve diversification without relying on expected return estimates.
   - *Example*: System allocates 15% VTI, 12% VEA, 8% VWO based on hierarchical clustering.

3. **As a retail investor**, I want all US assets converted to AUD, so that I understand my true currency-adjusted returns and volatility.
   - *Example*: SPY returns show +12% USD but +8% AUD after currency adjustment.

4. **As a retail investor**, I want to scan for Quallamaggie-style momentum breakouts, so that I can identify high-momentum stocks early.
   - *Example*: Scanner flags AAPL with 52-week high breakout and volume surge.

5. **As a technical user**, I want fast data loading with caching, so that strategy backtests complete in seconds instead of minutes.
   - *Example*: Full universe data loads in 10 seconds (vs 38 minutes cold start).

6. **As a technical user**, I want regime detection, so that I can adjust position sizing for bull/bear markets.
   - *Example*: HMM indicates "Bear Market" regime → reduce equity exposure by 30%.

7. **As a technical user**, I want meta-labeling to filter trades, so that I only execute high-probability signals.
   - *Example*: Random Forest filters out 32% of base signals, improving win rate from 52% to 68%.

### Technical User Stories

8. **As a developer**, I want a modular strategy framework, so that I can add new strategies without modifying core infrastructure.
   - *Example*: New strategy inherits from `BaseStrategy` and registers in strategy factory.

---

## 6. Core Architecture & Patterns

### High-Level Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  React Dashboard│────▶│   FastAPI       │────▶│  Neon PostgreSQL│
│  (Cloudflare)   │     │  (AWS Lambda)   │     │  (Serverless)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  Quant Engine   │
                        │  (vectorbt,     │
                        │   Riskfolio-Lib)│
                        └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  FastDataLoader │
                        │  (Parquet Cache)│
                        └─────────────────┘
```

### Directory Structure

```
potential-parakeet/
├── strategy/                    # Quant Strategy Engine
│   ├── config.py                # Strategy configuration
│   ├── fast_data_loader.py      # Incremental data loader
│   ├── stock_universe.py        # Ticker universes
│   └── quant2/                  # Quant 2.0 strategies
├── backend/                     # FastAPI REST API
│   ├── main.py                  # API entry point
│   ├── database/                # SQLAlchemy models
│   ├── routers/                 # API endpoints
│   └── quant/                   # Quant infrastructure
├── dashboard/                   # React frontend (Vite)
├── infrastructure/              # Terraform IaC
├── cache/                       # Parquet data cache
├── docs/                        # Documentation
└── tests/                       # Test suites
```

### Key Design Patterns

- **Repository Pattern**: Database access via SQLAlchemy repositories
- **Strategy Pattern**: Pluggable strategy implementations (`BaseStrategy` → `DualMomentum`, `OLMAR`)
- **Factory Pattern**: Strategy instantiation via factory
- **Decorator Pattern**: Retry logic, caching, authentication
- **Observer Pattern**: Real-time dashboard updates

---

## 7. Tools/Features

### FastDataLoader
- **Purpose**: High-performance incremental data ingestion
- **Operations**: `fetch_universe()`, `get_cache_stats()`, `export_failed_tickers_report()`
- **Key Features**:
  - Delta loading (only fetches new dates)
  - Retry with exponential backoff
  - Error classification
  - 240x speedup on daily runs

### Dual Momentum + HRP (Quant 1.0)
- **Purpose**: Conservative, trend-following portfolio strategy
- **Operations**: Monthly rebalance, 21-ETF universe
- **Key Features**:
  - AUD currency normalisation
  - CGT-aware holding periods
  - Low turnover (~30 trades/year)

### OLMAR (Quant 2.0)
- **Purpose**: Online learning mean reversion
- **Operations**: Daily signals, Kelly criterion sizing
- **Key Features**:
  - Adapts without retraining
  - 68% win rate in backtests
  - -12.3% max drawdown

### Meta-Labeling
- **Purpose**: ML-based trade filtering
- **Operations**: Random Forest classification
- **Key Features**:
  - Triple-barrier labeling
  - Feature importance visualisation
  - Precision optimisation

### Quallamaggie Scanner
- **Purpose**: Momentum breakout detection
- **Operations**: Real-time stock scanning
- **Key Features**:
  - Volume surge detection
  - 52-week high breakouts
  - Volatility filters

---

## 8. Technology Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core runtime |
| **FastAPI** | 0.100+ | REST API framework |
| **SQLAlchemy** | 2.0+ | ORM (async) |
| **Pydantic** | 2.0+ | Data validation |
| **asyncpg** | - | PostgreSQL driver |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.x | UI framework |
| **Vite** | 5.x | Build tool |
| **TailwindCSS** | 3.x | Styling |

### Quantitative Engine
| Library | Purpose |
|---------|---------|
| **vectorbt** | Vectorised backtesting |
| **Riskfolio-Lib** | Portfolio optimisation (HRP) |
| **scikit-learn** | ML models (meta-labeling, regime detection) |
| **pandas** | Data manipulation |
| **yfinance** | Market data API |

### Infrastructure
| Service | Purpose |
|---------|---------|
| **AWS Lambda** | Serverless compute |
| **Neon PostgreSQL** | Serverless database |
| **Cloudflare Pages** | Frontend hosting |
| **AWS S3** | Parquet cache storage |
| **AWS Secrets Manager** | Credential management |
| **Terraform** | Infrastructure as code |

---

## 9. Security & Configuration

### Authentication
- API token-based authentication
- AWS Secrets Manager for credential storage
- Environment-based configuration

### Configuration Management
```python
# Environment Variables
DATABASE_URL=postgres://...@neon.tech/...
CLOUDFLARE_API_TOKEN=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### Security Scope
- ✅ Input validation via Pydantic
- ✅ No secrets in logs
- ✅ Parameterised SQL queries
- ❌ Multi-tenant isolation (single-user MVP)
- ❌ OAuth2/SSO integration

---

## 10. API Specification

### Core Endpoints

#### GET /api/signals
Returns current trading signals for the portfolio.

```json
{
  "signals": [
    {
      "ticker": "VOO",
      "action": "BUY",
      "strength": 0.82,
      "regime": "BULL"
    }
  ]
}
```

#### GET /api/portfolio
Returns current portfolio allocation.

```json
{
  "allocations": [
    {"ticker": "VTI", "weight": 0.15},
    {"ticker": "VEA", "weight": 0.12}
  ],
  "total_value_aud": 125000.00
}
```

#### GET /api/health
Health check endpoint.

```json
{
  "status": "healthy",
  "database": "connected",
  "cache": "valid"
}
```

---

## 11. Success Criteria

### MVP Success Definition
A fully functional quantitative trading platform deployed to production with:
- Real-time dashboards accessible at `potential-parakeet.pages.dev`
- API responding at `api.potential-parakeet.com`
- Data refresh completing in <30 seconds daily

### Functional Requirements
- ✅ 96%+ test pass rate
- ✅ 240x data loading speedup (via caching)
- ✅ <500ms API response time
- ✅ Dashboard loads in <3 seconds
- ✅ Successful Cloudflare + AWS deployment

### Quality Indicators
- Sharpe Ratio > 0.8 (Quant 1.0)
- Sharpe Ratio > 1.0 (Quant 2.0)
- Max Drawdown < 30%
- Win Rate > 50%

---

## 12. Implementation Phases

### Phase 1: Core Infrastructure ✅
**Goal**: Establish data pipeline and basic strategy engine

**Deliverables**:
- ✅ FastDataLoader with incremental caching
- ✅ 741-ticker universe definition
- ✅ Parquet-based cache system
- ✅ Retry logic with error classification

**Validation**: `python strategy/fast_data_loader.py --test` completes successfully

---

### Phase 2: Strategy Implementation ✅
**Goal**: Implement Quant 1.0 and Quant 2.0 strategies

**Deliverables**:
- ✅ Dual Momentum + HRP (Quant 1.0)
- ✅ OLMAR, Meta-Labeling, Regime Detection (Quant 2.0)
- ✅ Quallamaggie Scanner
- ✅ Backtesting framework

**Validation**: Backtest results match expected performance metrics

---

### Phase 3: API & Dashboard ✅
**Goal**: Build user-facing interfaces

**Deliverables**:
- ✅ FastAPI REST API (96% test pass rate)
- ✅ React dashboards (Quant 1.0 & 2.0)
- ✅ Bi-temporal database schema
- ✅ 100 sample trades seeded

**Validation**: Dashboard displays live data; API health check passes

---

### Phase 4: Serverless Deployment 🔄
**Goal**: Production-ready cloud deployment

**Deliverables**:
- ✅ Cloudflare Pages (frontend)
- ✅ Terraform infrastructure
- ✅ Neon PostgreSQL integration
- 🔄 AWS Lambda API deployment
- 🔄 CI/CD pipeline

**Validation**: End-to-end user flow works on production URLs

---

## 13. Future Considerations

### Post-MVP Enhancements
- **Broker Integration**: Direct execution via Stake API, Interactive Brokers
- **Live Trading Mode**: Real-time signal execution with risk limits
- **Mobile App**: iOS/Android portfolio monitoring
- **Multi-Currency**: Support for GBP, EUR base currencies

### Integration Opportunities
- TradingView webhook integration
- Slack/Telegram notifications
- Portfolio tracking aggregators (Sharesight)

### Advanced Features
- Options strategy execution (VRP, Iron Condor)
- Alternative data sources (sentiment, satellite)
- Reinforcement learning strategy optimisation

---

## 14. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Data API Rate Limits** | Strategy execution delayed | Exponential backoff, 60s rate limit delay, local Parquet cache |
| **Currency Volatility** | AUD normalisation errors | Daily FX rate refresh, fallback to previous close |
| **Model Overfitting** | Poor out-of-sample performance | Walk-forward validation, holdout test sets, regime-aware testing |
| **Cloud Cost Overrun** | Exceed $5/month budget | Lambda concurrency limits, S3 lifecycle policies, usage alerts |
| **Database Connection Limits** | API failures under load | Neon connection pooling, async SQLAlchemy, circuit breaker pattern |

---

## 15. Appendix

### Related Documents
- [SETUP_GUIDE.md](../docs/SETUP_GUIDE.md) - AWS & Neon setup instructions
- [MIGRATION_PROGRESS.md](../docs/MIGRATION_PROGRESS.md) - Serverless migration status
- [FINAL_DEPLOYMENT_REPORT.md](../docs/FINAL_DEPLOYMENT_REPORT.md) - Deployment qualification
- [E2E_TESTING_GUIDE.md](../docs/testing/E2E_TESTING_GUIDE.md) - End-to-end testing guide

### Key Dependencies
| Package | URL |
|---------|-----|
| vectorbt | https://github.com/polakowo/vectorbt |
| Riskfolio-Lib | https://github.com/dcajasn/Riskfolio-Lib |
| yfinance | https://github.com/ranaroussi/yfinance |

### Repository
- **GitHub**: `alphawizards/potential-parakeet`
- **Frontend**: `https://potential-parakeet.pages.dev`
- **API**: `https://api.potential-parakeet.com`

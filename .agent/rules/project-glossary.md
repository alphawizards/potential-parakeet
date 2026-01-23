---
description: Domain-specific glossary and business rules for Australian Retail Investing.
trigger: always_on
---

# Project Glossary & Business Rules

Definitions and invariants specific to the Australian Retail Investing domain of `potential-parakeet`.

## Key Terms

*   **Quant 1.0:**
    *   *Definition:* Conservative, tax-aware strategies.
    *   *Focus:* Long-term hold (>12 months) to capture CGT Discount.
    *   *Examples:* Dual Momentum, Strategic Asset Allocation.
*   **Quant 2.0:**
    *   *Definition:* Aggressive, yield-focused strategies.
    *   *Focus:* High turnover permitted if yield > transaction costs.
    *   *Examples:* Stat Arb, Volatility Harvesting, Intraday Momentum.
*   **The AUD Standard:**
    *   *Rule:* All non-AUD assets (e.g., AAPL, SPY) must be normalized to AUD using daily FX rates before any return or volatility calculation.
*   **Friction:**
    *   *Constant:* **$3.00 AUD** flat fee per trade (Brokerage).
    *   *Slippage:* Assume 0.05% for large cap, 0.1% for mid-cap.

## Entity Definitions

*   **`FastDataLoader`:** The proprietary data ingestion engine. It manages the `cache/` directory (Parquet files). It is the *only* approved way to fetch OHLCV data.
*   **`Signal`:** A raw output from a strategy (Buy/Sell/Hold).
*   **`Order`:** A sized and vetted instruction sent to the broker API.

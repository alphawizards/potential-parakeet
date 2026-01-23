---
description: Hard technical constraints for the potential-parakeet project.
trigger: always_on
---

# Technology Stack & Constraints

This file defines the hard technical boundaries of the `potential-parakeet` project.

## Frontend / Edge
*   **Framework:** React (Vite)
    *   *Constraint:* Do NOT introduce Next.js, Astro, or Remix.
    *   *Reasoning:* Preserves existing Cloudflare Pages build pipeline (`npm run build`).
*   **Hosting:** Cloudflare Pages
*   **Styling:** TailwindCSS
    *   *Constraint:* Mobile-first, Dark-mode optimized.
*   **Build Output:** `dashboard/dist`

## Backend / Compute
*   **Runtime:** Python 3.10+ & Node.js 18+
*   **API Framework:** FastAPI (Python)
*   **Serverless:**
    *   AWS Lambda (Containerized Python)
    *   Cloudflare Workers (Node.js/JavaScript)
*   **Database:** Neon (Serverless PostgreSQL)
    *   *Driver:* `asyncpg` (Python), `postgres` (JS)
    *   *ORM:* SQLAlchemy (Async)

## Quantitative Engine
*   **Core Libraries:**
    *   `vectorbt` (Backtesting)
    *   `Riskfolio-Lib` (Portfolio Optimization)
    *   `pandas` (Data Manipulation - STRICTLY VECTORIZED)
    *   `scikit-learn` (ML Models)
*   **Data Ingestion:**
    *   `FastDataLoader` (Custom Parquet-based incremental cache).
    *   *Constraint:* **NEVER** bypass the cache. Always check `cache/` first.

## Infrastructure as Code (IaC)
*   **Source of Truth:** Terraform (`infrastructure/terraform`) for AWS.
*   **Edge Config:** `wrangler.toml` for Cloudflare.

---
description: Standard operating procedure for deploying potential-parakeet to Production.
---

# Workflow: Deploy to Production

Follow this strictly to ensure a safe release.

## 1. Pre-Flight Checks
*   [ ] **Build Verification:** Run `npm run build` locally.
    *   *Constraint:* Must exit with code 0.
*   [ ] **Test Suite:** Run `pytest` on the backend.
    *   *Constraint:* All critical quant tests must pass.
*   [ ] **Environment:** Check `wrangler.toml` (or `wrangler.json`) for correct production `env_vars`.

## 2. Frontend Deployment (Cloudflare Pages)
*   **Command:** `npx wrangler pages deploy dashboard/dist --project-name potential-parakeet`
*   **Verification:**
    *   Visit the Preview URL provided by Wrangler.
    *   Check the browser console for specific 404s on assets.

## 3. Backend Deployment (AWS/Workers)
*   **AWS:** Check Terraform state: `cd infrastructure/terraform && terraform plan`.
*   **Cloudflare Workers (if applicable):** `npx wrangler pages deploy dashboard/dist --project-name potential-parakeet`.
    *   *Note:* For Workers scripts (not Pages), use `npx wrangler deploy`.

## 4. Post-Deployment Verification
*   **Health Check:** `curl https://api.potential-parakeet.com/health` -> Expect `200 OK`.
*   **Live Smoke Test:** Log in to the dashboard and load "Today's Picks".

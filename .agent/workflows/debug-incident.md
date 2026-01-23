---
description: Protocol for diagnosing and fixing bugs or incidents.
---

# Workflow: Debug Incident

Use this when the user reports an error or a system failure.

## 1. Triage & Isolation
*   **Stop:** Do NOT blindly try fixes.
*   **Gather Data:**
    *   *Frontend:* Check Browser Console & Network Tab (HAR).
    *   *Backend:* Check CloudWatch Logs (AWS) or `wrangler tail` (Cloudflare).
    *   *Data:* Is the `cache/` corrupted?
*   *Verification:* Document the exact error message and stack trace.

## 2. Reproduction
*   Create a minimal reproduction script (Python) or test case (Jest/Playwright).
*   *Artifact:* `repro_incident_NAME.py`
*   *Verification:* Script must reproduce the error consistently.

## 3. Analysis (Root Cause)
*   Trace the error to the source (Code vs. Config vs. Data).
*   Check `memory/decisions.md`—did a recent change cause this?
*   *Verification:* Identify the exact line/config causing the issue.

## 4. Fix & Verify
*   Implement the fix.
*   **Regression Test:** Run the reproduction script from Step 2. It must now pass.
*   **System Test:** Run the full `pytest` suite to ensure no side effects.
*   *Verification:* All tests pass; no new errors in logs.

## 5. Post-Mortem
*   Update `memory/decisions.md` if a process changed.
*   Explain the "Why" to the user.
*   *Verification:* ADR entry created if architectural change was made.


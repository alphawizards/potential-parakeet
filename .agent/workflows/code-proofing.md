---
description: Generates the proof that the code works.
---

I am ready to ship this logic.

1. Generate a `pytest` unit test (for Python) or `Vitest` file (for JS) that covers:
   - The happy path.
   - One critical edge case (e.g., null data, API failure).
   *Verification:* Test file must run and pass with `pytest -v` or `npm test`.
   *Output:* `tests/test_{module_name}.py` or `dashboard/src/__tests__/{module}.test.ts`

2. Create a brief Markdown snippet for the `README.md` explaining how to deploy or run this specific module.
   *Verification:* README section includes prerequisites, commands, and expected output.

3. If this requires infrastructure, provide the CLI command to deploy it (e.g., `cdk deploy` or `wrangler pages deploy`).
   *Verification:* Command must be copy-pasteable and include all required flags.


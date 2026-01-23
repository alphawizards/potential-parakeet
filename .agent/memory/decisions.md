# Architecture Decision Records (ADR)

Use this file to log significant architectural decisions, their context, and consequences. This provides long-term memory for the agent.

## Format
```markdown
### [ADR-00X] Title
*   **Date:** YYYY-MM-DD
*   **Status:** Accepted / Deprecated / Proposed
*   **Context:** Why did we need to make this decision?
*   **Decision:** What did we do?
*   **Consequences:** What became easier? What became harder?
```

## Records

### [ADR-001] Adoption of Agent Configuration Architecture
*   **Date:** 2026-01-23
*   **Status:** Accepted
*   **Context:** The agent lacked a structured way to manage diverse constraints (tech stack vs. business rules) and persistent memory, leading to potential context drift.
*   **Decision:** Implemented the "Antigravity" architecture:
    *   `rules/` for static constraints.
    *   `workflows/` for repeatable procedures.
    *   `memory/` for long-term context.
*   **Consequences:**
    *   *Positive:* clearer separation of concerns; verifiable workflows; persistent architectural memory.
    *   *Negative:* Slight overhead in maintaining multiple documentation files.

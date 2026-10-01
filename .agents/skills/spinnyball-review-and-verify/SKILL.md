---
name: spinnyball-review-and-verify
description: >-
  Pre-completion 3-subagent review fleet and verification gate for SpinnyBall.
  Audits physical balance laws, workbench web worker & canvas UI, and Python research isolation.
---

# SpinnyBall Review Fleet & Verification Gate

Invoke this skill before committing or completing any feature, model addition, or refactor in SpinnyBall.

## 1. Automated Verification Checks
Before dispatching reviewers, run the deterministic test and code-graph suites:
```bash
# 1. Physics Engine Tests (19 tests, analytic convergence & balance laws)
node --test workbench/tests/physics.test.mjs

# 2. Python Launcher & Server Tests
python -m unittest discover -s workbench/tests -p "test_*.py"

# 3. Gauthier AST Code Graph Symbol Sanity
node scripts/code_graph.mjs --stats
```

## 2. The 3-Subagent Pro Review Fleet

Dispatch three independent reviewer subagents concurrently via `invoke_subagent` (`Model: "pro"`):

### Reviewer 1: Numerical Physics & Balance Law Auditor
- **Focus**:
  - Verify that energy and momentum balance residuals stay below the 0.1% reference scale threshold.
  - Verify that fixed-step integration maintains symplectic/analytic properties and the final step lands on the exact requested duration.
  - Verify that no synthetic or fabricated states are accepted as evidence on JSON import (`readExperiment`).
  - **Adversarial Gate (`hamel-shankar-eval-and-trace-taxonomy`)**: Check for **Broken Tests** (e.g. tests assuming unrealistic zero-drag orbit closure under surface collision) or **Poisoned Assumptions** (e.g. claiming mass-stream propulsion without center-of-mass momentum balance).

### Reviewer 2: Workbench UI & Web Worker Auditor
- **Focus**:
  - Inspect `workbench/worker.mjs` $\leftrightarrow$ `workbench/app.mjs` message passing.
  - Verify zero external npm or CDN dependencies (pure browser ES modules only).
  - Verify Canvas 2D rendering performance, scrubable timeline, and focus rings (`outline: 3px solid var(--orange)`).

### Reviewer 3: Research Isolation & Python Integrity Auditor
- **Focus**:
  - Ensure experimental Python code in `dynamics/`, `control_layer/`, or `scripts/` never gets imported into `workbench/`.
  - Verify that any Python modifications compile cleanly (`python -m py_compile`) and respect existing package structures.

## 3. Findings Consolidation
Categorize all findings into **Critical (blocking)**, **Important (non-blocking)**, and **Minor**. Resolve all Critical issues via TDD before committing.

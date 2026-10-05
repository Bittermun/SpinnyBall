# Trajectory Invariant Audit & Verification Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a zero-dependency, headless physical verification oracle and CLI that audits external trajectory files against orbital conservation laws with SpinnyBall's strict 0.1% balance threshold.

**Architecture:** Pure ES modules: `workbench/audit.mjs` provides mathematical trajectory inspection and CSV/JSON normalization functions; `workbench/verify-external.mjs` provides CLI streaming/file execution with JSON/summary output; `workbench/tests/audit.test.mjs` validates with Node's native test runner against synthetic Kepler orbits, noisy data, and decaying integrators.

**Tech Stack:** Node.js 20+ ES Modules (`node:test`, `node:assert/strict`, `node:fs/promises`, `node:readline`). Zero npm dependencies.

**Spec:** `docs/superpowers/specs/2026-10-05-trajectory-audit-oracle-design.md`

## Global Constraints
- Zero external runtime or test dependencies.
- Strict SI units.
- Balance threshold: `< 0.1%` (1e-3) of reference scale flags a warning.
- Pure JS ES module, runs in Node 20+.
- Supports both 2D planar and full 3D Cartesian coordinates.

## Review Focus
1. Near-parabolic orbits ($E_0 \to 0$): must not divide by zero; must use gravitational potential scale $\mu / r_0$.
2. Variable / non-uniform time steps $\Delta t$: must use weighted non-uniform finite-difference acceleration.
3. Malformed or single-point CSV/JSON: must throw clean, descriptive physical validation errors without crashing.
4. Central-body penetration: must flag trajectory points passing below $r_{\text{body}}$.
5. Preserves existing SpinnyBall test suite (`physics.test.mjs` must remain 100% green).

---

### Task 1: Mathematical Audit Core (`workbench/audit.mjs`) & Tests

**Files:**
- Create: `workbench/audit.mjs`
- Create: `workbench/tests/audit.test.mjs`

**Interfaces:**
- Produces:
  - `parseTrajectoryData(text, format?): Array<{ t: number, r: [number, number, number], v: [number, number, number] }>`
  - `sampleOrbitInvariants(sample, mu): { energy: number, h: [number, number, number], eVec: [number, number, number] }`
  - `auditOrbitTrajectory(samples, options): { passed: boolean, metrics: object, warnings: string[], sampleCount: number, duration: number }`

- [ ] **Step 1: Write the failing tests for audit core**
Create `workbench/tests/audit.test.mjs` testing perfect Kepler orbits (pass), Euler-drift unphysical orbits (fail), near-zero energy protection, and CSV parsing.

- [ ] **Step 2: Run test to verify it fails**
Run: `node --test workbench/tests/audit.test.mjs`
Expected: FAIL (module `audit.mjs` not found).

- [ ] **Step 3: Implement `workbench/audit.mjs`**
Implement vector math, invariant calculators ($\mathcal{E}$, $\vec{h}$, $\vec{e}$ Laplace-Runge-Lenz), variable-step acceleration residuals, CSV/JSON parser, and threshold validation.

- [ ] **Step 4: Run test to verify it passes**
Run: `node --test workbench/tests/audit.test.mjs`
Expected: PASS.

- [ ] **Step 5: Run existing physics test suite**
Run: `node --test workbench/tests/physics.test.mjs`
Expected: 19 tests pass.

---

### Task 2: Standalone CLI Oracle (`workbench/verify-external.mjs`)

**Files:**
- Create: `workbench/verify-external.mjs`

**Interfaces:**
- Consumes: `auditOrbitTrajectory`, `parseTrajectoryData` from `./audit.mjs`.
- CLI syntax:
  - `node workbench/verify-external.mjs <trajectory-file> [--mu <mu>] [--radius <r>] [--json] [--strict]`
  - Stdin support: `cat trajectory.csv | node workbench/verify-external.mjs -`

- [ ] **Step 1: Add CLI integration test in `workbench/tests/audit.test.mjs`**
Test executing `verify-external.mjs` via `child_process.execFile` on generated test files and validating exit codes (0 for pass, 1 for fail or flag `--strict`).

- [ ] **Step 2: Run test to verify it fails**
Run: `node --test workbench/tests/audit.test.mjs`
Expected: FAIL (`verify-external.mjs` does not exist).

- [ ] **Step 3: Implement `workbench/verify-external.mjs`**
Implement argument parsing, file reading / stdin streaming, formatting terminal summary or JSON output, and exit code logic.

- [ ] **Step 4: Run test to verify it passes**
Run: `node --test workbench/tests/audit.test.mjs`
Expected: PASS.

---

### Task 3: Documentation & Verification Evidence

**Files:**
- Modify: `README.md` (Add Oracle usage section)
- Modify: `ARCHITECTURE.md` (Document audit component)

- [ ] **Step 1: Update documentation with usage examples for external repositories**
- [ ] **Step 2: Run all repository tests (Node & Python tests)**
Run:
```bash
node --test workbench/tests/audit.test.mjs
node --test workbench/tests/physics.test.mjs
python -m unittest discover -s workbench/tests -p "test_*.py"
```
Expected: All suites pass cleanly.

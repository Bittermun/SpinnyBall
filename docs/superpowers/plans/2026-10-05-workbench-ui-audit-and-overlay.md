# Implementation Plan: UI External Audit & Kepler Analytic Overlay

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact Keplerian reference ellipse rendering to the orbit canvas and an external trajectory audit drag-and-drop / file picker dialog in the workbench.

**Architecture:** 
- `workbench/physics.mjs`: Export helper `keplerOrbitPoints(config, count = 120)` to compute exact parametric points $(x, y)$ for closed orbits.
- `workbench/app.mjs`: Integrate `keplerOrbitPoints` in `drawOrbit` as a faint reference curve; add drag-and-drop listener to the canvas stage and file picker button; call `parseTrajectoryData` and `auditOrbitTrajectory` from `audit.mjs` and display a clean audit modal.
- `workbench/index.html`: Add `#auditButton`, `#auditFile`, and `<dialog id="auditModal">`.
- `workbench/style.css`: Add styling for drag-over state and audit modal scorecard.

**Tech Stack:** Native HTML5 Drag and Drop, Canvas 2D API, ES Modules, zero external dependencies.

**Spec:** `docs/superpowers/specs/2026-10-05-workbench-ui-audit-and-overlay-design.md`

## Global Constraints
- Zero external dependencies.
- Retain SpinnyBall's strict invariant rule: uploaded files are audits/comparisons only, never internal ground-truth runs.
- Max audit file size: 8 MB.

---

### Task 1: Kepler Analytic Reference Generator in `physics.mjs`

**Files:**
- Modify: `workbench/physics.mjs`
- Test: `workbench/tests/physics.test.mjs`

- [x] **Step 1: Write test for `keplerOrbitPoints`**
Verify that points computed by `keplerOrbitPoints(config)` satisfy the exact Kepler equation and form a closed ellipse for circular/elliptical configurations.

- [x] **Step 2: Run test to verify it fails**
Run: `node --test workbench/tests/physics.test.mjs`
Expected: FAIL (`keplerOrbitPoints` is not defined).

- [x] **Step 3: Implement `keplerOrbitPoints(config, numPoints = 120)` in `physics.mjs`**
Use orbital elements ($a$, $e$) to generate $[x, y]$ parametric points along eccentric anomaly $E \in [0, 2\pi]$.

- [x] **Step 4: Run test to verify it passes**
Run: `node --test workbench/tests/physics.test.mjs`
Expected: PASS.

---

### Task 2: Web UI Audit Modal & Drag-and-Drop Integration

**Files:**
- Modify: `workbench/index.html`
- Modify: `workbench/style.css`
- Modify: `workbench/app.mjs`

- [x] **Step 1: Add HTML dialog and buttons in `workbench/index.html`**
Add `<button id="auditButton">Audit external file</button>`, `<input id="auditFile" type="file">`, and `<dialog id="auditModal">`.

- [x] **Step 2: Add CSS rules in `workbench/style.css`**
Add styles for `.stage.dragover`, `#auditModal`, scorecard badges, and metrics table.

- [x] **Step 3: Update `workbench/app.mjs`**
- Import `keplerOrbitPoints` from `./physics.mjs`.
- Render the analytic reference curve in `drawOrbit`.
- Import `parseTrajectoryData` and `auditOrbitTrajectory` from `./audit.mjs`.
- Wire drag-and-drop on `#stage` and `#auditButton` / `#auditFile`.
- Show `#auditModal` with scorecard metrics and warnings.

- [x] **Step 4: Run all automated verification tests**
Run:
```bash
node --test workbench/tests/physics.test.mjs workbench/tests/audit.test.mjs
```
Expected: All tests pass cleanly.


# Interactive Diagnostics and Orbital Speed Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users inspect exact retained samples and run, inspect, compare, and reproduce a deterministic sweep of orbital launch speed.

**Architecture:** Keep `physics.mjs` as the single-run numerical authority. Put sweep orchestration and serialization in a pure `sweep.mjs`, run it in a second module Worker, and leave ordinary playback on its existing Worker. Use the existing Canvas plots and one new sweep Canvas; UI-only code stays in `app.mjs`.

**Tech Stack:** Browser ES modules, Canvas 2D, module Web Workers, Node.js 20+ test runner, Python 3.10+ standard-library launcher. No npm packages, CDN resources, or build step.

**Spec:** `docs/superpowers/specs/2026-10-06-interactive-diagnostics-and-orbit-sweep-design.md`

## Global Constraints

- Read root and `workbench/AGENTS.md` before editing; preserve the zero-dependency/browser-worker boundary and research-code separation.
- Use SI internally. Speed is dimensionless × circular speed; displayed distance is km; report energy in J/kg and balance residuals as percent of documented scales.
- Preserve engine `1.0.0`, ordinary experiment schema `1`, 100,000-step single-run cap, and the existing 0.1% numerical warning convention. A sweep has its own schema `1`.
- Sweep count is an integer 3–21 and conservative total work is at most 200,000 integration steps. Never trust imported states, rows, or diagnostics.
- An initial positive-energy classification means “unbound in this idealized two-body model,” not “escaped within the finite run” or mission feasibility.
- Existing imported/ordinary runs, pinned comparisons, playback, audit overlays, and exports must keep working.

## Review Focus

1. **Different retained time grids:** inspecting a pinned comparison must display each run's actual nearest sample time; Task 2 tests this.
2. **Surface crossing:** a sweep point must show its last exterior state and “stopped before surface crossing”; Task 3 tests this.
3. **Forged or incompatible JSON:** import must recompute and reject unsupported versions; Task 5 tests this.
4. **Cancellation or tab switch during work:** late Worker messages must not replace current results; Task 4 exercises this.
5. **Hostile uploaded file name:** external audit rendering must insert it as text, never HTML; Task 1 checks this in a browser.

## File map

| File | Responsibility |
|---|---|
| `workbench/physics.mjs` | Correct the existing analytic orbit overlay orientation for sub-circular launch speed; otherwise keep single-run numerical semantics. |
| `workbench/plot-data.mjs` (new) | Pure nearest-retained-sample selection for plot inspection. |
| `workbench/sweep.mjs` (new) | Validate, run, and serialize orbital speed sweeps; no DOM or Worker globals. |
| `workbench/worker.mjs` | Dispatch ordinary runs and sweep requests; report sweep progress. |
| `workbench/app.mjs` | Plot inspection; sweep Worker lifecycle, controls, chart/table, import/export; safe audit dialog DOM. |
| `workbench/index.html`, `workbench/style.css` | Accessible inspection readout and orbit-only sweep panel. |
| `workbench/tests/physics.test.mjs`, `plot-data.test.mjs`, `sweep.test.mjs` | Analytic and edge-case tests. |
| `.github/workflows/workbench.yml`, `README.md`, `ARCHITECTURE.md`, `docs/WORKBENCH_MODELS.md`, `docs/WORKBENCH_VALIDATION.md` | CI and final contract/evidence. |

Before Task 1, work on a `codex/` branch or isolated worktree. Record `git status --short --branch`, the baseline commit, and the results of `node --test workbench/tests/physics.test.mjs workbench/tests/audit.test.mjs` and `python -m unittest discover -s workbench/tests -p "test_*.py"`. Preserve unrelated changes; never reset them. Do not treat the historical Python suite as a workbench gate.

### Task 1: Correct reference geometry and harden external-audit text

**Files:** Modify `workbench/physics.mjs`, `workbench/tests/physics.test.mjs`, and `workbench/app.mjs`.

**Interfaces:** `keplerOrbitPoints(config, numPoints = 120)` keeps its existing signature and return type. Audit dialog element IDs stay unchanged.

- [ ] **Step 1: Write a failing sub-circular reference test.** In `physics.test.mjs`, use `configFor('orbit', 0)` with `speed: 0.8`: the first reference point must equal `[radius, 0]` within floating tolerance, and the last must close on the first. Keep a super-circular case (`speed: 1.18`) to prevent reversing both orientations.
- [ ] **Step 2: Run the focused test.** `node --test workbench/tests/physics.test.mjs`; expect the new `0.8` alignment assertion to fail with current code.
- [ ] **Step 3: Fix only ellipse orientation.** In `keplerOrbitPoints`, for `speed < 1`, use the apoapsis-at-`+x` orientation (`x = a(e + cos E)`, `y = b sin E`); retain current periapsis orientation for `speed >= 1`. Do not alter integration or `orbitElements`.
- [ ] **Step 4: Replace audit `innerHTML` of untrusted text.** Build the existing scorecard using DOM nodes and `textContent` for the uploaded file name and warning strings. Static labels can remain markup. Preserve numeric formatting and visual status. Search `workbench/app.mjs` for any remaining file-derived `innerHTML`.
- [ ] **Step 5: Verify.** Run the physics and audit Node tests. In a local browser, audit a `File` whose name contains `<img src=x onerror=...>`; the dialog must display those characters as text, create no image node, and execute no handler. Run `git diff --check`; commit this focused repair.

### Task 2: Inspect exact diagnostic samples

**Files:** Create `workbench/plot-data.mjs` and `workbench/tests/plot-data.test.mjs`; modify `workbench/index.html`, `workbench/style.css`, and `workbench/app.mjs`.

**Interfaces:** Export `nearestSampleAtTime(samples, time) -> { index, sample } | null`. `samples` is the existing ascending `result.samples` array. Ties choose the earlier sample. Never synthesize a state.

- [ ] **Step 1: Write failing helper tests.** Cover empty samples, endpoint clamping, irregular times `[0, 1, 1.6, 4]` at `1.4`, exact matches, a tie, and non-finite requested time. Verify the return object is one of the original sample objects.
- [ ] **Step 2: Run `node --test workbench/tests/plot-data.test.mjs`;** expect missing-module/function failure.
- [ ] **Step 3: Implement `nearestSampleAtTime`** with binary search and the stated tie rule. Reject non-finite target time; do not interpolate. Run the focused tests until green.
- [ ] **Step 4: Add plot inspection UI.** Give both plot canvases `tabindex="0"` and a shared visible `#plotInspection` readout. Pointer move/tap maps the x coordinate into the chart's displayed time range, then selects the nearest retained sample. ArrowLeft/ArrowRight change its index by one; Home/End jump; Escape clears. Reuse the plot's current value conversion (`energyError * 100` for percent, `signal` with `signalLabel` for the measurement). Add a visible point marker and matching `aria-label`/text. Keep the ordinary replay cursor independent.
- [ ] **Step 5: Handle pinned runs honestly.** At the selected main-run time, call `nearestSampleAtTime(pinned.samples, selected.t)` separately. Print both actual times and values; do not claim simultaneous states when time grids differ. Clear inspection when the displayed model changes or no result exists.
- [ ] **Step 6: Verify in a browser.** Inspect orbit and spin via mouse and keyboard, a touch viewport, and two pinned runs with different `dt`/duration. Check focus indication, no interpolation claim, no horizontal overflow, and no console errors. Run focused Node tests and `git diff --check`; commit.

### Task 3: Build the deterministic sweep core

**Files:** Create `workbench/sweep.mjs` and `workbench/tests/sweep.test.mjs`.

**Interfaces:**

- `validateOrbitSweep(baseConfig, range) -> { baseConfig, range, speeds }`, where `range` has numeric `minSpeed`, `maxSpeed`, integer `count`.
- `runOrbitSweep(baseConfig, range, onProgress = () => {}) -> SweepResult`, with `rows` ordered by speed. Each row has `speed`, `initialEnergyJPerKg`, `classification`, `status`, `finalRadiusKm`, `finalTimeS`, `maxEnergyError`, `maxMomentumError`.
- `MAX_SWEEP_RUNS = 21`; `MAX_SWEEP_STEPS = 200000`. Use `simulate({ ...baseConfig, speed }, 2)` for each point. Return the validated base configuration and range in `SweepResult`.

- [ ] **Step 1: Write failing tests.** For circular orbit config and range 1.3–1.5 with 5 points, expect inclusive speeds `[1.3, 1.35, 1.4, 1.45, 1.5]` within tolerance. Check `initialEnergyJPerKg = mu / radius * (speed²/2 − 1)`, 1.4 `bound`, 1.45 `unbound`, `√2` itself non-negative, progress callbacks `(completed,total)` 1/5 through 5/5, and each row's residual maxima against direct `simulate(config, 2)`. Add invalid type/range/count/work-budget tests and a surface-stop case showing `finalRadiusKm > surface/1000` and `status === 'surface'`.
- [ ] **Step 2: Run `node --test workbench/tests/sweep.test.mjs`;** expect missing module failure.
- [ ] **Step 3: Implement validation and orchestration.** Validate with `validateConfig`; require model `orbit`, finite ordered speed bounds within 0.1–2, integer count 3–21, and `count * ceil(duration/dt) <= 200000`. Generate endpoints without rounding input values. Derive classification from analytic initial energy, not final radius. Call `onProgress` once per completed run. Keep only row summaries in the sweep result.
- [ ] **Step 4: Verify focused tests, both existing Node suites, and `git diff --check`;** commit. Do not add probability or uncertainty language.

### Task 4: Add a cancellable Worker and a complete sweep interaction

**Files:** Modify `workbench/worker.mjs`, `workbench/app.mjs`, `workbench/index.html`, and `workbench/style.css`.

**Interfaces:** Preserve the existing ordinary message `{ id, config } -> { id, result | error }`. New sweep messages are `{ id, kind: 'orbit-speed-sweep', config, range }`, progress `{ id, kind: 'sweep-progress', completed, total }`, and terminal `{ id, kind: 'sweep-result', result | error }`.

- [ ] **Step 1: Add an orbit-only sweep panel** near the existing result actions: min/max speed inputs with “× circular speed” units, integer count, Run sweep and Cancel buttons, progress text, a Canvas chart, an exact-value table, and a concise note about √2 and finite duration. Hide it in spin/exchange without deleting current ordinary results.
- [ ] **Step 2: Extend the Worker dispatcher** to call `runOrbitSweep` for new requests and post progress after each point; keep ordinary requests unchanged.
- [ ] **Step 3: Use a dedicated `sweepWorker` in `app.mjs`.** Reject dirty/unapplied main forms with an “Apply & run first” status. Validate synchronously before posting. Disable Run while busy, show Cancel, guard every response by request ID. Cancel with `terminate()`, increment request ID, and create a fresh Worker for the next sweep. Do the same on laboratory switch. Leave the ordinary Worker and displayed run unaffected.
- [ ] **Step 4: Render the sweep.** Draw final exterior distance (km) against speed, a labeled √2 vertical reference, and status-specific markers. Show a table with all row fields and units, including a non-misleading surface-stop label and balance residual percentages. Use a native button in each row so keyboard users can select it. Selection sets the main speed control and runs a normal full-sample experiment; keep the sweep visible and retain any pinned ordinary run.
- [ ] **Step 5: Exercise the browser flow.** Sweep the default orbit from 1.3 to 1.5 (5 points), inspect points on either side of √2, select a row, pin/compare, cancel a long sweep, switch labs during a sweep, and run again. Confirm no stale results, no UI lock-up or console errors, and a usable 390 px viewport. Run Node suites and `git diff --check`; commit.

### Task 5: Export and recompute sweep experiments

**Files:** Modify `workbench/sweep.mjs`, `workbench/tests/sweep.test.mjs`, `workbench/app.mjs`, `workbench/index.html`, and `workbench/style.css`.

**Interfaces:** Export `toSweepCSV(result) -> string` and `readOrbitSweep(value) -> { baseConfig, range }`. JSON result has `kind: 'orbit-speed-sweep'`, `sweepSchemaVersion: 1`, `engineVersion`, base configuration, range, generated rows, method, assumptions, units; the UI adds `exportedAt`.

- [ ] **Step 1: Write failing interchange tests.** CSV has one header and one row per speed, with unit-bearing column names and consistent column counts. `readOrbitSweep` accepts a valid result's version/config/range, rejects a foreign version or malformed range, and returns the same validated input even when uploaded `rows` contain forged physics or script-like strings. Ordinary `readExperiment` behavior must remain unchanged.
- [ ] **Step 2: Run the focused test** and confirm the new functions are absent.
- [ ] **Step 3: Implement pure serialization/validation.** Import reads only trusted configuration fields before rerunning through the sweep Worker. Do not copy uploaded rows, status or diagnostics into displayed results. Keep export order deterministic apart from UI-added `exportedAt`.
- [ ] **Step 4: Add Save sweep JSON / CSV actions and import detection.** Existing experiment JSON continues through `readExperiment`; `kind: 'orbit-speed-sweep'` goes through `readOrbitSweep`, switches to Orbital motion, applies the saved base configuration, and recomputes the sweep. Keep the 8 MB upload limit and give actionable errors for version/range/work-budget failures. Exports always describe the currently displayed sweep.
- [ ] **Step 5: Verify a browser save → import → recompute round trip** by comparing result rows after excluding `exportedAt`; verify forged rows are ignored and a normal experiment import still works. Run all Node suites and `git diff --check`; commit.

### Task 6: Documentation, CI, and final gate

**Files:** Modify `.github/workflows/workbench.yml`, `README.md`, `ARCHITECTURE.md`, `docs/WORKBENCH_MODELS.md`, and `docs/WORKBENCH_VALIDATION.md`. Add a representative sweep JSON fixture only if tests genuinely need it; avoid checked-in generated output otherwise.

- [ ] **Step 1: Update CI's Node command** to run `physics.test.mjs`, `audit.test.mjs`, `plot-data.test.mjs`, and `sweep.test.mjs`, plus syntax checks for every added `.mjs` file.
- [ ] **Step 2: Document one worked sweep.** From `python -m workbench`, use 1.3–1.5 with 5 points, identify the analytic √2 threshold, select 1.45 for ordinary replay, save JSON/CSV, and re-import JSON. Explain that initial energy classification and finite-duration distance are different observations. Document sample inspection and its exact-sample behavior.
- [ ] **Step 3: Update model/architecture/validation docs** with sweep work cap, two-sample retention versus all-step balance maxima, Worker cancellation, schema, units, analytic source, and recorded test/browser results. State that this is parameter sensitivity, not a statistical uncertainty distribution or a calibrated mission model.
- [ ] **Step 4: Run final checks:** `node --test workbench/tests/physics.test.mjs workbench/tests/audit.test.mjs workbench/tests/plot-data.test.mjs workbench/tests/sweep.test.mjs`; `python -m unittest discover -s workbench/tests -p "test_*.py"`; `node --check` on changed modules; `node scripts/code_graph.mjs --stats`; `git diff --check`. Report counts and any failures exactly.
- [ ] **Step 5: Perform a full browser pass** on localhost and a mobile viewport: ordinary model switch/replay, exact-sample inspection, pinning, sweep/progress/cancel, row selection, JSON/CSV save, JSON recomputation, audit upload, keyboard use, and console. Capture a current screenshot and record the commands/environment in `docs/WORKBENCH_VALIDATION.md`.
- [ ] **Step 6: Apply the repository's `spinnyball-review-and-verify` skill before the final implementation commit.** Resolve blocking findings, rerun affected checks, then commit and push the finished branch for review. Keep the legacy Python research CI status separate.

## Explicitly deferred

No probabilistic uncertainty distributions, Three.js rotor rewrite, Plotly/uPlot/Chart.js dependency, orbital perturbations, magnetic/thermal/control coupling, or schema migration for old ordinary experiments is part of this release. Those need their own physical question or dependency-contract decision. The previous Context7 assessment is a source of UI options, not evidence that any library validates the model.

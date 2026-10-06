# SpinnyBall architecture

The supported interactive experience is the self-contained `workbench/` directory. Older Python multiphysics research is a separate experimental path; see [research status](docs/RESEARCH_STATUS.md).

## Data flow

~~~text
controls / imported JSON -> validateConfig -> module Web Worker
                                                  |
                                             physics.mjs <- Node CLI
                                                  |
                                          immutable run result
                                                  |
                                   canvas / measurements / plots
                                                  |
                                      JSON experiment / sampled CSV
~~~

| File | Responsibility |
|---|---|
| workbench/physics.mjs | Model registry, SI configuration, deterministic integration, diagnostics and export schema |
| workbench/worker.mjs | Bounded calculation off UI thread; handles single runs and orbital sweeps |
| workbench/plot-data.mjs | Pure nearest-retained-sample binary search and inspection selection |
| workbench/sweep.mjs | Pure orbital speed sweep validation, execution, CSV formatting, and JSON interchange |
| workbench/audit.mjs | External trajectory validation against Keplerian two-body gravity |
| workbench/verify-external.mjs | CLI runner for external trajectory audit |
| workbench/app.mjs | Form state, replay, inspection, comparison, sweep lifecycle, import/export, and Canvas rendering |
| workbench/index.html, style.css | Responsive keyboard-operable interface, sweep panel, and model notes |
| workbench/run.mjs | Replays an export using the identical model engine |
| workbench/__main__.py | Standard-library localhost server serving only the workbench directory |
| workbench/tests/ | Independent analytic, convergence, balance, schema and launcher checks |

No database, build pipeline, external assets or network API is required. The workbench does not import legacy Python packages. Replay speed changes presentation only. Integration uses a fixed step with a shortened final step to land on the requested duration. Replay selects discrete retained states, without invented interpolation.

## Run contract

Engine version **1.0.0**, schema version **1**. Runs include normalized configuration, initial state, method, assumptions, source, state column names, samples and diagnostics. JSON export adds a timestamp. Import verifies the version and configuration and recomputes; uploaded samples are not treated as evidence.

Runs have at most 100,000 integration steps. Spin and spring models additionally reject excessive step/frequency products. Every step is checked for non-finite states and contributes to maximum balance residuals. Normally at most 1,200 trajectory samples are retained, including endpoints.

Energy diagnostics account for external work; momentum diagnostics account for external impulse. Spin checks the full inertial angular momentum vector. The 0.1% residual threshold is a numerical warning, not an uncertainty estimate or validation of real-world applicability.

## Diagnostics and sample inspection

Both time-series diagnostic plots (Signal and Energy balance) support exact-sample inspection via pointer hover/tap or keyboard focus (`tabindex="0"`, arrow keys, Home/End, Esc). The inspection readout displays the nearest retained sample without synthetic interpolation. When a comparison is pinned, the inspector queries each run's retained sample grid independently via `workbench/plot-data.mjs` (`nearestSampleAtTime`), displaying each run's own actual timestamp and value rather than assuming synchronized steps.

The external trajectory audit dialog strictly avoids `innerHTML` for untrusted user content; file names and audit warnings are rendered via safe DOM text nodes.

## Orbital speed sweep contract

Sweep schema version **1** (`kind: "orbit-speed-sweep"`). The orbital speed sweep evaluates parameter sensitivity across launch speed ratios $s \in [0.1, 2.0]$ with an integer point count of $3 \le N \le 21$, holding the currently applied orbit configuration constant. Sweeps are guarded by a conservative total-work limit of $N \times \lceil\text{duration}/\Delta t\rceil \le 200{,}000$ integration steps.

Sweeps run in a dedicated module Web Worker (`sweepWorker`), reporting progress after each evaluated point. The sweep worker can be cancelled at any time by the user or upon laboratory navigation without interfering with the primary simulation worker. Late worker messages are discarded.

Each sweep point runs the exact `simulate` engine with `sampleLimit: 2` (retaining only initial and final states). Balance residuals are monitored over all integration steps. Sweep results include analytic energy classification (`bound` for $\varepsilon < 0$, `unbound` for $\varepsilon \ge 0$), the analytic $\sqrt{2}$ threshold, last exterior distance, final time, and status. Points stopping at the central body are explicitly labeled `"stopped before surface crossing"`.

Sweep export produces schema 1 JSON or unit-bearing CSV. On JSON import, `readOrbitSweep` validates the engine version, sweep schema, base configuration, and sweep range, and **recomputes** all points deterministically; uploaded row data is discarded. Selecting a sweep row transfers that speed into the ordinary controls and triggers a full-resolution simulation.

## Extension contract

Add each model's parameters, equations, observations, domain limits and independent reference cases together. A force law is not validated because it produces an attractive visualization.

Exports describe the displayed run, including when form edits are unapplied. A pinned run keeps its original configuration and appears only alongside the same model. The old Python packaging metadata and dependencies apply to the research modules, not to the standalone laboratory.

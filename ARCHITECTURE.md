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
| workbench/worker.mjs | Bounded calculation off the UI thread; returns result or error |
| workbench/app.mjs | Form state, replay, comparison, import/export and Canvas rendering |
| workbench/index.html, style.css | Responsive keyboard-operable interface and model notes |
| workbench/run.mjs | Replays an export using the identical model engine |
| workbench/__main__.py | Standard-library localhost server serving only the workbench directory |
| workbench/tests/ | Independent analytic, convergence, balance, schema and launcher checks |

No database, build pipeline, external assets or network API is required. The workbench does not import legacy Python packages. Replay speed changes presentation only. Integration uses a fixed step with a shortened final step to land on the requested duration. Replay selects discrete retained states, without invented interpolation.

## Run contract

Engine version **1.0.0**, schema version **1**. Runs include normalized configuration, initial state, method, assumptions, source, state column names, samples and diagnostics. JSON export adds a timestamp. Import verifies the version and configuration and recomputes; uploaded samples are not treated as evidence.

Runs have at most 100,000 integration steps. Spin and spring models additionally reject excessive step/frequency products. Every step is checked for non-finite states and contributes to maximum balance residuals. Normally at most 1,200 trajectory samples are retained, including endpoints.

Energy diagnostics account for external work; momentum diagnostics account for external impulse. Spin checks the full inertial angular momentum vector. The 0.1% residual threshold is a numerical warning, not an uncertainty estimate or validation of real-world applicability.

## Extension contract

Add each model's parameters, equations, observations, domain limits and independent reference cases together. A force law is not validated because it produces an attractive visualization.

Exports describe the displayed run, including when form edits are unapplied. A pinned run keeps its original configuration and appears only alongside the same model. The old Python packaging metadata and dependencies apply to the research modules, not to the standalone laboratory.

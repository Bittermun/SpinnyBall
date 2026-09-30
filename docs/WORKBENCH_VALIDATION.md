# SpinnyBall workbench verification

Verification recorded on September 30, 2026 against the workbench implementation based on upstream commit `8551c5a`.

## Commands and results

| Check | Result |
|---|---|
| `node --test workbench/tests/physics.test.mjs` | 19 passed, 0 failed |
| `python -m unittest discover -s workbench/tests -p "test_*.py"` | 2 passed, 0 failed |
| `node --check workbench/app.mjs`, `worker.mjs`, `run.mjs` | Syntax checks passed |
| `python -m pytest tests/test_simulation_invariants.py -q` | 4 passed; Python 3.14 emitted existing pytest-asyncio deprecation warnings |
| `git diff --check` | No whitespace errors |
| `node workbench/run.mjs orbit output.json`, then `node workbench/run.mjs output.json replay.json` | Exact JSON result equality (excluding the CLI's separate status printout) |
| Local Chromium browser via Playwright CLI | Scenario switch, pin/clear, play/pause/reset, JSON and CSV download, model notes, spin and exchange navigation, fractional-value JSON import and rerun passed |
| 390 × 844 viewport | Visually inspected; no horizontal overflow or console errors observed |

The exported browser orbit JSON held 1,001 retained samples; its CSV had 1,001 data rows and 10 consistent columns. The default 1.18× orbit used 4,000 integration steps over 16,000 s. Its maximum scaled energy residual was approximately `7.29 × 10⁻⁷` (0.0000729% of μ/r₀); the specific angular momentum residual was approximately `9.03 × 10⁻¹⁵` of its scale. The displayed residual plot has sampled points; maxima in diagnostics include all integration steps.

The default middle-axis spin run's maximum scaled energy and inertial angular momentum residuals were approximately `3.72 × 10⁻¹²` and `1.02 × 10⁻¹⁰`. The command-line momentum exchange run used the drifting preset and produced approximately `4.58 × 10⁻⁵` maximum scaled energy residual, `1.16 × 10⁻¹⁵` momentum residual and `1.15 × 10⁻¹⁴ m` center-of-mass error. All were below the workbench's 0.1% numerical warning threshold.

## Independent reference checks

The tests compare a circular orbit's period and an elliptic trajectory to Kepler's equation, demonstrate second-order position convergence for Velocity Verlet, compare spherical/symmetric rotor motion to analytic solutions and fourth-order quaternion convergence, and compare spring motion with the reduced-mass oscillator plus constant-force center-of-mass solution. They also test external work and impulse accounting, a surface crossing, invalid and excessive inputs, rest cases and JSON/CSV consistency.

Agreement with these references supports the implementation within the specified models. The browser's 0.1% balance badge is a numerical diagnostic, not empirical accuracy. Energy balance alone cannot establish trajectory phase accuracy or the validity of omitted forces. Physical parameters and historical mass-stream claims have not been empirically validated by these tests.

## Scope of verification

The retained Python research tree and its large optional-dependency suite were not changed or claimed to pass as part of this workbench release. The local baseline's four `tests/test_simulation_invariants.py` cases passed during the initial audit; they do not cover the model and documentation issues recorded in [research status](RESEARCH_STATUS.md). GitHub CI runs the independent workbench checks on pushes and pull requests touching `workbench/`.

For a new model, extend independent reference cases and convergence tests before presenting its outcomes as reliable. For mission-level claims, further physical and empirical validation is required.

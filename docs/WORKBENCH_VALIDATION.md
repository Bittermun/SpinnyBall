# SpinnyBall workbench verification

Verification recorded on September 30, 2026 against the workbench implementation based on upstream commit `8551c5a`.

## Commands and results

### October 6, 2026 verification

| Check | Result |
|---|---|
| `node --test workbench/tests/physics.test.mjs workbench/tests/audit.test.mjs workbench/tests/plot-data.test.mjs workbench/tests/sweep.test.mjs` | 43 passed, 0 failed (~250 ms) |
| `python -m unittest discover -s workbench/tests -p "test_*.py"` | 2 passed, 0 failed |
| `node --check workbench/app.mjs`, `worker.mjs`, `run.mjs`, `audit.mjs`, `plot-data.mjs`, `sweep.mjs` | Syntax checks passed |
| `git diff --check` | No whitespace errors |
| `node scripts/code_graph.mjs --stats` | AST symbol graph verified |

### Worked orbital speed sweep example

Using the default circular orbit baseline ($r_0 = 1.85 \times 10^6$ m, duration $16{,}000$ s, $\Delta t = 4.0$ s, $4{,}000$ integration steps per point) with sweep parameters $s_{\text{min}} = 1.3$, $s_{\text{max}} = 1.5$, count = 5 ($20{,}000$ integration steps total, within the $200{,}000$-step budget):

| Speed ratio $s$ | Specific energy $\varepsilon$ (J/kg) | Classification | Status | Final radius $r_{\text{final}}$ (km) | Max scaled $\Delta E$ | Max $\Delta h / h_0$ |
|---|---|---|---|---|---|---|
| 1.30 | $-380{,}137.5$ | `bound` | complete | $10{,}142.32$ | $1.02 \times 10^{-6}$ (0.000102%) | $4.45 \times 10^{-15}$ |
| 1.35 | $-217{,}659.4$ | `bound` | complete | $12{,}889.50$ | $1.15 \times 10^{-6}$ (0.000115%) | $7.89 \times 10^{-15}$ |
| 1.40 | $-49{,}050.0$ | `bound` | complete | $15{,}380.44$ | $1.28 \times 10^{-6}$ (0.000128%) | $3.70 \times 10^{-15}$ |
| **$\sqrt{2} \approx 1.4142$** | **$0.0$** | **Analytic boundary** | — | — | — | — |
| 1.45 | $+125{,}690.6$ | `unbound` | complete | $17{,}687.04$ | $1.42 \times 10^{-6}$ (0.000142%) | $7.35 \times 10^{-15}$ |
| 1.50 | $+306{,}562.5$ | `unbound` | complete | $19{,}854.61$ | $1.57 \times 10^{-6}$ (0.000157%) | $1.06 \times 10^{-14}$ |

All points maintain maximum energy and momentum residuals well below the 0.1% numerical warning threshold ($< 0.00016\%$).
- Selecting row $s = 1.45$ loads $1.45$ into the ordinary orbit controls and triggers a full-resolution simulation run.
- Exporting sweep JSON and re-importing validates schema and range, recomputing all 5 points deterministically while ignoring any fabricated rows or diagnostics.

### September 30, 2026 baseline

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

On [PR #32](https://github.com/Bittermun/SpinnyBall/pull/32), both new workbench `verify` jobs passed. The existing [Python CI run](https://github.com/Bittermun/SpinnyBall/actions/runs/36734792286) remained red: Ruff reported 1,817 errors in retained Python paths, pytest stopped during collection on an undefined `Tuple` in `dynamics/cislunar_mascon.py`, and the parameter-consistency gate found identical mission metrics for YBCO and GdBCO. The [preceding `main` run](https://github.com/Bittermun/SpinnyBall/actions/runs/26312023650) had the same failing job steps (Ruff, parameter consistency, and Python tests). Its detailed logs have expired, so exact error equality cannot be established. These failures are unresolved legacy work; they are not workbench validation passes.

For a new model, extend independent reference cases and convergence tests before presenting its outcomes as reliable. For mission-level claims, further physical and empirical validation is required.

# SGMS — Spin-Gyro Magnetic Steering

Numerical simulation of a spinning magnetic sphere deflected laterally by a segmented pulsed coil array. The asymmetric pulse timing and amplitude profile produce a net transverse impulse while forward velocity is preserved.

## Current Scope

This repo currently treats the anchor work as a reduced-order package, not a high-fidelity field solver or rigid-body validation stack.

- Current implementation boundary: `L1-reduced-order`
- High-fidelity required now: `false`
- Current recommendation: keep README and artifacts aligned with stable reduced-order entrypoints only

Reopen Newton or FEMM work only if reviewer feedback, hardware design, or packet-level validation requirements make the current reduced-order assumptions insufficient. The written decision lives in [docs/anchor-validation-decision.md](/C:/Users/msunw/Desktop/SpinnyBall/docs/anchor-validation-decision.md).

## Files

| File | Description |
|---|---|
| `sgms_v1.py` | Python/SciPy RK45 simulation — the scientific reference |
| `sgms_anchor_v1.py` | Reduced-order moderate-`u` dynamic-anchor model with discrete-packet companion, sweep export, and plots |
| `sgms_anchor_control.py` | State-space + LQR analysis built directly on the reduced-order anchor model via `python-control` |
| `sgms_anchor_sensitivity.py` | Sobol sensitivity analysis for analytical anchor outputs via `SALib` |
| `sgms_anchor_pipeline.py` | Config-driven experiment runner that emits standardized artifacts, summaries, and a manifest |
| `sgms_anchor_report.py` | Self-contained HTML report generator for pipeline runs |
| `sgms_anchor_dashboard.py` | Static browser dashboard generator for reduced-order anchor exploration |
| `sgms_anchor_suite.py` | One-command runner that regenerates the nonlinear, control, and sensitivity anchor artifacts |
| `anchor_experiments.json` | Default experiment matrix for the config-driven anchor pipeline |
| `anchor_profiles.json` | Named paper and engineering parameter profiles with notes and provenance |
| `anchor_calibration.json` | Calibration/provenance tags for anchor parameters |
| `anchor_claims.json` | Claim-level metadata and the current high-fidelity decision |
| `requirements-anchor.txt` | Minimal Python dependencies for the anchor analysis stack |
| `docs/anchor-validation-decision.md` | Written decision deferring Newton/FEMM for the current phase |
| `index.html` | Self-contained Three.js interactive demo — live at [bittermun.github.io/SpinnyBall](https://bittermun.github.io/SpinnyBall/) |

## Physics summary

A 2 kg steel sphere (r = 46 mm) spins at 50,000 rpm, giving it a magnetic dipole moment μ ≈ 60 A·m² (placeholder — sweep variable). It transits an 8 m array of 6 pulsed coil segments at v_z = 15,000 m/s. Each segment fires a Gaussian pulse (σ = 50 µs) with individually tunable timing offset δᵢ and amplitude aᵢ. The asymmetry produces a net lateral force F_x = μ · dBx/dz while drag F_z remains small.

**Validated results at defaults:**
- Δvx = −0.00416 m/s (3 sig figs, converged at dt = 0.25 µs)
- Impulse magnitude = 0.00832 N·s
- Steering η = ∞ (drag disabled — set k_drag > 0 to see ratio)
- Rim speed = 241 m/s — SAFE

## Python simulation (`sgms_v1.py`)

Runs in Google Colab with no installs (NumPy, SciPy, Matplotlib — all standard).

```
colab.research.google.com → New notebook → paste cells in order → Runtime → Run all
```

Or locally: `py sgms_v1.py`

**Output:** 5 physics checks, timestep convergence report, 6-panel summary plot, 5-panel trajectory diagnostics, 3 parameter sweep plots, offset sensitivity test.

**Physics checks (all must pass before trusting numbers):**
1. Symmetric config → Δvx ≈ 0
2. μ = 0 → zero force exactly
3. Full antisymmetry flip (both delta and amp reversed) → sign reversal
4. Double μ → 2× impulse (linear regime confirmed, ratio = 2.000)
5. Bore-edge field ≤ 20 T (at default gradient = 50 T/m: 12.5 T — pass)

**Convergence:** Spatial force sampling is the limiting factor, not ODE tolerance. Converged to ~0.5% at dt = 0.25 µs. Tightening rtol/atol has no effect.

## Anchor simulation (`sgms_anchor_v1.py`)

This is a separate reduced-order model for the moderate-velocity recirculating counter-stream anchor use case. It does not claim to reuse the full pulsed-coil field solver from `sgms_v1.py`; instead it validates the anchor force law directly:

`F = λ u² θ`

with a symmetric counter-stream controller chosen so that:

`k_eff = λ u² g_gain`

Run locally with:

```bash
py -m unittest tests\test_sgms_anchor_v1.py
py sgms_anchor_v1.py
```

Generated artifacts:
- `sgms_anchor_v1_displacement.png` — continuum anchor displacement response
- `sgms_anchor_v1_packet_compare.png` — continuum vs discrete-packet response and force comparison
- `sgms_anchor_v1_sweep.png` — one-dimensional velocity sweep
- `sgms_anchor_v1_heatmaps.png` — `u`/`g_gain` heatmaps for stiffness and static offset
- `sgms_anchor_v1_grid.csv` — sweep table for paper or rig analysis

Current automated checks cover restoring sign, stiffness slope, seeded disturbance determinism, continuum period, discrete-packet convergence, and sweep export integrity.

## Control analysis (`sgms_anchor_control.py`)

This module linearizes the existing anchor model into:

`m_s x¨ + c_damp x˙ + k_eff x = u_force`

and uses `python-control` for LQR design on that plant. It is intended for controller sizing and Section 5 style control-limited analysis, not as a replacement for the nonlinear anchor simulation.

Install once:

```bash
py -m pip install control
```

Run locally with:

```bash
py -m unittest tests\test_sgms_anchor_control.py
py sgms_anchor_control.py
```

Generated artifact:
- `sgms_anchor_control_response.png` — open-loop vs closed-loop displacement plus control-force history

Current automated checks cover state-space matrix derivation, closed-loop pole stability, and improved closed-loop displacement area relative to open-loop response.

## Sensitivity analysis (`sgms_anchor_sensitivity.py`)

This module runs Sobol sensitivity analysis on the analytical anchor outputs rather than the ODE solver, so large sweeps stay fast and deterministic. The default problem spans:

- `u`
- `g_gain`
- `eps`
- `lam`

and reports indices for quantities such as `k_eff`, `period_s`, `static_offset_m`, and `packet_rate_hz`.

Install once:

```bash
py -m pip install SALib
```

Run locally with:

```bash
py -m unittest tests\test_sgms_anchor_sensitivity.py
py sgms_anchor_sensitivity.py
```

Generated artifacts:
- `sgms_anchor_sobol.csv` — first-order and total-order Sobol indices
- `sgms_anchor_sobol.png` — grouped bar plots of `S1` and `ST` by output

Current automated checks cover analytical metric consistency, Sobol sample sizing, low `eps` sensitivity for `k_eff`, and CSV export integrity.

## Combined runner (`sgms_anchor_suite.py`)

If you want the whole reduced-order anchor package in one shot, install the stack once:

```bash
py -m pip install -r requirements-anchor.txt
```

Then run:

```bash
py sgms_anchor_suite.py
```

This now delegates to the config-driven pipeline in `anchor_experiments.json` and emits a dated artifact directory under `artifacts/`.

## Config-driven pipeline (`sgms_anchor_pipeline.py`)

The fastest path for repeatable paper outputs is the pipeline runner:

```bash
py sgms_anchor_pipeline.py
```

It reads `anchor_experiments.json`, merges default parameters with named experiments, and writes:

- `artifacts/<run-label>/config_snapshot.json`
- `artifacts/<run-label>/dashboard_data.json`
- `artifacts/<run-label>/dashboard.html`
- `artifacts/<run-label>/manifest.json`
- `artifacts/<run-label>/profile_summary.csv`
- `artifacts/<run-label>/report.html`
- `artifacts/<run-label>/<experiment>/summary.json`
- `artifacts/<run-label>/<experiment>/metrics/*.csv`
- `artifacts/<run-label>/<experiment>/figures/*.png`

Each experiment currently generates:

- nonlinear anchor outputs
- controller trade study outputs
- robustness scenario outputs
- Sobol sensitivity outputs

The run-level `report.html` is self-contained and can be opened directly in a browser to review experiment summaries, controller comparisons, robustness rows, and sensitivity winners.

The run-level `dashboard.html` is a static interactive browser surface tied to the reduced-order model. It lets you vary `u`, `g_gain`, `eps`, and `c_damp` while viewing stiffness, period, static offset, and open vs proportional-trim response.

Run the pipeline tests with:

```bash
py -m unittest tests\test_sgms_anchor_pipeline.py
py -m unittest tests\test_sgms_anchor_report.py
py -m unittest tests\test_sgms_anchor_dashboard.py
```

## Parameter profiles (`anchor_profiles.json`)

The pipeline supports named profiles so paper-facing and engineering-facing runs are explicit rather than implied by raw parameter overrides.

Current profiles:

- `paper-baseline` — traceable reduced-order baseline for paper figures and tables
- `engineering-screen` — stronger-control engineering screen for trade studies

Experiments in `anchor_experiments.json` can reference a profile and then override only the parameters that differ. Profile metadata is preserved into:

- `summary.json`
- `manifest.json`
- `profile_summary.csv`
- `report.html`

Run the profile tests with:

```bash
py -m unittest tests\test_sgms_anchor_profiles.py
```

## Calibration and claims

`anchor_calibration.json` tags parameter values as memo baselines, reduced-order assumptions, placeholders, or engineering screens. This metadata is propagated into:

- `summary.json`
- `report.html`
- `dashboard.html`

`anchor_claims.json` records the current phase decision:

- higher-fidelity required now: `false`
- decision: `defer`
- current claim level: `L1-reduced-order`

That decision is documented in [anchor-validation-decision.md](/C:/Users/msunw/Desktop/SpinnyBall/docs/anchor-validation-decision.md).

Run the calibration and claim tests with:

```bash
py -m unittest tests\test_sgms_anchor_calibration.py
```

## Interactive demo (`index.html`)

Live at **https://bittermun.github.io/SpinnyBall/** — or open `index.html` directly in any browser. No server, no install.

**Three view modes:**
- **FORCES** (default) — red lateral arrow vs blue drag arrow. Efficiency claim made visual.
- **TRAJECTORY** — amber trail + real-scale inset canvas (µm vs m). True displacement magnitude.
- **FIELD** — top-down streamline view of the dipole field through active segments.

**Key buttons:**
- `FIRE` — pre-computes full transit, then animates replay
- `SYMMETRIC` — loads zero-asymmetry preset, shows ghost of previous asymmetric trail alongside straight line. This is the core physics argument.
- `VALIDATE` — runs dt convergence check (0.25 µs vs 0.50 µs) + reference match in-browser
- `PRESET ▾` — six presets: default, symmetric, flipped sign, zero dipole, high drag, max gradient

**Sign convention:** Physics gives Δvx negative (force toward −x). Scene flips to +x for visual clarity. Labeled throughout.

## Known limitations (V1 scope)

- Drag term is a geometric proxy. Eddy-current drag requires FEM or Biot-Savart skin-depth model. Do not trust E_loss numbers until k_drag is calibrated against independent calculation.
- μ = 60 A·m² is a placeholder. True value depends on magnet volume, remanence, and spin alignment. Use `sweep_mu()` to map the full sensitivity curve.
- Spin axis frozen at s = (0,0,1) by default. Precession available via `include_precession=True` — effect is small over one transit.
- No multi-ball interactions, thermal model, or relativistic corrections (v_z/c ≈ 5×10⁻⁵).

## Validation lineage

Independent MATLAB RK4 spec produced identical equations from a separate derivation. Force law, field model, drag, and precession all match. Python spec updated with MATLAB's timestep (0.25 µs) and 6-segment default. Physics checks confirmed in two independent codebases.

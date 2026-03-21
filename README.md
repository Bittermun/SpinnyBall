# SGMS — Spin-Gyro Magnetic Steering

Numerical simulation of a spinning magnetic sphere deflected laterally by a segmented pulsed coil array. The asymmetric pulse timing and amplitude profile produce a net transverse impulse while forward velocity is preserved.

## Files

| File | Description |
|---|---|
| `sgms_v1.py` | Python/SciPy RK45 simulation — the scientific reference |
| `sgms_demo.html` | Self-contained Three.js interactive demo — opens in any browser |

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

## Interactive demo (`sgms_demo.html`)

Open directly in any browser. No server, no install.

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

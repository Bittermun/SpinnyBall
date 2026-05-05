# SpinnyBall

Closed-loop gyroscopic mass-stream anchor for station-keeping in cislunar space.

## What it does

Spin-stabilized magnetic packets (50k RPM) circulate along an orbital circumference. Momentum-flux anchoring generates restoring force: F = λu²sin(θ). Flux-pinned superconducting bearings provide passive stiffness. Two material options: SmCo (passive, ~379K) or GdBCO (active cryogenic, ~77K).

## Physics

- Gyroscopic dynamics: I·ω̇ + ω×(I×ω) = τ
- Flux-pinning: Jc(B,T) = Jc0·(1-T/Tc)^n·f(B)
- Effective stiffness: k_eff = λu²g_gain + k_fp
- Centrifugal stress: σ = m·ω²r/(4πr) at operational spin rates

## Results (2026-05-05)

**Cascade boundary**: λ_crit ≈ 215/hr (stress test, N=1500). System stable at operational rates.

**Monte Carlo**: 256k realizations via JAX/XLA in 0.96s. T3 sweep extended to 3600s for rare-event statistics.

**Sobol (9 params, N=1024)**: Velocity dominates k_eff (81%). SmCo feasibility 0.3%, GdBCO 17.3%.

**Material comparison**: SmCo (15 km/s) = 280kg, passive cooling. GdBCO = ~2 MW cryocooler.

## Run it

```bash
poetry install
python src/sgms_anchor_v1.py
pytest tests/test_simulation_invariants.py -v
python check_damping.py
```

## Docs

- `docs/paper_manuscript.md` — full paper
- `docs/TECHNICAL_SPEC.md` — parameters and physics derivation
- `docs/RESEARCH_DATASET.md` — sweep results and data files
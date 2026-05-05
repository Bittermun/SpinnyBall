# Research Dataset

## Cascade Results (2026-05-05)

T3 sweep extended to 3600s simulation time for rare-event fault statistics (commit 8b454f5).

Zero cascades across 10⁻⁸–10⁻³/hr (N=3,000/point). Cascade onset at λ_crit ≈ 215/hr (N=1,500 stress test). 10⁶ margin over environmental fault rates.

## Velocity Scaling

k_eff ∝ u². At 15 km/s: k_eff = 909,000 N/m, 0.1% cost vs 500 m/s baseline.

## Sobol (9 params, N=1024, updated 2026-05-04)

Velocity dominates: 81% of k_eff variance. SmCo feasibility 0.3%, GdBCO 17.3%.

Material-by-structural combinations now available:
- SmCo: BFRP, CFRP, CNT_yarn
- GdBCO: BFRP, CFRP, CNT_yarn

## Data files

- `mission_analysis_results/sobol_smco.npz` — Sobol indices (20,480 samples)
- `mission_analysis_results/sobol_gdbco.npz` — Sobol indices
- `mission_analysis_results/*_sobol_indices.csv` — per-output S1/ST
- `mission_analysis_results/*_feasible.npy` — feasibility masks
- `mission_analysis_results/*_samples.npy` — parameter samples
- `mission_analysis_results/*_feasibility.png` — visualization

## Physics corrections (audited 2026-05-04)

- k_eff: lam·u²·g_gain + k_fp (corrected)
- mu: 3,330 A·m² for SmCo (was 60, 50x low)
- lam: derived from mp/spacing
- B0: 5.0 T (was 0.1 T, 50x increase)
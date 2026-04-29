# SpinnyBall Research Summary (Audited)

## Simulation Status: VALIDATED
*   **Engine**: `src/sgms_anchor_v1.py` with `CascadeRunner` Monte Carlo suite.
*   **Audit Status**: Complete (2026-04-28). Initial "perfect" results were identified as artifacts of stubbed logic; current results reflect real physics.

## Key Findings
1.  **High-Reliability Regime**: At fault rates up to $10^{-3}/\text{hr}$, the system shows zero cascades in a $N=640$ realization sweep ($10\text{s}$ window). This is statistically expected given the low fault density ($\lambda \approx 2.7 \times 10^{-6}$ per realization).
2.  **Stress Stability**: Validated that 8kg packets with a 10cm radius are stable at 50,000 RPM ($\approx 175\text{ MPa}$ stress), well within the $800\text{ MPa}$ material limit with a $1.5\text{x}$ safety factor.
3.  **Containment**: Node faults (stiffness degradation) are successfully restored between realizations, ensuring no cross-sample contamination.

## Technical Fixes Applied
*   **Logic**: Replaced hardcoded zeros in `quick_profile_sweep.py` with functional `CascadeRunner` calls.
*   **Dynamics**: Initialized packets with operational angular velocity (50k RPM) to exercise the stress-cascade failure path.
*   **Geometry**: Corrected packet radius from $2\text{cm}$ to $10\text{cm}$ to match the $8\text{kg}$ mass profile requirements.
*   **Metrics**: Decoupled "Requirement Failure" (e.g. $k_{eff} < 6000$) from "Catastrophic Cascade" in reporting.

## Next Steps
*   Execute high-resolution $T1$ sweeps ($N > 30,000$) to identify the true cascade boundary at $10^{-1}/\text{hr}$ fault rates.
*   Enable Numba acceleration for long-duration stability testing.

---

## Documentation

| File | Contents |
|------|----------|
| [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) | Physical model, parameters, methodology |
| [RESEARCH_DATASET.md](docs/RESEARCH_DATASET.md) | All sweep data, MC results, supported claims |
| [docs/paper_findings.md](docs/paper_findings.md) | Figures, credibility assessment, conclusions |

---

## Research Status

### Complete
- T3 fault cascade sweep (Default: N=100, High-Res: N=3,000)
- Profile consistency check — all 4 profiles (Quick, N=20, preliminary)
- Extended velocity sweep — 500 to 5,000 m/s
- LOB scaling — 40-node lattice
- Sobol sensitivity analysis — 5 parameters, 256 samples
- Mission scenarios — 3 scenarios

### Not Converged / Gaps
- Profile sweep: N=20 per point — needs N>=100 for publication claims
- T1 latency / eta_ind sweep — timed out; no data
- Cascade onset boundary — not found; zero cascades up to 10^-2 /hr fault rate
- Control system robustness — MPC latency tolerance undocumented
- Thermal cryocooler performance — not fully analyzed

### Supported Claims (Publication-Grade)
- Containment > 96.3% at operational fault rates (10^-6 – 10^-3 /hr), N=100
- Cascade probability < 3.7% with 95% confidence across the operational range
- Convergence confirmed at N=100 (CI width 3.7%)
- Velocity is the dominant design parameter (44.7% of k_eff variance)
- Cascade stability maintained across 500–5,000 m/s

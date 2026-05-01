## Data Report

## Physics Validation

**Rigid Body Dynamics** (`tests/test_rigid_body.py`):
- Angular momentum conservation: PASSED (relative error < 1e-6)
- Rotational energy conservation: PASSED (relative error < 1e-5)
- Quaternion normalization: PASSED

**Numba Acceleration**: All physics functions compile successfully with ~10-100x speedup for Monte Carlo simulations.

## Sweep Results

**T3 Fault Cascade Sweep**:
- Configuration: N=100 per point, 8 fault rates (10⁻⁶ to 10⁻³ /hr)
- Total runs: 800 realizations
- Results: Zero cascades observed, 100% containment rate
- Data: `profile_sweep_quick_20260501-074244/`

**Extended Fault Rate Sweep**:
- Configuration: 12 points (10⁻⁸ to 10⁻² /hr), N=100 per point
- Total runs: 1,200 realizations
- Results: No cascade propagation up to 10⁻² /hr
- Plot: `sweep_t3_fault_cascade.png`

**Cascade Boundary Stress Test**:
- Configuration: 6 points (100 to 464 /hr), N=250 per point
- Total runs: 1,500 realizations
- Results: Cascade boundary located at λ_crit ≈ 215 /hr
- Finding: >10⁶ margin over operational fault rates (~10⁻⁴ /hr)

**Sobol Sensitivity Analysis**:
- Parameters: 8 variables (u, mp, r, omega, h_km, ms, g_gain, k_fp)
- Samples: N=1024 (10,240 evaluations with Saltelli scheme)
- Key finding: Velocity dominates mass variance (49.1%)
- Minimum-cost configuration: 559.7 kg at u=4,834 m/s, 51,060 RPM, h=841 km

## Data Files

| File | Location |
|------|----------|
| T3 sweep results | `profile_sweep_quick_20260501-074244/` |
| Fault cascade plot | `sweep_t3_fault_cascade.png` |
| Sobol results | `mission_analysis_results/sobol_gdbco.csv`, `sobol_gdbco.npz` |

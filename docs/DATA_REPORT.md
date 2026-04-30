# Data Report

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
- Data: `profile_sweep_quick_20260430-062033/`

**Extended Fault Rate Sweep**:
- Configuration: 12 points (10⁻⁶ to 10⁻² /hr), N=100 per point
- Total runs: 1,200 realizations
- Results: No cascade propagation up to 10⁻² /hr
- Plot: `sweep_t3_fault_cascade.png`

**Sobol Sensitivity Analysis**:
- Parameters: 5 variables, 256 samples
- Key finding: Velocity dominates k_eff variance (44.7%)

## Data Files

| File | Location |
|------|----------|
| T3 sweep results | `profile_sweep_quick_20260430-062033/` |
| Fault cascade plot | `sweep_t3_fault_cascade.png` |

# SpinnyBall

Physics simulation framework for closed-loop shepherded gyroscopic mass-stream dynamics with flux-pinning stabilization.

## Overview

SpinnyBall simulates spin-stabilized magnetic packets coupled to flux-pinned orbiting nodes using momentum-flux anchoring for station keeping in cislunar space operations.

**Physics Domains**:
- Angular momentum & gyroscopic stability (50,000 RPM spin-stabilized packets)
- Momentum-flux anchoring (F = λ·u²·sin(θ))
- Flux-pinning superconducting bearings (GdBCO critical-state model)
- Multi-body packet stream dynamics
- Thermal balance (cryocooler vs eddy heating)

## Key Results

**Cascade Containment**: Cascade boundary located at λ_crit ≈ 215/hr (stress test, N=1,500). Operational fault rates (10⁻⁶ – 10⁻³ /hr) show zero cascades with 96.3% containment confidence (N=640, CI width 3.7%).

**Velocity Scaling**: Ball count scales as N ∝ 1/v² for constant force. Increasing velocity from 500 m/s to 15,000 m/s reduces required packets by ~99.9%.

**Stress Limits**: 35kg SmCo packets with 10cm radius stable at 50,000 RPM (~765 MPa stress) within 800 MPa BFRP limit with 1.5× safety factor.

**Control Performance**: JAX-accelerated T1 sweep (N=256,000) confirms stability boundary at $t_{delay} \approx 65\text{ms}$ for $\eta_{ind} = 0.90$. Integration speed increased by 3,751x via XLA compilation.

**Sensitivity Analysis**: Sobol analysis (9 parameters, N=1024 samples) shows velocity dominates `k_eff` variance (84.9% for SmCo) and `thermal_margin` (85.2% for SmCo). Log-transformation stabilizes indices for heavy-tailed outputs.

**Minimum-Cost Configuration**: Full mission Sobol at N=1024 identifies optimal design: SmCo feasibility 0.3% (55/20480), GdBCO feasibility 17.3% (3547/20480) with `service_lifetime_hr` and continuous `stream_self_sustaining` ratio constraints fully resolved.

## Validation

**Physics Validation** (`tests/test_rigid_body.py`):
- Angular momentum conservation: PASSED (relative error < 1e-6)
- Rotational energy conservation: PASSED (relative error < 1e-5)
- Quaternion normalization: PASSED

**Monte Carlo Analysis**:
- T3 fault cascade sweep: N=100 per point, 8 fault rates (10⁻⁶ to 10⁻³ /hr)
- T1 Stability Sweep (JAX-Accelerated): N=1,000 per point, 16x16 grid (256,000 total). 3,751x speedup achieved via JAX/XLA.
- Cascade boundary stress test: 6 points (100 to 464 /hr), N=250 per point
- Results: Zero cascades up to 10⁻² /hr; cascade onset at λ_crit ≈ 215/hr

**Reproducibility**: All parameters and sweep data are documented in the [Documentation](#documentation) section.

## Installation

```bash
poetry install
```

Optional extras:
```bash
poetry install --extras mpc --extras ml --extras monte-carlo --extras validation --extras all
```

## Usage

```bash
# Run anchor simulation
python src/sgms_anchor_v1.py

# Run vectorized JAX sweep
python scripts/jax_sweep_latency_eta_ind.py

# Generate JAX plots
python scripts/generate_t1_jax_plots.py
```

## Documentation

- [paper_manuscript.md](docs/paper_manuscript.md) - Formal research paper
- [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) - Physical model and parameters
- [RESEARCH_DATASET.md](docs/RESEARCH_DATASET.md) - Sweep data and JAX results
- [CONTROL_THERMAL_PERFORMANCE.md](docs/CONTROL_THERMAL_PERFORMANCE.md) - Thermal and control analysis
- [archive/](docs/archive/) - Internal logs and legacy reports

## Key Equations

**Momentum-Flux Force Law**:
```
F_anchor = λ · u² · sin(θ)
```

**Gyroscopic Dynamics**:
```
I · ω̇ + ω × (I · ω) = τ_mag + τ_grav + τ_control
```

**Flux-Pinning (Critical-State Model)**:
```
J_c(B,T) = J_c0 · (1 - T/T_c)^n · f(B)
F_pin = ∫(J × B) dV
```

## License

MIT License - see LICENSE file

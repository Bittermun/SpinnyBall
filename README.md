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

**Stress Limits**: Packets with 10cm radius stable at 50,000 RPM (~685 MPa stress) within 800 MPa BFRP limit with 1.5× safety factor.

**Sensitivity Analysis**: Sobol analysis (8 parameters, N=1024 samples) shows velocity dominates mass variance (49.1%).

**Minimum-Cost Configuration**: Full mission Sobol at N=1024 identifies optimal design: 559.7 kg total mass at u=4,834 m/s, 51,060 RPM, h=841 km with k_eff = 6,000–100,000 N/m constraint.

## Validation

**Physics Validation** (`tests/test_rigid_body.py`):
- Angular momentum conservation: PASSED (relative error < 1e-6)
- Rotational energy conservation: PASSED (relative error < 1e-5)
- Quaternion normalization: PASSED

**Monte Carlo Analysis**:
- T3 fault cascade sweep: N=100 per point, 8 fault rates (10⁻⁶ to 10⁻³ /hr)
- Extended fault rate sweep: 12 points (10⁻⁸ to 10⁻² /hr), N=100 per point
- Cascade boundary stress test: 6 points (100 to 464 /hr), N=250 per point
- Results: Zero cascades up to 10⁻² /hr; cascade onset at λ_crit ≈ 215/hr

**Reproducibility**: All parameters documented in [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md), sweep data available in `profile_sweep_quick_20260501-074244/`

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

# Run tests
pytest tests/

# Run sensitivity analysis
python src/sgms_anchor_sensitivity.py

# Generate plots
python scripts/generate_paper_plots.py
```

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

## Repository Structure

```
SpinnyBall/
├── src/                    # Core simulation modules
├── dynamics/               # Physics models (rigid body, flux-pinning, thermal)
├── control/                # MPC and control systems
├── monte_carlo/            # Cascade analysis
├── tests/                  # Unit tests
├── scripts/                # Analysis scripts
└── docs/                   # Documentation
```

## Documentation

- [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) - Physical model and parameters
- [RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md) - Key results
- [DATA_REPORT.md](docs/DATA_REPORT.md) - Sweep results

## License

MIT License - see LICENSE file

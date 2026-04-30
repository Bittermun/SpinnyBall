# SpinnyBall

Physics simulation framework for closed-loop shepherded gyroscopic mass-stream dynamics.

## Core Simulation

**Engine**: `src/sgms_anchor_v1.py` with Monte Carlo cascade analysis

**Physics Domains**:
- Angular momentum & gyroscopic stability (50,000 RPM spin-stabilized packets)
- Momentum-flux anchoring (F = λ·u²·sin(θ))
- Flux-pinning superconducting bearings (GdBCO critical-state model)
- Multi-body packet stream dynamics
- Thermal balance (cryocooler vs eddy heating)

## Key Results

**Cascade Containment**: At fault rates up to 10⁻³/hr, system shows zero cascades in N=640 realization sweep (10s window). Mean containment rate: 100% (all failures contained to ≤2 nodes).

**Velocity Scaling**: Ball count scales as N ∝ 1/v² for constant force. Increasing velocity from 500 m/s to 15,000 m/s reduces required packets by ~99.9%.

**Stress Limits**: 35kg SmCo packets with 10cm radius stable at 50,000 RPM (~765 MPa stress) within 800 MPa BFRP limit with 1.5× safety factor.

## Documentation

| File | Contents |
|------|----------|
| [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) | Physical model, parameters, methodology |
| [RESEARCH_DATASET.md](docs/RESEARCH_DATASET.md) | Sweep data, MC results |
| [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) | System specifications |

## Installation

```bash
poetry install
```

## Usage

```bash
python src/sgms_anchor_v1.py
pytest tests/
```

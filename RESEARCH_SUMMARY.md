# SpinnyBall

Physics simulation framework for closed-loop shepherded gyroscopic mass-stream dynamics.

## Core Simulation

**Engine**: `src/sgms_anchor_v1.py` with Monte Carlo cascade analysis.
**Accelerated Engine**: `scripts/jax_sweep_latency_eta_ind.py` (3700x faster).

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

**Control Stability**: JAX-accelerated sweeps (N=256,000) confirm a stable control regime for latency < 65ms at 90% induction efficiency. System exhibits 3,751x speedup over legacy CPU backends.

## Documentation

| File | Contents |
|------|----------|
| [TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) | Physical model, parameters, methodology |
| [RESEARCH_DATASET.md](docs/RESEARCH_DATASET.md) | Sweep data, MC results |
| [README.md](README.md) | Installation and Usage |

## Installation

```bash
poetry install
```

## Usage

```bash
python src/sgms_anchor_v1.py
python scripts/jax_sweep_latency_eta_ind.py
pytest tests/
```

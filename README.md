# SpinnyBall: Lunar Orbital Belt Logistics Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://github.com/bittermun/SpinnyBall/workflows/CI/badge.svg)](https://github.com/bittermun/SpinnyBall/actions)

**Status**: Physics Simulation Framework
**Target**: First-Principles Closed-Loop Shepherded Gyroscopic Mass-Stream Simulation
---

##  Executive Summary

SpinnyBall is a **first-principles digital twin** for closed-loop shepherded gyroscopic mass-stream simulation. It models spin-stabilized magnetic packets (Magnetic Kinetic Control Packets, or MKCPs) coupled to flux-pinned orbiting nodes, using momentum-flux anchoring for ultra-efficient station keeping and steering in cislunar space operations.

### Physics Foundation

The system leverages **six integrated physics domains**:

1. **Angular Momentum & Gyroscopic Stability** - Spin-stabilized packets at 50,000 RPM
2. **Network Force Generation & PID Control** - Active magnetic steering with full PID loops
3. **Wave Mechanics & Shockwave Dissipation** - Dispersive stream dynamics, wobble cascade prevention
4. **Passive Superconducting Flux-Pinning** - GdBCO zero-power stabilization layer
5. **Coupled Energy & Thermal Equilibrium** - Metabolic harvesting vs cryogenic cooling balance
6. **Advanced Predictive Diagnostics** - ML-based failure prediction (future phase)

### Physics Modeling

The system models the following physics domains:

**Orbital Dynamics**:
- J2 perturbation (Earth oblateness)
- Solar Radiation Pressure (SRP)
- Atmospheric drag
- Eclipse detection

**Thermal & Power Systems**:
- Coil switching loss models
- Cryocooler power integration
- Thermal feedback loops
- Eclipse-aware solar flux

**Electromagnetic Modeling**:
- Mutual inductance calculations
- Fringe field corrections
- Field gradient limits
- HTS current density modeling
---

## Momentum-Flux Anchoring

**Force Generation**:
- F_anchor = λ·u²·sin(θ)
- Where λ = mass flow rate (kg/m), u = relative velocity (m/s), θ = deflection angle (rad)

**Mechanism**: Lateral deflection of high-velocity mass stream transfers momentum to stations without dissipative losses.

## Installation

Install with poetry:

```powershell
poetry install
```

Install with optional extras:

```powershell
poetry install --extras mpc --extras ml --extras monte-carlo --extras validation --extras quaternion --extras backend --extras jax
poetry install --extras all
```

Available extras:
- mpc: CasADi and numba for model-predictive control
- ml: PyTorch for ML predictors
- monte-carlo: SALib and torch for sensitivity analysis
- validation: MuJoCo for physics validation
- quaternion: quaternion library for attitude representation
- backend: FastAPI, uvicorn, pydantic for digital twin API
- jax: JAX-accelerated thermal models
- all: All optional dependencies

## Usage

---

##  Technical Specifications

### Packet Parameters
| Property | Value | Notes |
|----------|-------|-------|
| Mass | 2 kg (individual), 10,000 kg (payload) | Scalable via clustering |
| Spin Rate | 50,000 RPM (833 Hz) | Gyroscopic stability |
| Stream Velocity | 15,000 m/s (orbital), 10 m/s (catch-relative) | Dual reference frames |
| Moment of Inertia | I_axial > I_transverse | Prolate spheroid geometry |
| Magnetic Moment | 10-200 A·m² (sweep variable) | Tunable per mission |

### Node/Stations
| Property | Value | Notes |
|----------|-------|-------|
| Station Mass | 1,000 kg (baseline) | Scales with payload |
| Flux-Pinning Stiffness | 4,500 N/m (GdBCO @ 77K) | Passive layer |
| Operating Temperature | 70-90 K | GdBCO superconducting range |

### Network Architecture
| Property | Value | Notes |
|----------|-------|-------|
| Node Count | 40 (global lattice) | Redundant mesh topology |
| Inter-Node Spacing | Variable (VPD controlled) | Wave scattering optimization |
| Tension Coupling | Global load distribution | Prevents cascade failures |

---

##  Resilience & Hardening

### Triple-Layer Stabilization

1. **Passive Layer** (Zero Power):
   - GdBCO flux-pinning stiffness
   - Critical-state model damping
   - Automatic transverse oscillation suppression

2. **Active Layer** (Low Power):
   - Full PID control (K_p, K_i, K_d)
   - Variable Packet Density (VPD) modulation
   - Lead-lag feed-forward impulse rejection

3. **Network Layer** (Distributed):
   - 40-node tension coupling
   - Shockwave dissipation via density gradients
   - Local failure isolation

### Failure Mode Modeling

The system models various failure scenarios including:
- Single node quench events
- Mass imbalance effects
- Shockwave propagation
- Thermal runaway conditions
- Control instability modes

---

##  Energy & Thermal Management

### Power Budget (Per Node)

| Component | Consumption | Notes |
|-----------|-------------|-------|
| Cryocooler (70-90K) | 5-8 W | Maintains GdBCO superconductivity |
| SiC Power Electronics | 1-2 W | Magnet drive circuits |
| Sensors & Comms | 0.5-1 W | Housekeeping |

### Thermal Limits

| Component | Max Temp | Failure Mode |
|-----------|----------|--------------|
| GdBCO Stator | 90 K | Quench → loss of pinning |
| Rotor Shaft | 150 K | Bearing degradation |
| Power Electronics | 400 K | SiC thermal runaway |

---

##  Performance Metrics

The simulation models:
- Station-keeping stiffness and precision
- Payload transport capacity and acceleration
- Network resilience and failure propagation
- Energy efficiency characteristics

---

##  Development Status

The project includes:
- ML-enhanced digital twin with wobble detection and thermal prediction
- VMD-IRCNN infrastructure with graceful fallback
- GPU training support on Python 3.11 with CUDA 12.1
- Anomaly detection and synthetic failure data generation
- Test suite with coverage requirements


---

##  Repository Structure

```
SpinnyBall/
├── sgms_v1.py                  # Core lateral deflection physics (RK45)
├── sgms_anchor_v1.py           # Reduced-order anchor model
├── sgms_anchor_control.py      # PID control logic
├── sgms_anchor_logistics.py    # Logistics event simulation with thermal balance
├── sgms_anchor_suite.py        # Config-driven experiment pipeline
├── sgms_anchor_calibration.py  # Anchor calibration routines
├── sgms_anchor_claims.py       # Anchor claim validation
├── sgms_anchor_dashboard.py    # Dashboard visualization
├── sgms_anchor_metabolism.py   # Metabolic yield calculations
├── sgms_anchor_mujoco.py       # 6-DOF high-fidelity validation (updated to paper targets)
├── sgms_anchor_pipeline.py     # Simulation pipeline orchestration
├── sgms_anchor_profiles.py     # Configuration profiles
├── sgms_anchor_report.py       # Report generation
├── sgms_anchor_resilience.py   # Resilience testing
├── sgms_anchor_sensitivity.py  # Sensitivity analysis
├── lob_scaling.py              # 40-node lattice scaling analysis
├── metabolic_yield.py          # Metabolic yield calculations
├── generate_lob_plot.py        # LOB plot generation
├── generate_paper_plots.py     # Paper-quality plot generation
├── index.html                  # Interactive visualization (Spin-Gyro Magnetic Steering)
├── digital_twin.html           # Digital twin dashboard with real-time simulation
├── dynamics/
│   ├── rigid_body.py           # Full 3D Euler dynamics with gyroscopic coupling
│   ├── gyro_matrix.py          # Gyroscopic coupling matrix
│   ├── multi_body.py           # Multi-body packet stream simulation
│   ├── stiffness_verification.py  # Stiffness verification utilities
│   └── stress_monitoring.py    # Stress monitoring utilities
├── control/
│   ├── mpc_controller.py       # Model-predictive control (with configuration modes)
│   ├── rom_predictor.py        # Reduced-order model predictor (with VMD-IRCNN integration)
│   ├── vmd_ircnn_stub.py       # VMD-IRCNN ML predictor stub
│   ├── vmd_decomposition.py    # True VMD decomposition with ADMM optimization
│   ├── ircnn_predictor.py      # True IRCNN predictor with invertible residual blocks
│   ├── training_data_generator.py  # Training data generator from simulator
│   ├── training_pipeline.py    # Training pipeline with checkpointing
│   ├── state_converter.py      # ROM-VMD-IRCNN state conversion
│   └── ml_integration.py       # ML integration layer with feature flags
├── config/
│   └── ml_config.json          # ML feature flag configuration
├── monte_carlo/
│   ├── cascade_runner.py       # Monte-Carlo cascade analysis
│   └── pass_fail_gates.py      # Pass/fail gates including thermal safety
├── backend/
│   └── app.py                  # FastAPI backend for digital twin
├── paper_model/
│   ├── sgms_physics_v1.py      # Paper model physics implementation
│   ├── grail_perturbation_stub.py  # GRAIL perturbation stub
│   └── gdbco_apc_catalog.json  # GdBCO APC catalog data
├── tests/                      # Unit tests
│   ├── test_rigid_body.py      # Core physics tests (placeholder parameters)
│   ├── test_multi_body.py      # Multi-body stream tests
│   ├── test_operational_scale.py  # NEW: Paper-target scale validation
│   ├── test_parameter_sweeps.py   # NEW: Parameter sweep validation
│   ├── test_rom_predictor.py
│   ├── test_monte_carlo.py
│   └── test_sgms_*.py          # Anchor module tests
├── legacy/                     # DEPRECATED: Old test implementations
│   └── test_mpc.py            # Moved from tests/ (uses placeholder parameters)
├── background/
│   ├── physics_simulator_audit.md    # Comprehensive audit
│   ├── backgroundinfo.txt            # Original requirements spec
│   ├── external_data_requirements.md # Material data needed
│   ├── production_gap_analysis.md    # Gap assessment
│   ├── priority_analysis.md          # Feature prioritization
│   ├── special_maneuver_analysis.md  # Gyro precession details
│   ├── station_keeping_analysis.md   # Station keeping proofs
│   ├── ideal_blueprint_alignment.md  # Blueprint alignment plan
│   └── post_mortem_analysis.md       # Post-mortem analysis
└── docs/
    ├── unified_geometry_table.md     # NEW: Consolidated geometry parameters with code-paper gap analysis
    └── anchor-validation-decision.md # Anchor validation decision
```

---

##  Profile System

The codebase now supports a **3-profile system** for interchangeable parameter sweeps:

### Material Profiles
- Located in `paper_model/gdbco_apc_catalog.json`
- Define flux-pinning stiffness ranges, damping ratios, and material properties
- Used by the Bean-London model for temperature/field-dependent stiffness

### Geometry Profiles
- Located in `geometry_profiles.json`
- Define shape, mass, radius, and inertia tensor parameters
- Support sphere and prolate_spheroid shapes
- Used for rigid-body dynamics calculations

### Environment Profiles
- Located in `environment_profiles.json`
- Define temperature, magnetic field, radiation flux, and gravity
- Support cryogenic (LN2, LHe), orbital, and surface environments
- Used for thermal and electromagnetic modeling

### Profile Resolution
Profiles are resolved in `sgms_anchor_profiles.py` with:
- Validation of required fields and type/range checks
- Override precedence: experiment params > profile params > environment profile
- Graceful error handling with descriptive messages

### Configuration Modes

The codebase now supports three configuration modes to align with different use cases:

### TEST Mode (Default)
- **Purpose**: Fast unit tests with small parameters for quick iteration
- **Parameters**: 0.05 kg mass, 0.02 m radius, 100 rad/s spin
- **Use case**: Unit test validation, development debugging
- **Usage**: `MPCController(configuration_mode=ConfigurationMode.TEST)`

### VALIDATION Mode
- **Purpose**: MuJoCo oracle validation with realistic parameters
- **Parameters**: 2.0 kg mass, 0.1 m radius, 5236 rad/s spin
- **Use case**: 6-DOF physics validation against MuJoCo oracle
- **Usage**: `MPCController(configuration_mode=ConfigurationMode.VALIDATION)`

### OPERATIONAL Mode
- **Purpose**: Paper-derived operational scale validation
- **Parameters**: 8.0 kg mass, 0.1 m radius, 5236 rad/s spin
- **Use case**: Operational-scale physics validation, paper verification
- **Usage**: `MPCController(configuration_mode=ConfigurationMode.OPERATIONAL)`

See `docs/unified_geometry_table.md` for complete parameter comparison and code-paper gap analysis.

---

##  Usage & Simulations

### Run Full Logistics Simulation

```bash
python sgms_anchor_logistics.py
```

**Output**: Displacement stability, thermal flux, catch precision metrics

### Run 40-Node Lattice with Blackout Test

```bash
python lob_scaling.py
```

**Output**: `lob_survivability_blackout.png` (drift analysis, cascade prevention)

### Run Sensitivity Analysis

```bash
python sgms_anchor_sensitivity.py
```

**Output**: Sobol indices for all parameters (identifies critical variables)

### Run MuJoCo 6-DOF Validation

```bash
python sgms_anchor_mujoco.py
```

**Output**: Spin stability verification under steering torques

### Generate Paper-Quality Plots

```bash
python generate_paper_plots.py
```

**Output**: Efficiency comparison, displacement spectra, energy budget figures

### Launch Interactive Dashboard

```bash
# Open index.html in browser
```

**Features**: Real-time parameter sweeps, efficiency mode visualization

### Run Tests

```bash
pytest tests/
```

### Quality Assurance Strategy

The project uses a **local testing strategy** without external CI/CD dependencies:

**1. Health Check Script** (Pre-commit)
```powershell
.\health_check.ps1
```
Runs comprehensive checks before committing:
- Python syntax validation
- Unit tests with coverage (70% threshold)
- Linting with ruff
- Phase 3 critical tests

**2. Watch Mode** (Development)
```powershell
.\watch_mode.ps1
```
Automatically runs tests on file changes during development for immediate feedback.

**Usage**:
- Run `.\health_check.ps1` before each commit
- Run `.\watch_mode.ps1` during development for continuous feedback

### Train VMD-IRCNN Models

**GPU Requirements**:
- NVIDIA GPU with CUDA support (RTX 4070 or similar recommended)
- PyTorch with CUDA enabled
- Minimum 8 GB VRAM for training

**Python Version Setup**:

For GPU training, use Python 3.11 (recommended):

```powershell
# Check available Python versions
py --list

# Install CUDA PyTorch for Python 3.11
py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional dependencies
py -3.11 -m pip install scipy jax jaxlib

# Run training with Python 3.11
py -3.11 train_vmd_ircnn.py
```

**Python Version Note**:
- Python 3.11: CUDA wheels available, GPU training recommended
- Python 3.14: CUDA wheels not yet available, training will run on CPU
- Python 3.10-3.13: CUDA wheels available for GPU acceleration

**Training Command**:

```bash
py -3.11 train_vmd_ircnn.py
```

This will:
1. Generate synthetic training data from the high-fidelity simulator
2. Train a wobble detection model (binary classification)
3. Train a trajectory prediction model (regression)
4. Save checkpoints and training metrics to `data/models/`

**Datasets**:
- Wobble dataset: 1000 samples (800 train, 200 val)
- Prediction dataset: 1000 samples (800 train, 200 val)
- Saved to `data/` for reuse

**Feature Flags**:
Edit `config/ml_config.json` to switch between stub and true implementations:
```json
{
  "vmd_implementation": "stub",  // or "true"
  "ircnn_implementation": "stub",  // or "true"
  "enable_training": false
}
```

### Poetry Entry Points

```bash
poetry run lob-scaling      # Run LOB scaling analysis
poetry run sgms-anchor      # Run anchor simulation
poetry run sgms-suite       # Run config-driven experiment suite
```

---

##  Key Equations Implemented

### Momentum-Flux Force Law
```
F_anchor = λ · u² · sin(θ)
```
Where:
- λ = mass flow rate (kg/m)
- u = relative velocity (m/s)
- θ = deflection angle (rad)

### Full PID Control
```
F_corr(t) = -K_p·e(t) - K_i·∫e(τ)dτ - K_d·de/dt
```
With anti-windup clamping and derivative filtering

### Flux-Pinning (Critical-State Model)
```
J_c(B,T) = J_c0 · (1 - T/T_c)^n · f(B)
F_pin = ∫(J × B) dV
```

### Thermal Balance
```
P_eddy + P_electronic = P_cryocooler + P_radiative + P_conductive
```
Steady-state constraint: T_stator < 90K

### Wave Propagation (Dispersive)
```
∂²u/∂t² = c²·∂²u/∂x² + α·∂⁴u/∂x⁴
v_phase(ω) = √(c² + α·k²)
```

---

##  Validation & Verification

### Digital Twin Dashboard

Start FastAPI backend for digital twin visualization:

```powershell
python -m backend.app
```

Open digital_twin.html in browser to access the dashboard with real-time simulation state and Monte-Carlo analysis.

#### Key Physics Upgrade

The governing equation for each spin-stabilized magnetic packet, ≤50 krpm is now:

\[
\mathbf{I} \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{I} \boldsymbol{\omega}) = \boldsymbol{\tau}_\text{mag} + \boldsymbol{\tau}_\text{grav} + \boldsymbol{\tau}_\text{solar} + \boldsymbol{\tau}_\text{tether}
\]

where the skew-symmetric term \(\boldsymbol{\omega} \times (\mathbf{I} \boldsymbol{\omega})\) **is** the gyroscopic coupling that produces precession and libration. This term is non-negotiable for gyroscopic stability claims.

#### New Modules

- **dynamics/**: Full 3D rigid-body dynamics with explicit gyroscopic coupling + quaternion attitudes
- **control/**: Model-predictive control (MPC) and reduced-order model (ROM) predictors
- **monte_carlo/**: Cascade risk assessment and pass/fail gates

### Validation References
- MATLAB RK4 reference implementation
- MuJoCo 6-DOF cross-check
- Analytical solutions for limiting cases
- Literature values for material properties
- Classical EM theory for eddy current heating
- Euler equations for gyroscopic stability
- Sobol sensitivity analysis

---

##  Documentation

See CONTRIBUTING.md for development guidelines and project structure.

---



##  License & Citations

**License**: MIT License (see LICENSE file)

**Suggested Citation**:
```
@misc{spinnyball2025,
  title = {SpinnyBall: Lunar Orbital Belt Logistics Engine - Gyroscopic Mass-Stream Digital Twin},
  author = {Bittermun},
  year = {2025},
  note = {GitHub repository: github.com/bittermun/SpinnyBall}
}

---

**Last Updated**: 2026-04-27
**Development Lead**: Bittermun

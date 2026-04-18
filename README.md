# Project Aethelgard: The LOB Logistics Engine

Project Aethelgard is a simulation framework for a high-velocity kinetic logistics infrastructure designed for the **Lunar Orbital Belt (LOB)**. It models the transport of payloads across cislunar space using a persistent magnetic stream coupled to flux-pinned orbiting nodes.

## 🚀 Technical Core
The simulation models momentum harvesting from a magnetic stream, converted into payload acceleration, and stabilized via active and passive restorative forces.

### 🔴 Simulation Parameters
- **Stream Velocity**: Configurable (default $1600\text{ m/s}$ in lattice simulations)
- **Stiffness ($k_{\rm fp}$)**: Configurable (default $10^5\text{ N/m}$)
- **Node Mass**: Configurable (default $1000\text{ kg}$)
- **Lattice Size**: Configurable (default 20 nodes, supports scaling to 40+)

## 🛡️ Resilience & Hardening
The simulation supports multi-node lattice configurations to study structural integrity under various failure conditions.

### 🌑 Node Blackout Simulation
The framework includes a "Total Quench" simulation where a node loses all flux-pinning ($k \to 0$).
- **Local Drift**: The failing node drifts at stream velocity
- **Lattice Tension**: Coupling prevents cascade collapse across the mesh
- **Configurable**: Supports testing different node counts and failure scenarios

## 🛠️ Usage

### � LOB Lattice Scaling
Run lattice simulations with configurable node counts and failure scenarios:

```powershell
python lob_scaling.py --nodes 20
python lob_scaling.py --audit  # Runs N=20, N=40, and node blackout scenarios
```

Parameters: `--u`, `--lam`, `--k_fp`, `--nodes`, `--audit`

### ⚓ Anchor Simulation
Run anchor stability simulations with parameter sweeps:

```powershell
python sgms_anchor_v1.py --audit
```

Parameters: `--u`, `--lam`, `--g_gain`, `--k_fp`, `--ms`, `--audit`

### 🔬 Experiment Suite
Run config-driven experiment pipeline:

```powershell
python sgms_anchor_suite.py --repro --output artifacts
```

---

## 🏗️ Repository Structure

- **`paper_model/`**: Physics models and perturbation stubs for research expansion
- **`tests/`**: Regression suite for anchor control, calibration, and pipeline
- **`docs/`**: Research strategy and expansion documentation
- **`artifacts/`**: Output directory for reproducibility runs via `--repro`

## Core Simulation Modules

- **`lob_scaling.py`**: Multi-node lattice scaling and survivability simulations
- **`sgms_anchor_v1.py`**: Dynamic anchor stability with parameter sweeps
- **`sgms_anchor_logistics.py`**: Logistics event simulation with thermal balance
- **`sgms_anchor_suite.py`**: Config-driven experiment pipeline
- **`index.html`**: Interactive SGMS visualization (Spin-Gyro Magnetic Steering)

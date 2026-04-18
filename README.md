# Lunar Orbital Belt Logistics Engine

This is a simulation framework for a kinetic logistics infrastructure designed for the Lunar Orbital Belt (LOB). It models payload transport across cislunar space using a magnetic stream coupled to orbiting nodes.

## Technical Core

The simulation models momentum harvesting from a magnetic stream, converted into payload acceleration, and stabilized via active and passive restorative forces.

## Simulation Parameters

- Stream Velocity: Configurable (default 1600 m/s in lattice simulations)
- Stiffness: Configurable (default 100000 N/m)
- Node Mass: Configurable (default 1000 kg)
- Lattice Size: Configurable (default 20 nodes, supports scaling to 40+)

## Resilience Testing

The simulation supports multi-node lattice configurations to study structural integrity under various failure conditions.

## Node Failure Simulation

The framework includes a simulation where a node loses all flux-pinning (stiffness goes to zero).

- Local Drift: The failing node drifts at stream velocity
- Lattice Tension: Coupling prevents failure propagation across the mesh
- Configurable: Supports testing different node counts and failure scenarios

## Usage

### LOB Lattice Scaling

Run lattice simulations with configurable node counts and failure scenarios:

```powershell
python lob_scaling.py --nodes 20
python lob_scaling.py --audit
```

Parameters: --u, --lam, --k_fp, --nodes, --audit

### Anchor Simulation

Run anchor stability simulations with parameter sweeps:

```powershell
python sgms_anchor_v1.py --audit
```

Parameters: --u, --lam, --g_gain, --k_fp, --ms, --audit

### Experiment Suite

Run config-driven experiment pipeline:

```powershell
python sgms_anchor_suite.py --repro --output artifacts
```

## Repository Structure

- paper_model/: Physics models and perturbation stubs for research expansion
- tests/: Regression suite for anchor control, calibration, and pipeline
- docs/: Research strategy and expansion documentation
- artifacts/: Output directory for reproducibility runs via --repro

## Core Simulation Modules

- lob_scaling.py: Multi-node lattice scaling and survivability simulations
- sgms_anchor_v1.py: Dynamic anchor stability with parameter sweeps
- sgms_anchor_logistics.py: Logistics event simulation with thermal balance
- sgms_anchor_suite.py: Config-driven experiment pipeline
- index.html: Interactive visualization (Spin-Gyro Magnetic Steering)

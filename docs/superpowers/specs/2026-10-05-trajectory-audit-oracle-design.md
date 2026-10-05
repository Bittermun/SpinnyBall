# Trajectory Invariant Audit & Verification Oracle

## Context & Intent
SpinnyBall is designed as a small scientific laboratory for motion, balance errors, and reproducible physics experiments. Its hallmark is mathematical accountability: energy and momentum balance residuals are computed and flagged if they exceed 0.1% of reference scale.

Currently, this accountability only applies internally to SpinnyBall's built-in simulation runs. External space simulation projects, flight dynamics scripts, orbital propagators, and researchers lack a lightweight, dependency-free oracle to independently verify their trajectory data against physical conservation laws.

## Goal & Architecture
Create a zero-dependency ES module audit core (`workbench/audit.mjs`), a standalone CLI oracle (`workbench/verify-external.mjs`), and comprehensive unit tests (`workbench/tests/audit.test.mjs`) that take external orbital trajectory samples (CSV or JSON, via file or stdin) and calculate:
1. **Specific Mechanical Energy conservation** ($\Delta \mathcal{E} / \mathcal{E}_{\text{scale}}$)
2. **Specific Angular Momentum vector conservation** ($\|\Delta \vec{h}\| / \|\vec{h}_0\|$)
3. **Laplace-Runge-Lenz (Eccentricity) vector drift** ($\|\Delta \vec{e}\|$)
4. **Parasitic numerical acceleration residual** ($\|\vec{a}_{\text{discrete}} - \vec{a}_{\text{gravity}}\|$)
5. **Surface collision boundary penetration detection**
6. **Pass/Fail Audit Grade** adhering to SpinnyBall's strict $< 0.1\%$ ($10^{-3}$) invariant threshold.

## Ingestion Formats Supported
- **SpinnyBall CSV**: `time_s,x,y,vx,vy,...`
- **Standard 3D Cartesian CSV**: `t,x,y,z,vx,vy,vz` or `time,x,y,z,vx,vy,vz` (flexible column matching)
- **JSON array of state objects or arrays**: `[{t, r: [x,y,z], v: [vx,vy,vz]}]` or `[{t, x, y, z, vx, vy, vz}]`

## Technical Constraints
- Pure JavaScript ES module. Zero external npm packages.
- Compatible with Node 20+ test runner (`node --test`).
- Strict physical invariants: $< 0.1\%$ warning threshold.
- Division-by-zero protection: scale energy using potential scale $\mu / r_0$ for near-parabolic orbits where $\mathcal{E}_0 \to 0$.
- Handle 2D ($x, y, v_x, v_y$) and 3D ($x, y, z, v_x, v_y, v_z$) trajectories seamlessly.

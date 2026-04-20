# Comprehensive Parameter Reference

Single source of truth for all simulation parameters across the SpinnyBall project.

## Reduced-Order Model Parameters

Parameters from `sgms_anchor_v1.py` DEFAULT_PARAMS - main simulation parameters for the reduced-order anchor model.

| Parameter | Symbol | Value | Units | Source | Usage | Status |
|-----------|--------|-------|-------|--------|-------|--------|
| Stream velocity | u | 10.0 | m/s | sgms_anchor_v1.py:33 | Momentum flux calculation | **PLACEHOLDER** - Debug value |
| Linear density | lam | 0.5 | kg/m | sgms_anchor_v1.py:34 | Stream mass per unit length | **PLACEHOLDER** |
| Packet mass | mp | 0.05 | kg | sgms_anchor_v1.py:35 | Packet mass for reduced-order model | **PLACEHOLDER** - Test convenience |
| Bias angle | theta_bias | 0.087 | rad | sgms_anchor_v1.py:36 | Stream angle offset | **PLACEHOLDER** |
| Control gain | g_gain | 0.05 | dimensionless | sgms_anchor_v1.py:37 | Feedback controller gain | **PLACEHOLDER** |
| Station mass | ms | 1000.0 | kg | sgms_anchor_v1.py:38 | Anchor station mass | **PLACEHOLDER** |
| Damping coefficient | c_damp | 4.0 | N·s/m | sgms_anchor_v1.py:39 | System damping | **PLACEHOLDER** |
| Epsilon | eps | 0.0 | dimensionless | sgms_anchor_v1.py:40 | Bias force scaling | **PLACEHOLDER** |
| Disturbance std | disturbance_theta_std | 0.0 | rad | sgms_anchor_v1.py:41 | Noise magnitude | **PLACEHOLDER** |
| Disturbance hold | disturbance_hold_s | 1.0 | s | sgms_anchor_v1.py:42 | Noise duration | **PLACEHOLDER** |
| Packet sigma | packet_sigma_s | 0.01 | s | sgms_anchor_v1.py:43 | Packet timing spread | **PLACEHOLDER** |
| Packet phase | packet_phase_s | 0.0 | s | sgms_anchor_v1.py:44 | Packet phase offset | **PLACEHOLDER** |
| Pinning stiffness | k_fp | 0.0 | N/m | sgms_anchor_v1.py:45 | Magnetic pinning stiffness | **PLACEHOLDER** |
| Initial position | x0 | 0.1 | m | sgms_anchor_v1.py:46 | Initial displacement | **PLACEHOLDER** |
| Initial velocity | v0 | 0.0 | m/s | sgms_anchor_v1.py:47 | Initial velocity | **PLACEHOLDER** |
| Max time | t_max | 400.0 | s | sgms_anchor_v1.py:48 | Simulation duration | **PLACEHOLDER** |
| Relative tolerance | rtol | 1e-8 | dimensionless | sgms_anchor_v1.py:49 | Integrator tolerance | **DERIVED** |
| Absolute tolerance | atol | 1e-10 | dimensionless | sgms_anchor_v1.py:50 | Integrator tolerance | **DERIVED** |
| Max step | max_step | 0.25 | s | sgms_anchor_v1.py:51 | Integrator max step | **DERIVED** |

## Geometry and Physics Parameters

Parameters from `docs/unified_geometry_table.md` - detailed geometry and physics parameters for validation.

**See:** [unified_geometry_table.md](unified_geometry_table.md) for complete table including:
- Mass parameters (packet mass variations)
- Dimensional parameters (radii, axes)
- Inertia tensor parameters
- Angular velocity parameters
- Linear velocity parameters
- Stiffness parameters
- Stress parameters
- Efficiency parameters
- Thermal parameters
- Simulation parameters

## Parameter Mapping Between Systems

| Concept | Reduced-Order Model | Geometry Table | Notes |
|---------|-------------------|----------------|-------|
| Stream velocity | u (10.0 m/s) | u_velocity (10-1600 m/s) | Different values, same concept |
| Packet mass | mp (0.05 kg) | mp (0.05-8.0 kg) | ROM uses test convenience value |
| Linear density | lam (0.5 kg/m) | lam (16.6667 kg/m) | Different values |
| Station mass | ms (1000.0 kg) | node_mass (1000.0 kg) | Same value |
| Pinning stiffness | k_fp (0.0) | k_fp (4500.0 N/m) | ROM sets to 0, geometry has value |

## Configuration Modes

### Test Mode (Fast Unit Tests)
- Purpose: Quick test execution
- Parameters: Small mass (0.05 kg), low velocity (10 m/s), simple geometry
- Files: test_*.py, mpc_controller.py

### Validation Mode (MuJoCo Oracle)
- Purpose: High-fidelity physics validation
- Parameters: Realistic mass (2.0 kg), prolate geometry, operational spin rate
- Files: sgms_anchor_mujoco.py

### Operational Mode (Paper Target)
- Purpose: Real-world deployment
- Parameters: Paper-derived values (8.0 kg, 0.1 m radius, 5236 rad/s)
- Status: Not fully implemented in code

## Critical Placeholders Requiring Resolution

1. **Stream velocity**: u=10.0 m/s in ROM vs 1600 m/s in multi_body - needs operational value
2. **Linear density**: lam=0.5 kg/m in ROM vs 16.6667 kg/m in geometry - needs reconciliation
3. **Packet mass**: mp=0.05 kg in ROM vs 8.0 kg paper target - needs configuration switch
4. **Pinning stiffness**: k_fp=0.0 in ROM vs 4500 N/m in geometry - needs ROM implementation
5. **Control gain**: g_gain=0.05 - needs tuning for operational performance

## Next Steps

1. Resolve placeholder values with paper-derived or operational values
2. Add configuration system to switch between test/validation/operational modes
3. Update ROM parameters to match geometry table where appropriate
4. Document parameter dependencies and scaling relationships

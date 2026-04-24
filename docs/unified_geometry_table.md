# Unified Geometry Table

## Overview
Comprehensive consolidation of geometric and physical parameters from the physics simulation core files (rigid_body.py, gyro_matrix.py, sgms_anchor_mujoco.py, multi_body.py, mpc_controller.py, test_rigid_body.py, cascade_runner.py).

---

## Mass Parameters

| Parameter | Symbol | Value | Units | Source File | Usage | Derivation |
|-----------|--------|-------|-------|-------------|-------|------------|
| Packet mass (MuJoCo) | mp | 2.0 | kg | sgms_anchor_mujoco.py:29 | MuJoCo validation | **PLACEHOLDER** - Paper specifies 8 kg |
| Packet mass (paper target) | mp | 8.0 | kg | Paper derivation | Operational | **DERIVED** - BFRP sleeve r≈0.1m, ρ=2500 kg/m³ |
| Packet mass (test) | mass | 0.05 | kg | test_rigid_body.py:150 | Unit test asymmetric body | **PLACEHOLDER** - Test convenience |
| Packet mass (MPC) | packet_mass | 0.05 | kg | mpc_controller.py:59 | MPC controller default | **PLACEHOLDER** - Test convenience |
| Node mass | node_mass | 1000.0 | kg | sgms_anchor_mujoco.py:36 | Station mass | **PLACEHOLDER** |
| Linear density | lam | 16.6667 | kg/m | sgms_anchor_mujoco.py:34 | Momentum flux (s=0.12m) | **PLACEHOLDER** |

**OPTIMIZABLE VARIABLES** (for mass sweep testing):
- Packet mass: Test sweep [0.05, 0.5, 2.0, 8.0] kg to validate scaling of:
  - Inertia tensor (I ∝ m·r²)
  - Centrifugal stress (σ ∝ m·ω²/r)
  - Angular momentum (L = I·ω)
  - Gyroscopic coupling magnitude

**Discrepancy**: Packet mass varies between 2.0 kg (MuJoCo), 0.05 kg (MPC/tests), 8.0 kg (paper). **CRITICAL GAP**: Code uses 4x-160x lower mass than paper target.

---

## Dimensional Parameters

| Parameter | Symbol | Value | Units | Source File | Usage | Derivation |
|-----------|--------|-------|-------|-------------|-------|------------|
| Semi-major axis | a | 0.1 | m | sgms_anchor_mujoco.py:30 | Prolate spheroid (x-axis) | **DERIVED** - Matches paper r≈0.1m |
| Semi-minor axis | b, c | 0.046 | m | sgms_anchor_mujoco.py:31 | Prolate spheroid (y,z axes) | **PLACEHOLDER** |
| Packet radius (default) | radius | 0.02 | m | multi_body.py:82 | Stress calculations | **PLACEHOLDER** - 5x smaller than paper |
| Packet radius (MPC) | packet_radius | 0.02 | m | mpc_controller.py:58 | MPC stress constraint | **PLACEHOLDER** - 5x smaller than paper |
| Packet radius (paper) | r | 0.1 | m | Paper derivation | Operational | **DERIVED** - BFRP sleeve at 8 kg, ρ=2500 kg/m³ |
| Test sphere radius | radius | 0.02 | m | test_rigid_body.py:151 | Unit test body | **PLACEHOLDER** - Test convenience |
| S-Node capture radius | capture_radius | 10.0 | m | multi_body.py:41 | Magnetic capture | **PLACEHOLDER** |
| S-Node release radius | release_radius | 5.0 | m | multi_body.py:42 | Magnetic release | **PLACEHOLDER** |

**OPTIMIZABLE VARIABLES** (for geometry sweep testing):
- Packet radius: Test sweep [0.02, 0.05, 0.1] m to validate scaling of:
  - Inertia tensor (I ∝ m·r²)
  - Centrifugal stress (σ ∝ m·ω²/r)
  - Gyroscopic coupling (ω × I·ω)

**Note**: Prolate spheroid geometry (a=0.1m, b=c=0.046m) used in MuJoCo validation. Sphere (r=0.02m) used in MPC/tests. **CRITICAL GAP**: Code uses 5x smaller radius than paper target (0.02m vs 0.1m).

---

## Inertia Tensor Parameters

| Parameter | Symbol | Value | Units | Source File | Usage | Derivation |
|-----------|--------|-------|-------|-------------|-------|------------|
| Prolate spheroid Ixx | ix | 0.4·m·b² | kg·m² | sgms_anchor_mujoco.py:77 | MuJoCo XML | **DERIVED** - First principles |
| Prolate spheroid Iyy | iy | 0.2·m·(a²+b²) | kg·m² | sgms_anchor_mujoco.py:78 | MuJoCo XML | **DERIVED** - First principles |
| Prolate spheroid Izz | iz | iy | kg·m² | sgms_anchor_mujoco.py:79 | MuJoCo XML | **DERIVED** - First principles |
| Sphere I (base) | I_sphere | (2/5)·m·r² | kg·m² | test_rigid_body.py:152 | Test body base | **DERIVED** - First principles |
| Asymmetric I | I | diag([I, 1.1I, 0.9I]) | kg·m² | test_rigid_body.py:154 | Test precession | **PLACEHOLDER** - Artificial asymmetry |
| MPC default I | I | diag([0.0001, 0.00011, 0.00009]) | kg·m² | mpc_controller.py:92 | MPC dynamics | **PLACEHOLDER** - Hardcoded |

**Computed values**:
- For m=2.0kg, b=0.046m: ix = 0.4·2.0·0.046² = 0.00169 kg·m²
- For m=2.0kg, a=0.1m, b=0.046m: iy = 0.2·2.0·(0.1²+0.046²) = 0.00285 kg·m²
- For m=0.05kg, r=0.02m: I_sphere = (2/5)·0.05·0.02² = 8e-6 kg·m²
- For m=8.0kg, r=0.1m (paper): I_sphere = (2/5)·8.0·0.1² = 0.032 kg·m²

**CRITICAL GAP**: Paper inertia (0.032 kg·m²) is 4000x larger than test inertia (8e-6 kg·m²).

---

## Angular Velocity Parameters

| Parameter | Symbol | Value | Units | Source File | Usage | Derivation |
|-----------|--------|-------|-------|-------------|-------|------------|
| Spin rate (MuJoCo) | omega_spin | 5236.0 | rad/s | sgms_anchor_mujoco.py:32 | ~50,000 RPM | **DERIVED** - Matches paper |
| Max spin rate (paper) | ω_max | 5657.0 | rad/s | Paper derivation | 54,000 RPM | **DERIVED** - σ_θ = ρr²ω² ≤ 800 MPa |
| Test spin rate | omega0 | [0, 100, 10] | rad/s | test_rigid_body.py:157 | ~950 RPM, induces precession | **PLACEHOLDER** - Test convenience |
| Angular velocity vector | ω | [ωx, ωy, ωz] | rad/s | rigid_body.py:13 | Body frame | **DERIVED** - First principles |

**OPTIMIZABLE VARIABLES** (for RPM sweep testing):
- Spin rate: Test sweep [100, 1000, 5000, 5236, 5657] rad/s to validate:
  - Centrifugal stress scaling (σ ∝ ω²)
  - Gyroscopic coupling scaling (ω × I·ω ∝ ω²)
  - Precession frequency
  - Stability threshold at ~20 k RPM (paper: 1D models fail above this)

---

## Linear Velocity Parameters

| Parameter | Symbol | Value | Units | Source File | Usage | Derivation |
|-----------|--------|-------|-------|-------------|-------|------------|
| Stream velocity (MuJoCo) | u_velocity | 10.0 | m/s | sgms_anchor_mujoco.py:33 | MuJoCo validation | **PLACEHOLDER** - Debug value |
| Stream velocity (multi-body) | stream_velocity | 1600.0 | m/s | multi_body.py:161 | Multi-body stream | **PLACEHOLDER** - No paper derivation |
| Orbital velocity (LEO) | v_orb | ~7600 | m/s | Paper context | Reference | **DERIVED** - LEO circular |

**OPTIMIZABLE VARIABLES** (for velocity sweep testing):
- Stream velocity: Test sweep [10, 100, 1000, 1600] m/s to validate:
  - Momentum flux scaling (F = λ·u²)
  - Dynamic pressure effects
  - Libration dynamics
  - Magnetic capture timing

**Discrepancy**: Factor of 160x difference between MuJoCo (10 m/s) and multi-body (1600 m/s) stream velocities. **GAP**: Paper does not specify operational stream velocity.

---

## Stiffness Parameters

| Parameter | Symbol | Value | Units | Source File | Usage |
|-----------|--------|-------|-------|-------------|-------|
| GdBCO pinning stiffness | k_fp | 4500.0 | N/m | sgms_anchor_mujoco.py:35 | Magnetic pinning |
| Minimum effective stiffness | k_eff_min | 6000.0 | N/m | mpc_controller.py:56 | MPC constraint |
| Minimum effective stiffness | k_eff_min | 6000.0 | N/m | cascade_runner.py:86 | Pass/fail gate |

**Discrepancy**: k_fp=4500 N/m used in MuJoCo, but gate requires k_eff≥6000 N/m. GdBCO stiffness below gate threshold.

---

## Stress Parameters

| Parameter | Symbol | Value | Units | Source File | Usage | Derivation |
|-----------|--------|-------|-------|-------------|-------|------------|
| Maximum centrifugal stress | max_stress | 1.2e9 | Pa | mpc_controller.py:55 | 1.2 GPa, SF=1.5 | **PLACEHOLDER** - Paper uses 800 MPa |
| Maximum centrifugal stress | stress_max | 1.2e9 | Pa | cascade_runner.py:85 | Pass/fail gate | **PLACEHOLDER** - Paper uses 800 MPa |
| Material limit (paper) | σ_allow | 8.0e8 | Pa | Paper derivation | 800 MPa, SF=1.5 | **DERIVED** - BFRP composite limit |
| Centrifugal acceleration | a | 2.74e6 | m/s² | Paper derivation | 279,000 g at 50k RPM | **DERIVED** - a = r·ω² |

**Stress formulas**:
- Paper (hoop stress): σ_θ = ρ·r²·ω² ≤ 800 MPa
- MPC (spherical): σ = m·ω² / (π·r)

**CRITICAL GAP**: Code uses 1.2 GPa (50% higher) vs paper 800 MPa. Different stress formulas used.

---

## Efficiency Parameters

| Parameter | Symbol | Value | Units | Source File | Usage |
|-----------|--------|-------|-------|-------------|-------|
| Minimum induction efficiency | eta_ind_min | 0.82 | dimensionless | multi_body.py:44 | S-Node constraint |
| Minimum induction efficiency | eta_ind_min | 0.82 | dimensionless | cascade_runner.py:84 | Pass/fail gate |
| Default induction efficiency | eta_ind | 1.0 | dimensionless | multi_body.py:81 | Packet default |

---

## Thermal Parameters

| Parameter | Symbol | Value | Units | Source File | Usage | Derivation |
|-----------|--------|-------|-------|-------------|-------|------------|
| Initial temperature | temperature | 300.0 | K | multi_body.py:83 | Packet temperature | **PLACEHOLDER** |
| Emissivity (code) | emissivity | 0.8 | dimensionless | multi_body.py:84 | Al/BFRP | **PLACEHOLDER** - Paper uses 0.85 |
| Emissivity (paper) | ε_p | 0.85 | dimensionless | Paper derivation | BFRP | **DERIVED** - Thermal balance |
| Specific heat | specific_heat | 900.0 | J/kg·K | multi_body.py:85 | Aluminum | **PLACEHOLDER** |
| Surface area (paper) | A_p | 0.2 | m² | Paper derivation | Radiation | **DERIVED** - Thermal balance |
| Total power (paper) | P_total | ≤200 | W | Paper derivation | Eddy + solar | **DERIVED** - Thermal balance |
| Steady-state temp | T_ss | <420 | K | Paper derivation | Thermal margin | **DERIVED** - T = (P/εσA)^(1/4) |

**CRITICAL GAP**: Code missing thermal ODE implementation, surface area, and power balance parameters from paper.

---

## Simulation Parameters

| Parameter | Symbol | Value | Units | Source File | Usage |
|-----------|--------|-------|-------|-------------|-------|
| Number of packets | num_packets | 40 | count | sgms_anchor_mujoco.py:37 | MuJoCo pool |
| Time horizon | time_horizon | 10.0 | s | cascade_runner.py:57 | Monte Carlo |
| Time step | dt | 0.01 | s | cascade_runner.py:58 | Monte Carlo |
| MPC horizon | horizon | 10 | steps | mpc_controller.py:50 | MPC prediction |
| MPC time step | dt | 0.01 | s | mpc_controller.py:51 | MPC integration |
| MuJoCo timestep | timestep | 0.0005 | s | sgms_anchor_mujoco.py:83 | MuJoCo integrator |

---

## State Vector Conventions

| Parameter | Format | Convention | Source File |
|-----------|--------|-------------|-------------|
| Quaternion | [qx, qy, qz, qw] | scalar-last (scipy) | rigid_body.py:12 |
| Quaternion (internal) | [w, x, y, z] | scalar-first | rigid_body.py:88 |
| Angular velocity | [ωx, ωy, ωz] | body frame (rad/s) | rigid_body.py:13 |
| State vector | [qx, qy, qz, qw, ωx, ωy, ωz] | 7 elements | rigid_body.py:14 |
| MuJoCo quaternion | [w, x, y, z] | scalar-first | sgms_anchor_mujoco.py:44 |

---

## Key Discrepancies Explained

### 1. Packet Mass: 2.0 kg (MuJoCo) vs 0.05 kg (MPC/tests) - 40x difference

**Context**: The MuJoCo validation uses PacketParams with mp=2.0 kg, representing a larger mass packet. The MPC controller and unit tests use 0.05 kg (50g), representing a smaller test packet.

**Impact**:
- Inertia scales with mass: I_MuJoCo ≈ 0.0017-0.0029 kg·m² vs I_test ≈ 8e-6 kg·m² (~200-350x difference)
- Angular momentum L = I·ω will be dramatically different for same spin rate
- Centrifugal stress σ ∝ m·ω²/r will scale with mass

**Root cause**: Likely intentional - MuJoCo validates high-fidelity physics on realistic parameters, while MPC/tests use simplified smaller packets for faster computation and easier debugging. However, this means MPC may not be validated at operational scale.

**Recommendation**: Add configuration parameter to scale packet mass between validation (2.0 kg) and test (0.05 kg) modes. Document which parameters correspond to which use case.

---

### 2. Stream Velocity: 10 m/s (MuJoCo) vs 1600 m/s (multi-body) - 160x difference

**Context**: MuJoCo validation uses u_velocity=10 m/s for packet stream velocity, while MultiBodyStream defaults to stream_velocity=1600 m/s.

**Impact**:
- Momentum flux F = λ·u² scales with velocity squared: 1600²/10² = 25,600x difference in steering forces
- Dynamic pressure q = 0.5·ρ·u² (if modeling atmospheric effects) would be vastly different
- Libration dynamics and magnetic capture timing will be affected

**Root cause**: MuJoCo validation may be using a reduced velocity for visualization/debugging purposes (10 m/s is easier to observe), while 1600 m/s may represent the actual operational stream velocity for the mass-stream system.

**Recommendation**: Clarify whether 10 m/s is a debug value or represents a different operational regime. If 1600 m/s is the target, update MuJoCo validation to use realistic velocity for meaningful oracle comparison. Add configuration to switch between debug/operational velocities.

---

### 3. Geometry: Prolate Spheroid (MuJoCo) vs Sphere (MPC/tests)

**Context**: MuJoCo models packets as prolate spheroids (a=0.1m, b=c=0.046m, aspect ratio ≈ 2.17:1), while MPC and tests use spheres (r=0.02m).

**Impact**:
- Inertia tensor is anisotropic for prolate spheroid (Ixx ≠ Iyy = Izz), enabling realistic gyroscopic precession
- Sphere has isotropic inertia (Ixx = Iyy = Izz), which simplifies dynamics but may not capture all precession effects
- Test body adds artificial asymmetry (diag([I, 1.1I, 0.9I])) to induce precession, but this is a different physical mechanism
- MuJoCo's prolate geometry is more physically realistic for spin-stabilized packets (elongated along spin axis)

**Root cause**: Prolate spheroid is the actual packet geometry (elongated for spin stabilization). Sphere is used in MPC/tests for simplicity and because asymmetric test inertia already induces precession for validation.

**Recommendation**: This is acceptable for testing purposes, but MPC should ideally use prolate spheroid inertia for operational deployment. Consider adding geometry parameter to MPC controller to switch between sphere (test) and prolate spheroid (operational).

---

### 4. Stiffness: k_fp=4500 N/m (MuJoCo) vs Gate Requirement k_eff≥6000 N/m - GdBCO Below Threshold

**Context**: MuJoCo validation uses GdBCO pinning stiffness k_fp=4500 N/m, but the pass/fail gate requires k_eff≥6000 N/m.

**Impact**:
- MuJoCo validation will always fail the k_eff gate with current parameters
- This suggests the validation is not testing the actual operational configuration
- GdBCO stiffness of 4500 N/m is 25% below the required 6000 N/m threshold

**Root cause**: Either:
- k_fp=4500 N/m is outdated/incorrect and should be updated to ≥6000 N/m
- The gate threshold of 6000 N/m is too conservative and should be lowered
- MuJoCo validation is testing a degraded/edge case scenario

**Recommendation**: Determine correct GdBCO stiffness value from specifications. If 4500 N/m is correct, the gate threshold is unrealistic. If 6000 N/m is correct, update MuJoCo validation to use k_fp≥6000 N/m. This is a critical configuration mismatch that affects validation credibility.

---

### 5. Spin Rate: 5236 rad/s (MuJoCo) vs 100 rad/s (test) - 52x difference

**Context**: MuJoCo validation uses omega_spin=5236 rad/s (~50,000 RPM), while unit tests use omega0=[0, 100, 10] rad/s (~950 RPM).

**Impact**:
- Centrifugal stress σ ∝ ω²: (5236/100)² ≈ 2742x difference in stress
- Angular momentum L = I·ω: ~52x difference (assuming similar inertia)
- Gyroscopic coupling term ω×(I·ω) scales quadratically with ω
- Test at 100 rad/s may not validate high-RPM operational regime

**Root cause**: 5236 rad/s (~50,000 RPM) is the operational spin rate for packets. 100 rad/s (~950 RPM) is used in tests for:
- Faster integration (larger time steps stable at lower ω)
- Easier visualization of precession dynamics
- Reduced numerical stiffness in differential equations

**Recommendation**: Add high-RPM test case at operational spin rate (5236 rad/s) to validate gyroscopic coupling at actual operating conditions. Current tests only validate low-RPM regime, which may not expose high-RPM instabilities or stress limits.

---

## Summary

The discrepancies fall into three categories:

1. **Intentional test simplification**: Sphere geometry, low spin rate (100 rad/s), small mass (0.05 kg) - acceptable for unit tests but need operational-scale validation
2. **Configuration mismatch**: Stream velocity (10 vs 1600 m/s), stiffness (4500 vs 6000 N/m) - needs resolution to ensure validation credibility
3. **Missing operational validation**: No tests at 50,000 RPM, no MPC validation with prolate spheroid geometry - should be added for completeness

**Priority actions**:
1. Resolve stiffness mismatch (k_fp vs k_eff gate)
2. Clarify stream velocity values (debug vs operational)
3. Add operational-scale test case (50,000 RPM, 8.0 kg, r=0.1m, prolate spheroid)
4. Document parameter mapping between test/validation/operational modes

---

## Code-Paper Gap Analysis

### Critical Gaps (High Priority)

| Parameter | Code Value | Paper Value | Gap Factor | Impact |
|-----------|------------|-------------|------------|--------|
| **Packet mass** | 0.05-2.0 kg | 8.0 kg | 4x-160x | Inertia, stress, angular momentum all wrong scale |
| **Packet radius** | 0.02 m | 0.1 m | 5x | Inertia 25x lower, stress 5x higher (wrong formula) |
| **Inertia tensor** | 8e-6 kg·m² | 0.032 kg·m² | 4000x | Gyroscopic dynamics at wrong scale |
| **Stress limit** | 1.2 GPa | 800 MPa | 1.5x | Different safety factor, different formula |
| **Thermal ODE** | Missing | Full radiation balance | N/A | No thermal validation |

### Moderate Gaps (Medium Priority)

| Parameter | Code Value | Paper Value | Gap Factor | Impact |
|-----------|------------|-------------|------------|--------|
| **Stream velocity** | 10-1600 m/s | Not specified | Unknown | Momentum flux unclear |
| **Spin rate (tests)** | 100 rad/s | 5236 rad/s | 52x | Low-RPM only, no high-RPM validation |
| **Emissivity** | 0.8 | 0.85 | 1.06x | Minor thermal difference |
| **Geometry** | Sphere | Prolate spheroid | N/A | Isotropic vs anisotropic inertia |

### Missing from Code (Paper Has)

1. **Thermal ODE implementation**: Paper has full radiation balance with steady-state solution
2. **Centrifugal acceleration metric**: Paper derives 279,000 g at 50k RPM
3. **Regenerative power harvesting**: Kinetic recovery during payload deceleration
4. **Cascade probability sensitivity**: dP/d(latency) ≈ 0.001 ms⁻¹
5. **System mass for 1 MW routing**: ~45 t for 100-station constellation

### Missing from Paper (Code Has)

1. **MPC controller implementation**: Code has full CasADi MPC, paper mentions it but no details
2. **Monte Carlo cascade runner**: Code has full MC framework, paper mentions cascade probability but no implementation
3. **VMD-IRCNN stub**: Code has placeholder, paper mentions as digital twin
4. **GdBCO stiffness**: Code has k_fp=4500 N/m, paper doesn't specify magnetic pinning stiffness
5. **S-Node capture/release radii**: Code has 10m/5m, paper doesn't specify

### Recommendations

1. **Update code to paper target parameters**:
   - Set default packet mass to 8.0 kg (configurable for tests)
   - Set default radius to 0.1 m (configurable for tests)
   - Update stress limit to 800 MPa with SF=1.5
   - Implement thermal ODE with radiation balance

2. **Add parameter sweep tests**:
   - Mass sweep: [0.05, 0.5, 2.0, 8.0] kg
   - Radius sweep: [0.02, 0.05, 0.1] m
   - RPM sweep: [100, 1000, 5000, 5236, 5657] rad/s
   - Velocity sweep: [10, 100, 1000, 1600] m/s

3. **Add missing paper-derived metrics**:
   - Centrifugal acceleration calculation
   - Thermal steady-state temperature

4. **Document configuration modes**:
   - `test_mode`: 0.05 kg, 0.02 m, 100 rad/s (fast unit tests)
   - `validation_mode`: 2.0 kg, 0.1 m, 5236 rad/s (MuJoCo oracle)
   - `operational_mode`: 8.0 kg, 0.1 m, 5236 rad/s (paper target)

# Control System & Thermal Performance Documentation

## MPC Latency Tolerance

### Overview
The Model-Predictive Controller (MPC) implements Smith predictor delay compensation to handle control latency in the gyroscopic mass-stream system.

### Key Parameters
- **Control Horizon**: N=10 steps
- **Time Step**: dt=0.01s (10ms)
- **Target Solve Time**: ≤30ms per control cycle
- **Delay Compensation**: Smith predictor with configurable delay_steps

### Latency Tolerance Range
Based on `control_layer/mpc_controller.py` and `tests/test_mpc_delay_compensation.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `delay_steps` | 0-20 | Number of control cycles to compensate (default: 5) |
| `dt_delay` | 0.01s | Time step for delay prediction |
| `enable_delay_compensation` | True/False | Enable/disable Smith predictor |
| **Effective Latency Range** | **0-200ms** | delay_steps × dt_delay |

### Smith Predictor Implementation
The Smith predictor advances the system state by `delay_steps` time steps to predict the effect of control actions before they are applied, compensating for:

1. **Sensor latency**: Delay in state measurements
2. **Computation latency**: MPC solve time (target ≤30ms)
3. **Actuation latency**: Delay in applying control torques

### Configuration Modes
The MPC controller supports three configuration modes with different latency characteristics:

| Mode | Mass (kg) | Radius (m) | Spin Rate (rad/s) | Use Case |
|------|-----------|------------|-------------------|----------|
| TEST | 0.05 | 0.02 | 100 | Fast unit tests |
| VALIDATION | 2.0 | 0.1 | 5236 | MuJoCo oracle validation |
| OPERATIONAL | 8.0 | 0.1 | 5236 | Paper target (operational) |

### Latency Performance
- **Target**: ≤30ms solve time per control cycle
- **Verification**: `MPCController.verify_mpc_latency(n_trials=10)` validates this target
- **Numerical Stability**: Tested for delay_steps = [1, 5, 10, 20]

### Constraints
The MPC optimizes subject to:
- Centrifugal stress ≤ 1.2 GPa (safety factor 1.5)
- k_eff ≥ 6,000 N/m
- η_ind ≥ 0.82

### Cost Function
Minimizes weighted sum of:
- Libration energy (weight: 1.0)
- Spacing deviation from target (weight: 0.5)
- Control effort (weight: 0.1)

---

## Cryocooler Performance

### Overview
The cryocooler model provides temperature-dependent cooling power for maintaining GdBCO superconductors at cryogenic temperatures (70-90K operating range).

### Specifications (Default: Thales LPT9310 Series)
Based on `dynamics/cryocooler_model.py`:

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Cooling Power @ 70K | 5.0 | W | Cooling capacity at base temperature |
| Cooling Power @ 80K | 8.0 | W | Cooling capacity at nominal temperature |
| Cooling Power @ 90K | 12.0 | W | Cooling capacity at upper limit |
| Input Power @ 70K | 50.0 | W | Electrical power at base temperature |
| Input Power @ 80K | 60.0 | W | Electrical power at nominal temperature |
| Input Power @ 90K | 80.0 | W | Electrical power at upper limit |
| Cooldown Time | 3600 | s | Time from 300K to 77K (1 hour) |
| Warmup Time | 60 | s | Time from 77K to 300K during quench (1 minute) |
| Mass | 5.0 | kg | Cryocooler mass |
| Volume | 0.01 | m³ | Cryocooler volume |
| Vibration Amplitude | 1e-6 | m | Microphonics level |

### Performance Characteristics

#### Cooling Power Curve
- **Interpolation**: Cubic spline fit through 70K, 80K, 90K data points
- **Range**: 70K (minimum) to 90K (maximum)
- **Below 70K**: Constant at 70K value (5W)
- **Above 90K**: Zero cooling (quench range)

#### Coefficient of Performance (COP)
- **Definition**: COP = Cooling Power / Input Power
- **70K**: 5.0W / 50.0W = 0.10
- **80K**: 8.0W / 60.0W = 0.13
- **90K**: 12.0W / 80.0W = 0.15

#### Input Power Curve
- **Interpolation**: Piecewise linear between data points
- **70K-80K**: Linear interpolation
- **80K-90K**: Linear interpolation
- **Below 70K**: Constant at 70K value (50W)
- **Above 90K**: Constant at 90K value (80W)

### Thermal Integration
The cryocooler integrates with:
1. **Lumped thermal model** (`dynamics/lumped_thermal.py`)
2. **Coil switching loss models** (eddy current heating)
3. **Eclipse-aware solar flux** (orbital thermal environment)
4. **Thermal feedback loops** (temperature-dependent performance)

### Operational Considerations

#### Cooldown Phase
- **Duration**: 1 hour from 300K to 77K
- **Power**: High input power during cooldown
- **Strategy**: Pre-cool before superconducting operation

#### Steady-State Operation
- **Temperature**: 77-90K (GdBCO critical temperature ~92K)
- **Power**: 50-80W input for 5-12W cooling
- **COP**: 0.10-0.15 (typical for cryocoolers)

#### Quench Event
- **Warmup Time**: 1 minute (rapid temperature rise)
- **Cause**: Temperature exceeds 90K threshold
- **Recovery**: Requires full cooldown cycle

### Design Implications
1. **Power Budget**: 50-80W continuous power per cryocooler
2. **Thermal Margin**: Operate at 77-80K for safety margin to 90K limit
3. **Redundancy**: Multiple cryocoolers may be needed for fault tolerance
4. **Vibration**: 1μm amplitude may affect sensitive measurements

---

## Integration Notes

### Control-Thermal Coupling
The MPC controller and cryocooler system interact through:
1. **Temperature-dependent material properties**: GdBCO flux-pinning stiffness varies with temperature
2. **Thermal stress**: Temperature gradients induce mechanical stress
3. **Power budget**: Cryocooler power competes with control system power

### Latency-Thermal Tradeoffs
- **Higher latency** → Reduced control authority → Increased thermal disturbances
- **Lower temperature** → Higher flux-pinning stiffness → Better control performance
- **Cryocooler power** → Heat generation → May affect nearby components

### Recommended Operating Points
Based on current analysis:

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| Control Latency | ≤30ms | Meets MPC target with delay compensation |
| Operating Temperature | 77-80K | Safe margin to 90K quench limit |
| Cryocooler COP | 0.10-0.13 | Optimal range for power efficiency |
| Delay Steps | 5-10 | Balances prediction accuracy vs computational cost |

---

## Validation Status

### MPC Latency
- **Tests**: `tests/test_mpc_delay_compensation.py`
- **Coverage**: Smith predictor, delay compensation, numerical stability
- **Status**: Validated for delay_steps = [1, 5, 10, 20]
- **Gap**: Full latency sweep (5-50ms) in progress via T1 sweep

### Cryocooler Performance
- **Tests**: `tests/test_cryocooler.py`, `tests/test_lumped_thermal.py`
- **Coverage**: Cooling power curve, input power, COP
- **Status**: Model validated against specifications
- **Gap**: Integration with full system thermal balance not fully analyzed

---

## References
- MPC Controller: `control_layer/mpc_controller.py`
- Cryocooler Model: `dynamics/cryocooler_model.py`
- MPC Delay Tests: `tests/test_mpc_delay_compensation.py`
- Cryocooler Tests: `tests/test_cryocooler.py`

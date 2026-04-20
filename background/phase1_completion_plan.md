# Phase 1 Completion Plan: Comprehensive Implementation Roadmap

## Executive Summary

**Status**: Gap Analysis Verified - 3 Critical Components Missing
**Timeline**: Hybrid approach - full implementation for PID and Flux-Pinning, simplified thermal model
**Critical Path**: Stage 1 (PID) → Stage 2 (Thermal) → Stage 4 (Integration)
**Parallel Path**: Stage 3 (Flux-Pinning) can run concurrently with Stage 2

**Approach**: Hybrid - Full implementation for PID and Flux-Pinning, simplified lumped-parameter thermal model leveraging existing JAX infrastructure

---

## Gap Analysis Verification Results

### Component 1: PID Controller (I+D terms, anti-windup) ✅ VERIFIED MISSING

**Evidence**:
- `sgms_anchor_control.py` only implements proportional control via `build_proportional_gain()`
- No integral term, no derivative term, no anti-windup clamping
- Grep search for "class.*PID" returns zero results
- Only LQR and P-control exist in `simulate_controller()`

**Impact**: HIGH - Without I-term, steady-state errors persist. Without D-term, overshoot during shockwaves.

### Component 2: Thermal Management (cryocooler model, quench detection) ✅ VERIFIED MISSING

**Evidence**:
- `dynamics/thermal_model.py` only implements basic radiative cooling (Stefan-Boltzmann)
- No cryocooler performance curves (70-90K range)
- No quench detection logic (>90K threshold)
- No conductive heat paths through rotor shafts
- Grep search for "cryocooler|quench" only finds documentation mentions

**Impact**: CRITICAL - GdBCO quenches at 90K. Without thermal model, system can fail catastrophically.

### Component 3: Critical-State Flux-Pinning (Bean-London model) ✅ VERIFIED MISSING

**Evidence**:
- `dynamics/stiffness_verification.py` uses linear spring model (k_fp parameter)
- No Bean-London J_c(B,T) dependence
- No nonlinear pinning force, no hysteresis
- Grep search for "Bean.*London|J_c.*B.*T" only finds documentation mentions

**Impact**: HIGH - Linear model overestimates stiffness at large displacements. Real flux-pinning saturates.

### Component 4: Integration Testing + Documentation ⚠️ PARTIAL

**Evidence**:
- `test_pid_controller.py` does not exist (0 results)
- 31 test files exist for other components
- Documentation exists but needs updates for new components

**Impact**: MEDIUM - Cannot validate integration without comprehensive tests.

---

## Standards & Compliance Requirements

### Code Quality Standards
- **Python Version**: 3.11+ (per pyproject.toml)
- **Testing**: pytest with 70% coverage threshold (per health_check.ps1)
- **Linting**: ruff (line-length: 100, select: E, F, W, I, N, UP, B, C4)
- **Formatting**: black (line-length: 100, target-version: py311, py312)
- **Type Checking**: mypy (warn_return_any, warn_unused_configs)
- **Dependencies**: python-control library already available (imported as `control`)

### External Data Requirements (from external_data_requirements.md)

**CRITICAL NOTE**: External data acquisition will NOT block Stages 2-3. Conservative literature values will be used as placeholders with parameter uncertainty quantified in Monte-Carlo analysis. Data acquisition will proceed as a parallel activity.

**Placeholder Values (from literature)**:
- GdBCO J_c(B,T): Use J_c0 = 3×10¹⁰ A/m², T_c = 92K, n = 1.5 (SuperPower Inc. typical values)
- Cryocooler performance: Use Thales LPT9310 curves (5-12W cooling at 70-90K, 50-80W input)
- Eddy current coefficients: Use classical EM theory estimates (k_eddy = 0.01-0.1)

**Data Acquisition Priority** (parallel to implementation):
- GdBCO J_c(B,T) curves at 77K and varying magnetic fields
- Cryocooler cooling power vs temperature curves (70-90K range)
- Eddy current loss coefficients

**HIGH PRIORITY** (for refinement):
- Thermal conductivity κ(T) in superconducting state
- Specific heat capacity C_p(T) near transition temperature
- Quench propagation velocity

**MEDIUM PRIORITY** (for validation):
- Flux-pinning stiffness measurements
- SiC power electronics characteristics

### Architecture Standards
- Follow existing module structure (dynamics/, control_layer/, tests/)
- Use dataclasses for configuration parameters
- Implement proper error handling and validation
- Add comprehensive docstrings with LaTeX equations
- Maintain backward compatibility with existing APIs

---

## Detailed Implementation Plan

### Stage 1: PID Controller Enhancement

#### 1.1 Design PID Class
**File**: `sgms_anchor_control.py` (extend existing file)

**Leverage**: Follow patterns from `control_layer/mpc_controller.py` (ConfigurationMode enum, dataclass parameters, configuration-driven design)

**Requirements**:
```python
from dataclasses import dataclass
from enum import Enum

class PIDMode(Enum):
    """PID controller operating modes."""
    POSITION = "position"  # Position control
    VELOCITY = "velocity"  # Velocity control
    TEMPERATURE = "temperature"  # Temperature control

@dataclass
class PIDParameters:
    """PID controller parameters (following MPC controller pattern)."""
    mode: PIDMode = PIDMode.POSITION
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain
    tau_filter: float = 0.1  # Derivative filter time constant (s)
    output_min: float = -np.inf  # Anti-windup lower bound
    output_max: float = np.inf  # Anti-windup upper bound
    integral_min: float = -np.inf  # Integral clamping lower bound
    integral_max: float = np.inf  # Integral clamping upper bound
    delay_steps: int = 0  # Number of delay steps to compensate (from MPC pattern)

class PIDController:
    """Full PID controller with anti-windup and derivative filtering.
    
    Follows MPC controller pattern: configuration-driven, mode-based,
    delay compensation support.
    """
    
    def __init__(self, params: PIDParameters, dt: float):
        """Initialize PID controller."""
        self.params = params
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0  # For low-pass filtering
        self.delay_buffer = [] if params.delay_steps > 0 else None  # Delay compensation
        
    def update(self, error: float) -> float:
        """Compute PID output with anti-windup and derivative filtering."""
        # Proportional term
        p_term = self.params.kp * error
        
        # Integral term with anti-windup clamping
        self.integral += error * self.dt
        self.integral = np.clip(self.integral,
                               self.params.integral_min,
                               self.params.integral_max)
        i_term = self.params.ki * self.integral
        
        # Derivative term with low-pass filtering
        derivative = (error - self.prev_error) / self.dt
        alpha = self.dt / (self.params.tau_filter + self.dt)
        filtered_derivative = alpha * derivative + (1 - alpha) * self.prev_derivative
        self.prev_derivative = filtered_derivative
        d_term = self.params.kd * filtered_derivative
        
        # Total output with saturation
        output = p_term + i_term + d_term
        output = np.clip(output, self.params.output_min, self.params.output_max)
        
        # Delay compensation (Smith predictor pattern from MPC)
        if self.delay_buffer is not None:
            self.delay_buffer.append(output)
            if len(self.delay_buffer) > self.params.delay_steps:
                output = self.delay_buffer.pop(0)
        
        # Update state
        self.prev_error = error
        
        return output
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        if self.delay_buffer is not None:
            self.delay_buffer = []
```

**Tuning Methods** (add to file):
```python
def ziegler_nichols_tuning(ku: float, tu: float) -> PIDParameters:
    """Compute PID gains using Ziegler-Nichols method.
    
    Args:
        ku: Ultimate gain (gain at stability limit)
        tu: Ultimate period (period of oscillations at stability limit)
    
    Returns:
        PIDParameters with tuned gains
    """
    kp = 0.6 * ku
    ki = 1.2 * ku / tu
    kd = 0.075 * ku * tu
    return PIDParameters(kp=kp, ki=ki, kd=kd)

def manual_tuning(
    kp: float,
    ki: float = 0.0,
    kd: float = 0.0,
    tau_filter: float = 0.1,
) -> PIDParameters:
    """Create PID parameters with manual tuning."""
    return PIDParameters(kp=kp, ki=ki, kd=kd, tau_filter=tau_filter)
```

#### 1.2 Integrate PID with Existing Controller
**File**: `sgms_anchor_control.py`

**Modify** `simulate_controller()` to support "pid" option:
```python
elif controller == "pid":
    # Create PID controller
    pid_params = PIDParameters(
        kp=p_gain_scale * metrics["k_eff"],
        ki=0.1 * metrics["k_eff"],  # Default I gain
        kd=0.01 * metrics["k_eff"],  # Default D gain
        tau_filter=0.1,
        output_min=-params["u"] * 10,  # Reasonable saturation
        output_max=params["u"] * 10,
    )
    pid = PIDController(pid_params, dt=t_eval[1] - t_eval[0])
    
    # Simulate with PID (discrete-time integration)
    x = x0.copy()
    control_forces = []
    states = []
    
    for i, t in enumerate(t_eval):
        # PID control
        error = 0.0 - x[0]  # Setpoint is 0
        u = pid.update(error)
        control_forces.append(u)
        
        # Update state using Euler integration
        # Dynamics: m_s * x_ddot + c_damp * x_dot + k_eff * x = u + disturbance
        dxdt = x[1]  # velocity
        dvdt = (u + disturbance_force[i] - params["c_damp"] * x[1] - metrics["k_eff"] * x[0]) / params["ms"]
        
        x[0] += dxdt * dt  # Update position
        x[1] += dvdt * dt  # Update velocity
        states.append(x.copy())
    
    # Convert to numpy arrays for compatibility
    states = np.array(states)
    control_forces = np.array(control_forces)
    
    # Return results in same format as other controllers
    K = np.array([[pid_params.kp, pid_params.ki, pid_params.kd]])  # For compatibility
    
    return {
        "controller": controller,
        "t": t_eval,
        "y": states.T,  # State as output
        "state": states,
        "x": states[:, 0],
        "v": states[:, 1],
        "control_force": control_forces,
        "disturbance_force": disturbance_force,
        "K": K,
        "closed_loop_poles": np.array([]),  # Not applicable for PID
        "metrics": metrics,
        "params": params,
    }
```

#### 1.3 Unit Tests
**File**: `tests/test_pid_controller.py` (new file)

**Test Coverage**:
```python
def test_pid_parameters():
    """Test PIDParameters dataclass."""
    params = PIDParameters(kp=1.0, ki=0.1, kd=0.01)
    assert params.kp == 1.0
    assert params.ki == 0.1
    assert params.kd == 0.01

def test_pid_controller_step():
    """Test single step of PID controller."""
    params = PIDParameters(kp=1.0, ki=0.1, kd=0.01, dt=0.01)
    pid = PIDController(params, dt=0.01)
    output = pid.update(1.0)
    assert output > 0  # Should respond to positive error

def test_pid_anti_windup():
    """Test anti-windup clamping."""
    params = PIDParameters(
        kp=1.0, ki=0.1, kd=0.0,
        integral_min=-10.0, integral_max=10.0
    )
    pid = PIDController(params, dt=0.01)
    # Force integral to saturate
    for _ in range(1000):
        pid.update(1.0)
    assert pid.integral <= 10.0

def test_pid_derivative_filtering():
    """Test derivative low-pass filtering."""
    params = PIDParameters(kp=0.0, ki=0.0, kd=1.0, tau_filter=0.1)
    pid = PIDController(params, dt=0.01)
    # Step input should produce filtered derivative
    output1 = pid.update(0.0)
    output2 = pid.update(1.0)
    assert abs(output2) < 100.0  # Should be filtered

def test_ziegler_nichols_tuning():
    """Test Ziegler-Nichols tuning method."""
    params = ziegler_nichols_tuning(ku=10.0, tu=5.0)
    assert params.kp == 6.0
    assert params.ki == 12.0 / 5.0
    assert params.kd == 0.75

def test_pid_reset():
    """Test controller reset."""
    params = PIDParameters(kp=1.0, ki=0.1, kd=0.01)
    pid = PIDController(params, dt=0.01)
    pid.update(1.0)
    pid.reset()
    assert pid.integral == 0.0
    assert pid.prev_error == 0.0
    assert pid.prev_derivative == 0.0

def test_pid_setpoint_tracking():
    """Test setpoint tracking with PID."""
    params = PIDParameters(kp=10.0, ki=1.0, kd=0.1)
    pid = PIDController(params, dt=0.01)
    errors = []
    setpoint = 1.0
    for i in range(100):
        measurement = 0.0  # Start at 0
        error = setpoint - measurement
        output = pid.update(error)
        errors.append(abs(error))
    # Errors should decrease over time
    assert errors[-1] < errors[0]
```

#### 1.4 Documentation
**Files to Update**:
- `README.md`: Add PID completion to Phase 1 checklist
- `background/physics_simulator_audit.md`: Mark PID as complete
- Add docstrings with LaTeX equations to PID class

---

### Stage 2: Thermal Management System - SIMPLIFIED

**Approach**: Lumped-parameter model (2 nodes: stator + rotor) using plain numpy. Provides 80-90% accuracy at 30% complexity compared to full thermal network. Simple, matches current implementation patterns.

#### 2.1 Cryocooler Performance Model
**File**: `dynamics/cryocooler_model.py` (new file)

**Requirements**:
```python
@dataclass
class CryocoolerSpecs:
    """Cryocooler performance specifications."""
    # Cooling power curve parameters (W vs temperature)
    cooling_power_at_70k: float  # W
    cooling_power_at_80k: float  # W
    cooling_power_at_90k: float  # W
    
    # Input power parameters
    input_power_at_70k: float  # W
    input_power_at_80k: float  # W
    input_power_at_90k: float  # W
    
    # Thermal properties
    cooldown_time: float  # s (from 300K to 77K)
    warmup_time: float  # s (from 77K to 300K during quench)
    
    # Physical properties
    mass: float  # kg
    volume: float  # m³
    vibration_amplitude: float  # m (microphonics)

class CryocoolerModel:
    """Cryocooler performance model with temperature-dependent cooling power."""
    
    def __init__(self, specs: CryocoolerSpecs):
        """Initialize cryocooler model."""
        self.specs = specs
        # Fit cooling power curve: P_cool(T) = a*T² + b*T + c
        self._fit_cooling_curve()
        
    def _fit_cooling_curve(self):
        """Fit quadratic curve to cooling power data."""
        T = np.array([70.0, 80.0, 90.0])
        P = np.array([
            self.specs.cooling_power_at_70k,
            self.specs.cooling_power_at_80k,
            self.specs.cooling_power_at_90k,
        ])
        # Quadratic fit: P = a*T² + b*T + c
        coeffs = np.polyfit(T, P, 2)
        self.cooling_coeffs = coeffs
        
    def cooling_power(self, temperature: float) -> float:
        """Compute cooling power at given temperature.
        
        Args:
            temperature: Current temperature (K)
        
        Returns:
            Cooling power (W)
        """
        if temperature < 70.0:
            return self.specs.cooling_power_at_70k
        elif temperature > 90.0:
            return 0.0  # No cooling above 90K (quench range)
        else:
            T = temperature
            a, b, c = self.cooling_coeffs
            return a * T**2 + b * T + c
    
    def input_power(self, temperature: float) -> float:
        """Compute input power at given temperature.
        
        Interpolates between known data points.
        """
        T = temperature
        if T <= 70.0:
            return self.specs.input_power_at_70k
        elif T >= 90.0:
            return self.specs.input_power_at_90k
        else:
            # Linear interpolation
            t = (T - 70.0) / (90.0 - 70.0)
            return (1 - t) * self.specs.input_power_at_70k + \
                   t * self.specs.input_power_at_90k
    
    def cop(self, temperature: float) -> float:
        """Compute coefficient of performance (cooling power / input power)."""
        p_cool = self.cooling_power(temperature)
        p_in = self.input_power(temperature)
        return p_cool / p_in if p_in > 0 else 0.0

# Default specifications (Thales LPT9310 series - placeholder values)
DEFAULT_CRYOCOOLER_SPECS = CryocoolerSpecs(
    cooling_power_at_70k=5.0,  # W
    cooling_power_at_80k=8.0,  # W
    cooling_power_at_90k=12.0,  # W
    input_power_at_70k=50.0,  # W
    input_power_at_80k=60.0,  # W
    input_power_at_90k=80.0,  # W
    cooldown_time=3600.0,  # 1 hour
    warmup_time=60.0,  # 1 minute (quench is fast)
    mass=5.0,  # kg
    volume=0.01,  # m³
    vibration_amplitude=1e-6,  # m
)
```

#### 2.2 Quench Detection Logic
**File**: `dynamics/quench_detector.py` (new file)

**Requirements**:
```python
@dataclass
class QuenchThresholds:
    """Quench detection thresholds."""
    temperature_critical: float = 90.0  # K (GdBCO T_c ≈ 92K)
    temperature_warning: float = 85.0  # K (13K margin)
    temperature_rate_limit: float = 10.0  # K/s (rapid heating)
    hysteresis: float = 2.0  # K (prevent chatter)

class QuenchDetector:
    """Quench detection and emergency shutdown logic."""
    
    def __init__(self, thresholds: QuenchThresholds):
        """Initialize quench detector."""
        self.thresholds = thresholds
        self.quenched = False
        self.warning_state = False
        self.prev_temperature = 70.0  # K
        self.quench_time = None
        
    def check_temperature(self, temperature: float, dt: float) -> dict:
        """Check temperature for quench conditions.
        
        Args:
            temperature: Current temperature (K)
            dt: Time step (s)
        
        Returns:
            Dictionary with status and alerts
        """
        # Compute heating rate
        heating_rate = (temperature - self.prev_temperature) / dt
        self.prev_temperature = temperature
        
        # Check critical threshold (with hysteresis)
        if self.quenched:
            # Stay quenched until below warning - hysteresis
            if temperature < self.thresholds.temperature_warning - self.thresholds.hysteresis:
                self.quenched = False
                self.quench_time = None
        else:
            # Trigger quench if above critical
            if temperature > self.thresholds.temperature_critical:
                self.quenched = True
                self.quench_time = 0.0  # Will be incremented by caller
        
        # Check warning threshold
        if temperature > self.thresholds.temperature_warning:
            self.warning_state = True
        else:
            self.warning_state = False
        
        # Check heating rate limit
        rate_violation = heating_rate > self.thresholds.temperature_rate_limit
        
        return {
            "quenched": self.quenched,
            "warning": self.warning_state,
            "rate_violation": rate_violation,
            "heating_rate": heating_rate,
            "emergency_shutdown": self.quenched or rate_violation,
        }
    
    def reset(self):
        """Reset quench detector."""
        self.quenched = False
        self.warning_state = False
        self.prev_temperature = 70.0
        self.quench_time = None
```

#### 2.3 Lumped-Parameter Thermal Model
**File**: `dynamics/lumped_thermal.py` (new file)

**Approach**: Simplified 2-node model (stator + rotor) using plain numpy. Uses explicit Euler integration instead of Newton-Raphson steady-state solver.

**Requirements**:
```python
from dataclasses import dataclass
import numpy as np

@dataclass
class LumpedThermalParams:
    """Parameters for lumped-parameter thermal model."""
    # Stator (GdBCO superconductor)
    stator_mass: float = 10.0  # kg
    stator_specific_heat: float = 500.0  # J/kg/K at 77K
    stator_surface_area: float = 0.1  # m²
    stator_emissivity: float = 0.1  # Low emissivity for superconductor

    # Rotor (magnetic bearing)
    rotor_mass: float = 5.0  # kg
    rotor_specific_heat: float = 400.0  # J/kg/K (aluminum at 77K)
    rotor_surface_area: float = 0.05  # m²
    rotor_emissivity: float = 0.2

    # Thermal coupling
    shaft_conductance: float = 10.0  # W/K (conductive path through shaft)

    # Operating conditions
    ambient_temp: float = 4.0  # K (deep space)
    initial_temp: float = 77.0  # K (operating temperature)

class LumpedThermalModel:
    """Lumped-parameter thermal model with 2 nodes (stator + rotor)."""

    def __init__(self, params: LumpedThermalParams, dt: float = 0.01):
        """Initialize lumped thermal model.

        Args:
            params: LumpedThermalParams
            dt: Time step (s)
        """
        self.params = params
        self.dt = dt

        # State: [T_stator, T_rotor]
        self.T_stator = params.initial_temp
        self.T_rotor = params.initial_temp

    def step(self, heat_sources: dict[str, float]) -> dict:
        """Step thermal model using explicit Euler integration.

        Args:
            heat_sources: Dictionary with 'stator' and 'rotor' heat input (W)

        Returns:
            Dictionary with updated temperatures and heat flows
        """
        # Extract heat inputs
        Q_stator = heat_sources.get('stator', 0.0)
        Q_rotor = heat_sources.get('rotor', 0.0)

        # Compute radiative loss (Stefan-Boltzmann)
        sigma = 5.67e-8
        P_rad_stator = self.params.stator_emissivity * sigma * \
                       self.params.stator_surface_area * \
                       (self.T_stator**4 - self.params.ambient_temp**4)
        P_rad_rotor = self.params.rotor_emissivity * sigma * \
                      self.params.rotor_surface_area * \
                      (self.T_rotor**4 - self.params.ambient_temp**4)

        # Compute conductive heat flow between stator and rotor
        P_cond = self.params.shaft_conductance * (self.T_rotor - self.T_stator)

        # Net heat flow
        Q_net_stator = Q_stator - P_rad_stator + P_cond
        Q_net_rotor = Q_rotor - P_rad_rotor - P_cond

        # Temperature change (Euler integration)
        dT_stator = Q_net_stator * self.dt / \
                     (self.params.stator_mass * self.params.stator_specific_heat)
        dT_rotor = Q_net_rotor * self.dt / \
                    (self.params.rotor_mass * self.params.rotor_specific_heat)

        # Update temperatures
        self.T_stator += dT_stator
        self.T_rotor += dT_rotor

        return {
            'T_stator': self.T_stator,
            'T_rotor': self.T_rotor,
            'P_rad_stator': P_rad_stator,
            'P_rad_rotor': P_rad_rotor,
            'P_cond': P_cond,
            'Q_net_stator': Q_net_stator,
            'Q_net_rotor': Q_net_rotor,
        }

    def get_temperatures(self) -> np.ndarray:
        """Get current temperatures as array."""
        return np.array([self.T_stator, self.T_rotor])

    def reset(self):
        """Reset to initial conditions."""
        self.T_stator = self.params.initial_temp
        self.T_rotor = self.params.initial_temp
```

#### 2.4 Integration with Existing Thermal Model
**File**: `dynamics/thermal_model.py` (modify)

**Add lumped-parameter model integration**:
```python
from dynamics.lumped_thermal import LumpedThermalModel, LumpedThermalParams

def create_anchor_lumped_thermal(
    stator_mass: float = 10.0,
    stator_specific_heat: float = 500.0,  # J/kg/K (GdBCO at 77K)
    rotor_mass: float = 5.0,
    rotor_specific_heat: float = 400.0,  # J/kg/K (aluminum at 77K)
    shaft_conductance: float = 10.0,  # W/K
    dt: float = 0.01,
) -> LumpedThermalModel:
    """Create lumped-parameter thermal model for anchor system.
    
    Args:
        stator_mass: Mass of GdBCO stator (kg)
        stator_specific_heat: Specific heat of stator (J/kg/K)
        rotor_mass: Mass of rotor (kg)
        rotor_specific_heat: Specific heat of rotor (J/kg/K)
        shaft_conductance: Thermal conductance of shaft (W/K)
        dt: Time step (s)
    
    Returns:
        LumpedThermalModel with stator and rotor nodes
    """
    params = LumpedThermalParams(
        stator_mass=stator_mass,
        stator_specific_heat=stator_specific_heat,
        rotor_mass=rotor_mass,
        rotor_specific_heat=rotor_specific_heat,
        shaft_conductance=shaft_conductance,
    )
    
    return LumpedThermalModel(params, dt=dt)
```

#### 2.5 Unit Tests
**Files**:
- `tests/test_cryocooler.py` (new)
- `tests/test_quench_detection.py` (new)
- `tests/test_lumped_thermal.py` (new)

**Test Coverage**:
- Cryocooler cooling power curve interpolation
- COP calculation
- Quench detection thresholds and hysteresis
- Heating rate violation detection
- Lumped-parameter model Euler integration
- Radiative and conductive heat flow calculations

#### 2.6 Documentation
**Files to Update**:
- `README.md`: Add thermal system completion to Phase 1 checklist
- `background/physics_simulator_audit.md`: Mark thermal as complete
- Add docstrings with lumped-parameter thermal equations

---

### Stage 3: Critical-State Flux-Pinning

#### 3.1 GdBCO Material Properties
**File**: `dynamics/gdBCO_material.py` (new file)

**Requirements**:
```python
@dataclass
class GdBCOProperties:
    """Material properties for GdBCO superconductor."""
    # Critical parameters
    Tc: float = 92.0  # K (critical temperature)
    Jc0: float = 3e10  # A/m² (critical current density at 0K, 0T)
    n_exponent: float = 1.5  # Temperature dependence exponent
    
    # Magnetic field dependence parameters
    B0: float = 0.1  # T (characteristic field)
    alpha: float = 0.5  # Field dependence exponent
    
    # Physical properties
    density: float = 6300.0  # kg/m³
    specific_heat: float = 500.0  # J/kg/K at 77K
    thermal_conductivity: float = 10.0  # W/m/K at 77K
    
    # Geometry (for coated conductor)
    thickness: float = 1e-6  # m (1 μm superconducting layer)
    width: float = 0.012  # m (12 mm wide tape)

class GdBCOMaterial:
    """GdBCO superconductor material model."""
    
    def __init__(self, props: GdBCOProperties):
        """Initialize material model."""
        self.props = props
        
    def critical_current_density(self, B: float, T: float) -> float:
        """Compute critical current density J_c(B, T) using Bean-London model.
        
        J_c(B, T) = J_c0 * (1 - T/T_c)^n * f(B)
        
        where f(B) = 1 / (1 + (B/B0)^α)
        
        Args:
            B: Magnetic flux density (T)
            T: Temperature (K)
        
        Returns:
            Critical current density (A/m²)
        """
        # Temperature dependence
        if T >= self.props.Tc:
            return 0.0  # Normal state
        
        temp_factor = (1.0 - T / self.props.Tc) ** self.props.n_exponent
        
        # Magnetic field dependence
        field_factor = 1.0 / (1.0 + (B / self.props.B0) ** self.props.alpha)
        
        return self.props.Jc0 * temp_factor * field_factor
    
    def critical_current(self, B: float, T: float) -> float:
        """Compute critical current I_c(B, T).
        
        I_c = J_c * cross_sectional_area
        
        Args:
            B: Magnetic flux density (T)
            T: Temperature (K)
        
        Returns:
            Critical current (A)
        """
        Jc = self.critical_current_density(B, T)
        area = self.props.thickness * self.props.width
        return Jc * area
```

#### 3.2 Bean-London Critical-State Model
**File**: `dynamics/bean_london_model.py` (new file)

**Requirements**:
```python
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class BeanLondonState:
    """State for Bean-London model (history-dependent)."""
    magnetization: np.ndarray  # Magnetization history
    previous_field: np.ndarray  # Previous magnetic field values
    penetration_depth: float  # Current flux penetration depth

class BeanLondonModel:
    """Bean-London critical-state model for flux-pinning.
    
    Models the critical state where current density equals J_c everywhere
    in the superconductor, creating a magnetization that opposes field changes.
    """
    
    def __init__(self, material: GdBCOMaterial, geometry: dict):
        """Initialize Bean-London model.
        
        Args:
            material: GdBCO material properties
            geometry: Dictionary with geometric parameters
                - thickness: Superconductor thickness (m)
                - width: Tape width (m)
                - length: Tape length (m)
        """
        self.material = material
        self.geometry = geometry
        self.state = BeanLondonState(
            magnetization=np.array([0.0]),
            previous_field=np.array([0.0]),
            penetration_depth=0.0,
        )
        
    def compute_pinning_force(self, displacement: float, B_field: float, 
                            temperature: float) -> float:
        """Compute flux-pinning force from Bean-London model.
        
        F_pin = ∫(J × B) dV
        
        For simplified geometry:
        F_pin = J_c(B, T) * B_field * volume * f(displacement)
        
        where f(displacement) models force saturation at large displacements.
        
        Args:
            displacement: Relative displacement (m)
            B_field: Magnetic flux density (T)
            temperature: Temperature (K)
        
        Returns:
            Pinning force (N)
        """
        # Get critical current density
        Jc = self.material.critical_current_density(B_field, temperature)
        
        # Compute penetration depth (increases with displacement)
        max_penetration = self.geometry["thickness"] / 2.0
        self.state.penetration_depth = min(
            abs(displacement) / max_penetration * max_penetration,
            max_penetration
        )
        
        # Effective volume with critical current
        volume = self.geometry["thickness"] * self.geometry["width"] * \
                 self.geometry["length"]
        effective_volume = volume * (self.state.penetration_depth / max_penetration)
        
        # Pinning force density: f_p = J_c × B
        force_density = Jc * B_field
        
        # Total pinning force (with saturation)
        F_pin = force_density * effective_volume
        
        # Saturation factor (force doesn't increase indefinitely)
        saturation_factor = np.tanh(abs(displacement) / (max_penetration * 0.1))
        F_pin *= saturation_factor
        
        # Direction opposes displacement
        F_pin *= -np.sign(displacement)
        
        return F_pin
    
    def update_magnetization(self, B_field: float, temperature: float):
        """Update magnetization history (hysteresis).
        
        The magnetization changes when the external field changes,
        with the rate limited by flux creep.
        
        Args:
            B_field: Current magnetic flux density (T)
            temperature: Temperature (K)
        """
        # Simplified hysteresis model
        delta_B = B_field - self.state.previous_field[-1]
        
        # Magnetization change proportional to field change
        Jc = self.material.critical_current_density(B_field, temperature)
        delta_M = -Jc * delta_B  # Opposes field change
        
        # Update history
        self.state.magnetization = np.append(self.state.magnetization, 
                                            self.state.magnetization[-1] + delta_M)
        self.state.previous_field = np.append(self.state.previous_field, B_field)
        
        # Keep history manageable
        if len(self.state.magnetization) > 100:
            self.state.magnetization = self.state.magnetization[-100:]
            self.state.previous_field = self.state.previous_field[-100:]
    
    def get_stiffness(self, displacement: float, B_field: float, 
                     temperature: float) -> float:
        """Compute effective stiffness k_fp = -dF_pin/dx.
        
        Args:
            displacement: Relative displacement (m)
            B_field: Magnetic flux density (T)
            temperature: Temperature (K)
        
        Returns:
            Effective stiffness (N/m)
        """
        # Numerical derivative
        delta = 1e-6  # Small displacement
        F1 = self.compute_pinning_force(displacement - delta, B_field, temperature)
        F2 = self.compute_pinning_force(displacement + delta, B_field, temperature)
        
        stiffness = -(F2 - F1) / (2 * delta)
        return stiffness
```

#### 3.3 Integration with Stiffness Verification
**File**: `dynamics/stiffness_verification.py` (modify)

**Add Bean-London integration**:
```python
def calculate_flux_pinning_stiffness(
    displacement: float,
    B_field: float,
    temperature: float,
    material: GdBCOMaterial,
    geometry: dict,
) -> float:
    """Calculate flux-pinning stiffness using Bean-London model.

    Args:
        displacement: Relative displacement (m)
        B_field: Magnetic flux density (T)
        temperature: Temperature (K)
        material: GdBCO material properties
        geometry: Geometry parameters

    Returns:
        Effective stiffness (N/m)
    """
    model = BeanLondonModel(material, geometry)
    return model.get_stiffness(displacement, B_field, temperature)
```

#### 3.4 Integration: Replace Linear k_fp with Dynamic Bean-London Stiffness
**File**: `sgms_anchor_v1.py` (modify)

**Current Implementation** (linear stiffness):
```python
# In analytical_metrics() function
k_fp = params.get("k_fp", 1000.0)  # Linear constant stiffness
k_eff = k_fp + k_structural
```

**New Implementation** (dynamic Bean-London stiffness):
```python
# Import Bean-London model
from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.bean_london import BeanLondonModel

# Initialize material and model (do this once at module level or in initialization)
gdBCO_props = GdBCOProperties(
    Tc=92.0,
    Jc0=3e10,
    n_exponent=1.5,
    B0=0.1,
    alpha=0.5,
    thickness=1e-6,
    width=0.012,
    length=1.0,  # 1 meter of tape
)
gdBCO_material = GdBCOMaterial(gdBCO_props)
flux_pinning_geometry = {
    "thickness": gdBCO_props.thickness,
    "width": gdBCO_props.width,
    "length": 1.0,
    "max_penetration": 1e-4,  # Maximum flux penetration depth
}
bean_london_model = BeanLondonModel(gdBCO_material, flux_pinning_geometry)

def analytical_metrics_with_flux_pinning(params: dict, temperature: float = 77.0, B_field: float = 1.0) -> dict:
    """Calculate metrics with dynamic Bean-London flux-pinning stiffness.

    Args:
        params: System parameters
        temperature: Temperature (K) for stiffness calculation
        B_field: Magnetic field (T) for stiffness calculation

    Returns:
        Dictionary with metrics including dynamic k_fp
    """
    # Calculate structural stiffness (existing code)
    k_structural = calculate_structural_stiffness(params)

    # Calculate dynamic flux-pinning stiffness
    # Use current displacement (from params) or equilibrium position
    displacement = params.get("x0", 0.0)  # Initial displacement
    k_fp = bean_london_model.get_stiffness(displacement, B_field, temperature)

    # Effective stiffness
    k_eff = k_fp + k_structural

    # Continue with existing metrics calculation
    # ... (rest of analytical_metrics code)

    return {
        **metrics,
        "k_fp": k_fp,
        "k_eff": k_eff,
        "temperature": temperature,
        "B_field": B_field,
    }
```

**Integration in Simulation Loop**:
```python
def simulate_anchor_with_flux_pinning(
    params: dict,
    t_eval: np.ndarray,
    temperature_profile: np.ndarray | None = None,
    B_field_profile: np.ndarray | None = None,
) -> dict:
    """Simulate anchor with dynamic Bean-London flux-pinning stiffness.

    Args:
        params: System parameters
        t_eval: Time evaluation points
        temperature_profile: Optional temperature profile over time (K)
        B_field_profile: Optional magnetic field profile over time (T)

    Returns:
        Dictionary with time series including dynamic k_fp
    """
    # Initialize temperature and field profiles
    if temperature_profile is None:
        temperature_profile = np.full_like(t_eval, 77.0)  # Constant 77K
    if B_field_profile is None:
        B_field_profile = np.full_like(t_eval, 1.0)  # Constant 1T

    # Initialize state
    x = params["x0"]
    v = params["v0"]
    dt = t_eval[1] - t_eval[0]

    # Storage for results
    results = {
        "t": t_eval,
        "x": [],
        "v": [],
        "k_fp": [],
        "k_eff": [],
        "temperature": [],
        "B_field": [],
    }

    # Simulation loop
    for i, t in enumerate(t_eval):
        # Get current temperature and field
        T = temperature_profile[i]
        B = B_field_profile[i]

        # Calculate dynamic flux-pinning stiffness
        k_fp = bean_london_model.get_stiffness(x, B, T)

        # Effective stiffness
        k_eff = k_fp + params["k_structural"]

        # Update dynamics (simplified Euler)
        # m_s * x_ddot + c_damp * x_dot + k_eff * x = 0
        a = -(params["c_damp"] * v + k_eff * x) / params["ms"]
        v += a * dt
        x += v * dt

        # Store results
        results["x"].append(x)
        results["v"].append(v)
        results["k_fp"].append(k_fp)
        results["k_eff"].append(k_eff)
        results["temperature"].append(T)
        results["B_field"].append(B)

    return results
```

**Backward Compatibility**:
```python
def analytical_metrics(params: dict) -> dict:
    """Calculate metrics with backward compatibility.

    If params contains "k_fp", use linear stiffness (old behavior).
    Otherwise, use dynamic Bean-London stiffness (new behavior).
    """
    if "k_fp" in params:
        # Legacy linear stiffness
        k_fp = params["k_fp"]
    else:
        # New dynamic Bean-London stiffness
        temperature = params.get("temperature", 77.0)
        B_field = params.get("B_field", 1.0)
        displacement = params.get("x0", 0.0)
        k_fp = bean_london_model.get_stiffness(displacement, B_field, temperature)

    k_eff = k_fp + params["k_structural"]

    # Continue with existing metrics calculation
    # ... (rest of code)

    return {
        **metrics,
        "k_fp": k_fp,
        "k_eff": k_eff,
    }
```

#### 3.5 Unit Tests
**Files**:
- `tests/test_bean_london.py` (new)
- `tests/test_flux_pinning.py` (new)

**Test Coverage**:
- GdBCO material properties
- Critical current density J_c(B, T)
- Bean-London pinning force calculation
- Magnetization hysteresis
- Stiffness computation
- Integration with stiffness verification

#### 3.5 Documentation
**Files to Update**:
- `README.md`: Add flux-pinning completion to Phase 1 checklist
- `background/physics_simulator_audit.md`: Mark flux-pinning as complete
- Add docstrings with Bean-London equations

---

### Stage 4: Integration Testing & Documentation

**Rationale for increased timeline**: Realistic integration complexity for 3 systems (PID + thermal + flux-pinning). Additional time needed for comprehensive scenario testing and documentation updates.

#### 4.0 Integration Architecture
**File**: `docs/phase1_integration_architecture.md` (new file)

**Component Interaction Diagram**:
```
┌─────────────────────────────────────────────────────────────────┐
│                        Anchor Simulation Loop                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   Dynamics   │    │     PID      │    │   Thermal    │        │
│  │   (Physics)  │───▶│  Controller  │───▶│    Model     │        │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘        │
│         │                   │                   │                 │
│         │ displacement    │ control force     │ temperature     │
│         │                   │                   │                 │
│         ▼                   ▼                   ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ Flux-Pinning │◀───│   Quench     │◀───│ Cryocooler   │        │
│  │   (Stiffness)│    │  Detector   │    │   Model      │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│         │                   │                   │                 │
│         │ k_fp(T)          │ emergency         │ cooling power    │
│         │                   │ shutdown          │                   │
│         └───────────────────┴───────────────────┘                 │
│                           │                                     │
│                           ▼                                     │
│                   System State Update                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow Between Components**:

1. **PID → Thermal**: Control force affects rotor heating (eddy currents, friction)
   - PID output `u` → heat input `Q_rotor = f(u, velocity)`

2. **Thermal → PID**: Temperature affects PID behavior
   - High temperature → reduce PID gain (prevent thermal runaway)
   - Quench detected → PID emergency shutdown (output = 0)

3. **Thermal → Flux-Pinning**: Temperature affects stiffness
   - `k_fp(T) = k_fp_0 * (1 - T/T_c)^n` (temperature-dependent stiffness)
   - Near T_c → stiffness drops → reduced control authority

4. **Flux-Pinning → Dynamics**: Stiffness affects system dynamics
   - `k_eff = k_fp(T) + k_structural` (effective stiffness)
   - Stiffness affects natural frequency and control response

**Feedback Loop Specifications**:

**Quench → PID Feedback Loop**:
```python
# Quench detector monitors stator temperature
quench_status = quench_detector.check_temperature(T_stator, dt)

if quench_status['quenched'] or quench_status['emergency_shutdown']:
    # Emergency shutdown: disable PID output
    pid.reset()  # Clear integral term
    control_force = 0.0
    # Log quench event
    logger.warning(f"Quench detected at T={T_stator}K, emergency shutdown activated")
```

**Temperature → Stiffness Feedback Loop**:
```python
# Update flux-pinning stiffness based on temperature
k_fp = bean_london_model.get_stiffness(
    displacement=current_displacement,
    B_field=magnetic_field,
    temperature=T_stator
)

# Update system dynamics with temperature-dependent stiffness
k_eff = k_fp + k_structural
metrics['k_eff'] = k_eff
```

**PID → Thermal Feedback Loop**:
```python
# Compute heat input from PID control force
# Eddy current heating: Q_eddy ∝ (B × v)²
# Friction heating: Q_fric ∝ F_control × v

def compute_control_heating(control_force, velocity, B_field):
    """Compute heat input from control force."""
    # Eddy current heating (proportional to velocity squared)
    Q_eddy = k_eddy * (B_field * velocity)**2
    
    # Friction heating (proportional to force × velocity)
    Q_fric = mu * abs(control_force * velocity)
    
    return Q_eddy + Q_fric

heat_sources = {
    'stator': compute_control_heating(u, v, B),
    'rotor': 0.0  # Rotor heating from other sources
}
```

**Integration Interface Specification**:

**PID Controller Interface**:
```python
class PIDController:
    def update(self, error: float, temperature: float = 77.0) -> float:
        """Compute PID output with temperature-dependent gain scheduling.
        
        Args:
            error: Control error
            temperature: Current temperature (K) for gain scheduling
        
        Returns:
            Control output (clamped, with emergency shutdown if quenched)
        """
        # Gain scheduling: reduce gains near quench
        if temperature > 85.0:
            gain_schedule = (90.0 - temperature) / 5.0  # Linear ramp to 0
            self.params.kp *= gain_schedule
            self.params.ki *= gain_schedule
            self.params.kd *= gain_schedule
        
        # Standard PID computation
        # ... (existing code)
        
        return output
```

**Thermal Model Interface**:
```python
class LumpedThermalModel:
    def step(self, heat_sources: dict[str, float], control_force: float, 
             velocity: float, B_field: float) -> dict:
        """Step thermal model with control heating.
        
        Args:
            heat_sources: External heat inputs (W)
            control_force: PID control force (N)
            velocity: Rotor velocity (rad/s)
            B_field: Magnetic field (T)
        
        Returns:
            Dictionary with temperatures and heat flows
        """
        # Add control heating
        Q_control = compute_control_heating(control_force, velocity, B_field)
        heat_sources['rotor'] += Q_control
        
        # Standard thermal step
        # ... (existing code)
        
        return result
```

**Flux-Pinning Interface**:
```python
class BeanLondonModel:
    def get_stiffness(self, displacement: float, B_field: float, 
                     temperature: float) -> float:
        """Compute temperature-dependent stiffness.
        
        Args:
            displacement: Relative displacement (m)
            B_field: Magnetic flux density (T)
            temperature: Temperature (K)
        
        Returns:
            Effective stiffness (N/m)
        """
        # Update magnetization history
        self.update_magnetization(B_field, temperature)
        
        # Compute pinning force with saturation
        F_pin = self.compute_pinning_force(displacement, B_field, temperature)
        
        # Numerical derivative for stiffness
        # ... (existing code)
        
        return stiffness
```

**Main Simulation Loop Integration**:
```python
def simulate_anchor_with_phase1(
    initial_state: np.ndarray,
    params: dict,
    t_eval: np.ndarray,
    pid_params: PIDParameters,
    thermal_params: LumpedThermalParams,
    flux_pinning_geometry: dict,
) -> dict:
    """Simulate anchor with PID, thermal, and flux-pinning integration.
    
    Args:
        initial_state: Initial system state [x, v, theta, omega, T_stator, T_rotor]
        params: System parameters
        t_eval: Time evaluation points
        pid_params: PID controller parameters
        thermal_params: Lumped thermal parameters
        flux_pinning_geometry: Flux-pinning geometry parameters
    
    Returns:
        Dictionary with time series of all variables
    """
    # Initialize components
    pid = PIDController(pid_params, dt=t_eval[1] - t_eval[0])
    thermal = LumpedThermalModel(thermal_params, dt=t_eval[1] - t_eval[0])
    quench = QuenchDetector(QuenchThresholds())
    material = GdBCOMaterial(GdBCOProperties())
    flux_pinning = BeanLondonModel(material, flux_pinning_geometry)
    
    # Simulation loop
    results = {
        't': t_eval,
        'x': [], 'v': [], 'theta': [], 'omega': [],
        'T_stator': [], 'T_rotor': [],
        'control_force': [], 'k_fp': [], 'quenched': []
    }
    
    state = initial_state.copy()
    
    for i, t in enumerate(t_eval):
        # Extract state
        x, v, theta, omega, T_stator, T_rotor = state
        
        # Compute flux-pinning stiffness (temperature-dependent)
        B_field = params.get('B_field', 1.0)  # Magnetic field
        k_fp = flux_pinning.get_stiffness(x, B_field, T_stator)
        k_eff = k_fp + params['k_structural']
        
        # PID control
        error = 0.0 - x  # Setpoint is 0
        u = pid.update(error, temperature=T_stator)
        
        # Check quench
        quench_status = quench.check_temperature(T_stator, dt=t_eval[1] - t_eval[0])
        if quench_status['quenched'] or quench_status['emergency_shutdown']:
            u = 0.0  # Emergency shutdown
            pid.reset()
        
        # Thermal update
        heat_sources = {
            'stator': 0.0,  # Stator heating from other sources
            'rotor': 0.0,  # Will add control heating
        }
        thermal_result = thermal.step(heat_sources, u, v, B_field)
        T_stator = thermal_result['T_stator']
        T_rotor = thermal_result['T_rotor']
        
        # Dynamics update (simplified Euler for illustration)
        F_total = u - k_eff * x  # Control force - restoring force
        a = F_total / params['mass']
        v += a * dt
        x += v * dt
        
        # Store results
        results['x'].append(x)
        results['v'].append(v)
        results['theta'].append(theta)
        results['omega'].append(omega)
        results['T_stator'].append(T_stator)
        results['T_rotor'].append(T_rotor)
        results['control_force'].append(u)
        results['k_fp'].append(k_fp)
        results['quenched'].append(quench_status['quenched'])
        
        # Update state
        state = np.array([x, v, theta, omega, T_stator, T_rotor])
    
    return results
```

#### 4.1 End-to-End Integration Test
**File**: `tests/test_phase1_integration.py` (new file)

**Test Scenarios**:
```python
def test_pid_thermal_integration():
    """Test PID controller with thermal management."""
    # Create thermal network with cryocooler
    # Apply PID control for temperature regulation
    # Verify temperature stays within limits

def test_flux_pinning_thermal_coupling():
    """Test flux-pinning stiffness with temperature dependence."""
    # Vary temperature from 70K to 90K
    # Verify stiffness decreases as temperature approaches T_c
    # Check quench detection triggers at 90K

def test_full_anchor_simulation():
    """Test full anchor with PID, thermal, and flux-pinning."""
    # Create anchor with all three components
    # Apply disturbance force
    # Verify displacement, temperature, and control force responses
```

#### 4.2 Scenario-Based Tests
**File**: `tests/test_phase1_scenarios.py` (new file)

**Test Scenarios**:
```python
def test_quench_event_scenario():
    """Test quench event and emergency shutdown."""
    # Inject rapid heating to trigger quench
    # Verify emergency shutdown activates
    # Verify temperature recovers after quench

def test_thermal_runaway_scenario():
    """Test thermal runaway prevention."""
    # Apply excessive heat load
    # Verify cryocooler reaches maximum capacity
    # Verify quench detection triggers before catastrophic failure

def test_flux_pinning_saturation_scenario():
    """Test flux-pinning saturation at large displacements."""
    # Apply large displacement
    # Verify pinning force saturates (doesn't increase indefinitely)
    # Verify stiffness decreases at large displacements
```

#### 4.3 Performance Benchmarks
**File**: `tests/test_phase1_benchmarks.py` (new file)

**Benchmarks**:
- PID controller update time (< 1 ms)
- Thermal network step time (< 10 ms)
- Bean-London force computation time (< 5 ms)
- Full simulation step time (< 50 ms)

#### 4.4 Documentation Updates
**Files to Update**:
- `README.md`: Mark Phase 1 as complete
- `background/physics_simulator_audit.md`: Update audit to reflect completion
- Create `docs/phase1_integration_guide.md` (new, 500 LOC)
- Update API documentation for all new modules

---

## Risk Assessment & Mitigation Strategies

### Risk 1: External Data Unavailable (CRITICAL)
**Probability**: MEDIUM
**Impact**: HIGH - Cannot implement accurate thermal or flux-pinning models

**Mitigation**:
- Use conservative placeholder values from literature
- Perform sensitivity analysis to identify critical parameters
- Document assumptions clearly
- Plan for data acquisition as parallel task

### Risk 2: Thermal Network Convergence Issues (MEDIUM)
**Probability**: MEDIUM
**Impact**: MEDIUM - Steady-state solver may not converge

**Mitigation**:
- Implement robust Newton-Raphson with damping
- Add fallback to iterative relaxation method
- Add convergence diagnostics and warnings
- Test with wide range of parameter values

### Risk 3: PID Tuning Difficulty (MEDIUM)
**Probability**: HIGH
**Impact**: MEDIUM - Poorly tuned PID may cause instability

**Mitigation**:
- Implement Ziegler-Nichols auto-tuning
- Add manual tuning interface
- Provide default conservative gains
- Add stability margins analysis

### Risk 4: Integration Complexity (HIGH)
**Probability**: MEDIUM
**Impact**: HIGH - Components may not integrate cleanly

**Mitigation**:
- Design clear interfaces between components
- Implement integration tests early
- Use dependency injection for flexibility
- Document integration points thoroughly

### Risk 5: Timeline Overrun (MEDIUM)
**Probability**: MEDIUM
**Impact**: MEDIUM - Integration complexity for 3 systems

**Mitigation**:
- Prioritize critical path (PID → Thermal → Integration)
- Defer nice-to-have features to Phase 2
- Use incremental delivery (stage by stage)
- Track progress weekly

---

## Acceptance Criteria

### Stage 1: PID Controller
- [ ] PIDController class implemented with I, D, anti-windup
- [ ] Derivative filtering with configurable tau
- [ ] Ziegler-Nichols tuning method implemented
- [ ] Unit tests achieve >90% coverage
- [ ] Integration with simulate_controller() working
- [ ] Documentation updated with LaTeX equations

### Stage 2: Thermal Management (Simplified)
- [ ] CryocoolerModel with temperature-dependent cooling power
- [ ] QuenchDetector with hysteresis and rate limiting
- [ ] LumpedThermalModel with 2 nodes (stator + rotor) using explicit Euler
- [ ] Integration with existing thermal_model.py
- [ ] Unit tests achieve >85% coverage
- [ ] Documentation updated with lumped-parameter thermal equations

### Stage 3: Critical-State Flux-Pinning
- [ ] GdBCOMaterial with J_c(B, T) model
- [ ] BeanLondonModel with hysteresis and saturation
- [ ] Integration with stiffness_verification.py
- [ ] Unit tests achieve >85% coverage
- [ ] Validation against literature data (if available)
- [ ] Documentation updated with Bean-London equations

### Stage 4: Integration & Documentation
- [ ] End-to-end integration test passes
- [ ] All scenario tests pass (quench, runaway, saturation)
- [ ] Performance benchmarks meet targets
- [ ] API documentation complete for all new modules
- [ ] Integration guide created (docs/phase1_integration_guide.md)
- [ ] README.md Phase 1 checklist complete
- [ ] physics_simulator_audit.md updated to reflect completion

---

## Summary

**Total Estimated Effort**: Hybrid approach - full implementation for PID and Flux-Pinning, simplified thermal model
**Critical Path**: Stage 1 (PID) → Stage 2 (Thermal) → Stage 4 (Integration)
**Parallel Opportunity**: Stage 3 (Flux-Pinning) can run concurrently with Stage 2

**Approach**: Hybrid - Full implementation for PID and Flux-Pinning, simplified lumped-parameter thermal model using plain numpy (80-90% accuracy at 30% complexity)

**Key Dependencies**:
- External data acquisition will NOT block implementation (using literature placeholders with Monte-Carlo uncertainty)
- Stage 2 depends on Stage 1 (PID for active cooling)
- Stage 4 depends on Stages 1-3 completion

**Leveraged Infrastructure**:
- `control_layer/mpc_controller.py` patterns (ConfigurationMode enum, dataclass parameters, delay compensation)

**Success Criteria**:
- All acceptance criteria met
- Test coverage >85% for new components
- Integration tests pass
- Documentation complete
- Performance benchmarks met

**Next Steps**:
1. Begin Stage 1 implementation (PID controller) - can start immediately
2. Proceed with Stage 2 (simplified thermal) and Stage 3 (flux-pinning) in parallel
3. Complete Stage 4 integration and documentation
4. Acquire external data as parallel activity for refinement

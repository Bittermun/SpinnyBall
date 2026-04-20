# Phase 1 Integration Architecture

This document describes the integration architecture for Phase 1 components: PID controller, thermal management, and flux-pinning.

## Component Interaction Diagram

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

## Data Flow Between Components

### 1. PID → Thermal
Control force affects rotor heating (eddy currents, friction)
- PID output `u` → heat input `Q_rotor = f(u, velocity)`
- Eddy current heating: `Q_eddy ∝ (B × v)²`
- Friction heating: `Q_fric ∝ F_control × v`

### 2. Thermal → PID
Temperature affects PID behavior
- High temperature → reduce PID gain (prevent thermal runaway)
- Quench detected → PID emergency shutdown (output = 0)
- Gain scheduling: `gain_schedule = (90.0 - T) / 5.0` for T > 85K

### 3. Thermal → Flux-Pinning
Temperature affects stiffness
- `k_fp(T) = k_fp_0 * (1 - T/T_c)^n` (temperature-dependent stiffness)
- Near T_c → stiffness drops → reduced control authority

### 4. Flux-Pinning → Dynamics
Stiffness affects system dynamics
- `k_eff = k_fp(T) + k_structural` (effective stiffness)
- Stiffness affects natural frequency and control response

## Feedback Loop Specifications

### Quench → PID Feedback Loop

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

### Temperature → Stiffness Feedback Loop

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

### PID → Thermal Feedback Loop

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

## Integration Interface Specifications

### PID Controller Interface

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

### Thermal Model Interface

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
        
        # Step thermal model
        result = self.step(heat_sources)
        
        return result
```

### Flux-Pinning Interface

```python
class BeanLondonModel:
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

## Usage Example

```python
from sgms_anchor_control import PIDController, PIDParameters
from dynamics.lumped_thermal import LumpedThermalModel, LumpedThermalParams
from dynamics.quench_detector import QuenchDetector, QuenchThresholds
from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.bean_london import BeanLondonModel

# Initialize components
pid = PIDController(PIDParameters(kp=100.0, ki=10.0, kd=1.0), dt=0.01)
thermal = LumpedThermalModel(LumpedThermalParams(), dt=0.01)
detector = QuenchDetector(QuenchThresholds())
material = GdBCOMaterial(GdBCOProperties())
flux_model = BeanLondonModel(material, geometry)

# Simulation loop
for i in range(n_steps):
    # Get current state
    displacement = x[i]
    velocity = v[i]
    T_stator = thermal.T_stator
    
    # Check for quench
    status = detector.check_temperature(T_stator, dt=0.01)
    
    if status['emergency_shutdown']:
        pid.reset()
        control_force = 0.0
    else:
        # Compute PID control
        error = setpoint - displacement
        control_force = pid.update(error, temperature=T_stator)
    
    # Update thermal model
    heat_sources = {'stator': 0.0, 'rotor': compute_control_heating(control_force, velocity, B)}
    thermal_result = thermal.step(heat_sources)
    
    # Update flux-pinning stiffness
    k_fp = flux_model.get_stiffness(displacement, B, thermal_result['T_stator'])
    k_eff = k_fp + k_structural
    
    # Update dynamics
    a = -(c_damp * velocity + k_eff * displacement - control_force) / ms
    velocity += a * dt
    displacement += velocity * dt
```

## Implementation Status

- ✅ PID Controller: Implemented in `sgms_anchor_control.py`
- ✅ Thermal Management: Implemented in `dynamics/cryocooler_model.py`, `dynamics/quench_detector.py`, `dynamics/lumped_thermal.py`
- ✅ Flux-Pinning: Implemented in `dynamics/gdBCO_material.py`, `dynamics/bean_london_model.py`
- ✅ Integration: Integrated in `sgms_anchor_v1.py` and `dynamics/stiffness_verification.py`
- ✅ Testing: Comprehensive unit and integration tests in `tests/`

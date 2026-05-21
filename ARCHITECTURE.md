# SpinnyBall Simulation Architecture v2.0

## Overview

This is a comprehensive physics simulation framework for the SpinnyBall orbital mechanics system, featuring:

- **Uncertainty Quantification**: All physics outputs include error bounds
- **Structure-Preserving Integration**: Symplectic integrators for long-term stability
- **Multi-Timescale Coupling**: Operator splitting with macro-step scheduling
- **Corrected Physics**: Fixed equations for Halbach field, slingshot energy, atmosphere

## Quick Start

```python
from sim.scheduler import MacroScheduler, SchedulerConfig
from sim.domains import (
    MechanicsStreamDomain, AttitudeFluxGyroDomain,
    ThermalAnchorDomain, OrbitalEnvironmentDomain
)

# Create full system
config = SchedulerConfig(macro_dt=1.0, save_interval=5.0)
scheduler = MacroScheduler(config)

# Register domains
scheduler.register_domain("mechanics", MechanicsStreamDomain(n_balls=10))
scheduler.register_domain("attitude", AttitudeFluxGyroDomain())
scheduler.register_domain("thermal", ThermalAnchorDomain())
scheduler.register_domain("orbital", OrbitalEnvironmentDomain())

# Run simulation
scheduler.initialize()
scheduler.run(100.0)  # 100 seconds
```

## Architecture Layers

### Layer 1: Physics Core (`sim/`)

#### Uncertainty System (`sim/uncertainty.py`)
```python
from sim.uncertainty import UncertainQuantity, from_relative

# Create quantity with uncertainty
B_field = from_relative(1.5, 0.05)  # 1.5 T ± 5%

# Arithmetic propagates uncertainty
B_total = B_field * 2  # Uncertainty scales linearly

# Check validity regime
is_valid, violations = B_total.check_validity(r=3.0)
```

**Key Features:**
- First-order error propagation (RSS of independent uncertainties)
- Validity regime tracking (parameter ranges where uncertainty applies)
- Statistical + systematic error separation

#### Integrators (`sim/integrators.py`)

| Integrator | Use Case | Symplectic | Order |
|------------|----------|------------|-------|
| `VelocityVerlet` | Conservative mechanics | Yes | 2 |
| `StormerVerlet` | Hamiltonian systems | Yes | 2 |
| `SymplecticEuler` | Dissipative mechanics | Yes | 1 |
| `RK4` | General ODEs | No | 4 |
| `AdaptiveRK45` | Orbital propagation | No | 5 |

```python
from sim.integrators import select_integrator

# Auto-select based on system type
integrator = select_integrator(
    system_type="conservative_mechanics",  # or "dissipative", "orbital", "general_ode"
    timescale=1e-3,  # seconds
    accuracy_required="high"
)
```

#### Domain Base (`sim/domain_base.py`)

All physics domains inherit from `DomainAdapter`:

```python
class MyDomain(DomainAdapter):
    @property
    def characteristic_timescale(self) -> TimeScale:
        return TimeScale.MILLI  # 1 ms
    
    def advance(self, state, t_start, t_end, inputs) -> AdvanceResult:
        # Physics integration here
        pass
    
    def compute_output(self, state, t, fidelity) -> DomainOutput:
        # Return outputs with uncertainty
        pass
```

#### Scheduler (`sim/scheduler.py`)

Macro-step scheduler coordinates multiple domains:

```python
from sim.scheduler import SchedulerConfig, CoupledInput

config = SchedulerConfig(
    macro_dt=1.0,        # 1 second macro steps
    adaptive_macro=True,
    save_interval=10.0   # Save every 10 seconds
)

scheduler = MacroScheduler(config)

# Couple domains: attitude eddy heating -> thermal
scheduler.register_coupling(
    "thermal",
    CoupledInput(
        source_domain="attitude",
        source_quantity="eddy_heating_stator",
        averaging="time_average"
    )
)
```

### Layer 2: Domain Adapters (`sim/domains/`)

#### Mechanics Stream (`mechanics_stream.py`)

Simulates mass stream dynamics:
- Interball magnetic forces (dipole-dipole with multipole corrections)
- Hoop tension restoring forces
- Shepherd station interactions

**Integrator**: Velocity Verlet (symplectic)

**Outputs**: 
- Kinetic energy with uncertainty
- Ball positions/velocities
- Mean spacing

```python
mechanics = MechanicsStreamDomain(
    n_balls=10,
    nominal_radius=100.0,  # meters
    ball_mass=100.0        # kg
)
```

#### Attitude Flux-Gyro (`attitude_fluxgyro.py`)

Attitude dynamics with flux pinning:
- Gyroscopic coupling
- Bean-London flux pinning model
- Eddy current heating (output to thermal)

**Integrator**: Symplectic Euler (handles damping)

**Outputs**:
- Angular velocity with uncertainty
- Eddy heating power (coupled to thermal)
- Angular momentum

```python
attitude = AttitudeFluxGyroDomain(
    mass=100.0,
    inertia_tensor=np.diag([10, 10, 5]),
    use_bean_london=True
)
```

#### Thermal Anchor (`thermal_anchor.py`)

Multi-mode thermal domain modeling temperature-dependent heat loads:
- Passive radiative equilibrium: Simulates high-Curie passive magnetics (e.g. Samarium-Cobalt Halbach arrays) operating without active cooling by balancing eddy-current hypervelocity heating and solar absorption against blackbody space radiation.
- Active cryogenic cooling: Simulates high-stiffness superconducting bearings (GdBCO) with a Stirling/pulse-tube cryocooler featuring a temperature-dependent coefficient of performance (COP) and cooling power.
- Multi-node thermal coupling (stator/rotor) and radiative transfer.

**Integrator**: Adaptive RK45

**Outputs**:
- Node temperatures with error bounds
- Eddy heating loads and cryocooler power consumption/COP (if active cooling enabled)

```python
thermal = ThermalAnchorDomain(
    stator_mass=10.0,
    rotor_mass=5.0,
    enable_cryocooler=False,      # Set True for active GdBCO cooling
    cryocooler_power_77k=5.0      # W (if enabled)
)
```

#### Orbital Environment (`orbit_env.py`)

Orbital mechanics with perturbations:
- 2-body + J2 oblateness
- Atmospheric drag (Jacchia model with F10.7/Ap)
- Solar radiation pressure
- Eclipse transitions

**Integrator**: Adaptive RK45 via scipy

**Outputs**:
- Position/velocity (ECI)
- Altitude with uncertainty
- Orbital energy

```python
orbital = OrbitalEnvironmentDomain(
    mass=100.0,
    area_drag=1.0,   # m²
    cd=2.2,          # Drag coefficient
    cr=1.8           # Reflectivity
)
```

### Layer 3: Corrected Physics (`dynamics/*_v2.py`)

#### Halbach Array (`halbach_array_v2.py`)

**Fixes from v1:**
- Internal field uses correct demagnetization factor N=1/3
- Multipole corrections for external field at r/R < 5
- Temperature-dependent magnetization with Curie-Weiss model
- Manufacturing imperfection factor (10% reduction)

```python
from dynamics.halbach_array_v2 import HalbachSphereV2

halbach = HalbachSphereV2(
    radius=0.05,      # 5 cm
    M_0=1.0e6,        # A/m
    temperature=293.0,
    material="NdFeB"
)

# Internal field with uncertainty
B_int = halbach.internal_field()
print(f"B = {B_int.value:.3f} ± {B_int.total_error:.3f} T")

# External field with multipole correction
B_ext = halbach.external_field([0.2, 0, 0], include_multipole=True)

# Check validity regime
regime = halbach.regime_info([0.2, 0, 0])
print(f"Regime: {regime['regime']}, Error: {regime['error_estimate']}")
```

#### Gravity Slingshot (`gravity_slingshot_v2.py`)

**Critical Fix:** Heliocentric energy computation

V1 computed `ΔE = ½(|v_inf_out|² - |v_inf_in|²)` which is **zero** (energy conserved in planet frame).

V2 correctly computes:
```
v_helio = v_planet + v_inf
ΔE = ½(|v_helio_out|² - |v_helio_in|²) = v_planet · (v_inf_out - v_inf_in)
```

```python
from dynamics.gravity_slingshot_v2 import (
    GravityBodyV2, GravitySlingshotCalculatorV2
)

earth = GravityBodyV2.earth()
calc = GravitySlingshotCalculatorV2(earth)

result = calc.compute_slingshot(
    v_inf_in=[0, -5000, 0],  # 5 km/s against Earth motion
    periapsis_radius=earth.radius + 500e3  # 500 km altitude
)

# This is now NON-ZERO (was ~0 in v1)
print(f"Energy gain: {result.energy_gain_heliocentric.value/1e6:.1f} MJ/kg")
```

#### Atmosphere (`atmosphere_v2.py`)

**Fixes from v1:**
- Replaced piecewise exponential (±40-60% error) with Jacchia model (±15-30%)
- Solar activity effects (F10.7 flux)
- Geomagnetic storm effects (Ap index)
- Proper uncertainty bounds

```python
from dynamics.atmosphere_v2 import (
    AtmosphereCalculatorV2, AtmosphereModel, SpaceWeatherConditions
)

# Jacchia model with solar conditions
calc = AtmosphereCalculatorV2(AtmosphereModel.EXPONENTIAL_JACCHIA)
calc.set_space_weather(SpaceWeatherConditions(
    f107=150,    # Solar flux
    f107a=150,   # 81-day average
    ap=15        # Geomagnetic index
))

# Get density with uncertainty
result = calc.compute_density(altitude_km=400)
print(f"ρ = {result.density:.3e} ± {result.total_uncertainty:.3e} kg/m³")

# Drag acceleration
a_drag = calc.compute_drag_acceleration(
    position_eci=[6871, 0, 0],  # km
    velocity_eci=[0, 7.5, 0],   # km/s
    mass=100.0,
    area=1.0
)
```

## Testing

Run integration tests:
```bash
python sim/test_integration.py
```

Tests validate:
1. Uncertainty arithmetic
2. Integrator energy conservation
3. Corrected Halbach field
4. Gravity slingshot heliocentric energy
5. Atmosphere model comparison
6. Domain adapters & scheduler
7. Cross-domain coupling
8. Full system integration

## Migration from v1

### Key Changes

| v1 | v2 |
|----|-----|
| `HalbachSphere` | `HalbachSphereV2` (corrected internal field) |
| `GravitySlingshot` | `GravitySlingshotCalculatorV2` (heliocentric energy) |
| Piecewise atmosphere | `AtmosphereCalculatorV2` (Jacchia model) |
| Explicit Euler | `VelocityVerlet` / `SymplecticEuler` / `RK45` |
| Raw floats | `UncertainQuantity` with error bounds |
| Ad-hoc coupling | `MacroScheduler` with operator splitting |

### Migration Example

```python
# v1 code
from dynamics.halbach_array import HalbachSphere
from dynamics.orbital_perturbations import get_orbital_perturbation_force

halbach = HalbachSphere(radius=0.05, M_0=1e6)
B = halbach.internal_field()  # Wrong: overestimated by 20-50%

# v2 code
from dynamics.halbach_array_v2 import HalbachSphereV2
from sim.uncertainty import from_relative

halbach = HalbachSphereV2(radius=0.05, M_0=1e6)
B = halbach.internal_field()  # Correct with demagnetization
print(f"B = {B.value:.3f} ± {100*B.relative_error:.1f}%")
```

## Performance

Typical performance on modern CPU:

| Domain | Fidelity | Time per Macro-Step |
|--------|----------|---------------------|
| Mechanics (10 balls) | APPROX | ~5 ms |
| Mechanics (10 balls) | PRECISE | ~50 ms |
| Attitude | APPROX | ~1 ms |
| Thermal | APPROX | ~0.1 ms |
| Orbital | APPROX | ~2 ms |

## References

### Physics Corrections

1. **Halbach Internal Field**: Griffiths, *Introduction to Electrodynamics*, Eq. 6.16
2. **Gravity Slingshot**: Bate, Mueller, White, *Fundamentals of Astrodynamics*, Eq. 7.16
3. **Atmosphere Model**: Jacchia, L.G. "New Static Models of the Thermosphere"
4. **Symplectic Integration**: Hairer, Lubich, Wanner, *Geometric Numerical Integration*

### Uncertainty Quantification

- First-order Taylor series propagation
- ISO Guide to the Expression of Uncertainty in Measurement (GUM)

## License

MIT License - See LICENSE file

## Contact

For questions or contributions, see the project repository.

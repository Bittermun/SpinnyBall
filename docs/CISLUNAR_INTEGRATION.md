# Cislunar Physics Integration: CR3BP & SPICE

## Overview

This directory contains high-fidelity cislunar dynamics modules for the SGMS simulation:

- **`dynamics/cislunar.py`** — Circular Restricted 3-Body Problem (CR3BP) propagator
- **`third_party/spice.py`** — SPICE ephemeris wrapper for high-precision body state queries
- **`examples/demo_cislunar_propagation.py`** — Demonstration script for 10-day cislunar propagation
- **`tests/test_cr3bp_spice.py`** — Comprehensive unit tests

## Features

### CR3BP Propagator (`dynamics/cislunar.py`)

The `CR3BPPropagator` class provides:

1. **Inertial Frame Integration**
   - Earth-Moon system dynamics (barycentric approach)
   - Adaptive RK45 integration with configurable tolerance
   - Support for optional Solar Radiation Pressure (SRP)

2. **Lagrange Point Computation**
   - L1-L5 locations in rotating frame
   - Used for trajectory design and station placement

3. **Flexible Configuration**
   ```python
   config = CR3BPConfig(
       mu=0.01215,          # Earth-Moon mass ratio
       rotating_frame=False, # Integrate in inertial frame
       include_srp=False,    # Optional SRP perturbation
       use_spice=False,      # Optional SPICE ephemeris updates
       rtol=1e-9,            # Relative tolerance
       atol=1e-12            # Absolute tolerance
   )
   propagator = CR3BPPropagator(config)
   ```

### SPICE Ephemeris Wrapper (`third_party/spice.py`)

The `SPICEWrapper` class provides:

1. **Kernel Management**
   - Automatic kernel loading from configurable directory
   - Support for standard NAIF kernels (de430.bsp, pck00010.tpc, etc.)
   - Graceful fallback if kernels unavailable

2. **Body State Queries**
   ```python
   spw = SPICEWrapper(kernel_dir=Path('./kernels'))
   moon_state = spw.get_body_state('MOON', time_jd=2460000.0)
   print(f"Moon position: {moon_state.position} km")
   print(f"Moon velocity: {moon_state.velocity} km/s")
   ```

3. **Convenience Methods**
   - `get_moon_position()`, `get_moon_velocity()`
   - `get_sun_position()`, `get_sun_direction()`
   - Time conversion (Julian Date ↔ UTC)

## Usage Examples

### Basic LEO Propagation

```python
from dynamics.cislunar import CR3BPPropagator, CR3BPConfig
import numpy as np

# Configure propagator
config = CR3BPConfig(rotating_frame=False)
prop = CR3BPPropagator(config)

# LEO circular orbit (400 km altitude)
R_earth = 6371.0  # km
r_leo = R_earth + 400.0
v_leo = np.sqrt(398600.4418 / r_leo)  # km/s

state0 = np.array([r_leo, 0.0, 0.0, 0.0, v_leo, 0.0])

# Propagate 1 day
t_eval = np.linspace(0, 86400, 1000)
sol = prop.propagate(state0, t_eval)

# Extract position at final time
r_final = sol.get_position(t_eval[-1])
print(f"Final position: {r_final} km")
```

### Cislunar Propagation (10 Days)

Run the included demonstration:

```bash
cd <repo_root>
python examples/demo_cislunar_propagation.py
```

Output:
- `results/cislunar_demo/trajectory.npz` — Full trajectory (positions, velocities, orbital radius/velocity)
- `results/cislunar_demo/summary.txt` — Summary statistics

### Lagrange Point Access

```python
prop = CR3BPPropagator()

# Get L1 location (Earth-Moon system)
l1 = prop.lagrange_point(1)
print(f"L1 position (normalized): {l1}")

# Get all Lagrange points
for i in range(1, 6):
    l_i = prop.lagrange_point(i)
    print(f"L{i}: {l_i}")
```

## Installation & Dependencies

### Optional: SPICE Integration

To use the SPICE wrapper for high-precision ephemerides:

1. **Install spiceypy**
   ```bash
   pip install spiceypy
   ```

2. **Download NAIF Kernels**
   - JPL NAIF GenericKernels: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
   - Required files:
     - `de430.bsp` (planetary ephemerides, ~100 MB)
     - `pck00010.tpc` (orientation constants)
     - `naif0012.tls` (leap seconds)
   
   - Place in `./kernels/` directory or specify via `kernel_dir` parameter

### No SPICE Required

The CR3BP propagator works standalone without SPICE. The wrapper will gracefully handle missing kernels:

```python
# This works even without SPICE
prop = CR3BPPropagator()
sol = prop.propagate(state0, t_eval)
```

## Physics Model Details

### CR3BP Equations of Motion (Inertial Frame)

The propagator integrates the full 3-body gravitational accelerations:

$$\ddot{\mathbf{r}} = -\frac{\mu_E}{r_E^3} \mathbf{r}_E - \frac{\mu_M}{r_M^3} \mathbf{r}_M$$

where:
- $\mathbf{r}$ = spacecraft position (ECI frame, km)
- $\mathbf{r}_E$ = position relative to Earth
- $\mathbf{r}_M$ = position relative to Moon
- $\mu_E = 398,600.4418$ km³/s² (Earth gravitational parameter)
- $\mu_M = 4,902.8005$ km³/s² (Moon gravitational parameter)

### Earth-Moon System Parameters

Standard values from NAIF ephemerides:
- **Mass parameter (μ)**: 0.01215 (dimensionless, Earth-Moon system)
- **Mean Earth-Moon distance**: 384,400 km
- **Lunar orbital period**: ≈ 27.3 days

### Integration Method

- **Algorithm**: Adaptive RK45 (Dormand-Prince method)
- **Default tolerances**: `rtol=1e-9`, `atol=1e-12`
- **Maximum timestep**: 60 seconds (configurable)

## Validation & Accuracy

### Expected Accuracy

For LEO orbits (400-2000 km altitude):
- **Position error vs. pure Earth gravity**: < 10 km/day (Moon's perturbation is small at LEO)
- **Short-term propagation (< 7 days)**: Position accuracy typically < 1 km with high tolerance settings

For cislunar orbits (approaching Moon):
- Full CR3BP dynamics required
- Validation against SPICE ephemeris (when enabled) provides ground truth

### Validation Tests

Run tests to verify implementation:

```bash
# All CR3BP/SPICE tests
python -m pytest tests/test_cr3bp_spice.py -v

# Non-slow tests only (quick)
python -m pytest tests/test_cr3bp_spice.py -v -m "not slow"

# Specific test class
python -m pytest tests/test_cr3bp_spice.py::TestCR3BPPropagator -v
```

## Integration with MultiBodyStream

The CR3BP propagator is designed to integrate with the packet stream dynamics:

```python
from dynamics.multi_body import MultiBodyStream
from dynamics.cislunar import CR3BPPropagator

# Create stream with CR3BP propagator
prop = CR3BPPropagator()
stream = MultiBodyStream(
    orbital_propagator=prop,
    # ... other parameters
)

# Propagate packet stream
stream.integrate(t_eval, ...)
```

See [`dynamics/multi_body.py`](../dynamics/multi_body.py) for integration examples.

## Next Steps (Planned)

1. **Lunar Mascon Gravity** (`dynamics/mascons.py`)
   - High-fidelity lunar gravity from GRAIL mission data
   - Spherical harmonic or mascon interpolant model

2. **SPICE Integration Extension**
   - Time-varying Moon position updates during long propagations
   - SRP with SPICE Sun direction

3. **Guided-Beam Electromagnetic Deflection and Shepherding Control**
   - Closed-loop electromagnetic packet stabilization and spacing maintenance
   - Δv budget tracking across cislunar ballistic corridors

4. **Monte Carlo Extension**
   - Cislunar perturbation sampling
   - Coherence lifetime and collision probability metrics

## References

- **Szebehely, V.** "Theory of Orbits: The Restricted Problem of Three Bodies" (1967)
- **Howell, K. C.** "Families of Orbits in the Vicinity of the Collinear Libration Points" (1984)
- **NAIF SPICE Toolkit**: https://naif.jpl.nasa.gov/naif/
- **Konopliv et al.** "The JPL Lunar Gravity Field to Spherical Harmonic Degree 660 from the GRAIL Mission" (2014)

## Authors & Acknowledgments

- CR3BP formulation follows standard references (Szebehely, NASA JPL)
- SPICE wrapper adapted from official NAIF spiceypy examples
- Integration methodology: scipy.integrate.solve_ivp (RK45)

## License

See [`LICENSE`](../LICENSE) in the repository root.

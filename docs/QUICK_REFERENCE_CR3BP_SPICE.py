"""Quick Reference: CR3BP and SPICE Integration

Fast lookup for common tasks and APIs.
"""

# =============================================================================
# CR3BP PROPAGATOR
# =============================================================================

# IMPORT
from dynamics.cislunar import CR3BPPropagator, CR3BPConfig
import numpy as np

# CONFIGURE
config = CR3BPConfig(
    mu=0.01215,           # Earth-Moon mass ratio
    rotating_frame=False,  # Use inertial (ECI) frame
    include_srp=False,     # No SRP perturbation
    use_spice=False,       # Don't require SPICE
    rtol=1e-9,             # Relative tolerance
    atol=1e-12             # Absolute tolerance
)

# CREATE PROPAGATOR
prop = CR3BPPropagator(config)

# INITIAL STATE: [x, y, z, vx, vy, vz] in km, km/s
# LEO example: 400 km circular orbit
r_earth = 6371.0
altitude = 400.0
r_orbit = r_earth + altitude
v_orbit = np.sqrt(398600.4418 / r_orbit)  # Orbital velocity

state0 = np.array([r_orbit, 0.0, 0.0, 0.0, v_orbit, 0.0])

# PROPAGATE
t_eval = np.linspace(0, 86400*10, 1000)  # 10 days, 1000 steps
sol = prop.propagate(state0, t_eval)

# ACCESS RESULTS
positions = sol.y[0:3, :]      # Shape: (3, N_time)
velocities = sol.y[3:6, :]     # Shape: (3, N_time)
times = sol.t                   # Time points

# QUERY AT SPECIFIC TIME
t_query = 43200.0  # 12 hours
pos_at_t = sol.get_position(t_query)     # Shape: (3,)
vel_at_t = sol.get_velocity(t_query)     # Shape: (3,)

# LAGRANGE POINTS
l1 = prop.lagrange_point(1)  # L1
l2 = prop.lagrange_point(2)  # L2
l3 = prop.lagrange_point(3)  # L3
l4 = prop.lagrange_point(4)  # L4 (triangular)
l5 = prop.lagrange_point(5)  # L5 (triangular)

# EARTH-MOON PARAMETERS
mu_earth = prop.MU_EARTH              # 398600.4418 km³/s²
mu_moon = prop.MU_MOON                # 4902.8005 km³/s²
r_em = prop.EARTH_MOON_DISTANCE       # 384400 km


# =============================================================================
# SPICE EPHEMERIS WRAPPER (Optional)
# =============================================================================

# IMPORT
from third_party.spice import SPICEWrapper
from pathlib import Path

# CREATE WRAPPER (graceful fallback if spiceypy not installed)
try:
    spw = SPICEWrapper(
        kernel_dir=Path('./kernels'),  # Directory with SPICE kernels
        auto_load_kernels=True,
        verbose=True
    )
except ImportError as e:
    print("spiceypy not installed; SPICE functionality disabled")
    spw = None

# QUERY BODY STATE
if spw:
    # Get Moon state at JD 2460000.0
    moon_state = spw.get_body_state('MOON', time_jd=2460000.0)
    
    moon_pos = moon_state.position      # [x, y, z] in km
    moon_vel = moon_state.velocity      # [vx, vy, vz] in km/s
    
    # Convenience methods
    moon_pos = spw.get_moon_position(time_jd=2460000.0)
    moon_vel = spw.get_moon_velocity(time_jd=2460000.0)
    sun_pos = spw.get_sun_position(time_jd=2460000.0)

# TIME CONVERSION
if spw:
    jd = 2460000.0
    utc_string = SPICEWrapper._jd_to_utc_string(jd)
    print(f"JD {jd} -> UTC {utc_string}")


# =============================================================================
# DEMO: 10-DAY CISLUNAR PROPAGATION
# =============================================================================

import numpy as np
from dynamics.cislunar import CR3BPPropagator, CR3BPConfig

# Run the demo
from examples.demo_cislunar_propagation import main

if __name__ == "__main__":
    exit_code = main()
    print(f"Demo exit code: {exit_code}")

# Or run via command line:
# $ python examples/demo_cislunar_propagation.py


# =============================================================================
# COMMON PATTERNS
# =============================================================================

# Pattern 1: High-precision short propagation (< 7 days)
config = CR3BPConfig(rtol=1e-10, atol=1e-13)
prop = CR3BPPropagator(config)
sol = prop.propagate(state0, np.linspace(0, 86400*7, 2000))

# Pattern 2: Long propagation (> 30 days, trade speed for speed)
config = CR3BPConfig(rtol=1e-8, atol=1e-11)
prop = CR3BPPropagator(config)
sol = prop.propagate(state0, np.linspace(0, 86400*90, 1000))

# Pattern 3: Extract orbital elements
from scipy.integrate import odeint
r_traj = np.linalg.norm(sol.y[0:3, :], axis=0)  # Distance from Earth
v_traj = np.linalg.norm(sol.y[3:6, :], axis=0)  # Speed

# Pattern 4: Check Moon proximity (for collision warning)
r_moon = prop.EARTH_MOON_DISTANCE
distances_to_moon = sol.get_distance_from_moon(sol.t)
min_distance = np.min(distances_to_moon)
if min_distance < 1000:  # Within 1000 km of Moon
    print(f"Warning: Closest approach to Moon: {min_distance:.1f} km")

# Pattern 5: Integrate with existing orbit_env.py for 2-body baseline
from sim.domains.orbit_env import OrbitalEnvironmentDomain

orbit_env = OrbitalEnvironmentDomain(mass=100.0)
# orbit_env propagates with Earth gravity + J2 + SRP + drag
# Compare with CR3BP (Earth + Moon gravity) for validation


# =============================================================================
# PHYSICS CONSTANTS
# =============================================================================

# Gravitational parameters (km³/s²)
MU_EARTH = 398600.4418
MU_MOON = 4902.8005
MU_SUN = 1.32712440018e11

# Orbital radii
R_EARTH = 6371.0  # km
R_MOON = 1737.0   # km

# Earth-Moon system
EARTH_MOON_DISTANCE = 384400.0  # km (mean)
EARTH_MOON_MASS_RATIO = 0.01215  # μ for CR3BP

# Typical orbital velocities
V_LEO_400 = np.sqrt(398600.4418 / (6371 + 400))  # ~7.67 km/s
V_EARTH_MOON_TRANSFER = 10.9  # km/s (trans-lunar injection)


# =============================================================================
# TESTING & VALIDATION
# =============================================================================

# Run unit tests
# $ python -m pytest tests/test_cr3bp_spice.py -v

# Run without slow tests
# $ python -m pytest tests/test_cr3bp_spice.py -v -m "not slow"

# Run specific test
# $ python -m pytest tests/test_cr3bp_spice.py::TestCR3BPPropagator::test_propagation_short_leo -v


# =============================================================================
# DOCUMENTATION REFERENCES
# =============================================================================

# Full documentation: docs/CISLUNAR_INTEGRATION.md
# Implementation report: docs/PHASE2_IMPLEMENTATION_REPORT.md
# Completion summary: docs/PHASE2_COMPLETION_SUMMARY.md
# Demo script: examples/demo_cislunar_propagation.py
# Unit tests: tests/test_cr3bp_spice.py

# Physics references:
# - Szebehely (1967): "Theory of Orbits: The Restricted Problem of Three Bodies"
# - Howell (1984): "Families of Orbits in the Vicinity of the Collinear Libration Points"
# - NAIF SPICE: https://naif.jpl.nasa.gov/


# =============================================================================
# ERROR HANDLING
# =============================================================================

# SPICE not available
try:
    spw = SPICEWrapper()
except ImportError:
    print("spiceypy required for SPICE functionality")
    # Fall back to fixed ephemeris or other method

# Propagation failure
try:
    sol = prop.propagate(state0, t_eval)
    if sol.status != 0:
        print(f"Integration failed: {sol.message}")
except Exception as e:
    print(f"Propagation error: {e}")

# Invalid state
try:
    bad_state = np.array([1.0, 2.0, 3.0])  # Wrong size
    sol = prop.propagate(bad_state, t_eval)
except ValueError as e:
    print(f"Invalid state: {e}")


# =============================================================================
# PERFORMANCE TIPS
# =============================================================================

# Tip 1: For many short propagations, reuse propagator object
prop = CR3BPPropagator()
for i in range(100):
    state = initial_states[i]
    sol = prop.propagate(state, t_eval)  # Reuses integrator

# Tip 2: Use lower tolerance for speed (trade accuracy)
config = CR3BPConfig(rtol=1e-7, atol=1e-10)  # Faster
prop = CR3BPPropagator(config)

# Tip 3: Reduce time steps if not using dense output
t_eval = np.linspace(0, 86400, 100)  # 100 steps instead of 1000

# Tip 4: For very long propagations, consider splitting into segments
t_segments = [
    np.linspace(0, 86400*30, 500),
    np.linspace(86400*30, 86400*60, 500),
]
state = state0
for t_eval_seg in t_segments:
    sol = prop.propagate(state, t_eval_seg)
    state = sol.y[:, -1]  # Use final state for next segment

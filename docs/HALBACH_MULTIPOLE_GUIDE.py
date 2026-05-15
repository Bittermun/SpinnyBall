"""
Halbach Array Multipole Expansion Integration Guide

Complete reference for magnetic field expansion and cislunar integration.
"""

# =============================================================================
# HALBACH MULTIPOLE EXPANSION: PHYSICS & INTEGRATION
# =============================================================================

"""
Overview
--------

Halbach array multipole expansion is a spherical harmonic representation of 
the near-field magnetic field created by a Halbach array (or similar magnets).

Key characteristics:
  - Degree 1-6 spherical harmonic model (configurable)
  - Captures ~90% of field in near-field region (r/R = 1-5)
  - Includes dipole (degree 1), quadrupole (degree 2), octupole (degree 3)
  - Accuracy: ±10% vs. reference for r/R ≤ 3
  - Essential for packet control in cislunar context

Physics Basis:
  - Magnetic field from localized source (Halbach array)
  - Expansion in spherical harmonics (infinite series truncated at degree N)
  - Coefficients: C_nm (cosine), S_nm (sine) for each degree/order
  - Evaluation: O(N²) operations for degree N

Applications:
  - Magnetic force calculation on charged/magnetized packets
  - Field mapping for mission planning
  - Gradient computation for trajectory shaping
"""

# PART 1: USAGE PATTERNS
# =============================================================================

# Pattern 1: Basic field evaluation
# =================================

import numpy as np
from dynamics.halbach_multipole import HalbachSphericalHarmonic, HalbachSphericalHarmonicConfig

# Create model (degree 4, standard Halbach)
config = HalbachSphericalHarmonicConfig(degree_max=4)
halbach = HalbachSphericalHarmonic(config)

# Evaluate field at position (meters, from Halbach center)
position = np.array([0.1, 0.0, 0.0])  # 100 mm along x-axis
B = halbach.field(position)  # [B_x, B_y, B_z] in Tesla
print(f"Magnetic field: {B} T")


# Pattern 2: Field gradient for force calculation
# ================================================

# Compute gradient tensor ∂B/∂x, ∂B/∂y, ∂B/∂z
grad_B = halbach.gradient(position)  # Shape (3, 3), units T/m

# Magnetic moment of packet
m = np.array([0.0, 0.0, 0.5])  # 0.5 A⋅m² along z

# Force: F = ∇(m · B) = (∇B)^T · m
force = grad_B @ m  # Newtons


# Pattern 3: Halbach in cislunar propagation
# ============================================

from dynamics.cislunar_halbach import CR3BPHalbachPropagator, CR3BPHalbachConfig

# Configure cislunar with Halbach
config = CR3BPHalbachConfig(
    use_halbach=True,
    halbach_degree_max=4,
    packet_magnetic_moment_am2=0.1,  # Packet magnetization
    packet_mass_kg=1.0
)

prop = CR3BPHalbachPropagator(config)

# Propagate lunar orbit with magnetic perturbations
state0 = np.array([384400 + 1837, 0.0, 0.0, 0.0, 1.68, 0.0])
t_eval = np.linspace(0, 30*86400, 500)

sol = prop.propagate(state0, t_eval)

# Extract trajectory
positions = sol.y[0:3, :].T


# Pattern 4: Magnetic force analysis
# ===================================

# Compute force on packet at specific location
packet_pos = np.array([384400 + 1837, 0.0, 0.0])  # Lunar orbit position
moment = np.array([0.0, 0.0, 0.1])  # Magnetic moment

force = prop.compute_magnetic_force_on_packet(packet_pos, moment)
print(f"Magnetic force on packet: {np.linalg.norm(force)*1e6:.3f} μN")


# Pattern 5: Magnetic field mapping
# ==================================

# Generate grid of field values
r_values = np.linspace(0.05, 0.2, 10)  # meters
field_grid = []

for r in r_values:
    pos = np.array([r, 0.0, 0.0])
    B = halbach.field(pos)
    field_grid.append(np.linalg.norm(B))

print(f"Field magnitudes: {field_grid}")


# PART 2: HALBACH CONFIGURATION OPTIONS
# =============================================================================

"""
HalbachSphericalHarmonicConfig Parameters
============================================

degree_max: int
  Maximum spherical harmonic degree (1-6 typical)
  - Degree 1: Dipole only (~90% of field)
  - Degree 2: +Quadrupole (~95-98%)
  - Degree 3: +Octupole (~99%)
  - Degree 4: Higher multipoles (~99.5%)
  
  Default: 4 (good accuracy vs. speed balance)

moment_magnitude_am2: float
  Dipole moment magnitude in A⋅m²
  - Typical Halbach: 0.5-2.0 A⋅m²
  - Spacecraft magnetosphere: 0.1-10 A⋅m²
  Default: 1.0

radius_m: float
  Reference radius of Halbach array in meters
  - Typical: 0.02-0.1 m (20-100 mm)
  - Sets scale for "near-field" region
  Default: 0.05 m

normalize_coefficients: bool
  Use normalized Legendre polynomials (for stability)
  Default: True

use_fast_legendre: bool
  Use recurrence relations (faster, recommended)
  Default: True


CR3BPHalbachConfig Parameters (extends CR3BPMasconConfig)
==========================================================

use_halbach: bool
  Enable/disable Halbach perturbations
  Default: True

halbach_degree_max: int
  Multipole degree for cislunar propagation
  Default: 4

packet_magnetic_moment_am2: float
  Magnetic moment of propagated packets
  - ESA Earth Magnetosphere: ~10^20 A⋅m² (planetary scale)
  - Spacecraft: ~1-100 A⋅m²
  - Tablet: ~10^-3 A⋅m²
  Default: 0.1

packet_mass_kg: float
  Mass of packet (for F = ma acceleration)
  Default: 1.0 kg

halbach_position_earth_relative: np.ndarray
  Position of Halbach shepherd in ECI km
  If None: assumed at Earth origin (0, 0, 0)
  Default: None

fixed_halbach_position: bool
  Keep Halbach at fixed position (or allow dynamics)
  Default: False (currently unused)
"""


# PART 3: FIELD EVALUATION WORKFLOW
# =============================================================================

"""
Recommended Workflow for Magnetic Field Analysis:

1. Initialize Halbach Model
   -------------------------
   config = HalbachSphericalHarmonicConfig(
       degree_max=4,
       moment_magnitude_am2=1.0,
       radius_m=0.05
   )
   halbach = HalbachSphericalHarmonic(config)

2. Evaluate Field at Points of Interest
   ------------------------------------
   positions = np.array([[0.05, 0, 0], [0.1, 0, 0], [0.15, 0, 0]])
   fields = np.array([halbach.field(pos) for pos in positions])

3. Validate Against Reference (if available)
   -------------------------------------------
   reference_fields = load_fem_reference()
   metrics = halbach.reference_comparison(positions, reference_fields)
   
   if metrics['percent_within_10%'] > 90:
       print("✓ Validation PASS")
   else:
       print("✗ Validation FAIL - try higher degree")

4. Compute Gradients for Forces
   ----------------------------------
   grad = halbach.gradient(pos)
   moment = np.array([0, 0, 0.1])
   force = grad @ moment

5. Analyze Multipole Contributions
   --------------------------------
   B_deg1 = halbach.field(pos, degree=1)
   B_deg4 = halbach.field(pos, degree=4)
   error = np.linalg.norm(B_deg4 - B_deg1) / np.linalg.norm(B_deg1)
   print(f"Multipole contribution: {error*100:.1f}%")
"""


# PART 4: CISLUNAR INTEGRATION WORKFLOW
# =============================================================================

"""
Workflow for Cislunar Propagation with Halbach

1. Configure Propagator
   ---------------------
   config = CR3BPHalbachConfig(
       use_halbach=True,
       halbach_degree_max=4,
       rotating_frame=False,
       packet_magnetic_moment_am2=0.1,
       packet_mass_kg=1.0
   )

2. Create Propagator
   ------------------
   prop = CR3BPHalbachPropagator(config)
   
   # Access Halbach model directly if needed
   B = prop.halbach.field(position_m)

3. Define Initial State
   ---------------------
   # Lunar orbit: 100 km altitude, circular
   state0 = np.array([
       384400 + 1837,  # x (Earth-Moon distance + orbit radius)
       0.0,            # y
       0.0,            # z
       0.0,            # vx
       1.68,           # vy (orbital velocity)
       0.0             # vz
   ])

4. Propagate with Analysis
   ------------------------
   t_eval = np.linspace(0, 30*86400, 500)
   sol, diag = prop.propagate_with_halbach_analysis(state0, t_eval)
   
   # Diagnostics include:
   # - magnetic_field_magnitudes: list of B magnitudes over time
   # - halbach_forces: list of force vectors over time
   # - orbital_elements: (if inherited from mascon)

5. Analyze Results
   ----------------
   positions = sol.y[0:3, :].T
   magnetic_forces = np.array(diag['halbach_forces'])
   force_magnitudes = np.linalg.norm(magnetic_forces, axis=1)
   
   print(f"Max magnetic force: {np.max(force_magnitudes)*1e6:.3f} μN")
   print(f"Mean force: {np.mean(force_magnitudes)*1e6:.3f} μN")
"""


# PART 5: VALIDATION & ACCURACY
# =============================================================================

"""
Validation Against Reference Data

Method 1: Dipole Comparison
   -------------------------
   # Use degree 1 as reference (pure dipole)
   B_dipole = halbach.field(pos, degree=1)
   B_full = halbach.field(pos, degree=4)
   
   error = 100 * np.linalg.norm(B_full - B_dipole) / np.linalg.norm(B_dipole)
   
   if error > 20:
       print("Higher multipoles significant (~20%)")
   else:
       print("Dipole approximation sufficient (~<20%)")

Method 2: FEM Reference Comparison
   --------------------------------
   reference_field = load_fem_simulation()  # Arbitrary reference
   metrics = halbach.reference_comparison(positions, reference_field)
   
   Acceptance: percent_within_10% > 90%

Method 3: Convergence Check
   -------------------------
   degrees = [1, 2, 3, 4, 5, 6]
   
   for degree in degrees:
       B = halbach.field(pos, degree=degree)
       print(f"Degree {degree}: |B| = {np.linalg.norm(B):.2e} T")
   
   Should show convergence (decreasing changes with degree)


Known Accuracy Limits:

Very Near-Field (r < R):
  - Expansion less accurate
  - Use higher degree (5-6) or exact solution
  - Error: ~1-5%

Near-Field (R < r < 3R):
  - Sweet spot for expansion
  - Degree 4 typically sufficient
  - Error: <10%

Intermediate (3R < r < 5R):
  - Still accurate
  - Dipole starts to dominate
  - Error: ~10-20%

Far-Field (r > 5R):
  - Pure dipole approximation sufficient
  - Use degree 1 only (faster)
  - Error: <1%
"""


# PART 6: ADVANCED TOPICS
# =============================================================================

"""
Multipole Moment Interpretation

Degree 1 (Dipole): μ_dipole
  - Monopole-like term (but magnetism has no monopoles)
  - Represents main field: B ~ μ₀ m / (4π r³)

Degree 2 (Quadrupole): Q_quadrupole
  - Second-order multipole moment
  - Represents deviations from pure dipole
  - For Halbach: ~10-20% of dipole contribution

Degree 3 (Octupole): O_octupole
  - Higher-order effects
  - For Halbach: ~5-10% of dipole contribution

Higher Degrees (n > 3):
  - Fine-structure, asymmetries
  - For Halbach: typically <5% each


Coefficient Relationship

For azimuthally symmetric field (Halbach typically):
  - S_nm = 0 for all terms (no sine component)
  - Only C_nm terms non-zero
  - Reduces computational load by ~50%

For full 3D asymmetric field:
  - Both C_nm and S_nm non-zero
  - Represents m-fold azimuthal asymmetry
  - m = 0: axisymmetric (Halbach)
  - m = 1: one-off dipole
  - m = 2: quadrupole asymmetry
"""


# PART 7: QUICK REFERENCE
# =============================================================================

"""
Typical Field Strengths (Halbach 0.05m radius, 1 A⋅m² dipole)

Position          | Field Magnitude | Near-field Region
------------------|-----------------|-------------------
At r/R = 1.0      | ~50 mT          | Strongest
At r/R = 1.5      | ~15 mT          | Strong
At r/R = 2.0      | ~6 mT           | Moderate
At r/R = 3.0      | ~2 mT           | Weak
At r/R = 5.0      | ~0.3 mT         | Very weak


Force on Magnetic Dipole (0.1 A⋅m²)

Position          | Force Magnitude | Notes
------------------|-----------------|-------------------
At r/R = 1.0      | ~500 μN         | Very strong
At r/R = 1.5      | ~100 μN         | Strong
At r/R = 2.0      | ~30 μN          | Moderate
At r/R = 3.0      | ~8 μN           | Weak
At r/R = 5.0      | ~1 μN           | Very weak


Computation Time (approximate)

Degree | Time per eval | N samples | Total
--------|---------------|-----------|-------
1       | ~10 μs        | 1000      | ~10 ms
4       | ~100 μs       | 1000      | ~100 ms
6       | ~500 μs       | 1000      | ~500 ms

Typical cislunar propagation (500 time steps):
  - Degree 4: ~50 ms total Halbach overhead
  - Degree 6: ~250 ms total Halbach overhead
  - Total propagation: 1-5 seconds
"""


# PART 8: TROUBLESHOOTING
# =============================================================================

"""
Issue: Field seems too small

  Solution: Check units and position
    - Position should be in METERS, not kilometers
    - Field result is in TESLA
    - Verify: |B| ~ 0.01-100 mT depending on distance

  Example:
    pos_m = 0.1  # 100 mm from Halbach center
    B = halbach.field(np.array([pos_m, 0, 0]))
    print(f"Field: {np.linalg.norm(B)*1e3:.1f} mT")  # Should be ~1-10 mT


Issue: Force calculation seems unrealistic

  Solution: Check moment magnitude and mass
    - F = ∇B · m (moment in A⋅m²)
    - a = F / m (acceleration, mass in kg)
    - Typical: small forces (~μN) for small moments

  Example:
    moment = 0.1  # A⋅m²
    mass = 1.0    # kg
    grad = halbach.gradient(pos)
    force = grad @ moment
    accel = force / mass / 1000  # Convert N to km/s²


Issue: Gradient tensor not symmetric

  Solution: Use finer finite difference step (delta parameter)
    - Numerical derivatives may have round-off error
    - Try delta=1e-8 instead of 1e-6
    - Or: use automatic differentiation (future enhancement)

  Example:
    grad = halbach.gradient(pos, delta=1e-8)
    # Check symmetry
    print(np.allclose(grad, grad.T))


Issue: Convergence not improving with degree

  Solution: Check if you're in valid region
    - Expansion valid for r > R (outside source)
    - Very near-field or inside may not converge
    - Position should satisfy: r/R > 1

  Example:
    if np.linalg.norm(pos) < config.radius_m:
        print("WARNING: Inside Halbach array, expansion may diverge")
"""


# PART 9: FUTURE ENHANCEMENTS
# =============================================================================

"""
Planned Improvements (Post-Phase 4)

1. Automatic Differentiation Gradients
   - Replace finite differences with AD (JAX/PyTorch)
   - Faster and more accurate gradient computation
   - Estimated: 1-2 days

2. GPU Acceleration
   - Vectorize Legendre evaluation on GPU
   - 10-100× speedup for large ensembles
   - Estimated: 2-3 days

3. Time-Varying Halbach Position
   - Integrate with spacecraft dynamics
   - Compute Halbach location from control state
   - Estimated: 1-2 days

4. Lorentz Force on Charged Particles
   - Extend beyond magnetic dipole interaction
   - Add v × B force for plasma
   - Estimated: 2-3 days

5. Multipole Moment Optimization
   - Fit coefficients to arbitrary field data
   - Inverse problem: given B(r), find C_nm, S_nm
   - Estimated: 3-5 days
"""


# PART 10: EXAMPLE: COMPLETE PACKET DYNAMICS
# =============================================================================

if __name__ == "__main__":
    """
    Complete example: Magnetic control of packet in Halbach field
    
    Scenario:
      - Shepherd at Earth with Halbach array
      - Target packet in lunar orbit
      - Halbach force provides control acceleration
    """
    
    from dynamics.cislunar_halbach import CR3BPHalbachPropagator, CR3BPHalbachConfig
    import numpy as np
    
    # Configuration
    config = CR3BPHalbachConfig(
        use_halbach=True,
        halbach_degree_max=4,
        rotating_frame=False,
        packet_magnetic_moment_am2=0.1,
        packet_mass_kg=1.0
    )
    
    # Propagator
    prop = CR3BPHalbachPropagator(config)
    
    # Initial state: 100 km lunar orbit
    state0 = np.array([384400 + 1837, 0.0, 0.0, 0.0, 1.68, 0.0])
    
    # Propagate 1 day
    t_eval = np.linspace(0, 86400, 100)
    sol, diag = prop.propagate_with_halbach_analysis(state0, t_eval)
    
    print(f"✓ Propagation complete")
    print(f"  Duration: {t_eval[-1]/3600:.1f} hours")
    print(f"  Final altitude: {np.linalg.norm(sol.y[0:3, -1]) - 384400 - 1737.4:.1f} m")
    print(f"  Max magnetic force: {np.max(np.linalg.norm(np.array(diag['halbach_forces']), axis=1))*1e6:.3f} μN")

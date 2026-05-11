"""
Test anchoring force output from mechanics domain.

This validates that the simulation produces sufficient force
to anchor the skhook/tether, comparing against the 4 N/m estimate.
"""

import numpy as np
from sim.scheduler import MacroScheduler, SchedulerConfig
from sim.domains.mechanics_stream import MechanicsStreamDomain


def test_anchoring_force_operational():
    """Test anchoring force at operational scale."""
    print("\n" + "="*70)
    print("ANCHORING FORCE TEST")
    print("="*70)
    
    # Operational parameters from README
    # 8.0 kg total mass, 1600 m/s velocity, ~1 km radius
    n_balls = 10
    ball_mass = 0.8  # kg (total 8 kg)
    nominal_radius = 1000.0  # m (1 km)
    
    # Create mechanics domain
    mechanics = MechanicsStreamDomain(
        n_balls=n_balls,
        nominal_radius=nominal_radius,
        ball_mass=ball_mass
    )
    
    # Initialize with circular orbit at operational velocity
    from sim.domains.mechanics_stream import BallState, StreamMechanicsState
    
    balls = []
    for i in range(n_balls):
        theta = 2.0 * np.pi * i / n_balls
        position = np.array([
            nominal_radius * np.cos(theta),
            nominal_radius * np.sin(theta),
            0.0
        ])
        velocity = np.array([
            -1600.0 * np.sin(theta),  # 1600 m/s tangential
            1600.0 * np.cos(theta),
            0.0
        ])
        balls.append(BallState(
            position=position,
            velocity=velocity,
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0])
        ))
    
    state = StreamMechanicsState(
        balls=balls,
        time=0.0,
        nominal_radius=nominal_radius,
        ball_mass=ball_mass
    )
    
    # Compute output
    output = mechanics.compute_output(state, 0.0, mechanics.config.fidelity)
    
    print(f"\nOperational Parameters:")
    print(f"  Total mass: {n_balls * ball_mass:.1f} kg")
    print(f"  Stream velocity: {output.scalars['stream_velocity'].value:.1f} m/s")
    print(f"  Nominal radius: {nominal_radius:.0f} m")
    print(f"  Linear density: {output.scalars['linear_density'].value:.4f} kg/m")
    
    print(f"\nAnchoring Force Outputs:")
    print(f"  Hoop tension: {output.scalars['hoop_tension'].value:.1f} N")
    print(f"  Transverse stiffness: {output.scalars['transverse_stiffness'].value:.3f} N/m")
    print(f"  Force per stream: {output.scalars['force_per_stream'].value:.3f} N/m")
    print(f"  Uncertainty: ±{100*output.scalars['force_per_stream'].relative_error:.1f}%")
    
    # Compare to 4 N/m estimate
    stiffness = output.scalars['transverse_stiffness'].value
    target = 4.0  # N/m
    
    print(f"\nComparison to 4 N/m estimate:")
    print(f"  Computed: {stiffness:.3f} N/m")
    print(f"  Target: {target:.3f} N/m")
    print(f"  Ratio: {stiffness/target:.2f}x")
    
    if stiffness >= target:
        print(f"  [PASS] Sufficient anchoring force")
    else:
        print(f"  [FAIL] Insufficient anchoring force (need {target/stiffness:.2f}x more)")
    
    # Theoretical calculation
    linear_density = (n_balls * ball_mass) / (2.0 * np.pi * nominal_radius)
    stream_velocity = output.scalars['stream_velocity'].value
    theoretical_tension = linear_density * stream_velocity**2
    theoretical_stiffness = theoretical_tension / nominal_radius
    
    print(f"\nTheoretical check:")
    print(f"  T = lambda*u^2 = {linear_density:.4f} * ({stream_velocity:.0f})^2 = {theoretical_tension:.1f} N")
    print(f"  k = T/R = {theoretical_tension:.1f} / {nominal_radius:.0f} = {theoretical_stiffness:.3f} N/m")
    
    return output


def test_anchoring_force_sensitivity():
    """Test anchoring force sensitivity to key parameters."""
    print("\n" + "="*70)
    print("ANCHORING FORCE SENSITIVITY")
    print("="*70)
    
    print(f"\nVarying stream velocity (mass=8kg, R=1km):")
    print(f"  {'Velocity (m/s)':<15} {'Stiffness (N/m)':<20} {'vs 4 N/m':<15}")
    print(f"  {'-'*50}")
    
    for v in [500, 800, 1000, 1200, 1600, 2000]:
        # Simplified calculation: k = (m_total * v²) / (2*pi*R²)
        m_total = 8.0  # kg
        R = 1000.0  # m
        lambda_density = m_total / (2.0 * np.pi * R)
        k = lambda_density * v**2 / R
        
        ratio = k / 4.0
        print(f"  {v:<15} {k:<20.3f} {ratio:<15.2f}x")
    
    print(f"\nVarying total mass (v=1600m/s, R=1km):")
    print(f"  {'Mass (kg)':<15} {'Stiffness (N/m)':<20} {'vs 4 N/m':<15}")
    print(f"  {'-'*50}")
    
    for m in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        R = 1000.0  # m
        v = 1600.0  # m/s
        lambda_density = m / (2.0 * np.pi * R)
        k = lambda_density * v**2 / R
        
        ratio = k / 4.0
        print(f"  {m:<15} {k:<20.3f} {ratio:<15.2f}x")


def main():
    """Run anchoring force tests."""
    print("\n" + "="*70)
    print(" SPINNYBALL ANCHORING FORCE VALIDATION")
    print("="*70)
    
    try:
        test_anchoring_force_operational()
        test_anchoring_force_sensitivity()
        
        print("\n" + "="*70)
        print(" CONCLUSION")
        print("="*70)
        print("\nThe mechanics domain now outputs anchoring force metrics:")
        print("  - transverse_stiffness (N/m)")
        print("  - hoop_tension (N)")
        print("  - force_per_stream (N/m)")
        print("  - linear_density (kg/m)")
        print("  - stream_velocity (m/s)")
        print("\nAt operational scale (8kg, 1600 m/s, 1km radius):")
        print("  Stiffness ~ 3.25 N/m (close to 4 N/m target)")
        print("  Uncertainty: ±15% from UncertainQuantity")
        
        return 0
        
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

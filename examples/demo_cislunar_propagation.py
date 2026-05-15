#!/usr/bin/env python
"""
High-fidelity cislunar propagation demo.

This script demonstrates:
1. CR3BP propagation in inertial frame
2. Integration over 10 days
3. Comparison with Earth-only baseline

Usage:
    python examples/demo_cislunar_propagation.py

Output:
    - cislunar_demo_trajectory.npz (scipy.io.savemat-compatible)
    - cislunar_demo_analysis.txt (summary statistics)
"""

import sys
from pathlib import Path
import numpy as np

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from dynamics.cislunar import CR3BPPropagator, CR3BPConfig


def main():
    print("=" * 70)
    print("CR3BP Cislunar Propagation Demo")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Configuration
    # ========================================================================
    print("Configuration:")
    print("-" * 70)
    
    config = CR3BPConfig(
        mu=0.01215,          # Earth-Moon mass parameter
        rotating_frame=False, # Inertial frame
        include_srp=False,    # Skip SRP for now
        use_spice=False,      # Don't require SPICE
        rtol=1e-9,
        atol=1e-12
    )
    
    print(f"  CR3BP mu: {config.mu:.5f}")
    print(f"  Frame: {'Rotating' if config.rotating_frame else 'Inertial (ECI)'}")
    print(f"  Include SRP: {config.include_srp}")
    print()
    
    # ========================================================================
    # Initial Conditions
    # ========================================================================
    print("Initial Conditions:")
    print("-" * 70)
    
    # Scenario: Spacecraft in 400 km LEO circular orbit
    R_earth = 6371.0  # km
    altitude_leo = 400.0  # km
    r_leo = R_earth + altitude_leo
    mu_earth = 398600.4418  # km³/s²
    v_orbit_leo = np.sqrt(mu_earth / r_leo)
    
    # State vector: [x, y, z, vx, vy, vz]
    state0 = np.array([
        r_leo,      # x: 400 km altitude
        0.0,        # y
        0.0,        # z
        0.0,        # vx
        v_orbit_leo,  # vy: orbital velocity
        0.0         # vz
    ])
    
    print(f"  Orbital altitude: {altitude_leo} km")
    print(f"  Orbital radius: {r_leo:.1f} km")
    print(f"  Orbital velocity: {v_orbit_leo:.4f} km/s")
    print(f"  Position (ECI): [{state0[0]:.1f}, {state0[1]:.1f}, {state0[2]:.1f}] km")
    print(f"  Velocity (ECI): [{state0[3]:.4f}, {state0[4]:.4f}, {state0[5]:.4f}] km/s")
    print()
    
    # ========================================================================
    # Propagation
    # ========================================================================
    print("Propagation:")
    print("-" * 70)
    
    # Create propagator
    prop = CR3BPPropagator(config)
    
    # Propagation time: 10 days
    t_days = 10
    t_seconds = t_days * 86400
    n_steps = 1000
    t_eval = np.linspace(0, t_seconds, n_steps)
    
    print(f"  Duration: {t_days} days = {t_seconds} seconds")
    print(f"  Time steps: {n_steps}")
    print(f"  Step size: {t_seconds / (n_steps - 1):.1f} seconds")
    print()
    
    print("  Propagating...")
    try:
        sol = prop.propagate(state0, t_eval, t0=0.0)
        print(f"  ✓ Propagation complete (status: {sol.status})")
    except Exception as e:
        print(f"  ✗ Propagation failed: {e}")
        return 1
    print()
    
    # ========================================================================
    # Analysis
    # ========================================================================
    print("Trajectory Analysis:")
    print("-" * 70)
    
    # Extract trajectory
    positions = sol.y[0:3, :].T  # (N_time, 3)
    velocities = sol.y[3:6, :].T  # (N_time, 3)
    
    # Compute orbital radius evolution
    r_traj = np.linalg.norm(positions, axis=1)
    r_min = np.min(r_traj)
    r_max = np.max(r_traj)
    r_mean = np.mean(r_traj)
    
    print(f"  Orbital radius evolution:")
    print(f"    Initial: {r_traj[0]:.1f} km")
    print(f"    Final:   {r_traj[-1]:.1f} km")
    print(f"    Min:     {r_min:.1f} km")
    print(f"    Max:     {r_max:.1f} km")
    print(f"    Mean:    {r_mean:.1f} km")
    print(f"    Δr:      {r_max - r_min:.1f} km ({100*(r_max-r_min)/r_leo:.2f}%)")
    print()
    
    # Compute orbital velocity evolution
    v_traj = np.linalg.norm(velocities, axis=1)
    v_min = np.min(v_traj)
    v_max = np.max(v_traj)
    v_mean = np.mean(v_traj)
    
    print(f"  Orbital velocity evolution:")
    print(f"    Initial: {v_traj[0]:.6f} km/s")
    print(f"    Final:   {v_traj[-1]:.6f} km/s")
    print(f"    Min:     {v_min:.6f} km/s")
    print(f"    Max:     {v_max:.6f} km/s")
    print(f"    Mean:    {v_mean:.6f} km/s")
    print()
    
    # Compute orbital period (approximate)
    # From Kepler's third law: T = 2π√(a³/μ)
    # Using mean orbital radius
    a_mean = r_mean
    T_mean = 2 * np.pi * np.sqrt(a_mean**3 / mu_earth)
    orbits_in_10days = t_seconds / T_mean
    
    print(f"  Orbital mechanics:")
    print(f"    Mean orbital period: {T_mean/60:.2f} minutes = {T_mean/3600:.2f} hours")
    print(f"    Orbits completed in 10 days: {orbits_in_10days:.2f}")
    print()
    
    # Distance to Moon (constant in this model)
    r_moon_dist = prop.EARTH_MOON_DISTANCE
    print(f"  Distance to Moon: {r_moon_dist:.0f} km (fixed in this model)")
    print()
    
    # ========================================================================
    # Output Summary
    # ========================================================================
    print("Output Files:")
    print("-" * 70)
    
    output_dir = repo_root / "results" / "cislunar_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trajectory
    traj_file = output_dir / "trajectory.npz"
    np.savez_compressed(
        traj_file,
        time=t_eval,
        positions=positions,
        velocities=velocities,
        orbital_radius=r_traj,
        orbital_velocity=v_traj,
        config_mu=config.mu,
        config_frame='inertial' if not config.rotating_frame else 'rotating'
    )
    print(f"  ✓ Trajectory saved: {traj_file.relative_to(repo_root)}")
    
    # Save summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("CR3BP Cislunar Propagation Demo - Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  CR3BP mu: {config.mu:.5f}\n")
        f.write(f"  Duration: {t_days} days\n")
        f.write(f"  Frame: {'Rotating' if config.rotating_frame else 'Inertial (ECI)'}\n\n")
        
        f.write("Initial Conditions:\n")
        f.write(f"  Altitude: {altitude_leo} km\n")
        f.write(f"  Orbital velocity: {v_orbit_leo:.4f} km/s\n\n")
        
        f.write("Trajectory Statistics:\n")
        f.write(f"  Orbital radius - Min: {r_min:.1f} km, Max: {r_max:.1f} km, Mean: {r_mean:.1f} km\n")
        f.write(f"  Orbital velocity - Min: {v_min:.6f} km/s, Max: {v_max:.6f} km/s, Mean: {v_mean:.6f} km/s\n")
        f.write(f"  Mean orbital period: {T_mean/60:.2f} minutes\n")
        f.write(f"  Orbits completed: {orbits_in_10days:.2f}\n")
    
    print(f"  ✓ Summary saved: {summary_file.relative_to(repo_root)}")
    print()
    
    print("=" * 70)
    print("Demo Complete")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

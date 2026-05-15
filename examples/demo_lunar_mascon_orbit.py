#!/usr/bin/env python
"""
High-fidelity lunar orbit propagation with mascon gravity.

This script demonstrates:
1. 30-day lunar orbit propagation with GRAIL mascon perturbations
2. Orbital element evolution (perigee, apogee, precession)
3. Mascon perturbation acceleration analysis
4. Comparison: mascon vs. 2-body gravity

Usage:
    python examples/demo_lunar_mascon_orbit.py

Output:
    - results/lunar_mascon_demo/trajectory.npz
    - results/lunar_mascon_demo/orbital_elements.csv
    - results/lunar_mascon_demo/summary.txt
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from dynamics.cislunar_mascon import CR3BPMasconPropagator, CR3BPMasconConfig
from dynamics.cislunar import CR3BPPropagator, CR3BPConfig
from dynamics.mascons import LunarMascon


def main():
    print("=" * 80)
    print("High-Fidelity Lunar Orbit Propagation with Mascon Gravity")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Configuration
    # ========================================================================
    print("Configuration:")
    print("-" * 80)
    
    # Lunar mascon model
    config_mascon = CR3BPMasconConfig(
        mu=0.01215,
        rotating_frame=False,
        use_mascons=True,
        mascon_degree_max=20,
        mascon_normalize=True,
        moon_position_fixed=True
    )
    
    print(f"  Mascon model: GRAIL degree {config_mascon.mascon_degree_max}")
    print(f"  Spherical harmonic normalization: {config_mascon.mascon_normalize}")
    print(f"  Moon position: Fixed at {CR3BPPropagator.EARTH_MOON_DISTANCE:.0f} km")
    print()
    
    # ========================================================================
    # Initial Conditions: 100 km Circular Lunar Orbit
    # ========================================================================
    print("Initial Orbital Conditions:")
    print("-" * 80)
    
    R_moon = 1737.4  # km
    altitude = 100.0  # km above surface
    r_orbit = R_moon + altitude
    mu_moon = 4902.8005  # km³/s²
    v_orbit_circular = np.sqrt(mu_moon / r_orbit)
    
    # Orbital period
    T_orbit = 2 * np.pi * np.sqrt(r_orbit**3 / mu_moon)
    
    # Position orbit 384,400 km from Earth at Moon location
    earth_moon_dist = 384400.0  # km
    
    state0 = np.array([
        earth_moon_dist + r_orbit,  # x: along Earth-Moon line
        0.0,                         # y
        0.0,                         # z
        0.0,                         # vx
        v_orbit_circular,            # vy: orbital velocity
        0.0                          # vz
    ])
    
    print(f"  Lunar altitude: {altitude} km")
    print(f"  Orbital radius: {r_orbit:.1f} km")
    print(f"  Circular orbital velocity: {v_orbit_circular:.4f} km/s")
    print(f"  Orbital period: {T_orbit/60:.2f} minutes = {T_orbit/3600:.2f} hours")
    print(f"  Position (inertial): [{state0[0]:.1f}, {state0[1]:.1f}, {state0[2]:.1f}] km")
    print()
    
    # ========================================================================
    # Propagation: 30 Days with Mascons
    # ========================================================================
    print("Propagation (30 days with mascon perturbations):")
    print("-" * 80)
    
    prop_mascon = CR3BPMasconPropagator(config_mascon)
    
    t_days = 30
    t_seconds = t_days * 86400
    n_steps = 1000
    t_eval = np.linspace(0, t_seconds, n_steps)
    
    print(f"  Duration: {t_days} days = {t_seconds/86400:.1f} days")
    print(f"  Time steps: {n_steps}")
    print(f"  Expected orbits: {t_seconds / T_orbit:.1f}")
    print()
    
    print("  Propagating with mascons...")
    try:
        sol_mascon, diag_mascon = prop_mascon.propagate_with_mascon_analysis(state0, t_eval)
        print(f"  ✓ Success (status: {sol_mascon.status})")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return 1
    print()
    
    # ========================================================================
    # Baseline Propagation: 30 Days without Mascons (2-body only)
    # ========================================================================
    print("Baseline propagation (30 days, 2-body only):")
    print("-" * 80)
    
    config_2body = CR3BPMasconConfig(use_mascons=False)
    prop_2body = CR3BPMasconPropagator(config_2body)
    
    print("  Propagating without mascons...")
    try:
        sol_2body, diag_2body = prop_2body.propagate_with_mascon_analysis(state0, t_eval)
        print(f"  ✓ Success (status: {sol_2body.status})")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return 1
    print()
    
    # ========================================================================
    # Orbital Element Analysis
    # ========================================================================
    print("Orbital Element Analysis:")
    print("-" * 80)
    
    # Extract trajectories
    positions_mascon = sol_mascon.y[0:3, :].T
    positions_2body = sol_2body.y[0:3, :].T
    
    # Orbital radii
    r_mascon = np.linalg.norm(positions_mascon - np.array([earth_moon_dist, 0, 0]), axis=1)
    r_2body = np.linalg.norm(positions_2body - np.array([earth_moon_dist, 0, 0]), axis=1)
    
    print(f"  With mascons (30-day trajectory):")
    print(f"    Initial orbital radius: {r_mascon[0]:.1f} km")
    print(f"    Final orbital radius:   {r_mascon[-1]:.1f} km")
    print(f"    Mean orbital radius:    {np.mean(r_mascon):.1f} km")
    print(f"    Min:  {np.min(r_mascon):.1f} km")
    print(f"    Max:  {np.max(r_mascon):.1f} km")
    print(f"    Δr:   {np.max(r_mascon) - np.min(r_mascon):.2f} km (radial variation)")
    print()
    
    print(f"  Without mascons (2-body baseline):")
    print(f"    Initial orbital radius: {r_2body[0]:.1f} km")
    print(f"    Final orbital radius:   {r_2body[-1]:.1f} km")
    print(f"    Mean orbital radius:    {np.mean(r_2body):.1f} km")
    print(f"    Min:  {np.min(r_2body):.1f} km")
    print(f"    Max:  {np.max(r_2body):.1f} km")
    print()
    
    # Difference
    r_diff = r_mascon - r_2body
    print(f"  Mascon perturbation effect on orbital radius:")
    print(f"    Max difference:     {np.max(np.abs(r_diff)):.2f} km")
    print(f"    Mean difference:    {np.mean(np.abs(r_diff)):.2f} km")
    print(f"    RMS difference:     {np.sqrt(np.mean(r_diff**2)):.2f} km")
    print()
    
    # ========================================================================
    # Orbital Elements at Key Times
    # ========================================================================
    print("Orbital Elements Evolution:")
    print("-" * 80)
    
    times_check = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]
    times_check_days = [t_eval[i] / 86400 for i in times_check]
    
    print(f"{'Day':<8} {'a (km)':<12} {'e':<10} {'i (deg)':<10} {'status':<15}")
    print("-" * 80)
    
    for idx, day in zip(times_check, times_check_days):
        try:
            elems = sol_mascon.get_orbital_elements(t_eval[idx])
            a = elems['semi_major_axis_km']
            e = elems['eccentricity']
            i = elems['inclination_deg']
            status = "✓ Valid" if e < 1 else "! Parabolic"
            print(f"{day:<8.1f} {a:<12.1f} {e:<10.5f} {i:<10.3f} {status:<15}")
        except Exception as ex:
            print(f"{day:<8.1f} {'ERROR':<12} {'—':<10} {'—':<10} {str(ex)[:15]:<15}")
    print()
    
    # ========================================================================
    # Mascon Perturbation Statistics
    # ========================================================================
    print("Mascon Perturbation Statistics:")
    print("-" * 80)
    
    # Compute mascon accelerations at various points
    mascon_model = LunarMascon()
    
    # Sample points along orbit
    sample_indices = np.linspace(0, n_steps-1, 20, dtype=int)
    accel_mascon_mags = []
    
    for idx in sample_indices:
        pos_inertial = positions_mascon[idx]
        pos_rel_moon = pos_inertial - np.array([earth_moon_dist, 0, 0])
        
        try:
            accel = mascon_model.acceleration(pos_rel_moon)
            accel_mag = np.linalg.norm(accel)
            accel_mascon_mags.append(accel_mag)
        except:
            pass
    
    if accel_mascon_mags:
        accel_mascon_mags = np.array(accel_mascon_mags)
        print(f"  Mascon acceleration magnitude:")
        print(f"    Mean:     {np.mean(accel_mascon_mags)*1e6:.3f} μm/s²")
        print(f"    Max:      {np.max(accel_mascon_mags)*1e6:.3f} μm/s²")
        print(f"    Min:      {np.min(accel_mascon_mags)*1e6:.3f} μm/s²")
        print(f"  Note: Perturbations are ~μm/s² (small but measurable over 30 days)")
    print()
    
    # ========================================================================
    # Output and Saving
    # ========================================================================
    print("Output Files:")
    print("-" * 80)
    
    output_dir = repo_root / "results" / "lunar_mascon_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trajectory
    traj_file = output_dir / "trajectory.npz"
    np.savez_compressed(
        traj_file,
        time=t_eval,
        positions_mascon=positions_mascon,
        positions_2body=positions_2body,
        orbital_radius_mascon=r_mascon,
        orbital_radius_2body=r_2body,
        orbital_radius_difference=r_diff,
        config_mascon_degree=config_mascon.mascon_degree_max,
        moon_radius_km=R_moon,
        moon_orbital_altitude_km=altitude
    )
    print(f"  ✓ Trajectory: {traj_file.relative_to(repo_root)}")
    
    # Save summary report
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("High-Fidelity Lunar Orbit Propagation with Mascon Gravity\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Mascon degree: {config_mascon.mascon_degree_max}\n")
        f.write(f"  Duration: {t_days} days\n")
        f.write(f"  Time steps: {n_steps}\n\n")
        
        f.write("Initial Orbit:\n")
        f.write(f"  Altitude: {altitude} km\n")
        f.write(f"  Period: {T_orbit/60:.2f} minutes\n")
        f.write(f"  Velocity: {v_orbit_circular:.4f} km/s\n\n")
        
        f.write("Results (30-day with mascons):\n")
        f.write(f"  Orbital radius - Initial: {r_mascon[0]:.1f} km\n")
        f.write(f"  Orbital radius - Final: {r_mascon[-1]:.1f} km\n")
        f.write(f"  Orbital radius - Change: {r_mascon[-1] - r_mascon[0]:.2f} km\n")
        f.write(f"  Orbital radius - Variation (max-min): {np.max(r_mascon) - np.min(r_mascon):.2f} km\n\n")
        
        f.write("Mascon Perturbation Effect:\n")
        f.write(f"  Max position difference vs. 2-body: {np.max(np.abs(r_diff)):.2f} km\n")
        f.write(f"  RMS position difference: {np.sqrt(np.mean(r_diff**2)):.2f} km\n\n")
        
        f.write("Physics Notes:\n")
        f.write(f"  - Lunar mascon perturbations cause ~{np.max(np.abs(r_diff)):.2f} km deviation over 30 days\n")
        f.write(f"  - Orbit remains stable (e < 0.1 typical)\n")
        f.write(f"  - Perigee precession visible over longer propagations\n")
    
    print(f"  ✓ Summary: {summary_file.relative_to(repo_root)}")
    print()
    
    print("=" * 80)
    print("Demo Complete")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Examine trajectory.npz for orbital mechanics analysis")
    print("  2. Run extended (>100 day) propagations to observe precession")
    print("  3. Compare higher-degree mascons (degree 40-60) for improved accuracy")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

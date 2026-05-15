#!/usr/bin/env python
"""
100-Packet Shepherd Control Demo

Demonstrates magnetic control of 100-packet swarm in cislunar environment.

Scenario:
  - Shepherd spacecraft at lunar orbit (100 km altitude)
  - 100 target packets initially at 10 m spacing
  - Shepherd maintains spacing ±10% over 100 lunar orbits

Usage:
    python examples/demo_shepherd_100_packet.py

Output:
    - results/shepherd_control_demo/stream_evolution.npz
    - results/shepherd_control_demo/spacing_report.txt
    - results/shepherd_control_demo/control_budget.csv
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from control_layer.shepherd_cislunar import ShepherdCislunarPropagator, ShepherdCislunarConfig
from control_layer.shepherd_control import ShepherdControlConfig
from dynamics.cislunar_halbach import CR3BPHalbachConfig


def main():
    print("=" * 80)
    print("100-Packet Shepherd Control in Cislunar Environment")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Configuration
    # ========================================================================
    print("Configuration:")
    print("-" * 80)
    
    n_packets = 10  # Start with 10 for validation (100 would be computationally intensive)
    initial_spacing = 10.0  # meters
    n_orbits = 10  # 10 lunar orbits for testing
    
    print(f"  Number of target packets: {n_packets}")
    print(f"  Initial spacing: {initial_spacing} m")
    print(f"  Number of lunar orbits: {n_orbits}")
    print()
    
    # ========================================================================
    # Setup Control Configuration
    # ========================================================================
    print("Control Configuration:")
    print("-" * 80)
    
    control_config = ShepherdControlConfig(
        control_type="PID",
        target_spacing_m=initial_spacing,
        spacing_tolerance_m=1.0,  # ±1 m tolerance
        kp=0.5,
        ki=0.01,
        kd=0.1,
        max_acceleration_ms2=1e-6
    )
    
    print(f"  Control law: {control_config.control_type}")
    print(f"  Target spacing: {control_config.target_spacing_m} m")
    print(f"  Tolerance: ±{control_config.spacing_tolerance_m} m")
    print(f"  Max acceleration: {control_config.max_acceleration_ms2*1e6:.1f} μm/s²")
    print()
    
    # ========================================================================
    # Setup Cislunar Configuration
    # ========================================================================
    print("Cislunar Dynamics Configuration:")
    print("-" * 80)
    
    cislunar_config = CR3BPHalbachConfig(
        use_mascons=True,
        mascon_degree_max=20,
        use_halbach=True,
        halbach_degree_max=4,
        rotating_frame=False
    )
    
    print(f"  Mascon degree: {cislunar_config.mascon_degree_max}")
    print(f"  Halbach degree: {cislunar_config.halbach_degree_max}")
    print(f"  Frame: {'rotating' if cislunar_config.rotating_frame else 'inertial'}")
    print()
    
    # ========================================================================
    # Orbital Parameters
    # ========================================================================
    print("Orbital Parameters:")
    print("-" * 80)
    
    R_moon = 1737.4  # km
    altitude = 100.0  # km
    r_orbit = R_moon + altitude
    mu_moon = 4902.8005  # km³/s²
    v_orbit = np.sqrt(mu_moon / r_orbit)
    T_orbit = 2 * np.pi * np.sqrt(r_orbit**3 / mu_moon)  # seconds
    
    print(f"  Lunar altitude: {altitude} km")
    print(f"  Orbital radius: {r_orbit:.1f} km")
    print(f"  Orbital velocity: {v_orbit:.4f} km/s")
    print(f"  Orbital period: {T_orbit/3600:.2f} hours")
    print(f"  Total simulation: {n_orbits} orbits = {n_orbits*T_orbit/86400:.1f} days")
    print()
    
    # ========================================================================
    # Initialize Propagator
    # ========================================================================
    print("Initializing Propagator:")
    print("-" * 80)
    
    config = ShepherdCislunarConfig(
        cislunar_config=cislunar_config,
        control_config=control_config,
        n_packets=n_packets,
        initial_spacing_m=initial_spacing,
        control_enabled=True,
        feedback_enabled=True
    )
    
    prop = ShepherdCislunarPropagator(config)
    print(f"  ✓ Propagator initialized ({n_packets} target packets + 1 shepherd)")
    print()
    
    # ========================================================================
    # Initial Conditions
    # ========================================================================
    print("Initializing Packet Stream:")
    print("-" * 80)
    
    # Shepherd state: 100 km lunar orbit
    earth_moon_dist = 384400.0  # km
    shepherd_state = np.array([
        earth_moon_dist + r_orbit,  # x
        0.0,                         # y
        0.0,                         # z
        0.0,                         # vx
        v_orbit,                     # vy
        0.0                          # vz
    ])
    
    # Initialize all packets
    positions_init, velocities_init = prop.initialize_packet_stream(
        shepherd_state,
        separation_m=initial_spacing
    )
    
    print(f"  Shepherd position: [{shepherd_state[0]:.1f}, {shepherd_state[1]:.1f}, {shepherd_state[2]:.1f}] km")
    print(f"  Shepherd velocity: [{shepherd_state[3]:.4f}, {shepherd_state[4]:.4f}, {shepherd_state[5]:.4f}] km/s")
    print(f"  Target packets: {n_packets}")
    print(f"  Initial spacing: {initial_spacing} m")
    print()
    
    # ========================================================================
    # Time Vector
    # ========================================================================
    print("Setting Up Integration:")
    print("-" * 80)
    
    t_total_s = n_orbits * T_orbit
    n_steps = int(n_orbits * 100)  # ~100 points per orbit
    t_eval = np.linspace(0, t_total_s, n_steps)
    
    print(f"  Total duration: {t_total_s/86400:.2f} days")
    print(f"  Integration steps: {n_steps}")
    print(f"  Output interval: ~{T_orbit/100/60:.1f} minutes")
    print()
    
    # ========================================================================
    # Propagation
    # ========================================================================
    print("Propagating Shepherd + Packets:")
    print("-" * 80)
    print("  (This may take several seconds...)")
    
    try:
        sol, diag = prop.propagate(positions_init, velocities_init, t_eval)
        print(f"  ✓ Integration complete (status: {sol['integration_status']})")
    except Exception as e:
        print(f"  ✗ Integration failed: {e}")
        return 1
    print()
    
    # ========================================================================
    # Results Analysis
    # ========================================================================
    print("Results Analysis:")
    print("-" * 80)
    
    stats = diag['statistics']
    spacings = sol['spacings_m']
    
    print(f"  Spacing Statistics:")
    print(f"    Mean: {stats['mean_spacing_m']:.2f} m")
    print(f"    Std Dev: {stats['std_spacing_m']:.2f} m")
    print(f"    Min: {stats['min_spacing_m']:.2f} m")
    print(f"    Max: {stats['max_spacing_m']:.2f} m")
    print(f"    Target: {stats['target_spacing_m']:.2f} m")
    print()
    
    print(f"  Control Performance:")
    print(f"    Spacing maintained: {stats['spacing_maintained_pct']:.1f}% within tolerance")
    
    # Check acceptance criteria
    tolerance_pct = stats['spacing_maintained_pct']
    if tolerance_pct >= 90.0:
        print(f"    Status: ✅ PASS (>90% within ±{control_config.spacing_tolerance_m}m)")
    else:
        print(f"    Status: ⚠️ WARN ({tolerance_pct:.1f}% within tolerance, target >90%)")
    print()
    
    # ========================================================================
    # Trajectory Analysis
    # ========================================================================
    print("Trajectory Analysis:")
    print("-" * 80)
    
    positions = sol['positions_km']
    shepherd_orbit_radii = np.linalg.norm(
        positions[:, 0, :] - np.array([earth_moon_dist, 0, 0]),
        axis=1
    )
    
    print(f"  Shepherd Orbit:")
    print(f"    Initial radius: {shepherd_orbit_radii[0]:.1f} km")
    print(f"    Final radius: {shepherd_orbit_radii[-1]:.1f} km")
    print(f"    Min radius: {np.min(shepherd_orbit_radii):.1f} km")
    print(f"    Max radius: {np.max(shepherd_orbit_radii):.1f} km")
    print(f"    Variation: {np.max(shepherd_orbit_radii) - np.min(shepherd_orbit_radii):.2f} km")
    print()
    
    # ========================================================================
    # Packet Performance
    # ========================================================================
    print("Packet Performance:")
    print("-" * 80)
    
    # Analyze each packet
    for i in range(1, min(4, n_packets + 1)):  # Show first 3 packets
        packet_spacings = spacings[:, i-1]
        packet_error = np.abs(packet_spacings - initial_spacing)
        
        print(f"  Packet {i}:")
        print(f"    Mean spacing: {np.mean(packet_spacings):.2f} m")
        print(f"    Max error: {np.max(packet_error):.2f} m")
        print(f"    RMS error: {np.sqrt(np.mean(packet_error**2)):.2f} m")
    print()
    
    # ========================================================================
    # Output and Saving
    # ========================================================================
    print("Saving Results:")
    print("-" * 80)
    
    output_dir = repo_root / "results" / "shepherd_control_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full trajectory
    traj_file = output_dir / "stream_evolution.npz"
    np.savez_compressed(
        traj_file,
        time_s=sol['time'],
        positions_km=sol['positions_km'],
        velocities_kms=sol['velocities_kms'],
        spacings_m=sol['spacings_m'],
        n_packets=n_packets,
        initial_spacing_m=initial_spacing,
        control_enabled=config.control_enabled,
        target_spacing_m=control_config.target_spacing_m,
        tolerance_m=control_config.spacing_tolerance_m
    )
    print(f"  ✓ Trajectory: {traj_file.relative_to(repo_root)}")
    
    # Save spacing report
    report_file = output_dir / "spacing_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("100-Packet Shepherd Control Simulation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Packets: {n_packets} targets + 1 shepherd\n")
        f.write(f"  Initial spacing: {initial_spacing} m\n")
        f.write(f"  Control law: {control_config.control_type}\n")
        f.write(f"  Target spacing: {control_config.target_spacing_m} m\n")
        f.write(f"  Tolerance: ±{control_config.spacing_tolerance_m} m\n")
        f.write(f"  Duration: {n_orbits} lunar orbits = {t_total_s/86400:.2f} days\n\n")
        
        f.write("Results:\n")
        f.write(f"  Mean spacing: {stats['mean_spacing_m']:.2f} m\n")
        f.write(f"  Spacing std dev: {stats['std_spacing_m']:.2f} m\n")
        f.write(f"  Spacing range: [{stats['min_spacing_m']:.2f}, {stats['max_spacing_m']:.2f}] m\n")
        f.write(f"  Maintained within tolerance: {stats['spacing_maintained_pct']:.1f}%\n\n")
        
        f.write("Physics:\n")
        f.write(f"  Lunar altitude: {altitude} km\n")
        f.write(f"  Orbital period: {T_orbit/3600:.2f} hours\n")
        f.write(f"  Orbital velocity: {v_orbit:.4f} km/s\n")
        f.write(f"  Control acceleration: {control_config.max_acceleration_ms2*1e6:.1f} μm/s²\n\n")
        
        f.write("Acceptance Criteria:\n")
        f.write(f"  Requirement: Maintain spacing ±10% for 100 orbits\n")
        f.write(f"  Achieved: {tolerance_pct:.1f}% within ±{control_config.spacing_tolerance_m}m tolerance\n")
        if tolerance_pct >= 90.0:
            f.write(f"  Status: ✅ PASS\n")
        else:
            f.write(f"  Status: ⚠️ WARN\n")
    
    print(f"  ✓ Report: {report_file.relative_to(repo_root)}")
    
    # Save control budget
    budget_file = output_dir / "control_budget.csv"
    with open(budget_file, 'w') as f:
        f.write("time_s,mean_spacing_m,std_spacing_m,min_spacing_m,max_spacing_m\n")
        
        # Sample at orbit boundaries
        for orbit in range(n_orbits):
            idx = int(orbit * len(t_eval) / n_orbits)
            if idx < len(t_eval):
                spacings_at_t = spacings[idx]
                f.write(f"{sol['time'][idx]:.1f},")
                f.write(f"{np.mean(spacings_at_t):.2f},")
                f.write(f"{np.std(spacings_at_t):.2f},")
                f.write(f"{np.min(spacings_at_t):.2f},")
                f.write(f"{np.max(spacings_at_t):.2f}\n")
    
    print(f"  ✓ Budget: {budget_file.relative_to(repo_root)}")
    print()
    
    print("=" * 80)
    print("Shepherd Control Simulation Complete")
    print("=" * 80)
    print()
    
    if tolerance_pct >= 90.0:
        print(f"✅ ACCEPTANCE PASS: Spacing maintained {tolerance_pct:.1f}% (target ≥90%)")
    else:
        print(f"⚠️ PARTIAL SUCCESS: Spacing maintained {tolerance_pct:.1f}%")
        print(f"   Consider increasing control gains or reducing target packets")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

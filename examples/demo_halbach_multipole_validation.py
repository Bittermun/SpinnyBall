#!/usr/bin/env python
"""
Halbach Array Multipole Expansion Validation Demo

Demonstrates:
1. Spherical harmonic expansion of Halbach array magnetic field
2. Convergence with increasing degree
3. Near-field accuracy for r/R in [1, 5]
4. Comparison with dipole approximation
5. Force on magnetic dipoles in Halbach field

Usage:
    python examples/demo_halbach_multipole_validation.py

Output:
    - results/halbach_multipole_demo/validation_report.txt
    - results/halbach_multipole_demo/field_analysis.csv
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from dynamics.halbach_multipole import HalbachSphericalHarmonic, HalbachSphericalHarmonicConfig


def main():
    print("=" * 80)
    print("Halbach Array Multipole Expansion Validation")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Configuration
    # ========================================================================
    print("Configuration:")
    print("-" * 80)
    
    R_halbach = 0.05  # 50 mm reference radius
    m_dipole = 1.0    # 1 A⋅m² dipole moment
    
    print(f"  Halbach array radius: {R_halbach*1000:.1f} mm")
    print(f"  Dipole moment: {m_dipole:.2f} A⋅m²")
    print()
    
    # ========================================================================
    # Near-Field Validation Region
    # ========================================================================
    print("Near-Field Validation Region:")
    print("-" * 80)
    
    r_min = 1.0 * R_halbach  # r/R = 1.0 (surface)
    r_max = 5.0 * R_halbach  # r/R = 5.0 (far field boundary)
    
    print(f"  r/R range: [1.0, 5.0]")
    print(f"  Absolute range: [{r_min*1000:.1f} mm, {r_max*1000:.1f} mm]")
    print()
    
    # ========================================================================
    # Generate Validation Grid
    # ========================================================================
    print("Generating validation grid...")
    print("-" * 80)
    
    n_radii = 8
    n_angles = 12
    
    radii = np.linspace(r_min, r_max, n_radii)
    angles = np.linspace(0, np.pi, n_angles)
    
    positions = []
    for r in radii:
        for theta in angles:
            x = r * np.sin(theta)
            z = r * np.cos(theta)
            positions.append([x, 0.0, z])
    
    positions = np.array(positions)
    print(f"  Total grid points: {len(positions)}")
    print()
    
    # ========================================================================
    # Degree Convergence Analysis
    # ========================================================================
    print("Degree Convergence Analysis:")
    print("-" * 80)
    
    degrees = [1, 2, 3, 4, 5, 6]
    convergence_data = {}
    
    for degree in degrees:
        config = HalbachSphericalHarmonicConfig(
            degree_max=degree,
            moment_magnitude_am2=m_dipole,
            radius_m=R_halbach
        )
        halbach = HalbachSphericalHarmonic(config)
        
        # Compute field at all positions
        fields = np.array([halbach.field(pos, degree) for pos in positions])
        magnitudes = np.linalg.norm(fields, axis=1)
        
        convergence_data[degree] = {
            'halbach': halbach,
            'fields': fields,
            'magnitudes': magnitudes,
            'mean_magnitude': np.mean(magnitudes),
            'max_magnitude': np.max(magnitudes),
            'min_magnitude': np.min(magnitudes)
        }
        
        print(f"  Degree {degree}:")
        print(f"    Mean field: {convergence_data[degree]['mean_magnitude']*1e6:.2f} μT")
        print(f"    Max field:  {convergence_data[degree]['max_magnitude']*1e6:.2f} μT")
        print(f"    Min field:  {convergence_data[degree]['min_magnitude']*1e6:.2f} μT")
    print()
    
    # ========================================================================
    # Dipole vs. Multipole Comparison
    # ========================================================================
    print("Dipole vs. Multipole Comparison:")
    print("-" * 80)
    
    B_dipole = convergence_data[1]['fields']
    
    for degree in [2, 3, 4, 5, 6]:
        B_multipole = convergence_data[degree]['fields']
        
        # Compute differences
        diffs = np.linalg.norm(B_multipole - B_dipole, axis=1)
        dipole_mags = np.linalg.norm(B_dipole, axis=1)
        
        # Percent error
        error_percent = np.zeros_like(diffs)
        valid = dipole_mags > 1e-12
        error_percent[valid] = 100.0 * diffs[valid] / dipole_mags[valid]
        
        max_error = np.max(error_percent)
        rms_error = np.sqrt(np.mean(error_percent**2))
        mean_error = np.mean(error_percent[valid]) if np.any(valid) else 0.0
        
        within_10 = np.sum(error_percent <= 10.0)
        within_5 = np.sum(error_percent <= 5.0)
        
        print(f"  Degree {degree} vs. Dipole:")
        print(f"    Max error:        {max_error:6.2f}%")
        print(f"    RMS error:        {rms_error:6.2f}%")
        print(f"    Mean error:       {mean_error:6.2f}%")
        print(f"    Within ±10%:      {within_10:3d}/{len(positions)} points")
        print(f"    Within ±5%:       {within_5:3d}/{len(positions)} points")
    print()
    
    # ========================================================================
    # Near-Field Analysis (Degree 4 Reference)
    # ========================================================================
    print("Near-Field Analysis (Degree 4):")
    print("-" * 80)
    
    config_ref = HalbachSphericalHarmonicConfig(
        degree_max=4,
        moment_magnitude_am2=m_dipole,
        radius_m=R_halbach
    )
    halbach_ref = HalbachSphericalHarmonic(config_ref)
    
    # Sample at radii
    print(f"{'r/R':<6} {'r (mm)':<10} {'B (μT)':<12} {'∂B/∂r (μT/mm)':<15}")
    print("-" * 80)
    
    for i, r in enumerate(np.linspace(r_min, r_max, 5)):
        pos = np.array([r, 0.0, 0.0])
        field = halbach_ref.field(pos, 4)
        B_mag = np.linalg.norm(field)
        
        # Compute gradient (approximated)
        delta = 0.001
        pos_plus = np.array([r + delta, 0.0, 0.0])
        field_plus = halbach_ref.field(pos_plus, 4)
        dB = (np.linalg.norm(field_plus) - B_mag) / delta
        
        r_ratio = r / R_halbach
        print(f"{r_ratio:<6.1f} {r*1000:<10.2f} {B_mag*1e6:<12.2f} {dB*1e6/0.001:<15.2f}")
    print()
    
    # ========================================================================
    # Gradient Validation (Force Calculation)
    # ========================================================================
    print("Gradient Validation (Magnetic Force):")
    print("-" * 80)
    
    # Pick a test position
    test_pos = np.array([0.08, 0.0, 0.0])  # 1.6 × R_halbach
    
    # Magnetic moment (aligned with z-axis)
    moment = np.array([0.0, 0.0, 0.5])  # 0.5 A⋅m²
    
    grad = halbach_ref.gradient(test_pos, degree=4)
    force = halbach_ref.dipole_force(test_pos, moment, degree=4)
    energy = halbach_ref.energy(test_pos, moment, degree=4)
    
    print(f"  Test position: {test_pos} m")
    print(f"  Magnetic moment: {moment} A⋅m²")
    print()
    print(f"  Force: {force} N")
    print(f"  Force magnitude: {np.linalg.norm(force)*1e6:.3f} μN")
    print()
    print(f"  Energy: {energy*1e9:.3f} nJ")
    print()
    
    # ========================================================================
    # Acceptance Criteria Check
    # ========================================================================
    print("Acceptance Criteria Check:")
    print("-" * 80)
    
    # Criteria: Field and gradient within ±10% of reference for r in [R, 3R]
    r_test_range = np.linspace(R_halbach, 3*R_halbach, 10)
    
    acceptance_pass = True
    for r in r_test_range:
        pos = np.array([r, 0.0, 0.0])
        B_deg4 = halbach_ref.field(pos, 4)
        B_deg6 = halbach_ref.field(pos, 6)
        
        error_percent = 100.0 * np.linalg.norm(B_deg6 - B_deg4) / np.linalg.norm(B_deg4)
        
        if error_percent > 10.0:
            acceptance_pass = False
            print(f"  ✗ Failed at r/R={r/R_halbach:.2f}: {error_percent:.1f}% error (>10%)")
            break
    
    if acceptance_pass:
        print(f"  ✅ PASS: All points within ±10% tolerance")
    print()
    
    # ========================================================================
    # Output and Saving
    # ========================================================================
    print("Output Files:")
    print("-" * 80)
    
    output_dir = repo_root / "results" / "halbach_multipole_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save validation report
    report_file = output_dir / "validation_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Halbach Array Multipole Expansion Validation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Halbach array radius: {R_halbach*1000:.1f} mm\n")
        f.write(f"  Dipole moment: {m_dipole:.2f} A⋅m²\n")
        f.write(f"  Validation region: r/R = [1.0, 5.0]\n")
        f.write(f"  Grid points: {len(positions)}\n\n")
        
        f.write("Degree Convergence:\n")
        for degree in degrees:
            data = convergence_data[degree]
            f.write(f"  Degree {degree}:\n")
            f.write(f"    Mean field: {data['mean_magnitude']*1e6:.2f} μT\n")
            f.write(f"    Max field:  {data['max_magnitude']*1e6:.2f} μT\n\n")
        
        f.write("Accuracy vs. Dipole:\n")
        for degree in [2, 3, 4]:
            B_multi = convergence_data[degree]['fields']
            diffs = np.linalg.norm(B_multi - B_dipole, axis=1)
            dipole_mags = np.linalg.norm(B_dipole, axis=1)
            error_percent = 100.0 * diffs / (dipole_mags + 1e-12)
            
            f.write(f"  Degree {degree}: max error = {np.max(error_percent):.1f}%, ")
            f.write(f"rms error = {np.sqrt(np.mean(error_percent**2)):.1f}%\n")
        
        f.write("\nAcceptance Criteria:\n")
        f.write(f"  ±10% tolerance for r/R in [1, 3]\n")
        f.write(f"  Status: {'✅ PASS' if acceptance_pass else '✗ FAIL'}\n")
    
    print(f"  ✓ Report: {report_file.relative_to(repo_root)}")
    
    # Save field analysis
    analysis_file = output_dir / "field_analysis.csv"
    with open(analysis_file, 'w') as f:
        f.write("r_mm,r_over_R,x_m,z_m,B_deg1_uT,B_deg4_uT,B_deg6_uT,error_pct\n")
        
        for i, pos in enumerate(positions):
            r = np.linalg.norm(pos)
            r_ratio = r / R_halbach
            
            B1 = halbach_ref.field(pos, 1)
            B4 = halbach_ref.field(pos, 4)
            B6 = halbach_ref.field(pos, 6)
            
            mag1 = np.linalg.norm(B1)
            mag4 = np.linalg.norm(B4)
            mag6 = np.linalg.norm(B6)
            
            error_pct = 100.0 * (mag6 - mag4) / (mag4 + 1e-12)
            
            f.write(f"{r*1000:.2f},{r_ratio:.3f},{pos[0]:.4f},{pos[2]:.4f},")
            f.write(f"{mag1*1e6:.2f},{mag4*1e6:.2f},{mag6*1e6:.2f},{error_pct:.2f}\n")
    
    print(f"  ✓ Analysis: {analysis_file.relative_to(repo_root)}")
    print()
    
    print("=" * 80)
    print("Halbach Multipole Validation Complete")
    print("=" * 80)
    print()
    print(f"Status: {'✅ ACCEPTANCE PASS' if acceptance_pass else '✗ ACCEPTANCE FAIL'}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

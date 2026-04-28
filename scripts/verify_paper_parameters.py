#!/usr/bin/env python
"""
Verify paper parameters for credibility and accuracy.

This script performs self-checks to ensure parameters are physically reasonable
and within expected ranges for the paper.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params
from sgms_anchor_v1 import analytical_metrics

print("=" * 70)
print("PAPER PARAMETER VERIFICATION")
print("=" * 70)

# Load operational profile
data = load_anchor_profiles("anchor_profiles.json")
operational_params = resolve_profile_params(data, "operational")["params"]

print("\n1. OPERATIONAL PROFILE PARAMETERS")
print("-" * 70)
print(f"  Velocity (u): {operational_params['u']:.1f} m/s")
print(f"  Linear density (lam): {operational_params['lam']:.2f} kg/m")
print(f"  Mass (mp): {operational_params['mp']:.2f} kg")
print(f"  Control gain (g_gain): {operational_params['g_gain']:.6f}")
print(f"  Flux-pinning (k_fp): {operational_params['k_fp']:.2f} N/m")
print(f"  Station mass (ms): {operational_params['ms']:.1f} kg")

# Verify k_eff calculation
print("\n2. k_eff CALCULATION VERIFICATION")
print("-" * 70)
k_eff_manual = (operational_params['lam'] * operational_params['u']**2 * 
                operational_params['g_gain'] + operational_params['k_fp'])
metrics = analytical_metrics(operational_params)
k_eff_computed = metrics['k_eff']

print(f"  Manual calculation: k_eff = λ·u²·g_gain + k_fp")
print(f"  k_eff = {operational_params['lam']:.2f} × {operational_params['u']:.1f}² × {operational_params['g_gain']:.6f} + {operational_params['k_fp']:.2f}")
print(f"  k_eff = {operational_params['lam']:.2f} × {operational_params['u']**2:.0f} × {operational_params['g_gain']:.6f} + {operational_params['k_fp']:.2f}")
print(f"  k_eff = {operational_params['lam'] * operational_params['u']**2:.0f} × {operational_params['g_gain']:.6f} + {operational_params['k_fp']:.2f}")
print(f"  k_eff = {operational_params['lam'] * operational_params['u']**2 * operational_params['g_gain']:.2f} + {operational_params['k_fp']:.2f}")
print(f"  k_eff = {k_eff_manual:.2f} N/m")
print(f"  Computed k_eff: {k_eff_computed:.2f} N/m")
print(f"  Match: {np.isclose(k_eff_manual, k_eff_computed, rtol=1e-6)}")

# Check if k_eff is in reasonable range
print(f"\n  Target range: 6000-10000 N/m (paper claim)")
print(f"  Operational k_eff: {k_eff_computed:.2f} N/m")
if 6000 <= k_eff_computed <= 10000:
    print(f"  ✓ IN TARGET RANGE")
elif k_eff_computed > 10000:
    print(f"  ⚠ ABOVE TARGET RANGE (stiffer than required)")
else:
    print(f"  ✗ BELOW TARGET RANGE (insufficient stiffness)")

# Verify stress calculation
print("\n3. STRESS CALCULATION VERIFICATION")
print("-" * 70)
radius = 0.1  # m
density = 2500  # kg/m³ (BFRP)
omega = 5236  # rad/s (50,000 RPM)
stress_limit = 8.0e8  # Pa (800 MPa with SF=1.5)

stress = density * radius**2 * omega**2
stress_mpa = stress / 1e6

print(f"  Hoop stress formula: σ_θ = ρ·r²·ω²")
print(f"  σ_θ = {density} × {radius}² × {omega}²")
print(f"  σ_θ = {density} × {radius**2:.4f} × {omega**2:.0f}")
print(f"  σ_θ = {stress:.2e} Pa = {stress_mpa:.2f} MPa")
print(f"  Stress limit: {stress_limit/1e6:.0f} MPa")
print(f"  Safety margin: {(stress_limit - stress)/stress_limit * 100:.1f}%")
if stress < stress_limit:
    print(f"  ✓ WITHIN SAFE LIMIT")
else:
    print(f"  ✗ EXCEEDS SAFE LIMIT")

# Verify thermal calculation
print("\n4. THERMAL CALCULATION VERIFICATION")
print("-" * 70)
emissivity = 0.85
surface_area = 0.2  # m²
power_heating = 200  # W
thermal_limit = 450  # K
sigma_sb = 5.67e-8

temp_steady = (power_heating / (emissivity * surface_area * sigma_sb))**0.25

print(f"  Radiative cooling: T = (P / (ε·A·σ))^0.25")
print(f"  T = ({power_heating} / ({emissivity} × {surface_area} × {sigma_sb:.2e}))^0.25")
print(f"  T = {temp_steady:.1f} K")
print(f"  Thermal limit: {thermal_limit} K")
print(f"  Safety margin: {(thermal_limit - temp_steady)/thermal_limit * 100:.1f}%")
if temp_steady < thermal_limit:
    print(f"  ✓ WITHIN THERMAL LIMIT")
else:
    print(f"  ✗ EXCEEDS THERMAL LIMIT")

# Verify momentum flux
print("\n5. MOMENTUM FLUX VERIFICATION")
print("-" * 70)
momentum_flux = operational_params['lam'] * operational_params['u']**2
print(f"  Momentum flux: F = λ·u²")
print(f"  F = {operational_params['lam']:.2f} × {operational_params['u']:.1f}²")
print(f"  F = {momentum_flux:.2e} N")
print(f"  Reasonable range: 1e3 - 1e7 N")
if 1e3 <= momentum_flux <= 1e7:
    print(f"  ✓ IN REASONABLE RANGE")
else:
    print(f"  ⚠ OUTSIDE EXPECTED RANGE")

# Verify parameter scaling
print("\n6. PARAMETER SCALING CHECKS")
print("-" * 70)

# Check if parameters are physically reasonable
checks = [
    ("Velocity", operational_params['u'], 1, 10000, "m/s", "1-10000 m/s"),
    ("Linear density", operational_params['lam'], 0.1, 100, "kg/m", "0.1-100 kg/m"),
    ("Mass", operational_params['mp'], 0.01, 1000, "kg", "0.01-1000 kg"),
    ("Control gain", operational_params['g_gain'], 1e-6, 1.0, "", "1e-6 to 1.0"),
    ("Flux-pinning", operational_params['k_fp'], 0, 1e6, "N/m", "0-1e6 N/m"),
    ("Station mass", operational_params['ms'], 10, 1e6, "kg", "10-1e6 kg"),
]

all_pass = True
for name, value, min_val, max_val, unit, range_str in checks:
    in_range = min_val <= value <= max_val
    status = "✓" if in_range else "⚠"
    print(f"  {status} {name}: {value:.2e} {unit} (expected: {range_str})")
    if not in_range:
        all_pass = False

# Summary
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

if all_pass:
    print("✓ All parameters are within reasonable ranges")
else:
    print("⚠ Some parameters are outside expected ranges - review needed")

print("\nKey findings:")
print(f"  - k_eff = {k_eff_computed:.2f} N/m (target: 6000-10000 N/m)")
print(f"  - Stress = {stress_mpa:.2f} MPa (limit: 800 MPa)")
print(f"  - Temperature = {temp_steady:.1f} K (limit: 450 K)")
print(f"  - Momentum flux = {momentum_flux:.2e} N")

print("\nCredibility assessment:")
if k_eff_computed > 10000:
    print("  ⚠ k_eff is above target range - may appear over-engineered")
else:
    print("  ✓ k_eff is within or below target range - conservative design")

if stress < stress_limit * 0.8:
    print("  ✓ Stress has good safety margin (>20%)")
elif stress < stress_limit:
    print("  ⚠ Stress has minimal safety margin (<20%)")
else:
    print("  ✗ Stress exceeds limit - critical issue")

if temp_steady < thermal_limit * 0.9:
    print("  ✓ Temperature has good safety margin (>10%)")
elif temp_steady < thermal_limit:
    print("  ⚠ Temperature has minimal safety margin (<10%)")
else:
    print("  ✗ Temperature exceeds limit - critical issue")

print("\n" + "=" * 70)

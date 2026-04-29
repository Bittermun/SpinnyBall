#!/usr/bin/env python
"""
Create comprehensive metrics table for different parameter configurations.

This script generates a table showing how different parameter values
affect key metrics like k_eff, force, period, and offset.

Usage:
    python scripts/parameter_metrics_table.py

Output:
    - Prints metrics table to console
    - Saves detailed results to parameter_metrics_table.json
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params
from src.sgms_anchor_v1 import analytical_metrics
import numpy as np

print("=" * 80)
print("COMPREHENSIVE PARAMETER METRICS TABLE")
print("=" * 80)

# Define parameter variations to test
test_configs = [
    # (name, params_overrides)
    ("Baseline (paper-baseline)", {"u": 10.0, "mp": 0.05, "g_gain": 0.05, "k_fp": 0.0}),
    ("Operational (current)", {"u": 1600.0, "mp": 8.0, "g_gain": 0.00014, "k_fp": 6000.0}),
    ("Low Velocity", {"u": 100.0, "mp": 8.0, "g_gain": 0.00014, "k_fp": 6000.0}),
    ("Mid Velocity", {"u": 600.0, "mp": 8.0, "g_gain": 0.00014, "k_fp": 6000.0}),
    ("High Velocity", {"u": 1200.0, "mp": 8.0, "g_gain": 0.00014, "k_fp": 6000.0}),
    ("Low Mass", {"u": 600.0, "mp": 0.5, "g_gain": 0.00014, "k_fp": 6000.0}),
    ("Mid Mass", {"u": 600.0, "mp": 4.0, "g_gain": 0.00014, "k_fp": 6000.0}),
    ("High Mass", {"u": 600.0, "mp": 8.0, "g_gain": 0.00014, "k_fp": 6000.0}),
]

# Base parameters (use operational as base)
try:
    if not Path("anchor_profiles.json").exists():
        print(f"ERROR: anchor_profiles.json not found")
        sys.exit(1)
    print(f"DEBUG: Loading anchor_profiles.json...")
    data = load_anchor_profiles("anchor_profiles.json")
    print(f"DEBUG: Resolving operational profile...")
    resolved = resolve_profile_params(data, "operational")
    base_params = resolved["params"]
    profile_meta = resolved["profile"]
    print(f"DEBUG: Profile resolved successfully.")
except (KeyError, ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as e:
    print(f"ERROR: Failed to resolve operational profile: {e}")
    sys.exit(1)

# Display profile information
print(f"\nBase Profile: {profile_meta.get('name', 'unknown')}")
print(f"  Category: {profile_meta.get('category', 'unspecified')}")
if profile_meta.get("material_profile"):
    print(f"  Material: {profile_meta['material_profile'].get('name', 'unknown')}")
if profile_meta.get("geometry_profile"):
    print(f"  Geometry: {profile_meta['geometry_profile'].get('name', 'unknown')}")
if profile_meta.get("environment_profile"):
    print(f"  Environment: {profile_meta['environment_profile'].get('name', 'unknown')}")
print()

# Table header
print(f"\n{'Config':<25} {'u (m/s)':>10} {'mp (kg)':>10} {'k_eff (N/m)':>15} {'Force (N)':>15} {'Period (s)':>12}")
print("-" * 90)

results = []

for name, overrides in test_configs:
    print(f"DEBUG: Testing config: {name}")
    params = base_params.copy()
    params.update(overrides)
    
    # Compute metrics
    print(f"DEBUG: Computing metrics for {name}...")
    metrics = analytical_metrics(params)
    print(f"DEBUG: Metrics computed.")
    
    k_eff = metrics["k_eff"]
    force = metrics["force_per_stream_n"]
    period = metrics["period_s"]
    
    print(f"{name:<25} {params['u']:>10.1f} {params['mp']:>10.2f} {k_eff:>15.2f} {force:>15.2f} {period:>12.3f}")
    
    results.append({
        "config": name,
        "u": params["u"],
        "mp": params["mp"],
        "k_eff": k_eff,
        "force": force,
        "period": period,
    })

print("\n" + "=" * 80)
print("MOMENTUM FLUX (F = λ·u²)")
print("=" * 80)

print(f"\n{'Config':<25} {'u (m/s)':>10} {'λ (kg/m)':>12} {'Momentum (N)':>15}")
print("-" * 65)
for r in results:
    momentum = r["config"].split()[0]  # Simplified
    lam = base_params["lam"]
    momentum_force = lam * r["u"]**2
    print(f"{r['config']:<25} {r['u']:>10.1f} {lam:>12.2f} {momentum_force:>15.2f}")

print("\n" + "=" * 80)
print("METRICS TABLE COMPLETE")
print("=" * 80)

# Save results
with open("parameter_metrics_table.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to parameter_metrics_table.json")

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
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params
from sgms_anchor_v1 import analytical_metrics
import numpy as np
import json

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
data = load_anchor_profiles("anchor_profiles.json")
base_params = resolve_profile_params(data, "operational")["params"]

# Table header
print(f"\n{'Config':<25} {'u (m/s)':>10} {'mp (kg)':>10} {'k_eff (N/m)':>15} {'Force (N)':>15} {'Period (s)':>12}")
print("-" * 90)

results = []

for name, overrides in test_configs:
    params = base_params.copy()
    params.update(overrides)
    
    # Compute metrics
    metrics = analytical_metrics(params)
    
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

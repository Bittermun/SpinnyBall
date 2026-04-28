#!/usr/bin/env python
"""
Run sensitivity analysis to find optimal parameters.

This script performs Sobol sensitivity analysis on anchor parameters
and identifies optimal ranges for key metrics like k_eff.

Usage:
    python scripts/analyze_sensitivity.py

Output:
    - Prints sensitivity analysis results
    - Saves optimal parameters to optimal_parameters.json
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params
from sgms_anchor_sensitivity import run_sobol_sensitivity
from sgms_anchor_v1 import analytical_metrics, simulate_anchor
from sgms_anchor_pipeline import export_fmeca_json
import numpy as np
import json

print("=" * 70)
print("SENSITIVITY ANALYSIS FOR OPTIMAL PARAMETERS")
print("=" * 70)

# Load operational profile as baseline
try:
    if not Path("anchor_profiles.json").exists():
        print(f"ERROR: anchor_profiles.json not found")
        sys.exit(1)
    data = load_anchor_profiles("anchor_profiles.json")
    resolved = resolve_profile_params(data, "operational")
    baseline_params = resolved["params"]
    profile_meta = resolved["profile"]
except (KeyError, ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as e:
    print(f"ERROR: Failed to resolve operational profile: {e}")
    sys.exit(1)

print(f"\nBase Profile: {profile_meta.get('name', 'unknown')}")
print(f"  Category: {profile_meta.get('category', 'unspecified')}")
if profile_meta.get("material_profile"):
    print(f"  Material: {profile_meta['material_profile'].get('name', 'unknown')}")
if profile_meta.get("geometry_profile"):
    print(f"  Geometry: {profile_meta['geometry_profile'].get('name', 'unknown')}")
if profile_meta.get("environment_profile"):
    print(f"  Environment: {profile_meta['environment_profile'].get('name', 'unknown')}")

print(f"\nBaseline Parameters:")
print(f"  Mass (mp): {baseline_params['mp']:.2f} kg")
print(f"  Velocity (u): {baseline_params['u']:.2f} m/s")
print(f"  Linear density (lam): {baseline_params['lam']:.2f} kg/m")
print(f"  Control gain (g_gain): {baseline_params['g_gain']:.6f}")
print(f"  Flux-pinning (k_fp): {baseline_params['k_fp']:.2f} N/m")

# Run Sobol sensitivity analysis
print(f"\n{'='*70}")
print("RUNNING SOBOL SENSITIVITY ANALYSIS")
print(f"{'='*70}")

problem = {
    "num_vars": 5,
    "names": ["u", "g_gain", "eps", "lam", "mp"],
    "bounds": [
        [5.0, 1600.0],    # u (m/s)
        [0.0001, 0.001],  # g_gain (tuned range around operational)
        [0.0, 1e-3],      # eps
        [0.1, 20.0],      # lam (kg/m)
        [0.05, 8.0],      # mp (kg)
    ],
}

N = 256
print(f"\nRunning Sobol sensitivity analysis with {N} samples...")
result = run_sobol_sensitivity(problem=problem, N=N, base_params=baseline_params)
print(f"Completed sensitivity analysis")

print(f"\n{'='*70}")
print("SENSITIVITY ANALYSIS RESULTS")
print(f"{'='*70}")

si = result["indices"]
print(f"\nFirst-order Sobol indices (parameter importance):")
for i, name in enumerate(problem["names"]):
    s1 = si["k_eff"]["S1"][i] if i < len(si["k_eff"]["S1"]) else 0
    print(f"  {name:10s}: {s1:.4f}")

# Find optimal parameters for k_eff
print(f"\n{'='*70}")
print("FINDING OPTIMAL PARAMETERS FOR k_eff")
print(f"{'='*70}")

samples = result["samples"]
outputs = result["outputs"]
k_eff_values = outputs["k_eff"]

# Find indices where k_eff is in optimal range (6000-10000 N/m)
optimal_mask = (k_eff_values >= 6000) & (k_eff_values <= 10000)
optimal_indices = np.where(optimal_mask)[0]

print(f"\nSamples with k_eff in optimal range [6000, 10000] N/m: {len(optimal_indices)}/{len(k_eff_values)}")

if len(optimal_indices) > 0:
    print(f"\nOptimal parameter ranges (from samples in target range):")
    optimal_samples = samples[optimal_indices]
    for i, name in enumerate(problem["names"]):
        values = optimal_samples[:, i]
        print(f"  {name:10s}: [{np.min(values):.4f}, {np.max(values):.4f}] (mean: {np.mean(values):.4f})")
    
    # Find best sample (closest to 8000 N/m midpoint)
    target_k_eff = 8000.0
    distances = np.abs(k_eff_values[optimal_indices] - target_k_eff)
    best_idx = optimal_indices[np.argmin(distances)]
    
    print(f"\nBest sample (k_eff = {k_eff_values[best_idx]:.2f} N/m):")
    for i, name in enumerate(problem["names"]):
        print(f"  {name:10s}: {samples[best_idx, i]:.4f}")
else:
    print(f"\nNo samples found in optimal range. Finding closest...")
    target_k_eff = 8000.0
    distances = np.abs(k_eff_values - target_k_eff)
    best_idx = np.argmin(distances)
    print(f"\nClosest sample (k_eff = {k_eff_values[best_idx]:.2f} N/m):")
    for i, name in enumerate(problem["names"]):
        print(f"  {name:10s}: {samples[best_idx, i]:.4f}")

print(f"\n{'='*70}")
print("SENSITIVITY ANALYSIS COMPLETE")
print(f"{'='*70}")

# Save results
results_summary = {
    "baseline": baseline_params,
    "sensitivity": {name: float(si["k_eff"]["S1"][i]) if i < len(si["k_eff"]["S1"]) else 0 for i, name in enumerate(problem["names"])},
    "best_sample": {
        "k_eff": float(k_eff_values[best_idx]),
        "parameters": {name: float(samples[best_idx, i]) for i, name in enumerate(problem["names"])},
    },
}

with open("optimal_parameters.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to optimal_parameters.json")

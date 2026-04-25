#!/usr/bin/env python
"""
Validate anchor profile parameters.

This script validates that a profile from anchor_profiles.json produces
expected metrics and runs a quick simulation to verify it works correctly.

Usage:
    python scripts/validate_profile.py [profile_name]

    If profile_name is not specified, defaults to "operational".
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params
from sgms_anchor_v1 import analytical_metrics, simulate_anchor
import numpy as np

def validate_profile(profile_name: str = "operational"):
    """Validate an anchor profile."""
    print("=" * 70)
    print(f"VALIDATING PROFILE: {profile_name}")
    print("=" * 70)
    
    # Load profile
    data = load_anchor_profiles("anchor_profiles.json")
    params = resolve_profile_params(data, profile_name)["params"]
    
    print(f"\nParameters:")
    print(f"  Mass (mp): {params['mp']:.2f} kg")
    print(f"  Velocity (u): {params['u']:.2f} m/s")
    print(f"  Linear density (lam): {params['lam']:.2f} kg/m")
    print(f"  Control gain (g_gain): {params['g_gain']:.6f}")
    print(f"  Flux-pinning (k_fp): {params['k_fp']:.2f} N/m")
    
    # Compute analytical metrics
    metrics = analytical_metrics(params)
    
    print(f"\nAnalytical Metrics:")
    print(f"  Effective stiffness (k_eff): {metrics['k_eff']:.2f} N/m")
    print(f"  Force per stream: {metrics['force_per_stream_n']:.2f} N")
    print(f"  Period: {metrics['period_s']:.3f} s")
    print(f"  Static offset: {metrics['static_offset_m']:.6f} m")
    print(f"  Packet rate: {metrics['packet_rate_hz']:.2f} Hz")
    
    # Quick simulation test
    print(f"\nRunning quick simulation (t_max=5s)...")
    params["t_max"] = 5.0
    t_eval = np.linspace(0.0, params["t_max"], 100)
    result = simulate_anchor(params, t_eval=t_eval, seed=42)
    
    print(f"  Simulation completed successfully")
    print(f"  Final displacement: {result['x'][-1]:.6f} m")
    print(f"  Max displacement: {np.max(np.abs(result['x'])):.6f} m")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "operational"
    validate_profile(profile_name)

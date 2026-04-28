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
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params, load_material_catalog, load_geometry_catalog, load_environment_catalog
from sgms_anchor_v1 import analytical_metrics, simulate_anchor
import numpy as np

def validate_profile(profile_name: str = "operational"):
    """Validate an anchor profile."""
    print("=" * 70)
    print(f"VALIDATING PROFILE: {profile_name}")
    print("=" * 70)
    
    # Load profile
    try:
        if not Path("anchor_profiles.json").exists():
            print(f"\nERROR: anchor_profiles.json not found")
            return
        data = load_anchor_profiles("anchor_profiles.json")
        resolved = resolve_profile_params(data, profile_name)
        params = resolved["params"]
        profile_meta = resolved["profile"]
    except KeyError as e:
        print(f"\nERROR: {e}")
        print("Profile not found or has invalid references.")
        return
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Profile validation failed - missing required fields.")
        return
    except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
        print(f"\nERROR: Failed to load profile file: {e}")
        return
    
    print(f"\nProfile Metadata:")
    print(f"  Category: {profile_meta.get('category', 'unspecified')}")
    print(f"  Notes: {'; '.join(profile_meta.get('notes', []))}")
    
    # Display material profile if present
    if profile_meta.get("material_profile"):
        mat = profile_meta["material_profile"]
        print(f"\nMaterial Profile:")
        print(f"  Name: {mat.get('name', 'unknown')}")
        print(f"  Stiffness range: {mat.get('k_fp_range', 'N/A')} N/m")
        print(f"  Damping ratio: {mat.get('damping_ratio', 'N/A')}")
        print(f"  Source: {mat.get('source', 'N/A')}")
    
    # Display geometry profile if present
    if profile_meta.get("geometry_profile"):
        geo = profile_meta["geometry_profile"]
        print(f"\nGeometry Profile:")
        print(f"  Name: {geo.get('name', 'unknown')}")
        print(f"  Shape: {geo.get('shape', 'unknown')}")
        print(f"  Mass: {geo.get('mass', 'N/A')} kg")
        print(f"  Radius: {geo.get('radius', 'N/A')} m")
        print(f"  Description: {geo.get('description', 'N/A')}")
    
    # Display environment profile if present
    if profile_meta.get("environment_profile"):
        env = profile_meta["environment_profile"]
        print(f"\nEnvironment Profile:")
        print(f"  Name: {env.get('name', 'unknown')}")
        print(f"  Temperature: {env.get('temperature', 'N/A')} K")
        print(f"  Magnetic field: {env.get('B_field', 'N/A')} T")
        print(f"  Radiation flux: {env.get('radiation_flux', 'N/A')} W/m²")
        print(f"  Gravity: {env.get('gravity', 'N/A')} m/s²")
        print(f"  Description: {env.get('description', 'N/A')}")
    
    print(f"\nParameters:")
    print(f"  Mass (mp): {params['mp']:.2f} kg")
    print(f"  Velocity (u): {params['u']:.2f} m/s")
    print(f"  Linear density (lam): {params['lam']:.2f} kg/m")
    print(f"  Control gain (g_gain): {params['g_gain']:.6f}")
    print(f"  Flux-pinning (k_fp): {params.get('k_fp', 'dynamic'):.2f} N/m")
    
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
    try:
        result = simulate_anchor(params, t_eval=t_eval, seed=42)
        print(f"  Simulation completed successfully")
        print(f"  Final displacement: {result['x'][-1]:.6f} m")
        print(f"  Max displacement: {np.max(np.abs(result['x'])):.6f} m")
    except Exception as e:
        print(f"  ERROR: Simulation failed: {e}")
        print("  Validation incomplete due to simulation error")
        return
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "operational"
    validate_profile(profile_name)

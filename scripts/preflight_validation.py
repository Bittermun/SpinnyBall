"""
Pre-flight validation script for comprehensive sweep campaign.
Validates profile system and runs quick smoke tests.
"""

import sys
from pathlib import Path

# Test profile loading
print("=== Pre-flight Validation ===\n")

print("1. Testing profile system loading...")
try:
    from sgms_anchor_profiles import (
        load_anchor_profiles,
        load_material_catalog,
        load_geometry_catalog,
        load_environment_catalog,
        resolve_profile_params,
    )
    
    # Load all catalogs
    anchor_profiles = load_anchor_profiles("anchor_profiles.json")
    material_catalog = load_material_catalog("paper_model/gdbco_apc_catalog.json")
    geometry_catalog = load_geometry_catalog("geometry_profiles.json")
    environment_catalog = load_environment_catalog("environment_profiles.json")
    
    print(f"   ✓ Loaded {len(anchor_profiles['profiles'])} anchor profiles")
    print(f"   ✓ Loaded {len(material_catalog.get('material_profiles', {}))} material profiles")
    print(f"   ✓ Loaded {len(geometry_catalog['geometry_profiles'])} geometry profiles")
    print(f"   ✓ Loaded {len(environment_catalog['environment_profiles'])} environment profiles")
    
    # Test resolving each profile
    profile_names = [p['name'] for p in anchor_profiles['profiles']]
    print(f"\n2. Testing profile resolution for all {len(profile_names)} profiles...")
    for name in profile_names:
        try:
            resolved = resolve_profile_params(anchor_profiles, name)
            print(f"   ✓ Resolved profile: {name}")
        except Exception as e:
            print(f"   ✗ Failed to resolve {name}: {e}")
            sys.exit(1)
    
    print("\n3. Profile system validation: PASSED ✓")
    
except Exception as e:
    print(f"\n✗ Profile system validation FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing import of sweep modules...")
try:
    import sweep_fault_cascade
    import sweep_latency_eta_ind
    print("   ✓ sweep_fault_cascade imported")
    print("   ✓ sweep_latency_eta_ind imported")
except Exception as e:
    print(f"   ✗ Failed to import sweep modules: {e}")
    sys.exit(1)

print("\n5. Testing import of main simulation modules...")
try:
    import sgms_anchor_logistics
    import sgms_anchor_sensitivity
    import lob_scaling
    print("   ✓ sgms_anchor_logistics imported")
    print("   ✓ sgms_anchor_sensitivity imported")
    print("   ✓ lob_scaling imported")
except Exception as e:
    print(f"   ✗ Failed to import simulation modules: {e}")
    sys.exit(1)

print("\n=== All pre-flight validations PASSED ✓ ===")
print("Ready to launch comprehensive sweep campaign.")

"""
Compatibility shim for older scripts importing `sgms_anchor_profiles` from the
project root.

The canonical implementation lives in `src/sgms_anchor_profiles.py`.
"""
from __future__ import annotations

from src.sgms_anchor_profiles import *  # noqa: F403
from src.sgms_anchor_profiles import (  # noqa: F401
    load_anchor_profiles,
    resolve_profile_params,
    export_profile_summary_csv,
    _validate_environment_profile,
    _validate_geometry_profile,
    _validate_material_profile,
    load_environment_catalog,
    load_geometry_catalog,
    load_material_catalog,
)

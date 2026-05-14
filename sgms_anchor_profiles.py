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
)

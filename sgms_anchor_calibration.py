"""
Calibration and provenance helpers for anchor parameters.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_anchor_calibration(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_calibrated_params(
    calibration_data: dict,
    params: dict,
    profile_name: str | None = None,
) -> dict:
    resolved = params.copy()
    provenance = {}

    entries = {}
    entries.update(calibration_data.get("defaults", {}))
    if profile_name:
        entries.update(calibration_data.get("profiles", {}).get(profile_name, {}))

    for key, meta in entries.items():
        if key not in resolved:
            resolved[key] = meta.get("value")
        provenance[key] = {
            "status": meta.get("status", "unknown"),
            "source": meta.get("source", "unspecified"),
            "calibrated_value": meta.get("value"),
        }

    return {
        "params": resolved,
        "provenance": provenance,
    }

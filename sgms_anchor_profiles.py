"""
Named parameter profiles and provenance helpers for anchor experiments.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sgms_anchor_v1 import DEFAULT_PARAMS


def load_anchor_profiles(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _profile_lookup(profile_data: dict) -> dict[str, dict]:
    return {profile["name"]: profile for profile in profile_data.get("profiles", [])}


def resolve_profile_params(
    profile_data: dict,
    profile_name: str,
    overrides: dict | None = None,
    base_params: dict | None = None,
) -> dict:
    lookup = _profile_lookup(profile_data)
    if profile_name not in lookup:
        raise KeyError(f"Unknown profile: {profile_name}")

    profile = lookup[profile_name]
    params = DEFAULT_PARAMS.copy() if base_params is None else base_params.copy()
    params.update(profile.get("params", {}))
    if overrides:
        params.update(overrides)

    return {
        "profile": {
            "name": profile["name"],
            "category": profile.get("category", "unspecified"),
            "notes": profile.get("notes", []),
            "provenance": profile.get("provenance", {}),
        },
        "params": params,
    }


def build_profile_summary_rows(experiments: list[dict]) -> list[dict]:
    rows = []
    for experiment in experiments:
        profile = experiment.get("profile", {})
        params = experiment.get("params", {})
        rows.append(
            {
                "experiment": experiment["name"],
                "profile": profile.get("name", "direct-defaults"),
                "category": profile.get("category", "unspecified"),
                "u": params.get("u"),
                "lam": params.get("lam"),
                "g_gain": params.get("g_gain"),
                "eps": params.get("eps"),
                "c_damp": params.get("c_damp"),
                "notes": " | ".join(profile.get("notes", [])),
            }
        )
    return rows


def export_profile_summary_csv(rows: list[dict], filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = list(rows[0].keys())
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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


def load_material_catalog(path: str | Path = "paper_model/gdbco_apc_catalog.json") -> dict:
    """Load material catalog from JSON file.
    
    Args:
        path: Path to material catalog JSON file
        
    Returns:
        Dictionary with material profiles
    """
    path = Path(path)
    if not path.exists():
        return {"material_profiles": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_geometry_catalog(path: str | Path = "geometry_profiles.json") -> dict:
    """Load geometry catalog from JSON file.
    
    Args:
        path: Path to geometry catalog JSON file
        
    Returns:
        Dictionary with geometry profiles
    """
    path = Path(path)
    if not path.exists():
        return {"geometry_profiles": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_environment_catalog(path: str | Path = "environment_profiles.json") -> dict:
    """Load environment catalog from JSON file.
    
    Args:
        path: Path to environment catalog JSON file
        
    Returns:
        Dictionary with environment profiles
    """
    path = Path(path)
    if not path.exists():
        return {"environment_profiles": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _profile_lookup(profile_data: dict) -> dict[str, dict]:
    return {profile["name"]: profile for profile in profile_data.get("profiles", [])}


def _material_lookup(material_data: dict) -> dict[str, dict]:
    return {name: profile for name, profile in material_data.get("material_profiles", {}).items()}


def _geometry_lookup(geometry_data: dict) -> dict[str, dict]:
    return {name: profile for name, profile in geometry_data.get("geometry_profiles", {}).items()}


def _environment_lookup(environment_data: dict) -> dict[str, dict]:
    return {name: profile for name, profile in environment_data.get("environment_profiles", {}).items()}


def _validate_material_profile(material_profile: dict) -> None:
    """Validate that required material profile fields are present and valid."""
    required_fields = ["name"]
    for field in required_fields:
        if field not in material_profile:
            raise ValueError(f"Material profile missing required field: {field}")
    
    if "k_fp_range" in material_profile:
        k_fp_range = material_profile["k_fp_range"]
        if not isinstance(k_fp_range, list) or len(k_fp_range) != 2:
            raise ValueError("Material profile k_fp_range must be a list of 2 numbers")
        if not all(isinstance(x, (int, float)) for x in k_fp_range):
            raise ValueError("Material profile k_fp_range values must be numbers")
        if k_fp_range[0] < 0 or k_fp_range[1] < 0:
            raise ValueError("Material profile k_fp_range values must be non-negative")


def _validate_geometry_profile(geometry_profile: dict) -> None:
    """Validate that required geometry profile fields are present and valid."""
    required_fields = ["name", "shape", "mass", "radius"]
    for field in required_fields:
        if field not in geometry_profile:
            raise ValueError(f"Geometry profile missing required field: {field}")
    
    if not isinstance(geometry_profile["mass"], (int, float)):
        raise ValueError("Geometry profile mass must be a number")
    if geometry_profile["mass"] <= 0:
        raise ValueError("Geometry profile mass must be positive")
    
    if not isinstance(geometry_profile["radius"], (int, float)):
        raise ValueError("Geometry profile radius must be a number")
    if geometry_profile["radius"] <= 0:
        raise ValueError("Geometry profile radius must be positive")


def _validate_environment_profile(environment_profile: dict) -> None:
    """Validate that required environment profile fields are present and valid."""
    required_fields = ["name", "temperature", "B_field"]
    for field in required_fields:
        if field not in environment_profile:
            raise ValueError(f"Environment profile missing required field: {field}")
    
    if not isinstance(environment_profile["temperature"], (int, float)):
        raise ValueError("Environment profile temperature must be a number")
    if environment_profile["temperature"] < 0:
        raise ValueError("Environment profile temperature must be non-negative (in Kelvin)")
    
    if not isinstance(environment_profile["B_field"], (int, float)):
        raise ValueError("Environment profile B_field must be a number")
    if environment_profile["B_field"] < 0:
        raise ValueError("Environment profile B_field must be non-negative (in Tesla)")


def resolve_profile_params(
    profile_data: dict,
    profile_name: str,
    overrides: dict | None = None,
    base_params: dict | None = None,
    material_catalog_path: str | Path = "paper_model/gdbco_apc_catalog.json",
    geometry_catalog_path: str | Path = "geometry_profiles.json",
    environment_catalog_path: str | Path = "environment_profiles.json",
) -> dict:
    lookup = _profile_lookup(profile_data)
    if profile_name not in lookup:
        raise KeyError(f"Unknown profile: {profile_name}")

    profile = lookup[profile_name]
    params = DEFAULT_PARAMS.copy() if base_params is None else base_params.copy()
    params.update(profile.get("params", {}))
    if overrides:
        params.update(overrides)

    # Resolve material profile if specified
    material_profile = None
    if profile.get("material_profile"):
        material_data = load_material_catalog(material_catalog_path)
        material_lookup = _material_lookup(material_data)
        material_name = profile["material_profile"]
        if material_name in material_lookup:
            material_profile = material_lookup[material_name]
            _validate_material_profile(material_profile)
        else:
            raise KeyError(f"Unknown material profile: {material_name}")

    # Resolve geometry profile if specified
    geometry_profile = None
    if profile.get("geometry_profile"):
        geometry_data = load_geometry_catalog(geometry_catalog_path)
        geometry_lookup = _geometry_lookup(geometry_data)
        geometry_name = profile["geometry_profile"]
        if geometry_name in geometry_lookup:
            geometry_profile = geometry_lookup[geometry_name]
            _validate_geometry_profile(geometry_profile)
        else:
            raise KeyError(f"Unknown geometry profile: {geometry_name}")

    # Resolve environment profile if specified
    environment_profile = None
    if profile.get("environment_profile"):
        environment_data = load_environment_catalog(environment_catalog_path)
        environment_lookup = _environment_lookup(environment_data)
        environment_name = profile["environment_profile"]
        if environment_name in environment_lookup:
            environment_profile = environment_lookup[environment_name]
            _validate_environment_profile(environment_profile)
            # Override params with environment values, but allow experiment params to take precedence
            # Check both profile params and experiment overrides
            profile_params = profile.get("params", {})
            overrides = overrides or {}
            if "temperature" in environment_profile and "temperature" not in profile_params and "temperature" not in overrides:
                params["temperature"] = environment_profile["temperature"]
            if "B_field" in environment_profile and "B_field" not in profile_params and "B_field" not in overrides:
                params["B_field"] = environment_profile["B_field"]
        else:
            raise KeyError(f"Unknown environment profile: {environment_name}")

    return {
        "profile": {
            "name": profile["name"],
            "category": profile.get("category", "unspecified"),
            "notes": profile.get("notes", []),
            "provenance": profile.get("provenance", {}),
            "material_profile": material_profile,
            "geometry_profile": geometry_profile,
            "environment_profile": environment_profile,
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

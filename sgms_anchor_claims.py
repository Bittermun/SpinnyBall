"""
Claim-level and validation-decision helpers for anchor runs.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_anchor_claims(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_claim_context(claim_data: dict, profile_name: str | None = None) -> dict:
    profile_claim = {}
    if profile_name:
        profile_claim = claim_data.get("profiles", {}).get(profile_name, {})
    return {
        "phase_decision": claim_data.get("phase_decision", {}),
        "profile_claim": profile_claim,
    }

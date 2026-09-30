#!/usr/bin/env python3
"""
Autonomous one-shot research run orchestrator.

This script executes a full simulation campaign with:
1. Preflight validation
2. Coarse T1/T3 sweeps
3. Automatic refinement around interesting boundaries
4. Plausibility gates and provenance-aware artifact export
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .sweep_fault_cascade import analyze_containment_threshold, plot_t3_results, run_t3_sweep
    from .sweep_latency_eta_ind import analyze_stability_boundary, plot_t1_results, run_t1_sweep
except ImportError:
    # Support direct script execution: python scripts/autonomous_research_run.py
    from sweep_fault_cascade import analyze_containment_threshold, plot_t3_results, run_t3_sweep
    from sweep_latency_eta_ind import analyze_stability_boundary, plot_t1_results, run_t1_sweep


SUCCESS_THRESHOLD = 0.95
CASCADE_LIMIT = 1e-6
CONTAINMENT_TARGET = 0.95


@dataclass(frozen=True)
class SweepPreset:
    """Preset parameters for run scale."""

    t1_latency_points: int
    t1_eta_points: int
    t1_realizations: int
    t3_fault_points: int
    t3_realizations: int
    t3_time_horizon_s: float


PRESETS: dict[str, SweepPreset] = {
    "quick": SweepPreset(
        t1_latency_points=4,
        t1_eta_points=3,
        t1_realizations=10,
        t3_fault_points=5,
        t3_realizations=20,
        t3_time_horizon_s=600.0,
    ),
    "standard": SweepPreset(
        t1_latency_points=10,
        t1_eta_points=8,
        t1_realizations=50,
        t3_fault_points=8,
        t3_realizations=100,
        t3_time_horizon_s=3600.0,
    ),
    "research": SweepPreset(
        t1_latency_points=16,
        t1_eta_points=12,
        t1_realizations=120,
        t3_fault_points=12,
        t3_realizations=200,
        t3_time_horizon_s=3600.0,
    ),
}


def to_jsonable(data: Any) -> Any:
    """Convert numpy-heavy structures into JSON-safe objects."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, dict):
        return {str(k): to_jsonable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_jsonable(v) for v in data]
    return data


def save_json(path: Path, payload: Any) -> None:
    """Write JSON payload with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def run_preflight_validation(repo_root: Path) -> None:
    """Execute the existing preflight validation script."""
    script_path = repo_root / "scripts" / "preflight_validation.py"
    command = [sys.executable, str(script_path)]
    completed = subprocess.run(command, cwd=repo_root, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Preflight validation failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def derive_t1_refinement_bounds(
    latency_values: np.ndarray,
    eta_values: np.ndarray,
    success_rate_grid: np.ndarray,
    success_threshold: float = SUCCESS_THRESHOLD,
    latency_padding_ms: float = 5.0,
    eta_padding: float = 0.01,
) -> tuple[float, float, float, float] | None:
    """
    Find refinement bounds around the T1 transition surface near success_threshold.
    """
    crossing_latencies: list[float] = []
    crossing_etas: list[float] = []

    for eta_idx, eta_value in enumerate(eta_values):
        row = success_rate_grid[eta_idx]
        for latency_idx in range(len(latency_values) - 1):
            left_ok = row[latency_idx] >= success_threshold
            right_ok = row[latency_idx + 1] >= success_threshold
            if left_ok != right_ok:
                crossing_latencies.extend([latency_values[latency_idx], latency_values[latency_idx + 1]])
                crossing_etas.append(eta_value)

    if not crossing_latencies:
        return None

    latency_min = max(float(np.min(latency_values)), min(crossing_latencies) - latency_padding_ms)
    latency_max = min(float(np.max(latency_values)), max(crossing_latencies) + latency_padding_ms)
    eta_min = max(float(np.min(eta_values)), min(crossing_etas) - eta_padding)
    eta_max = min(float(np.max(eta_values)), max(crossing_etas) + eta_padding)

    if latency_max <= latency_min or eta_max <= eta_min:
        return None
    return latency_min, latency_max, eta_min, eta_max


def derive_t3_refinement_bounds(
    fault_rates: np.ndarray,
    cascade_probability: np.ndarray,
    containment_rate: np.ndarray,
    cascade_limit: float = CASCADE_LIMIT,
    containment_target: float = CONTAINMENT_TARGET,
) -> tuple[float, float] | None:
    """
    Find refinement bounds around first significant T3 transition.
    """
    transition_indices = np.where(
        (cascade_probability > cascade_limit) | (containment_rate < containment_target)
    )[0]

    if transition_indices.size == 0:
        return None

    center = int(transition_indices[0])
    left = max(0, center - 1)
    right = min(len(fault_rates) - 1, center + 1)

    low = float(fault_rates[left])
    high = float(fault_rates[right])
    if high <= low:
        return None
    return low, high


def needs_guaranteed_fault_probe(
    fault_events_total_per_point: list[int] | np.ndarray,
    sanity_warnings: list[str] | None,
) -> bool:
    """Detect whether low-rate region is under-excited and needs a guaranteed-fault probe."""
    if len(fault_events_total_per_point) == 0:
        return True
    if min(int(v) for v in fault_events_total_per_point) == 0:
        return True
    if sanity_warnings and any(warning for warning in sanity_warnings):
        return True
    return False


def run_t1_campaign(preset: SweepPreset, output_dir: Path) -> dict[str, Any]:
    """Run coarse T1 then refine near transition boundary."""
    coarse = run_t1_sweep(
        latency_range=(5.0, 50.0),
        eta_ind_range=(0.8, 0.95),
        n_latency_points=preset.t1_latency_points,
        n_eta_points=preset.t1_eta_points,
        n_realizations_per_point=preset.t1_realizations,
    )
    coarse_analysis = analyze_stability_boundary(coarse)
    plot_t1_results(coarse, output_file=str(output_dir / "t1_coarse.png"))
    save_json(output_dir / "t1_coarse.json", {"results": coarse, "analysis": coarse_analysis})

    refine_bounds = derive_t1_refinement_bounds(
        latency_values=coarse["latency_values"],
        eta_values=coarse["eta_ind_values"],
        success_rate_grid=coarse["success_rate_grid"],
    )

    refined_payload: dict[str, Any] | None = None
    if refine_bounds is not None:
        latency_min, latency_max, eta_min, eta_max = refine_bounds
        refined = run_t1_sweep(
            latency_range=(latency_min, latency_max),
            eta_ind_range=(eta_min, eta_max),
            n_latency_points=max(6, preset.t1_latency_points),
            n_eta_points=max(6, preset.t1_eta_points),
            n_realizations_per_point=max(2 * preset.t1_realizations, 50),
        )
        refined_analysis = analyze_stability_boundary(refined)
        plot_t1_results(refined, output_file=str(output_dir / "t1_refined.png"))
        refined_payload = {"bounds": refine_bounds, "results": refined, "analysis": refined_analysis}
        save_json(output_dir / "t1_refined.json", refined_payload)

    return {
        "coarse": {"analysis": coarse_analysis},
        "refined": refined_payload["analysis"] if refined_payload else None,
        "refinement_bounds": refine_bounds,
    }


def run_t3_campaign(preset: SweepPreset, output_dir: Path) -> dict[str, Any]:
    """Run coarse T3, probe plausibility, then refine around threshold."""
    coarse = run_t3_sweep(
        fault_rate_range=(1e-6, 1e-3),
        n_fault_rate_points=preset.t3_fault_points,
        cascade_threshold=1.05,
        containment_threshold=2,
        n_nodes=10,
        n_realizations_per_point=preset.t3_realizations,
        time_horizon=preset.t3_time_horizon_s,
        fault_injection_mode="rate",
    )
    coarse_analysis = analyze_containment_threshold(coarse)
    plot_t3_results(coarse, output_file=str(output_dir / "t3_coarse.png"))
    save_json(output_dir / "t3_coarse.json", {"results": coarse, "analysis": coarse_analysis})

    guaranteed_probe_payload: dict[str, Any] | None = None
    if needs_guaranteed_fault_probe(
        coarse.get("fault_events_total_per_point", []),
        coarse.get("sanity_warnings", []),
    ):
        guaranteed_probe = run_t3_sweep(
            fault_rate_range=(1e-6, 1e-3),
            n_fault_rate_points=preset.t3_fault_points,
            cascade_threshold=1.05,
            containment_threshold=2,
            n_nodes=10,
            n_realizations_per_point=max(50, preset.t3_realizations // 2),
            time_horizon=preset.t3_time_horizon_s,
            fault_injection_mode="guaranteed",
            n_guaranteed_faults=1,
        )
        guaranteed_analysis = analyze_containment_threshold(guaranteed_probe)
        guaranteed_probe_payload = {"results": guaranteed_probe, "analysis": guaranteed_analysis}
        save_json(output_dir / "t3_guaranteed_probe.json", guaranteed_probe_payload)

    refine_bounds = derive_t3_refinement_bounds(
        fault_rates=coarse["fault_rates"],
        cascade_probability=coarse["cascade_probability"],
        containment_rate=coarse["containment_rate"],
    )

    refined_payload: dict[str, Any] | None = None
    if refine_bounds is not None:
        low, high = refine_bounds
        refined = run_t3_sweep(
            fault_rate_range=(low, high),
            n_fault_rate_points=max(8, preset.t3_fault_points),
            cascade_threshold=1.05,
            containment_threshold=2,
            n_nodes=10,
            n_realizations_per_point=max(2 * preset.t3_realizations, 100),
            time_horizon=preset.t3_time_horizon_s,
            fault_injection_mode="rate",
        )
        refined_analysis = analyze_containment_threshold(refined)
        plot_t3_results(refined, output_file=str(output_dir / "t3_refined.png"))
        refined_payload = {"bounds": refine_bounds, "results": refined, "analysis": refined_analysis}
        save_json(output_dir / "t3_refined.json", refined_payload)

    return {
        "coarse": {"analysis": coarse_analysis},
        "guaranteed_probe": guaranteed_probe_payload["analysis"] if guaranteed_probe_payload else None,
        "refined": refined_payload["analysis"] if refined_payload else None,
        "refinement_bounds": refine_bounds,
    }


def get_git_revision(repo_root: Path) -> str | None:
    """Resolve git revision if available."""
    completed = subprocess.run(
        ["git", "--no-pager", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Run autonomous SpinnyBall research campaign")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="standard",
        help="Run scale preset",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight validation script",
    )
    return parser


def run() -> Path:
    """CLI entrypoint implementation."""
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "sweep_results" / f"autonomous_run_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    preset = PRESETS[args.preset]

    if not args.skip_preflight:
        run_preflight_validation(repo_root)

    t1_summary = run_t1_campaign(preset, output_dir)
    t3_summary = run_t3_campaign(preset, output_dir)

    summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "preset": args.preset,
        "preset_config": asdict(preset),
        "git_revision": get_git_revision(repo_root),
        "output_dir": str(output_dir),
        "t1": t1_summary,
        "t3": t3_summary,
    }
    save_json(output_dir / "autonomous_summary.json", summary)

    print(f"Autonomous research run complete. Outputs: {output_dir}")
    return output_dir


def main() -> None:
    """Process-level entrypoint."""
    run()


if __name__ == "__main__":
    main()

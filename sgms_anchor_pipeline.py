"""
Config-driven runner for the reduced-order anchor analysis stack.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sgms_anchor_calibration import load_anchor_calibration, resolve_calibrated_params
from sgms_anchor_claims import load_anchor_claims, resolve_claim_context
from sgms_anchor_control import (
    controller_trade_study,
    export_robustness_csv,
    export_trade_study_csv,
    plot_robustness,
    plot_trade_study,
    run_robustness_scenarios,
)
from sgms_anchor_dashboard import build_dashboard_payload, write_dashboard_html
from sgms_anchor_profiles import (
    build_profile_summary_rows,
    export_profile_summary_csv,
    load_anchor_profiles,
    resolve_profile_params,
)
from sgms_anchor_report import build_report_payload, write_report_html
from sgms_anchor_sensitivity import (
    export_sobol_indices_csv,
    plot_sobol_indices,
    run_sobol_sensitivity,
)
from sgms_anchor_v1 import (
    DEFAULT_PARAMS,
    estimate_period,
    export_sweep_csv,
    plot_anchor_response,
    plot_continuum_vs_packet,
    plot_sweep_heatmaps,
    plot_velocity_sweep,
    simulate_anchor,
    simulate_discrete_anchor,
    sweep_anchor_grid,
    sweep_velocity,
)
from monte_carlo.pass_fail_gates import create_default_gate_set


def load_experiment_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _slugify(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")


def _json_dump(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _make_time_grid(t_max: float, num_points: int) -> np.ndarray:
    return np.linspace(0.0, float(t_max), int(num_points))


def _build_disturbance(spec: dict | None, t_eval: np.ndarray) -> np.ndarray:
    disturbance = np.zeros_like(t_eval)
    if not spec:
        return disturbance
    start = float(spec.get("start", 0.0))
    end = float(spec.get("end", start))
    force = float(spec.get("force", 0.0))
    disturbance[(t_eval >= start) & (t_eval <= end)] = force
    return disturbance


def _merge_params(default_params: dict, overrides: dict | None) -> dict:
    params = default_params.copy()
    if overrides:
        params.update(overrides)
    return params


def _run_anchor_outputs(params: dict, experiment_dir: Path) -> tuple[dict, list[str]]:
    files = []
    metrics_dir = experiment_dir / "metrics"
    figures_dir = experiment_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    t_eval = _make_time_grid(params["t_max"], 6000 if params["t_max"] >= 400 else 2000)
    continuum = simulate_anchor(params, t_eval=t_eval, seed=7)
    discrete = simulate_discrete_anchor(params, t_eval=t_eval, seed=7)

    displacement_png = figures_dir / "anchor_displacement.png"
    packet_png = figures_dir / "anchor_packet_compare.png"
    plot_anchor_response(continuum, filename=str(displacement_png))
    plot_continuum_vs_packet(continuum, discrete, filename=str(packet_png))
    files.extend([str(displacement_png), str(packet_png)])

    sweep = sweep_velocity(params)
    sweep_png = figures_dir / "anchor_velocity_sweep.png"
    plot_velocity_sweep(sweep, filename=str(sweep_png))
    files.append(str(sweep_png))

    grid_rows = sweep_anchor_grid(params)
    grid_csv = metrics_dir / "anchor_grid.csv"
    heatmap_png = figures_dir / "anchor_heatmaps.png"
    export_sweep_csv(grid_rows, grid_csv)
    plot_sweep_heatmaps(grid_rows, eps=1e-3, filename=str(heatmap_png))
    files.extend([str(grid_csv), str(heatmap_png)])

    anchor_summary = {
        "continuum_metrics": continuum["metrics"],
        "continuum_period_s": estimate_period(continuum["t"], continuum["x"]),
        "discrete_period_s": estimate_period(discrete["t"], discrete["x"]),
    }
    return anchor_summary, files


def _run_trade_study(params: dict, trade_spec: dict, experiment_dir: Path) -> tuple[dict, list[str]]:
    files = []
    metrics_dir = experiment_dir / "metrics"
    figures_dir = experiment_dir / "figures"
    t_eval = _make_time_grid(trade_spec.get("t_max", params["t_max"]), trade_spec.get("num_points", 2000))
    disturbance = _build_disturbance(trade_spec.get("disturbance"), t_eval)
    study = controller_trade_study(
        params=params,
        controllers=tuple(trade_spec.get("controllers", ["open", "p", "lqr"])),
        t_eval=t_eval,
        disturbance_force=disturbance,
        p_gain_scale=float(trade_spec.get("p_gain_scale", 1.0)),
    )

    trade_csv = metrics_dir / "controller_trade_study.csv"
    trade_png = figures_dir / "controller_trade_study.png"
    export_trade_study_csv(study["rows"], trade_csv)
    plot_trade_study(study, filename=str(trade_png))
    files.extend([str(trade_csv), str(trade_png)])
    return {"rows": study["rows"]}, files


def _run_robustness(params: dict, robustness_spec: dict, experiment_dir: Path) -> tuple[dict, list[str]]:
    files = []
    metrics_dir = experiment_dir / "metrics"
    figures_dir = experiment_dir / "figures"
    t_eval = _make_time_grid(robustness_spec.get("t_max", params["t_max"]), robustness_spec.get("num_points", 2000))
    disturbance = _build_disturbance(robustness_spec.get("base_disturbance"), t_eval)
    robustness = run_robustness_scenarios(
        base_params=params,
        scenarios=robustness_spec.get("scenarios", []),
        controller=str(robustness_spec.get("controller", "lqr")),
        t_eval=t_eval,
        base_disturbance=disturbance,
        p_gain_scale=float(robustness_spec.get("p_gain_scale", 1.0)),
    )
    robustness_csv = metrics_dir / "robustness_summary.csv"
    robustness_png = figures_dir / "robustness.png"
    export_robustness_csv(robustness["rows"], robustness_csv)
    plot_robustness(robustness, filename=str(robustness_png))
    files.extend([str(robustness_csv), str(robustness_png)])
    return {"rows": robustness["rows"]}, files


def _run_sensitivity(params: dict, sensitivity_spec: dict, experiment_dir: Path) -> tuple[dict, list[str]]:
    files = []
    metrics_dir = experiment_dir / "metrics"
    figures_dir = experiment_dir / "figures"
    outputs = tuple(sensitivity_spec.get("outputs", ["k_eff", "period_s", "static_offset_m"]))
    result = run_sobol_sensitivity(
        N=int(sensitivity_spec.get("N", 256)),
        outputs=outputs,
        calc_second_order=bool(sensitivity_spec.get("calc_second_order", False)),
        base_params=params,
        seed=sensitivity_spec.get("seed"),
    )
    sobol_csv = metrics_dir / "sobol_indices.csv"
    sobol_png = figures_dir / "sobol_indices.png"
    export_sobol_indices_csv(result["indices"], result["problem"]["names"], sobol_csv)
    plot_sobol_indices(result["indices"], result["problem"]["names"], filename=str(sobol_png))
    files.extend([str(sobol_csv), str(sobol_png)])
    compact = {}
    for output, data in result["indices"].items():
        compact[output] = {
            "S1": {name: float(data["S1"][i]) for i, name in enumerate(result["problem"]["names"])},
            "ST": {name: float(data["ST"][i]) for i, name in enumerate(result["problem"]["names"])},
        }
    return compact, files


def export_fmeca_json(results: dict) -> dict:
    """
    Map experiment results to FMECA v1.2 failure modes.

    FMECA (Failure Modes, Effects, and Criticality Analysis) mapping:
    - FM-01: Spin decay → check ω_final vs ω_initial
    - FM-06: Hitch slip → check capture efficiency η_ind
    - FM-09: Shepherd AI latency → check MPC latency
    - FM-12: Thermal runaway → check temperature limits
    - FM-15: Structural failure → check stress limits

    Args:
        results: Experiment results dictionary with metrics

    Returns:
        FMECA risk matrix with failure modes, severity, probability, and risk
    """
    fmeca = {
        "schema_version": "1.2",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "failure_modes": {},
        "kill_criteria": {},
    }

    # Extract metrics from results
    metrics = results.get("anchor", {}).get("continuum_metrics", {})

    # FM-01: Spin decay (severity 8 - high)
    # Check if spin rate decays significantly
    omega_initial = metrics.get("omega_initial", 1.0)
    omega_final = metrics.get("omega_final", 1.0)
    spin_decay_ratio = omega_final / omega_initial if abs(omega_initial) > 1e-10 else 0.0
    spin_decay_prob = max(0.0, 1.0 - spin_decay_ratio)  # Higher decay = higher probability

    fmeca["failure_modes"]["FM-01"] = {
        "mode": "Spin decay",
        "description": "Loss of rotational kinetic energy",
        "severity": 8,
        "probability": float(spin_decay_prob),
        "risk": 8 * spin_decay_prob,
        "status": "PASS" if spin_decay_ratio > 0.9 else "FAIL" if spin_decay_ratio < 0.5 else "WARNING",
    }

    # FM-06: Hitch slip (severity 9 - critical)
    # Check induction efficiency
    eta_ind = metrics.get("eta_ind", 0.9)
    hitch_slip_prob = max(0.0, 1.0 - eta_ind / 0.82)  # Below threshold = higher probability

    fmeca["failure_modes"]["FM-06"] = {
        "mode": "Hitch slip",
        "description": "Loss of magnetic coupling during induction",
        "severity": 9,
        "probability": float(hitch_slip_prob),
        "risk": 9 * hitch_slip_prob,
        "status": "PASS" if eta_ind >= 0.82 else "FAIL",
    }

    # FM-09: Shepherd AI latency (severity 6 - medium)
    # Check control latency from trade study
    trade_results = results.get("trade_study", {}).get("rows", [])
    max_latency = 0.0
    for row in trade_results:
        latency = row.get("max_latency_ms", 0.0)
        max_latency = max(max_latency, latency)

    latency_prob = max(0.0, (max_latency - 10.0) / 20.0)  # Above 10ms = increasing probability

    fmeca["failure_modes"]["FM-09"] = {
        "mode": "Shepherd AI latency",
        "description": "Control loop latency exceeds real-time requirements",
        "severity": 6,
        "probability": float(min(1.0, latency_prob)),
        "risk": 6 * min(1.0, latency_prob),
        "status": "PASS" if max_latency <= 10.0 else "WARNING" if max_latency <= 30.0 else "FAIL",
    }

    # FM-12: Thermal runaway (severity 10 - catastrophic)
    # Check temperature limits
    temp_max = metrics.get("temperature_max", 300.0)
    thermal_prob = max(0.0, (temp_max - 400.0) / 100.0)  # Above 400K = increasing probability

    fmeca["failure_modes"]["FM-12"] = {
        "mode": "Thermal runaway",
        "description": "Temperature exceeds safe operating limits",
        "severity": 10,
        "probability": float(min(1.0, thermal_prob)),
        "risk": 10 * min(1.0, thermal_prob),
        "status": "PASS" if temp_max <= 400.0 else "WARNING" if temp_max <= 450.0 else "FAIL",
    }

    # FM-15: Structural failure (severity 10 - catastrophic)
    # Check stress limits
    stress_max = metrics.get("stress_max", 0.0)
    stress_limit = 0.8e9  # 800 MPa with SF=1.5
    stress_prob = max(0.0, (stress_max - stress_limit) / stress_limit)

    fmeca["failure_modes"]["FM-15"] = {
        "mode": "Structural failure",
        "description": "Centrifugal stress exceeds material limits",
        "severity": 10,
        "probability": float(min(1.0, stress_prob)),
        "risk": 10 * min(1.0, stress_prob),
        "status": "PASS" if stress_max <= stress_limit else "FAIL",
    }

    # Kill criteria flags
    fmeca["kill_criteria"] = {
        "energy_dissipation_exceeded": spin_decay_ratio < 0.95,  # >5% energy loss
        "misalignment_exceeded": metrics.get("max_displacement", 0.0) > 0.1,  # >10cm
        "induction_failed": eta_ind < 0.82,
        "thermal_limit_exceeded": temp_max > 450.0,
        "stress_limit_exceeded": stress_max > stress_limit,
        "any_kill_criteria": any([
            spin_decay_ratio < 0.95,
            metrics.get("max_displacement", 0.0) > 0.1,
            eta_ind < 0.82,
            temp_max > 450.0,
            stress_max > stress_limit,
        ]),
    }

    return fmeca


def run_experiment_suite(
    config_path: str | Path,
    output_root: str | Path = "artifacts",
    run_label: str | None = None,
) -> dict:
    config_path = Path(config_path)
    config = load_experiment_config(config_path)
    defaults = config.get("defaults", {})
    default_params = _merge_params(DEFAULT_PARAMS, defaults.get("params"))
    profiles_path = defaults.get("profiles_path")
    profile_data = load_anchor_profiles(profiles_path) if profiles_path else {"profiles": []}
    material_catalog_path = defaults.get("material_catalog_path", "paper_model/gdbco_apc_catalog.json")
    geometry_catalog_path = defaults.get("geometry_catalog_path", "geometry_profiles.json")
    environment_catalog_path = defaults.get("environment_catalog_path", "environment_profiles.json")
    calibration_path = defaults.get("calibration_path")
    calibration_data = load_anchor_calibration(calibration_path) if calibration_path else {"defaults": {}, "profiles": {}}
    claims_path = defaults.get("claims_path")
    claims_data = load_anchor_claims(claims_path) if claims_path else {"phase_decision": {}, "profiles": {}}
    output_root = Path(output_root)
    if run_label is None:
        run_label = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    skipped_experiments = []
    
    manifest = {
        "run_label": run_label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "profiles_path": str(profiles_path) if profiles_path else None,
        "calibration_path": str(calibration_path) if calibration_path else None,
        "claims_path": str(claims_path) if claims_path else None,
        "experiments": [],
    }
    summaries_by_slug = {}
    profile_rows_input = []

    _json_dump(config, run_dir / "config_snapshot.json")

    for experiment in config.get("experiments", []):
        name = experiment["name"]
        slug = _slugify(name)
        experiment_dir = run_dir / slug
        experiment_dir.mkdir(parents=True, exist_ok=True)

        if experiment.get("profile"):
            try:
                resolved = resolve_profile_params(
                    profile_data,
                    experiment["profile"],
                    overrides=experiment.get("params"),
                    base_params=default_params,
                    material_catalog_path=material_catalog_path,
                    geometry_catalog_path=geometry_catalog_path,
                    environment_catalog_path=environment_catalog_path,
                )
                params = resolved["params"]
                profile_meta = resolved["profile"]
                # Pass material_profile to params for physics modules
                if profile_meta.get("material_profile"):
                    params["material_profile"] = profile_meta["material_profile"]
                # Pass geometry_profile to params for physics modules
                if profile_meta.get("geometry_profile"):
                    params["geometry_profile"] = profile_meta["geometry_profile"]
                # Pass environment_profile to params for physics modules
                if profile_meta.get("environment_profile"):
                    params["environment_profile"] = profile_meta["environment_profile"]
            except (KeyError, ValueError, FileNotFoundError, OSError, json.JSONDecodeError) as e:
                print(f"ERROR: Failed to resolve profile '{experiment['profile']}': {e}")
                print(f"Skipping experiment: {name}")
                skipped_experiments.append({"name": name, "profile": experiment["profile"], "error": str(e)})
                continue
        else:
            params = _merge_params(default_params, experiment.get("params"))
            profile_meta = {
                "name": "direct-defaults",
                "category": "unspecified",
                "notes": [],
                "provenance": {},
            }

        calibration = resolve_calibrated_params(
            calibration_data,
            params,
            profile_name=profile_meta["name"] if profile_meta["name"] != "direct-defaults" else None,
        )
        params = calibration["params"]
        claim_context = resolve_claim_context(
            claims_data,
            profile_name=profile_meta["name"] if profile_meta["name"] != "direct-defaults" else None,
        )

        summary = {
            "name": name,
            "slug": slug,
            "params": params,
            "profile": profile_meta,
            "calibration": calibration,
            "claim_context": claim_context,
        }
        files = []

        anchor_summary, anchor_files = _run_anchor_outputs(params, experiment_dir)
        summary["anchor"] = anchor_summary
        files.extend(anchor_files)

        trade_spec = defaults.get("trade_study", {}).copy()
        trade_spec.update(experiment.get("trade_study", {}))
        trade_summary, trade_files = _run_trade_study(params, trade_spec, experiment_dir)
        summary["trade_study"] = trade_summary
        files.extend(trade_files)

        robustness_spec = defaults.get("robustness", {}).copy()
        robustness_spec.update(experiment.get("robustness", {}))
        robustness_summary, robustness_files = _run_robustness(params, robustness_spec, experiment_dir)
        summary["robustness"] = robustness_summary
        files.extend(robustness_files)

        sensitivity_spec = defaults.get("sensitivity", {}).copy()
        sensitivity_spec.update(experiment.get("sensitivity", {}))
        sensitivity_summary, sensitivity_files = _run_sensitivity(params, sensitivity_spec, experiment_dir)
        summary["sensitivity"] = sensitivity_summary
        files.extend(sensitivity_files)

        summary_path = experiment_dir / "summary.json"
        _json_dump(summary, summary_path)
        summaries_by_slug[slug] = summary
        profile_rows_input.append(summary)

        # Export FMECA risk matrix
        fmeca = export_fmeca_json(summary)
        fmeca_path = experiment_dir / "fmeca_risk.json"
        _json_dump(fmeca, fmeca_path)
        files.append(str(fmeca_path))

        manifest["experiments"].append(
            {
                "name": name,
                "slug": slug,
                "summary_path": str(summary_path),
                "fmeca_path": str(fmeca_path),
                "profile": profile_meta,
                "claim_context": claim_context,
                "files": files,
            }
        )

    profile_rows = build_profile_summary_rows(profile_rows_input)
    if profile_rows:
        profile_csv = run_dir / "profile_summary.csv"
        export_profile_summary_csv(profile_rows, profile_csv)
        manifest["profile_summary_path"] = str(profile_csv)

    report_payload = build_report_payload(manifest, summaries_by_slug)
    report_path = run_dir / "report.html"
    write_report_html(report_payload, report_path)
    manifest["report_path"] = str(report_path)
    manifest["validation_decision"] = claims_data.get("phase_decision", {})

    dashboard_payload = build_dashboard_payload(manifest, summaries_by_slug)
    dashboard_data_path = run_dir / "dashboard_data.json"
    _json_dump(dashboard_payload, dashboard_data_path)
    dashboard_path = run_dir / "dashboard.html"
    write_dashboard_html(dashboard_payload, dashboard_path)
    manifest["dashboard_data_path"] = str(dashboard_data_path)
    manifest["dashboard_path"] = str(dashboard_path)
    
    if skipped_experiments:
        manifest["skipped_experiments"] = skipped_experiments
        print(f"\nWARNING: {len(skipped_experiments)} experiment(s) skipped due to errors")
        for skipped in skipped_experiments:
            print(f"  - {skipped['name']}: {skipped['error']}")
    
    _json_dump(manifest, run_dir / "manifest.json")
    return manifest


def main() -> None:
    manifest = run_experiment_suite("anchor_experiments.json")
    print(f"Run label: {manifest['run_label']}")
    print(f"Experiments: {len(manifest['experiments'])}")


if __name__ == "__main__":
    main()
    main()

"""
T3 Sweep: Fault rate [10^-6 to 10^-3] / hr with cascade threshold analysis.

Sweep parameters:
- fault_rate ∈ [10^-6, 10^-3] / hr
- cascade_threshold = 1.05
- Target: Containment in ≤2 nodes, 95% of runs
- Question: Does the system contain failures or amplify them?
"""

import os
import sys

# Add the project root to the system path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

import matplotlib.pyplot as plt
import numpy as np

from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.rigid_body import RigidBody
from monte_carlo.cascade_runner import (
    CascadeRunner,
    MonteCarloConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stream_factory_with_nodes(n_nodes: int = 10):
    """Create a stream factory with specified number of nodes."""
    def factory():
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        # Create multiple packets with spatial distribution
        n_packets = 5
        stream_vel = 100.0
        spacing = 10.0
        packets = []
        for p_id in range(n_packets):
            position = np.array([p_id * spacing, 0.0, 0.0])
            velocity = np.array([stream_vel, 0.0, 0.0])
            packets.append(Packet(
                id=p_id,
                body=RigidBody(mass, I, position=position, velocity=velocity),
                eta_ind=0.9,
            ))

        # Create nodes with stiffness
        nodes = []
        for i in range(n_nodes):
            node = SNode(
                id=i,
                position=np.array([i * 10.0, 0.0, 0.0]),  # Spaced 10m apart
                max_packets=10,
                eta_ind_min=0.82,
                k_fp=6000.0,  # Flux-pinning stiffness (>= feasibility gate)
            )
            nodes.append(node)

        stream = MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=stream_vel)
        return stream
    return factory


def run_t3_sweep(
    fault_rate_range: tuple[float, float] = (1e-6, 1e-3),
    n_fault_rate_points: int = 8,
    cascade_threshold: float = 1.05,
    containment_threshold: int = 2,
    n_nodes: int = 10,
    n_realizations_per_point: int = 100,
    time_horizon: float = 3600.0,  # Extended to 1 hour for rare-event (1e-6/hr) fault statistics
    enable_cascade_propagation: bool = False,  # NEW: Enable cascade propagation
    fault_injection_mode: str = "rate",  # NEW: Fault injection mode
    n_guaranteed_faults: int = 0,  # NEW: Guaranteed faults
    pass_fail_eta_ind_min: float = 0.82,
    pass_fail_stress_max: float = 1.2e9,
    pass_fail_k_eff_min: float = 6000.0,
) -> dict:
    """
    Run T3 sweep: fault rate vs cascade/containment metrics.

    Args:
        fault_rate_range: (min_per_hr, max_per_hr)
        n_fault_rate_points: Number of fault rate points (log scale)
        cascade_threshold: Stiffness reduction factor for cascade
        containment_threshold: Max nodes allowed for containment success
        n_nodes: Number of nodes in the lattice
        n_realizations_per_point: Monte-Carlo runs per fault rate
        time_horizon: Simulation time horizon (s). Default 3600s (1 hour) for rare-event fault statistics at 1e-6/hr and below.
        enable_cascade_propagation: Enable neighbor load redistribution (Root Cause #2)
        fault_injection_mode: "rate", "guaranteed", or "poisson" (Root Cause #1)
        n_guaranteed_faults: Number of guaranteed faults per realization

    Returns:
        Dictionary with sweep results
    """
    fault_rates = np.logspace(np.log10(fault_rate_range[0]), np.log10(fault_rate_range[1]), n_fault_rate_points)

    # Results storage
    cascade_probability = []
    nodes_affected_mean = []
    nodes_affected_std = []
    containment_rate = []
    success_rate = []

    # NEW: Diagnostic tracking - Trust Strategy #1
    fault_events_total_per_point = []
    sanity_warnings = []

    logger.info(f"Starting T3 sweep: {n_fault_rate_points} fault rate points, {n_realizations_per_point} runs each")
    logger.info(f"Total Monte-Carlo runs: {n_fault_rate_points * n_realizations_per_point}")

    # NEW: Pre-flight sanity check - Trust Strategy #2
    expected_faults_min = fault_rates[0] * time_horizon * n_nodes / 3600.0
    if expected_faults_min < 0.01 and fault_injection_mode == "rate":
        logger.warning(
            f"Pre-flight check: Expected faults at lowest rate = {expected_faults_min:.4f} per realization with time_horizon={time_horizon}s. "
            f"Consider using fault_injection_mode='guaranteed' or extending time_horizon beyond 3600s for rates below 1e-6/hr."
        )

    for fault_rate in fault_rates:
        logger.info(f"Fault rate: {fault_rate:.2e} /hr")

        # Configure Monte-Carlo
        config = MonteCarloConfig(
            n_realizations=n_realizations_per_point,
            time_horizon=time_horizon,
            dt=0.01,
            fault_rate=fault_rate,
            cascade_threshold=cascade_threshold,
            containment_threshold=containment_threshold,
            # NEW: Root Cause fixes
            enable_cascade_propagation=enable_cascade_propagation,
            fault_injection_mode=fault_injection_mode,
            n_guaranteed_faults=n_guaranteed_faults,
            pass_fail_gates={
                "eta_ind": (pass_fail_eta_ind_min, ">="),
                "stress": (pass_fail_stress_max, "<="),
                "k_eff": (pass_fail_k_eff_min, ">="),
            },
        )

        # Run Monte-Carlo with individual result tracking
        runner = CascadeRunner(config)
        stream_factory = create_stream_factory_with_nodes(n_nodes=n_nodes)

        # Run individual realizations to track nodes_affected
        individual_results = []
        for i in range(n_realizations_per_point):
            stream = stream_factory()
            result = runner.run_realization(stream, i)
            individual_results.append(result)

        # Calculate aggregated statistics
        success_count = sum(1 for r in individual_results if r.success)
        cascade_count = sum(1 for r in individual_results if r.cascade_occurred)

        cascade_probability.append(cascade_count / n_realizations_per_point)
        success_rate.append(success_count / n_realizations_per_point)

        # Calculate nodes affected statistics
        nodes_affected_list = [r.nodes_affected for r in individual_results]
        nodes_affected_mean.append(np.mean(nodes_affected_list))
        nodes_affected_std.append(np.std(nodes_affected_list))

        # Calculate containment rate
        containment_count = sum(1 for r in individual_results if r.containment_successful)
        containment_rate.append(containment_count / n_realizations_per_point)

        # NEW: Track diagnostic counters - Trust Strategy #1
        faults_at_this_point = sum(r.fault_events_injected for r in individual_results)
        fault_events_total_per_point.append(faults_at_this_point)

        # Check sanity - Trust Strategy #2
        if faults_at_this_point == 0 and fault_rate > 0 and fault_injection_mode == "rate":
            sanity_warning = f"NO FAULTS INJECTED at fault_rate={fault_rate:.2e}/hr"
            sanity_warnings.append(sanity_warning)
            logger.warning(sanity_warning)
        else:
            sanity_warnings.append("")

        logger.info(f"  Faults injected: {faults_at_this_point}, Mean per realization: {faults_at_this_point/n_realizations_per_point:.2f}")

    return {
        'fault_rates': fault_rates,
        'cascade_probability': np.array(cascade_probability),
        'nodes_affected_mean': np.array(nodes_affected_mean),
        'nodes_affected_std': np.array(nodes_affected_std),
        'containment_rate': np.array(containment_rate),
        'success_rate': np.array(success_rate),
        'cascade_threshold': cascade_threshold,
        'containment_threshold': containment_threshold,
        'n_nodes': n_nodes,
        # NEW: Diagnostic tracking - Trust Strategy #1 & #4
        'fault_events_total_per_point': fault_events_total_per_point,
        'sanity_warnings': sanity_warnings,
    }


def _make_json_serializable(obj):
    """Convert numpy arrays / scalars in results to plain Python types for json.dump."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    return obj


def plot_t3_results(results: dict, output_file: str = 'sweep_t3_fault_cascade.png'):
    """Plot T3 sweep results."""
    fault_rates = results['fault_rates']
    cascade_probability = results['cascade_probability']
    containment_rate = results['containment_rate']
    success_rate = results['success_rate']

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Cascade probability vs fault rate (log-log)
    axes[0].loglog(fault_rates, cascade_probability, 'ro-', markersize=6, linewidth=2, label='Cascade Probability')
    axes[0].axhline(1e-6, color='green', linestyle='--', linewidth=2, label='Target (<10⁻⁶)')
    axes[0].axhline(1e-4, color='orange', linestyle=':', linewidth=1, label='FMECA residual (10⁻⁴)')
    axes[0].set_xlabel('Fault Rate (per hour)')
    axes[0].set_ylabel('Cascade Probability')
    axes[0].set_title('Cascade Probability vs Fault Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Containment rate and success rate vs fault rate
    ax2 = axes[1]
    ax2.semilogx(fault_rates, containment_rate * 100, 'bo-', markersize=6, linewidth=2, label='Containment Rate (%)')
    ax2.semilogx(fault_rates, success_rate * 100, 'gs-', markersize=6, linewidth=2, label='Success Rate (%)')
    ax2.axhline(95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
    ax2.set_xlabel('Fault Rate (per hour)')
    ax2.set_ylabel('Rate (%)')
    ax2.set_title('Containment & Success Rate vs Fault Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    logger.info(f"Saved T3 sweep plot: {output_file}")


def analyze_containment_threshold(results: dict) -> dict:
    """Analyze containment threshold from sweep results."""
    fault_rates = results['fault_rates']
    cascade_probability = results['cascade_probability']
    containment_rate = results['containment_rate']

    # Find fault rate where cascade probability exceeds 1e-6
    cascade_threshold_idx = np.where(cascade_probability > 1e-6)[0]
    if len(cascade_threshold_idx) > 0:
        cascade_threshold_fault_rate = fault_rates[cascade_threshold_idx[0]]
    else:
        cascade_threshold_fault_rate = fault_rates[-1]  # Max fault rate tested

    # Find fault rate where containment rate drops below 95%
    containment_threshold_idx = np.where(containment_rate < 0.95)[0]
    if len(containment_threshold_idx) > 0:
        containment_threshold_fault_rate = fault_rates[containment_threshold_idx[0]]
    else:
        containment_threshold_fault_rate = fault_rates[-1]

    # Determine overall system behavior
    if np.mean(cascade_probability) < 1e-6:
        system_behavior = "contains_failures"
    elif np.mean(cascade_probability) > 1e-4:
        system_behavior = "amplifies_failures"
    else:
        system_behavior = "mixed"

    return {
        'cascade_threshold_fault_rate': cascade_threshold_fault_rate,
        'containment_threshold_fault_rate': containment_threshold_fault_rate,
        'system_behavior': system_behavior,
        'mean_cascade_probability': np.mean(cascade_probability),
        'mean_containment_rate': np.mean(containment_rate),
    }


def _parse_cli_args():
    import argparse
    parser = argparse.ArgumentParser(description="T3 Sweep: fault rate vs cascade/containment metrics")

    # Sweep shape
    parser.add_argument("--fault_rate_min", type=float, default=1e-6)
    parser.add_argument("--fault_rate_max", type=float, default=1e-2)
    parser.add_argument("--n_fault_rate_points", type=int, default=12)

    # Physics / gates
    parser.add_argument("--cascade_threshold", type=float, default=1.05)
    parser.add_argument("--containment_threshold", type=int, default=2)
    parser.add_argument("--n_nodes", type=int, default=10)
    parser.add_argument("--pass_fail_eta_ind_min", type=float, default=0.82)
    parser.add_argument("--pass_fail_stress_max", type=float, default=1.2e9)
    parser.add_argument("--pass_fail_k_eff_min", type=float, default=6000.0)

    # Monte Carlo scale
    parser.add_argument("--n_realizations_per_point", type=int, default=200)
    parser.add_argument("--time_horizon", type=float, default=3600.0)
    parser.add_argument("--dt", type=float, default=0.01)

    # Interesting coupling toggles
    parser.add_argument("--enable_cascade_propagation", action="store_true", default=False)
    parser.add_argument("--enable_thermal_quench", action="store_true", default=False)
    parser.add_argument("--quench_detection_enabled", action="store_true", default=False)

    # Fault injection mode
    parser.add_argument("--fault_injection_mode", choices=["rate", "guaranteed", "poisson"], default="rate")
    parser.add_argument("--n_guaranteed_faults", type=int, default=0)

    # Artifacts
    parser.add_argument("--out_dir", type=str, default="sweep_results")
    parser.add_argument("--save_plot", action="store_true", default=False)
    parser.add_argument("--no_plot", action="store_true", default=False)

    # Runtime
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true", default=False)
    return parser.parse_args()


def _timestamp():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    args = _parse_cli_args()

    # Smoke test presets (fast, still exercises pipeline)
    if args.smoke_test:
        args.n_fault_rate_points = min(args.n_fault_rate_points, 4)
        args.n_realizations_per_point = min(args.n_realizations_per_point, 20)
        args.time_horizon = min(args.time_horizon, 60.0)  # 60s is enough for basic wiring

        # Ensure failures happen even in short horizon
        if args.fault_injection_mode == "rate":
            # With short horizon, rate-based injection may lead to zero injected faults.
            # Switch to guaranteed for smoke test.
            args.fault_injection_mode = "guaranteed"
            args.n_guaranteed_faults = max(args.n_guaranteed_faults, 1)

    if args.seed is not None:
        np.random.seed(args.seed)

    results = run_t3_sweep(
        fault_rate_range=(args.fault_rate_min, args.fault_rate_max),
        n_fault_rate_points=args.n_fault_rate_points,
        cascade_threshold=args.cascade_threshold,
        containment_threshold=args.containment_threshold,
        n_nodes=args.n_nodes,
        n_realizations_per_point=args.n_realizations_per_point,
        time_horizon=args.time_horizon,
        enable_cascade_propagation=args.enable_cascade_propagation,
        fault_injection_mode=args.fault_injection_mode,
        n_guaranteed_faults=args.n_guaranteed_faults,
        pass_fail_eta_ind_min=args.pass_fail_eta_ind_min,
        pass_fail_stress_max=args.pass_fail_stress_max,
        pass_fail_k_eff_min=args.pass_fail_k_eff_min,
    )

    # Attach run metadata (for provenance)
    results["run_metadata"] = {
        "fault_rate_range": [args.fault_rate_min, args.fault_rate_max],
        "n_fault_rate_points": args.n_fault_rate_points,
        "cascade_threshold": args.cascade_threshold,
        "containment_threshold": args.containment_threshold,
        "n_nodes": args.n_nodes,
        "n_realizations_per_point": args.n_realizations_per_point,
        "time_horizon": args.time_horizon,
        "enable_cascade_propagation": args.enable_cascade_propagation,
        "enable_thermal_quench": args.enable_thermal_quench,
        "quench_detection_enabled": args.quench_detection_enabled,
        "fault_injection_mode": args.fault_injection_mode,
        "n_guaranteed_faults": args.n_guaranteed_faults,
        "dt": args.dt,
        "seed": args.seed,
        "smoke_test": args.smoke_test,
        "artifact_timestamp": _timestamp(),
    }

    # Analyze containment threshold
    analysis = analyze_containment_threshold(results)
    results["analysis"] = analysis

    # Save JSON artifacts
    _ensure_dir(args.out_dir)
    ts = results["run_metadata"]["artifact_timestamp"]
    json_path = os.path.join(args.out_dir, f"t3_sweep_results_{ts}.json")
    import json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            _make_json_serializable(results),
            f,
            indent=2
        )
    logger.info(f"Saved sweep results JSON: {json_path}")

    # Plot results (optional)
    if args.save_plot and not args.no_plot:
        plot_path = os.path.join(args.out_dir, f"t3_sweep_plot_{ts}.png")
        plot_t3_results(results, output_file=plot_path)

    # Print summary
    print("\n=== T3 SWEEP SUMMARY ===")
    print(f"System behavior: {analysis['system_behavior']}")
    print(f"Mean cascade probability: {analysis['mean_cascade_probability']:.2e}")
    print(f"Mean containment rate: {analysis['mean_containment_rate']*100:.1f}%")
    print(f"\nFault rate where cascade probability > 10^-6: {analysis['cascade_threshold_fault_rate']:.2e} /hr")
    print(f"Fault rate where containment rate < 95%: {analysis['containment_threshold_fault_rate']:.2e} /hr")

    print("\nDetailed results:")
    for fr, cp, cr, sr in zip(
        results['fault_rates'],
        results['cascade_probability'],
        results['containment_rate'],
        results['success_rate'], strict=False
    ):
        print(f"  fault_rate={fr:.2e}: cascade={cp:.2e}, containment={cr*100:.1f}%, success={sr*100:.1f}%")

    print("\nConclusion:")
    if analysis['system_behavior'] == "contains_failures":
        print("✓ System contains failures - cascade probability remains low")
    elif analysis['system_behavior'] == "amplifies_failures":
        print("✗ System amplifies failures - cascade probability high")
    else:
        print("⚠ System shows mixed behavior - depends on fault rate")

    if analysis['mean_containment_rate'] >= 0.95:
        print("✓ Containment in ≤2 nodes achieved in ≥95% of runs")
    else:
        print("✗ Containment in ≤2 nodes not achieved consistently")

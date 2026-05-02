"""
T3 Sweep: Fault rate [10^-6 to 10^-3] / hr with cascade threshold analysis.

Sweep parameters:
- fault_rate ∈ [10^-6, 10^-3] / hr
- cascade_threshold = 1.05
- Target: Containment in ≤2 nodes, 95% of runs
- Question: Does the system contain failures or amplify them?
"""

import sys
import os

# Add the project root to the system path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging

from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig, Perturbation, PerturbationType
from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.rigid_body import RigidBody

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
    fault_rate_range: Tuple[float, float] = (1e-6, 1e-3),
    n_fault_rate_points: int = 8,
    cascade_threshold: float = 1.05,
    containment_threshold: int = 2,
    n_nodes: int = 10,
    n_realizations_per_point: int = 100,
    time_horizon: float = 10.0,
    enable_cascade_propagation: bool = False,  # NEW: Enable cascade propagation
    fault_injection_mode: str = "rate",  # NEW: Fault injection mode
    n_guaranteed_faults: int = 0,  # NEW: Guaranteed faults
) -> Dict:
    """
    Run T3 sweep: fault rate vs cascade/containment metrics.

    Args:
        fault_rate_range: (min_per_hr, max_per_hr)
        n_fault_rate_points: Number of fault rate points (log scale)
        cascade_threshold: Stiffness reduction factor for cascade
        containment_threshold: Max nodes allowed for containment success
        n_nodes: Number of nodes in the lattice
        n_realizations_per_point: Monte-Carlo runs per fault rate
        time_horizon: Simulation time horizon (s)
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
            f"Pre-flight check: Expected faults at lowest rate = {expected_faults_min:.4f} per realization. "
            f"This is very low - consider using fault_injection_mode='guaranteed' or increasing time_horizon."
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
                "eta_ind": (0.82, ">="),
                "stress": (1.2e9, "<="),
                "k_eff": (6000.0, ">="),
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


def plot_t3_results(results: Dict, output_file: str = 'sweep_t3_fault_cascade.png'):
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


def analyze_containment_threshold(results: Dict) -> Dict:
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


if __name__ == "__main__":
    # Run sweep with extended range to find cascade boundary
    results = run_t3_sweep(
        fault_rate_range=(1e-6, 1e-2),  # Extended to 1e-2 to find cascade boundary
        n_fault_rate_points=12,  # More points for finer resolution
        cascade_threshold=1.05,
        containment_threshold=2,
        n_nodes=10,
        n_realizations_per_point=100,  # High-resolution for convergence
        time_horizon=10.0,
    )

    # Plot results
    plot_t3_results(results)

    # Analyze containment threshold
    analysis = analyze_containment_threshold(results)

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
        results['success_rate']
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

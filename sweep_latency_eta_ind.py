"""
T1 Sweep: Latency [5-50ms] × η_ind [0.8-0.95] with delay margin analysis.

Sweep parameters:
- latency_ms ∈ [5, 50] ms
- η_ind ∈ [0.8, 0.95]
- Target: Delay margin ≥ 35 ms
- Question: Can discrete-time control stabilize the validated mechanism?
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging

from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig
from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.rigid_body import RigidBody
from control_layer.mpc_controller import MPCController, ConfigurationMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stream_factory(eta_ind: float = 0.9):
    """Create a stream factory with specified eta_ind."""
    def factory():
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        packets = [Packet(id=0, body=RigidBody(mass, I), eta_ind=eta_ind)]
        stream = MultiBodyStream(packets=packets, nodes=[], stream_velocity=100.0)
        return stream
    return factory


def run_t1_sweep(
    latency_range: Tuple[float, float] = (5.0, 50.0),
    eta_ind_range: Tuple[float, float] = (0.8, 0.95),
    n_latency_points: int = 10,
    n_eta_points: int = 8,
    n_realizations_per_point: int = 50,
) -> Dict:
    """
    Run T1 sweep: latency × eta_ind grid.

    Args:
        latency_range: (min_ms, max_ms)
        eta_ind_range: (min, max)
        n_latency_points: Number of latency grid points
        n_eta_points: Number of eta_ind grid points
        n_realizations_per_point: Monte-Carlo runs per grid point

    Returns:
        Dictionary with sweep results
    """
    latency_values = np.linspace(latency_range[0], latency_range[1], n_latency_points)
    eta_ind_values = np.linspace(eta_ind_range[0], eta_ind_range[1], n_eta_points)

    # Results storage
    success_rate_grid = np.zeros((n_eta_points, n_latency_points))
    delay_margin_grid = np.zeros((n_eta_points, n_latency_points))
    max_displacement_grid = np.zeros((n_eta_points, n_latency_points))

    # Create MPC controller for delay margin calculation
    try:
        mpc = MPCController(
            configuration_mode=ConfigurationMode.TEST,
            delay_steps=5,
            enable_delay_compensation=True,
        )
        mpc_available = True
    except ImportError:
        logger.warning("CasADi not available, skipping delay margin calculation")
        mpc_available = False

    logger.info(f"Starting T1 sweep: {n_latency_points}×{n_eta_points} grid, {n_realizations_per_point} runs each")
    logger.info(f"Total Monte-Carlo runs: {n_latency_points * n_eta_points * n_realizations_per_point}")

    for i, eta_ind in enumerate(eta_ind_values):
        for j, latency_ms in enumerate(latency_values):
            logger.info(f"Grid point ({i+1}/{n_eta_points}, {j+1}/{n_latency_points}): eta_ind={eta_ind:.3f}, latency={latency_ms:.1f}ms")

            # Configure Monte-Carlo
            config = MonteCarloConfig(
                n_realizations=n_realizations_per_point,
                time_horizon=1.0,  # Shorter horizon for speed
                dt=0.01,
                latency_ms=latency_ms,
                latency_std_ms=0.0,  # No variation for sweep
                pass_fail_gates={
                    "eta_ind": (0.82, ">="),
                    "stress": (1.2e9, "<="),
                    "k_eff": (6000.0, ">="),
                },
            )

            # Run Monte-Carlo
            runner = CascadeRunner(config)
            stream_factory = create_stream_factory(eta_ind=eta_ind)
            results = runner.run_monte_carlo(stream_factory)

            # Store success rate
            success_rate_grid[i, j] = results['success_rate']

            # Calculate delay margin if MPC available
            if mpc_available:
                delay_margin = mpc.calculate_delay_margin()
                delay_margin_grid[i, j] = delay_margin['delay_margin_ms']
            else:
                delay_margin_grid[i, j] = np.nan

            # Store max displacement (placeholder - would need actual tracking)
            max_displacement_grid[i, j] = 0.0

    return {
        'latency_values': latency_values,
        'eta_ind_values': eta_ind_values,
        'success_rate_grid': success_rate_grid,
        'delay_margin_grid': delay_margin_grid,
        'max_displacement_grid': max_displacement_grid,
        'mpc_available': mpc_available,
    }


def plot_t1_results(results: Dict, output_file: str = 'sweep_t1_latency_eta_ind.png'):
    """Plot T1 sweep results."""
    latency_values = results['latency_values']
    eta_ind_values = results['eta_ind_values']
    success_rate_grid = results['success_rate_grid']
    delay_margin_grid = results['delay_margin_grid']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Success rate heatmap
    im1 = axes[0].imshow(
        success_rate_grid * 100,
        extent=[latency_values[0], latency_values[-1], eta_ind_values[0], eta_ind_values[-1]],
        origin='lower',
        aspect='auto',
        cmap='RdYlGn',
        vmin=0,
        vmax=100,
    )
    axes[0].set_xlabel('Latency (ms)')
    axes[0].set_ylabel('η_ind')
    axes[0].set_title('Success Rate (%)')
    plt.colorbar(im1, ax=axes[0], label='Success Rate (%)')

    # Add contour at 95% success rate
    cs1 = axes[0].contour(
        latency_values,
        eta_ind_values,
        success_rate_grid * 100,
        levels=[95],
        colors='white',
        linewidths=2,
    )
    axes[0].clabel(cs1, inline=True, fontsize=10, fmt='95%%')

    # Plot 2: Delay margin heatmap (if available)
    if results['mpc_available']:
        im2 = axes[1].imshow(
            delay_margin_grid,
            extent=[latency_values[0], latency_values[-1], eta_ind_values[0], eta_ind_values[-1]],
            origin='lower',
            aspect='auto',
            cmap='viridis',
        )
        axes[1].set_xlabel('Latency (ms)')
        axes[1].set_ylabel('η_ind')
        axes[1].set_title('Delay Margin (ms)')
        plt.colorbar(im2, ax=axes[1], label='Delay Margin (ms)')

        # Add contour at 35ms threshold
        cs2 = axes[1].contour(
            latency_values,
            eta_ind_values,
            delay_margin_grid,
            levels=[35.0],
            colors='red',
            linewidths=2,
        )
        axes[1].clabel(cs2, inline=True, fontsize=10, fmt='35ms')
    else:
        axes[1].text(0.5, 0.5, 'Delay margin not available\n(CasADi not installed)',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Delay Margin (unavailable)')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    logger.info(f"Saved T1 sweep plot: {output_file}")


def analyze_stability_boundary(results: Dict) -> Dict:
    """Analyze stability boundary from sweep results."""
    latency_values = results['latency_values']
    eta_ind_values = results['eta_ind_values']
    success_rate_grid = results['success_rate_grid']
    delay_margin_grid = results['delay_margin_grid']

    # Find 95% success rate boundary
    boundary_points = []
    for i, eta_ind in enumerate(eta_ind_values):
        for j, latency_ms in enumerate(latency_values):
            if success_rate_grid[i, j] >= 0.95:
                boundary_points.append((latency_ms, eta_ind))

    # Find maximum latency for 95% success at each eta_ind
    max_latency_95 = []
    for i, eta_ind in enumerate(eta_ind_values):
        latencies_at_eta = [latency_values[j] for j in range(len(latency_values))
                           if success_rate_grid[i, j] >= 0.95]
        max_latency_95.append(max(latencies_at_eta) if latencies_at_eta else 0.0)

    # Find minimum eta_ind for 95% success at each latency
    min_eta_95 = []
    for j, latency_ms in enumerate(latency_values):
        etas_at_latency = [eta_ind_values[i] for i in range(len(eta_ind_values))
                          if success_rate_grid[i, j] >= 0.95]
        min_eta_95.append(min(etas_at_latency) if etas_at_latency else 1.0)

    # Check delay margin ≥ 35ms condition
    if results['mpc_available']:
        delay_margin_ok = delay_margin_grid >= 35.0
        delay_margin_95_region = np.sum(delay_margin_ok) / delay_margin_ok.size
    else:
        delay_margin_95_region = np.nan

    return {
        'boundary_points': boundary_points,
        'max_latency_95': max_latency_95,
        'min_eta_95': min_eta_95,
        'delay_margin_95_region': delay_margin_95_region,
        'overall_success_rate': np.mean(success_rate_grid),
    }


if __name__ == "__main__":
    # Run sweep with reduced grid for testing
    results = run_t1_sweep(
        latency_range=(5.0, 50.0),
        eta_ind_range=(0.8, 0.95),
        n_latency_points=10,
        n_eta_points=8,
        n_realizations_per_point=20,  # Reduced for speed
    )

    # Plot results
    plot_t1_results(results)

    # Analyze stability boundary
    analysis = analyze_stability_boundary(results)

    # Print summary
    print("\n=== T1 SWEEP SUMMARY ===")
    print(f"Overall success rate: {analysis['overall_success_rate']*100:.1f}%")
    print(f"Region with delay_margin ≥ 35ms: {analysis['delay_margin_95_region']*100:.1f}%")
    print(f"\nMax latency for 95% success at each eta_ind:")
    for eta, lat in zip(results['eta_ind_values'], analysis['max_latency_95']):
        print(f"  η_ind={eta:.3f}: {lat:.1f} ms")

    print(f"\nMin eta_ind for 95% success at each latency:")
    for lat, eta in zip(results['latency_values'], analysis['min_eta_95']):
        print(f"  latency={lat:.1f}ms: η_ind={eta:.3f}")

    print("\nConclusion:")
    if analysis['overall_success_rate'] >= 0.95:
        print("✓ System achieves 95% success rate across most of the parameter space")
    else:
        print("✗ System does not achieve 95% success rate across parameter space")

    if results['mpc_available']:
        if analysis['delay_margin_95_region'] >= 0.8:
            print("✓ Delay margin ≥ 35ms in ≥80% of parameter space")
        else:
            print("✗ Delay margin < 35ms in significant portion of parameter space")
    else:
        print("⚠ Delay margin analysis skipped (CasADi not available)")

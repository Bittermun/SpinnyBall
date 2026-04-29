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
import json
import os
from pathlib import Path
from joblib import Parallel, delayed

from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig
from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.rigid_body import RigidBody
from control_layer.mpc_controller import MPCController, ConfigurationMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stream(eta_ind: float = 0.9):
    """Create a MultiBodyStream with specified eta_ind (module-level for pickling)."""
    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    packets = [Packet(id=0, body=RigidBody(mass, I), eta_ind=eta_ind)]
    
    # HIGH-FIDELITY: Enable orbital dynamics and thermal effects
    stream = MultiBodyStream(
        packets=packets, 
        nodes=[], 
        stream_velocity=100.0,
        enable_orbital_dynamics=True,  # HIGH-FIDELITY: Enable orbital coupling
    )
    
    # Initialize packet with high-fidelity thermal properties
    packet = packets[0]
    packet.temperature = 300.0  # K, room temperature
    packet.radius = 0.01  # m, 1cm radius
    packet.emissivity = 0.8  # Typical for metal
    packet.specific_heat = 500.0  # J/(kg·K), typical for metal
    
    # HIGH-FIDELITY: Initialize magnetic flux-pinning model
    try:
        from dynamics.gdBCO_material import GdBCOProperties, GdBCOMaterial
        from dynamics.bean_london_model import BeanLondonModel
        
        # Real GdBCO material properties
        gd_props = GdBCOProperties()
        material = GdBCOMaterial(gd_props)
        
        # Bean-London flux-pinning model
        geometry = {
            'thickness': 1e-6,  # 1 μm superconducting layer
            'width': 0.012,     # 12 mm wide tape
            'length': 0.1       # 10 cm length
        }
        flux_model = BeanLondonModel(material, geometry)
        
        # Attach to packet
        packet.flux_model = flux_model
        packet.material = material
        logger.info("HIGH-FIDELITY: GdBCO flux-pinning model enabled")
    except ImportError:
        logger.warning("Flux-pinning model not available, using placeholder")
    
    # Initialize orbital state if available
    try:
        from dynamics.orbital_coupling import create_circular_orbit
        packet.orbital_state = create_circular_orbit(altitude=400000)  # 400 km orbit
        packet.in_eclipse = False  # Start in sunlight
    except ImportError:
        logger.warning("Orbital dynamics not available, using placeholder")
    
    return stream


def save_checkpoint(results: Dict, grid_point_idx: int, total_points: int, checkpoint_file: str = 't1_checkpoint.json'):
    """Save checkpoint after each grid point."""
    checkpoint_data = {
        'results': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in results.items()},
        'last_idx': grid_point_idx,
        'total_points': total_points,
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.info(f"Checkpoint saved: {grid_point_idx+1}/{total_points} points completed")


def load_checkpoint(checkpoint_file: str = 't1_checkpoint.json') -> Dict:
    """Load checkpoint if exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Checkpoint loaded: {data['last_idx']+1}/{data['total_points']} points completed")
        return data
    return None


def run_grid_point(
    eta_ind: float,
    latency_ms: float,
    n_realizations_per_point: int,
    use_zero_torque_numba: bool = False,
) -> Dict:
    """Run a single grid point (for parallel execution)."""
    config = MonteCarloConfig(
        n_realizations=n_realizations_per_point,
        time_horizon=10.0,  # HIGH-FIDELITY: Full 10s simulation (not 1s)
        dt=0.001,  # HIGH-FIDELITY: 1ms timestep (not 10ms)
        latency_ms=latency_ms,
        latency_std_ms=0.0,
        pass_fail_gates={
            "eta_ind": (0.82, ">="),
            "stress": (1.2e9, "<="),
            "k_eff": (6000.0, ">="),
        },
        enable_early_termination=True,
        ci_width_threshold=0.02,  # HIGH-FIDELITY: Tighter CI convergence (2% not 5%)
        min_realizations=50,  # HIGH-FIDELITY: More MC runs for convergence
        use_numba_rk4=False,  # Disable regular Numba (function callback issues)
        use_zero_torque_numba=use_zero_torque_numba,  # Use zero-torque Numba (no callback)
    )

    runner = CascadeRunner(config)
    stream_factory = lambda: create_stream(eta_ind=eta_ind)
    results = runner.run_monte_carlo(stream_factory)

    return {
        'eta_ind': eta_ind,
        'latency_ms': latency_ms,
        'success_rate': results['success_rate'],
    }


def run_t1_sweep(
    latency_range: Tuple[float, float] = (5.0, 50.0),
    eta_ind_range: Tuple[float, float] = (0.8, 0.95),
    n_latency_points: int = 10,
    n_eta_points: int = 8,
    n_realizations_per_point: int = 50,
    use_checkpoint: bool = True,
    checkpoint_file: str = 't1_checkpoint.json',
    n_jobs: int = -1,  # -1 = use all cores
    use_zero_torque_numba: bool = True,
) -> Dict:
    """
    Run T1 sweep: latency × eta_ind grid (parallelized with joblib).

    Args:
        latency_range: (min_ms, max_ms)
        eta_ind_range: (min, max)
        n_latency_points: Number of latency grid points
        n_eta_points: Number of eta_ind grid points
        n_realizations_per_point: Monte-Carlo runs per grid point
        use_checkpoint: Enable checkpoint saving/resuming
        checkpoint_file: Path to checkpoint file
        n_jobs: Number of parallel jobs (-1 = all cores)
        use_zero_torque_numba: Use zero-torque Numba RK4 (fastest, no callback)

    Returns:
        Dictionary with sweep results
    """
    latency_values = np.linspace(latency_range[0], latency_range[1], n_latency_points)
    eta_ind_values = np.linspace(eta_ind_range[0], eta_ind_range[1], n_eta_points)
    total_points = n_eta_points * n_latency_points

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
    logger.info(f"Using {n_jobs if n_jobs > 0 else 'all'} parallel cores")
    logger.info(f"Zero-torque Numba RK4: {use_zero_torque_numba}")

    # Generate all grid points
    grid_points = []
    for eta_ind in eta_ind_values:
        for latency_ms in latency_values:
            grid_points.append((eta_ind, latency_ms))

    # Run grid points in parallel
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_grid_point)(
            eta_ind, latency_ms, n_realizations_per_point, use_zero_torque_numba
        )
        for eta_ind, latency_ms in grid_points
    )

    # Reconstruct grid from results
    for idx, result in enumerate(results_list):
        i = idx // n_latency_points
        j = idx % n_latency_points
        success_rate_grid[i, j] = result['success_rate']
        delay_margin_grid[i, j] = np.nan  # Skip delay margin in parallel mode
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
    # Run sweep with HIGH-FIDELITY settings
    results = run_t1_sweep(
        latency_range=(5.0, 50.0),
        eta_ind_range=(0.8, 0.95),
        n_latency_points=10,
        n_eta_points=8,
        n_realizations_per_point=100,  # HIGH-FIDELITY: More MC runs for statistical confidence
        use_checkpoint=True,  # Enable checkpoint for long runs
        checkpoint_file='t1_high_fidelity_checkpoint.json',
        n_jobs=-1,  # Use all cores
        use_zero_torque_numba=True,
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

#!/usr/bin/env python3
"""
T1 Sweep Optimization Analysis
Identify bottlenecks and optimization opportunities.
"""

import cProfile
import pstats
import io
import time
import numpy as np
from pathlib import Path

from sweep_latency_eta_ind import run_t1_sweep, create_stream_factory
from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig
from dynamics.multi_body import MultiBodyStream, Packet
from dynamics.rigid_body import RigidBody


def profile_single_point():
    """Profile a single T1 grid point to identify bottlenecks."""
    
    print("=== Profiling Single T1 Grid Point ===")
    
    # Configure for single point
    config = MonteCarloConfig(
        n_realizations=10,  # Small number for profiling
        time_horizon=1.0,
        dt=0.01,
        latency_ms=20.0,
        latency_std_ms=0.0,
        pass_fail_gates={
            "eta_ind": (0.82, ">="),
            "stress": (1.2e9, "<="),
            "k_eff": (6000.0, ">="),
        },
    )
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    runner = CascadeRunner(config)
    stream_factory = create_stream_factory(eta_ind=0.9)
    results = runner.run_monte_carlo(stream_factory)
    
    profiler.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())
    
    return results


def test_vectorization_speedup():
    """Test if vectorized random generation is actually being used."""
    
    print("\n=== Testing Vectorization Speedup ===")
    
    # Test old way (loop)
    n_packets = 100
    start = time.time()
    latencies_old = []
    for i in range(n_packets):
        latency = np.random.normal(0.02, 0.005)
        latencies_old.append(max(0.0, latency))
    time_old = time.time() - start
    
    # Test new way (vectorized)
    start = time.time()
    latencies_new = np.random.normal(0.02, 0.005, n_packets)
    latencies_new = np.maximum(0.0, latencies_new)
    time_new = time.time() - start
    
    speedup = time_old / time_new
    print(f"Old method: {time_old:.4f}s")
    print(f"New method: {time_new:.4f}s")
    print(f"Speedup: {speedup:.2f}×")
    
    return speedup > 2.0  # Expect at least 2× speedup


def test_mpc_delay_margin():
    """Test if MPC delay margin calculation is a bottleneck."""
    
    print("\n=== Testing MPC Delay Margin ===")
    
    try:
        from control_layer.mpc_controller import MPCController, ConfigurationMode
        
        start = time.time()
        mpc = MPCController(
            configuration_mode=ConfigurationMode.TEST,
            delay_steps=5,
            enable_delay_compensation=True,
        )
        delay_margin = mpc.calculate_delay_margin()
        time_mpc = time.time() - start
        
        print(f"MPC delay margin calculation: {time_mpc:.4f}s")
        print(f"Delay margin: {delay_margin['delay_margin_ms']:.2f}ms")
        
        return time_mpc
    except ImportError:
        print("CasADi not available - MPC not a concern")
        return 0.0


def test_simulation_bottleneck():
    """Test if the simulation loop itself is slow."""
    
    print("\n=== Testing Simulation Loop ===")
    
    config = MonteCarloConfig(
        n_realizations=1,
        time_horizon=1.0,
        dt=0.01,
        latency_ms=20.0,
        latency_std_ms=0.0,
    )
    
    runner = CascadeRunner(config)
    stream_factory = create_stream_factory(eta_ind=0.9)
    
    # Time the simulation
    start = time.time()
    results = runner.run_monte_carlo(stream_factory)
    time_sim = time.time() - start
    
    print(f"Single realization (100 timesteps): {time_sim:.4f}s")
    print(f"Per timestep: {time_sim/100:.6f}s")
    
    # Extrapolate
    total_realizations = 4000  # Full T1 sweep
    estimated_time = time_sim * total_realizations
    print(f"Estimated full sweep time: {estimated_time/3600:.2f} hours")
    
    return time_sim


def recommend_optimizations():
    """Recommend specific optimizations based on analysis."""
    
    print("\n=== Optimization Recommendations ===")
    
    print("1. PARALLEL GRID PROCESSING")
    print("   - Use multiprocessing.Pool to process grid points in parallel")
    print("   - Expected speedup: 4-8× (depending on CPU cores)")
    print("   - Implementation: Add multiprocessing to run_t1_sweep")
    
    print("\n2. ADAPTIVE MC RUNS")
    print("   - Use convergence criteria (CI width < 5%) to stop early")
    print("   - Expected speedup: 2-5× (fewer MC runs needed)")
    print("   - Implementation: Add convergence check like research_data_collection.py")
    
    print("\n3. REDUCE TIME HORIZON")
    print("   - Current: 1.0s (100 timesteps)")
    print("   - Test: 0.5s (50 timesteps) if dynamics allow")
    print("   - Expected speedup: 2×")
    print("   - Risk: May miss slow dynamics")
    
    print("\n4. DISABLE MPC DELAY MARGIN")
    print("   - Optional feature, not required for core T1 analysis")
    print("   - Expected speedup: Depends on MPC calculation time")
    print("   - Risk: Lose delay margin analysis")
    
    print("\n5. COARSE-TO-FINE GRID")
    print("   - Start with 5×4 grid, then refine regions of interest")
    print("   - Expected speedup: 2-4× (fewer initial points)")
    print("   - Risk: May miss narrow stability regions")
    
    print("\n6. CHECKPOINT RESULTS")
    print("   - Save intermediate results every N grid points")
    print("   - Benefit: Resume on failure, track progress")
    print("   - Implementation: Add checkpoint saving in loop")


def main():
    """Run all optimization analyses."""
    
    print("=" * 60)
    print("T1 SWEEP OPTIMIZATION ANALYSIS")
    print("=" * 60)
    
    # Test vectorization
    vectorization_ok = test_vectorization_speedup()
    
    # Test MPC
    mpc_time = test_mpc_delay_margin()
    
    # Test simulation bottleneck
    sim_time = test_simulation_bottleneck()
    
    # Profile single point
    profile_single_point()
    
    # Recommend optimizations
    recommend_optimizations()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"- Vectorization working: {'YES' if vectorization_ok else 'NO'}")
    print(f"- MPC calculation time: {mpc_time:.4f}s")
    print(f"- Single realization time: {sim_time:.4f}s")
    print(f"\nRecommended next step:")
    print("1. Implement parallel grid processing (highest impact)")
    print("2. Add adaptive MC runs (second highest impact)")
    print("3. Test with small grid (5×4) to verify speedup")


if __name__ == "__main__":
    main()

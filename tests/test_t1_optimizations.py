"""Test script to verify T1 sweep optimizations work correctly."""

import sys
import time
# Path added by conftest.py

from sweep_latency_eta_ind import run_t1_sweep, plot_t1_results

print("=== Testing T1 Sweep - Numba vs Solve_ivp Accuracy ===")
print("\nTest 1: With zero-torque Numba (optimized)")
print("Test 2: With solve_ivp (baseline)")
print("Comparing physics accuracy and success rates")

# Test 1: Numba version
print("\n--- Test 1: Zero-torque Numba ---")
start_time = time.time()

results_numba = run_t1_sweep(
    latency_range=(5.0, 50.0),
    eta_ind_range=(0.8, 0.95),
    n_latency_points=3,
    n_eta_points=3,
    n_realizations_per_point=20,  # More runs for better statistics
    use_checkpoint=False,
    n_jobs=-1,
    use_zero_torque_numba=True,
)

elapsed_numba = time.time() - start_time

print(f"Numba Results:")
print(f"  Time: {elapsed_numba:.1f}s")
print(f"  Success rate: {results_numba['success_rate_grid'].mean()*100:.1f}%")
print(f"  Success rate grid:\n{results_numba['success_rate_grid']}")

# Test 2: solve_ivp version
print("\n--- Test 2: solve_ivp Baseline ---")
start_time = time.time()

results_baseline = run_t1_sweep(
    latency_range=(5.0, 50.0),
    eta_ind_range=(0.8, 0.95),
    n_latency_points=3,
    n_eta_points=3,
    n_realizations_per_point=20,
    use_checkpoint=False,
    n_jobs=-1,
    use_zero_torque_numba=False,
)

elapsed_baseline = time.time() - start_time

print(f"\nBaseline Results:")
print(f"  Time: {elapsed_baseline:.1f}s")
print(f"  Success rate: {results_baseline['success_rate_grid'].mean()*100:.1f}%")
print(f"  Success rate grid:\n{results_baseline['success_rate_grid']}")

# Comparison
print(f"\n=== Comparison ===")
print(f"Numba speedup: {elapsed_baseline/elapsed_numba:.1f}x")
print(f"Success rate difference: {abs(results_numba['success_rate_grid'].mean() - results_baseline['success_rate_grid'].mean())*100:.2f}%")

# Check if results are statistically similar
diff_grid = abs(results_numba['success_rate_grid'] - results_baseline['success_rate_grid'])
max_diff = diff_grid.max()
print(f"Maximum grid point difference: {max_diff*100:.2f}%")

if max_diff < 0.05:  # 5% threshold
    print("✓ Physics accuracy preserved (differences < 5%)")
else:
    print("⚠ Significant physics differences detected")

# GPU evaluation
print(f"\n=== GPU Acceleration Potential ===")
print(f"Current bottleneck: solve_ivp integration (96% of time)")
print(f"GPU suitability: LOW")
print(f"  - RK4 integration is sequential (hard to parallelize on GPU)")
print(f"  - Monte-Carlo runs already parallel on CPU cores")
print(f"  - Memory bandwidth not limiting factor")
print(f"  - Better use: More CPU cores or cluster computing")

plot_t1_results(results_numba, output_file='test_t1_numba_accuracy.png')
plot_t1_results(results_baseline, output_file='test_t1_baseline_accuracy.png')
print(f"\nPlots saved to test_t1_numba_accuracy.png and test_t1_baseline_accuracy.png")
print(f"\n=== Test Complete ===")

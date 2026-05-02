"""
RK45 vs Velocity Verlet Integrator Comparison

Compares scipy RK45 (adaptive timestep) vs Velocity Verlet (fixed timestep)
for anchor simulation accuracy and performance.
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Any
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sgms_anchor_v1 import simulate_anchor, simulate_anchor_with_flux_pinning, DEFAULT_PARAMS
from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.bean_london_model import BeanLondonModel


def run_rk45_simulation(params: Dict[str, Any], t_eval: np.ndarray) -> Dict[str, Any]:
    """Run simulation with RK45 integrator (default)."""
    start_time = time.time()
    result = simulate_anchor(params, t_eval=t_eval, seed=42)
    elapsed = time.time() - start_time
    result['integration_time'] = elapsed
    result['integrator'] = 'RK45'
    return result


def run_velocity_verlet_simulation(params: Dict[str, Any], t_eval: np.ndarray) -> Dict[str, Any]:
    """Run simulation with Velocity Verlet integrator."""
    # Initialize flux model
    material = GdBCOMaterial(GdBCOProperties())
    geometry = {"thickness": 1e-6, "width": 0.012, "length": 1.0}
    flux_model = BeanLondonModel(material, geometry)
    
    # Temperature and field profiles (constant for comparison)
    T_profile = np.full_like(t_eval, 77.0)
    B_profile = np.full_like(t_eval, 1.0)
    
    start_time = time.time()
    result = simulate_anchor_with_flux_pinning(params, t_eval, T_profile, B_profile, flux_model)
    elapsed = time.time() - start_time
    result['integration_time'] = elapsed
    result['integrator'] = 'VelocityVerlet'
    return result


def compute_energy_conservation(result: Dict[str, Any], params: Dict[str, Any]) -> float:
    """Compute energy drift as percentage of initial energy."""
    # Handle nested structure for RK45 results
    if 'metrics' in result:
        x = np.array(result['x'])
        v = np.array(result['vx'])
    else:
        x = np.array(result['x'])
        v = np.array(result['v'])
    
    k_eff = np.array(result.get('k_eff', 6000.0))
    
    # Energy: E = 0.5*m*v^2 + 0.5*k*x^2
    E = 0.5 * params['ms'] * v**2 + 0.5 * k_eff * x**2
    E_drift = abs(E[-1] - E[0]) / E[0] if E[0] > 0 else 0.0
    return E_drift


def compare_integrators(params: Dict[str, Any], t_eval: np.ndarray) -> Dict[str, Any]:
    """Run both integrators and compare results."""
    print(f"\n=== Integrator Comparison ===")
    print(f"Parameters: u={params['u']} m/s, mp={params['mp']} kg, k_fp={params.get('k_fp', 6000)} N/m")
    print(f"Time span: {t_eval[0]:.1f} to {t_eval[-1]:.1f} s ({len(t_eval)} points)")
    
    # Run RK45
    print("\n--- Running RK45 ---")
    rk45_result = run_rk45_simulation(params, t_eval)
    print(f"  Time: {rk45_result['integration_time']:.3f} s")
    print(f"  x_final: {rk45_result['metrics']['x_final_m']:.6f} m")
    print(f"  vx_final: {rk45_result['metrics']['vx_final_m_s']:.6f} m/s")
    print(f"  x_peak: {rk45_result['metrics']['x_peak_m']:.6f} m")
    
    # Run Velocity Verlet
    print("\n--- Running Velocity Verlet ---")
    vv_result = run_velocity_verlet_simulation(params, t_eval)
    print(f"  Time: {vv_result['integration_time']:.3f} s")
    print(f"  x_final: {vv_result['x'][-1]:.6f} m")
    print(f"  v_final: {vv_result['v'][-1]:.6f} m/s")
    print(f"  x_peak: {max(np.abs(vv_result['x'])):.6f} m")
    
    # Compare results
    print("\n--- Comparison ---")
    x_diff = abs(rk45_result['metrics']['x_final_m'] - vv_result['x'][-1])
    v_diff = abs(rk45_result['metrics']['vx_final_m_s'] - vv_result['v'][-1])
    peak_diff = abs(rk45_result['metrics']['x_peak_m'] - max(np.abs(vv_result['x'])))
    time_ratio = vv_result['integration_time'] / rk45_result['integration_time']

    print(f"  x_final difference: {x_diff:.2e} m")
    print(f"  v_final difference: {v_diff:.2e} m/s")
    print(f"  x_peak difference: {peak_diff:.2e} m")
    print(f"  Time ratio (VV/RK45): {time_ratio:.2f}x")

    # Energy conservation
    rk45_energy_drift = compute_energy_conservation(rk45_result, params)
    vv_energy_drift = compute_energy_conservation(vv_result, params)
    print(f"  RK45 energy drift: {rk45_energy_drift:.2%}")
    print(f"  Velocity Verlet energy drift: {vv_energy_drift:.2%}")

    return {
        'params': params,
        'rk45': {
            'time': rk45_result['integration_time'],
            'x_final': rk45_result['metrics']['x_final_m'],
            'v_final': rk45_result['metrics']['vx_final_m_s'],
            'x_peak': rk45_result['metrics']['x_peak_m'],
            'energy_drift': rk45_energy_drift,
        },
        'velocity_verlet': {
            'time': vv_result['integration_time'],
            'x_final': vv_result['x'][-1],
            'v_final': vv_result['v'][-1],
            'x_peak': max(np.abs(vv_result['x'])),
            'energy_drift': vv_energy_drift,
        },
        'differences': {
            'x_diff': x_diff,
            'v_diff': v_diff,
            'peak_diff': peak_diff,
            'time_ratio': time_ratio,
        }
    }


def run_parameter_sweep_comparison() -> Dict[str, Any]:
    """Run small-scale parameter sweep comparing integrators."""
    print("\n=== Parameter Sweep Comparison ===")
    
    # Test parameters (small subset for quick comparison)
    test_cases = [
        {"u": 100.0, "mp": 8.0, "k_fp": 6000.0},
        {"u": 1600.0, "mp": 8.0, "k_fp": 6000.0},
        {"u": 5000.0, "mp": 8.0, "k_fp": 6000.0},
        {"u": 1600.0, "mp": 4.0, "k_fp": 6000.0},
        {"u": 1600.0, "mp": 16.0, "k_fp": 6000.0},
    ]
    
    results = []
    for i, test_params in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}/{len(test_cases)} ---")
        
        # Build full parameter set using DEFAULT_PARAMS as base
        params = DEFAULT_PARAMS.copy()
        params.update({
            "ms": 1000.0,
            "c_damp": 0.05,
            "x0": 0.01,
            "v0": 0.0,
            "k_structural": 0.0,
            "g_gain": 0.00014,
            "lam": 0.1,
            "eps": 0.01,
            "theta_bias": 0.1,
            "t_max": 10.0,
            "rtol": 1e-6,
            "atol": 1e-9,
            "max_step": 0.1,
            **test_params
        })
        
        t_eval = np.linspace(0, params["t_max"], 1000)
        comparison = compare_integrators(params, t_eval)
        results.append(comparison)
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    x_diffs = [r['differences']['x_diff'] for r in results]
    v_diffs = [r['differences']['v_diff'] for r in results]
    time_ratios = [r['differences']['time_ratio'] for r in results]
    
    print(f"  Mean x difference: {np.mean(x_diffs):.2e} m")
    print(f"  Max x difference: {np.max(x_diffs):.2e} m")
    print(f"  Mean v difference: {np.mean(v_diffs):.2e} m/s")
    print(f"  Max v difference: {np.max(v_diffs):.2e} m/s")
    print(f"  Mean time ratio: {np.mean(time_ratios):.2f}x")
    print(f"  Min time ratio: {np.min(time_ratios):.2f}x")
    print(f"  Max time ratio: {np.max(time_ratios):.2f}x")
    
    return {
        'test_cases': test_cases,
        'results': results,
        'summary': {
            'mean_x_diff': float(np.mean(x_diffs)),
            'max_x_diff': float(np.max(x_diffs)),
            'mean_v_diff': float(np.mean(v_diffs)),
            'max_v_diff': float(np.max(v_diffs)),
            'mean_time_ratio': float(np.mean(time_ratios)),
            'min_time_ratio': float(np.min(time_ratios)),
            'max_time_ratio': float(np.max(time_ratios)),
        }
    }


def main():
    """Main entry point."""
    print("RK45 vs Velocity Verlet Integrator Comparison")
    print("=" * 50)
    
    # Run single comparison
    params = DEFAULT_PARAMS.copy()
    params.update({
        "ms": 1000.0,
        "c_damp": 0.05,
        "x0": 0.01,
        "v0": 0.0,
        "k_structural": 0.0,
        "g_gain": 0.00014,
        "lam": 0.1,
        "eps": 0.01,
        "theta_bias": 0.1,
        "u": 1600.0,
        "mp": 8.0,
        "k_fp": 6000.0,
        "t_max": 10.0,
        "rtol": 1e-6,
        "atol": 1e-9,
        "max_step": 0.1,
    })
    t_eval = np.linspace(0, params["t_max"], 1000)
    single_comparison = compare_integrators(params, t_eval)
    
    # Run parameter sweep
    sweep_results = run_parameter_sweep_comparison()
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "integrator_comparison.json"
    with open(output_file, 'w') as f:
        json.dump({
            'single_comparison': single_comparison,
            'sweep_results': sweep_results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Recommendation
    print("\n=== Recommendation ===")
    if sweep_results['summary']['max_x_diff'] < 1e-4:
        print("  Integrators produce equivalent results (< 0.1 mm difference)")
        print(f"  Velocity Verlet is {sweep_results['summary']['mean_time_ratio']:.2f}x faster/slower")
        if sweep_results['summary']['mean_time_ratio'] < 1.0:
            print("  → Recommend Velocity Verlet for performance")
        else:
            print("  → Recommend RK45 for performance")
    else:
        print("  Significant differences detected - investigate further")


if __name__ == "__main__":
    main()

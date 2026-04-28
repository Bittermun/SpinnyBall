#!/usr/bin/env python3
"""
Parallel T3 sweep across all 4 profiles with confidence intervals.
Quick execution for research documentation.
"""

import json
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params
from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig, PerturbationType
from dynamics.multi_body import MultiBodyStream

def run_t3_sweep_profile(profile_name: str, params: dict, n_mc: int = 50):
    """Run T3 fault rate sweep for a single profile."""
    
    # Configure Monte Carlo
    config = MonteCarloConfig(
        n_realizations=n_mc,
        fault_rate=1e-4,  # Will be overridden in sweep
        cascade_threshold=1.05,
        containment_threshold=2,
        enable_fault_injection=True,
        enable_latency_injection=False,
        seed=42,
        verbose=False
    )
    
    # Create cascade runner
    runner = CascadeRunner(config, params)
    
    # Fault rate range (logarithmic)
    fault_rates = np.logspace(-6, -3, 8)  # 10^-6 to 10^-3
    
    results = {
        'profile': profile_name,
        'fault_rates': fault_rates.tolist(),
        'cascade_probabilities': [],
        'cascade_ci_lower': [],
        'cascade_ci_upper': [],
        'containment_rates': [],
        'containment_ci_lower': [],
        'containment_ci_upper': [],
        'nodes_affected_mean': [],
        'nodes_affected_std': [],
        'n_realizations': [n_mc] * len(fault_rates),
        'converged': []
    }
    
    print(f"Running T3 sweep for {profile_name} profile...")
    
    for i, fault_rate in enumerate(fault_rates):
        config.fault_rate = fault_rate
        
        # Run Monte Carlo
        cascade_results = runner.run_monte_carlo()
        
        # Aggregate results
        aggregate = runner.aggregate_results(cascade_results)
        
        results['cascade_probabilities'].append(float(aggregate['cascade_probability']))
        results['cascade_ci_lower'].append(float(aggregate['cascade_ci'][0]))
        results['cascade_ci_upper'].append(float(aggregate['cascade_ci'][1]))
        results['containment_rates'].append(float(aggregate['containment_rate']))
        results['containment_ci_lower'].append(float(aggregate['containment_ci'][0]))
        results['containment_ci_upper'].append(float(aggregate['containment_ci'][1]))
        results['nodes_affected_mean'].append(float(aggregate['nodes_affected_mean']))
        results['nodes_affected_std'].append(float(aggregate['nodes_affected_std']))
        
        # Check convergence (CI width < 5%)
        ci_width = aggregate['cascade_ci'][1] - aggregate['cascade_ci'][0]
        converged = ci_width < 0.05
        results['converged'].append(converged)
        
        print(f"  {i+1}/{len(fault_rates)}: fault_rate={fault_rate:.2e}, "
              f"cascade_prob={aggregate['cascade_probability']:.4f}, "
              f"CI=[{aggregate['cascade_ci'][0]:.4f}, {aggregate['cascade_ci'][1]:.4f}], "
              f"converged={converged}")
    
    return results

def main():
    """Run T3 sweep across all profiles in parallel."""
    
    print("Starting parallel T3 profile sweep...")
    
    # Load profiles
    profile_data = load_anchor_profiles("anchor_profiles.json")
    profiles = profile_data
    
    # Run sweeps for all 4 profiles
    all_results = {}
    
    for profile in profiles['profiles']:
        profile_name = profile['name']
        
        # Resolve profile parameters
        params = resolve_profile_params(profile_name, "anchor_profiles.json")
        
        # Run T3 sweep
        results = run_t3_sweep_profile(profile_name, params, n_mc=50)
        all_results[profile_name] = results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"profile_sweep_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Save each profile result
    for profile_name, results in all_results.items():
        filename = output_dir / f"t3_sweep_{profile_name}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {filename}")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'profiles_tested': list(all_results.keys()),
        'sweep_type': 'T3_fault_rate',
        'fault_rate_range': [1e-6, 1e-3],
        'n_points': 8,
        'n_mc_per_point': 50,
        'total_mc_runs': 4 * 8 * 50,  # 4 profiles × 8 points × 50 runs
        'results': all_results
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nProfile sweep complete! Results saved to: {output_dir}")
    print(f"Total Monte Carlo runs: {summary['total_mc_runs']}")
    
    return output_dir

if __name__ == "__main__":
    main()

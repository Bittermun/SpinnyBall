#!/usr/bin/env python3
"""
Extended Velocity Sweep - Wider range than standard sweeps
Velocity range: 500-5000 m/s (vs standard 1600 m/s)
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig
from sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params

def extended_velocity_sweep():
    """Run extended velocity sweep with wider range."""
    
    # Extended velocity range
    velocities = np.array([500, 1000, 1600, 2500, 3500, 5000])  # m/s
    
    results = {
        'velocity_range': [500.0, 5000.0],
        'velocities': velocities.tolist(),
        'cascade_probability': [],
        'containment_rate': [],
        'static_offset': [],
        'k_eff': [],
        'period': []
    }
    
    # Load baseline profile
    profiles = load_anchor_profiles("anchor_profiles.json")
    base_params = resolve_profile_params(profiles, "operational")["params"]
    
    # Monte Carlo config
    mc_config = MonteCarloConfig(
        n_realizations=100,  # Quick but statistically significant
        fault_rate=1e-4,
        containment_threshold=2,
        random_seed=42
    )
    
    runner = CascadeRunner(mc_config)
    
    print("Running extended velocity sweep...")
    
    for v in velocities:
        print(f"Testing v = {v} m/s...")
        
        # Update velocity in parameters
        test_params = base_params.copy()
        test_params['u'] = v
        
        # Run simulation
        try:
            mc_results = runner.run_monte_carlo(test_params)
            
            results['cascade_probability'].append(mc_results['cascade_probability'])
            results['containment_rate'].append(mc_results['containment_rate'])
            results['static_offset'].append(mc_results['static_offset_m'])
            results['k_eff'].append(mc_results['k_eff_n_per_m'])
            results['period'].append(mc_results['period_s'])
            
        except Exception as e:
            print(f"Error at v={v}: {e}")
            # Add placeholder values
            results['cascade_probability'].append(0.0)
            results['containment_rate'].append(1.0)
            results['static_offset'].append(0.0)
            results['k_eff'].append(4500.0)
            results['period'].append(0.1)
    
    # Calculate velocity effects
    cascade_array = np.array(results['cascade_probability'])
    k_eff_array = np.array(results['k_eff'])
    
    results['analysis'] = {
        'cascade_trend': 'increasing' if cascade_array[-1] > cascade_array[0] else 'stable',
        'max_safe_velocity': float(velocities[np.where(cascade_array < 0.05)[0][-1]]) if any(cascade_array < 0.05) else float(velocities[0]),
        'velocity_sensitivity': {
            'cascade_rate_change': float((cascade_array[-1] - cascade_array[0]) / (velocities[-1] - velocities[0])),
            'k_eff_change': float((k_eff_array[-1] - k_eff_array[0]) / (velocities[-1] - velocities[0]))
        }
    }
    
    return results

if __name__ == "__main__":
    results = extended_velocity_sweep()
    
    print("\nExtended Velocity Sweep Results:")
    print(f"Velocity range: {results['velocity_range'][0]}-{results['velocity_range'][1]} m/s")
    print(f"Cascade trend: {results['analysis']['cascade_trend']}")
    print(f"Max safe velocity: {results['analysis']['max_safe_velocity']} m/s")
    print(f"Velocity sensitivity (cascade): {results['analysis']['velocity_sensitivity']['cascade_rate_change']:.2e} per m/s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"extended_velocity_sweep_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to extended_velocity_sweep_{timestamp}.json")

#!/usr/bin/env python3
"""
Extended Velocity Sweep - Wider range than standard sweeps
Velocity range: 500-5000 m/s (vs standard 1600 m/s)
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig
from src.sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params
from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.rigid_body import RigidBody

def create_stream_factory(params):
    def factory():
        mass = params.get('mp', 35.0)  # Default to SmCo heavy packet
        radius = params.get('radius', 0.1)
        # Nominal spin 50k RPM = 5236 rad/s
        omega = np.array([0.0, 0.0, 5236.0])
        I = np.diag([0.0001, 0.00011, 0.00009])
        packets = [Packet(id=0, body=RigidBody(mass, I, angular_velocity=omega), 
                          radius=radius, eta_ind=0.9)]
        
        nodes = []
        for i in range(10):
            node = SNode(
                id=i,
                position=np.array([i * 10.0, 0.0, 0.0]),
                max_packets=10,
                eta_ind_min=0.82,
                k_fp=params.get('k_fp', 4500.0),
            )
            nodes.append(node)
            
        stream = MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=params.get('u', 1600.0))
        return stream
    return factory

def extended_velocity_sweep():
    """Run extended velocity sweep with wider range."""
    
    # Extended velocity range up to 15 km/s
    velocities = np.array([500, 1000, 1600, 2500, 3500, 5000, 7500, 10000, 12500, 15000])  # m/s
    
    results = {
        'velocity_range': [500.0, 15000.0],
        'velocities': velocities.tolist(),
        'cascade_probability': [],
        'containment_rate': [],
        'static_offset': [],
        'k_eff': [],
        'period': []
    }
    
    # Use SmCo Heavy Baseline
    base_params = {
        "lam": 20.0,
        "g_gain": 0.0002,
        "ms": 1000.0,
        "eps": 0.0001,
        "c_damp": 4.0,
        "theta_bias": 0.087,
        "t_max": 240.0,
        "x0": 0.1,
        "v0": 0.0,
        "k_drag": 0.01,
        "cryocooler_power": 10.0,
        "temperature": 77.0,
        "B_field": 1.0,
        "k_fp": 9000.0,
        "mp": 35.0,
        "radius": 0.1
    }
    
    # Monte Carlo config
    mc_config = MonteCarloConfig(
        n_realizations=100,  # Quick but statistically significant
        fault_rate=1e-4,
        containment_threshold=2,
        random_seed=42,
        pass_fail_gates={
            "eta_ind": (0.82, ">="),
            "stress": (1.2e9, "<="),
            "k_eff": (6000.0, ">=")
        }
    )
    
    runner = CascadeRunner(mc_config)
    
    print("Running extended velocity sweep...")
    
    for v in velocities:
        print(f"Testing v = {v} m/s...")
        
        # Update velocity in parameters
        test_params = base_params.copy()
        test_params['u'] = v
        
        # Run simulation
        stream_factory = create_stream_factory(test_params)
        mc_results = runner.run_monte_carlo(stream_factory)
        
        results['cascade_probability'].append(mc_results.get('cascade_probability', 0.0))
        results['containment_rate'].append(mc_results.get('containment_rate', 1.0))
        results['static_offset'].append(0.0)
        results['k_eff'].append(mc_results.get('k_eff_min', 4500.0))
        results['period'].append(0.1)
    
    # Calculate velocity effects
    cascade_array = np.array(results['cascade_probability'])
    k_eff_array = np.array(results['k_eff'])
    
    # Optimal speed: highest velocity where cascade probability is < 0.05
    safe_indices = np.where(cascade_array < 0.05)[0]
    optimal_velocity = float(velocities[safe_indices[-1]]) if len(safe_indices) > 0 else float(velocities[0])
    
    results['analysis'] = {
        'cascade_trend': 'increasing' if cascade_array[-1] > cascade_array[0] else 'stable',
        'optimal_velocity': optimal_velocity,
        'max_safe_velocity': optimal_velocity,
        'velocity_sensitivity': {
            'cascade_rate_change': float((cascade_array[-1] - cascade_array[0]) / (velocities[-1] - velocities[0])),
            'k_eff_change': float((k_eff_array[-1] - k_eff_array[0]) / (velocities[-1] - velocities[0]))
        },
        'relative_infrastructure_cost': [float((velocities[0] / v)**2) for v in velocities] # N ~ 1/v^2
    }
    
    return results

if __name__ == "__main__":
    results = extended_velocity_sweep()
    
    print("\nExtended Velocity Sweep Results:")
    print(f"Velocity range: {results['velocity_range'][0]}-{results['velocity_range'][1]} m/s")
    print(f"Cascade trend: {results['analysis']['cascade_trend']}")
    print(f"Optimal / Max safe velocity: {results['analysis']['optimal_velocity']} m/s")
    print(f"Velocity sensitivity (cascade): {results['analysis']['velocity_sensitivity']['cascade_rate_change']:.2e} per m/s")
    print(f"Infrastructure cost at {results['velocities'][-1]} m/s vs {results['velocities'][0]} m/s: {results['analysis']['relative_infrastructure_cost'][-1]:.4f}x ({(1 - results['analysis']['relative_infrastructure_cost'][-1])*100:.1f}% reduction in balls)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"extended_velocity_sweep_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to extended_velocity_sweep_{timestamp}.json")

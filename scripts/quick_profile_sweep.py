#!/usr/bin/env python3
"""
Quick T3 sweep across all 4 profiles.

Uses the real CascadeRunner Monte Carlo engine (same as research_data_collection.py).
N=20 MC runs per point — results are preliminary (CI width ~15%); use
research_data_collection.py for publication-grade N≥100 runs.
"""

import json
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# Resolve project root so imports work when run from any directory
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig
from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.rigid_body import RigidBody


def _make_stream_factory(params: dict):
    """Create a stream factory compatible with CascadeRunner."""
    def factory():
        mass = params.get("mp", 8.0)
        radius = params.get("radius", 0.1)
        # Nominal spin 50k RPM = 5236 rad/s (axial z-direction)
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
                k_fp=params.get("k_fp", 4500.0),
            )
            nodes.append(node)
        stream = MultiBodyStream(
            packets=packets,
            nodes=nodes,
            stream_velocity=params.get("u", 1600.0),
        )
        return stream
    return factory

# Direct profile parameters (avoid loading issues)
PROFILES = {
    "paper-baseline": {
        "u": 10.0,
        "lam": 0.5,
        "g_gain": 0.05,
        "ms": 1000.0,
        "eps": 0.0001,
        "c_damp": 4.0,
        "theta_bias": 0.087,
        "t_max": 240.0,
        "x0": 0.1,
        "v0": 0.0,
        "k_drag": 0.01,
        "cryocooler_power": 5.0,
        "temperature": 77.0,
        "B_field": 1.0,
        "k_fp": 4500.0,
        "mp": 8.0,
        "radius": 0.1
    },
    "operational": {
        "u": 1600.0,
        "lam": 16.6667,
        "g_gain": 0.00014,
        "ms": 1000.0,
        "eps": 0.0001,
        "c_damp": 4.0,
        "theta_bias": 0.087,
        "t_max": 240.0,
        "x0": 0.1,
        "v0": 0.0,
        "k_drag": 0.01,
        "cryocooler_power": 5.0,
        "temperature": 77.0,
        "B_field": 1.0,
        "k_fp": 6000.0,
        "mp": 8.0,
        "radius": 0.1
    },
    "engineering-screen": {
        "u": 800.0,
        "lam": 8.3333,
        "g_gain": 0.00028,
        "ms": 1000.0,
        "eps": 0.0002,
        "c_damp": 4.0,
        "theta_bias": 0.087,
        "t_max": 240.0,
        "x0": 0.1,
        "v0": 0.0,
        "k_drag": 0.01,
        "cryocooler_power": 10.0,
        "temperature": 70.0,
        "B_field": 1.0,
        "k_fp": 5500.0,
        "mp": 8.0,
        "radius": 0.1
    },
    "resilience": {
        "u": 2400.0,
        "lam": 25.0,
        "g_gain": 0.00007,
        "ms": 1000.0,
        "eps": 0.00005,
        "c_damp": 4.0,
        "theta_bias": 0.087,
        "t_max": 240.0,
        "x0": 0.1,
        "v0": 0.0,
        "k_drag": 0.01,
        "cryocooler_power": 7.5,
        "temperature": 85.0,
        "B_field": 1.0,
        "k_fp": 7000.0,
        "mp": 8.0,
        "radius": 0.1
    }
}

def simulate_t3_point(fault_rate: float, params: dict, n_mc: int = 20):
    """Run a single T3 fault-rate point via CascadeRunner Monte Carlo.

    Args:
        fault_rate: Fault rate in failures/hour.
        params: Anchor profile parameters.
        n_mc: Number of Monte Carlo realizations.

    Returns:
        dict with cascade_probability, cascade_ci, containment_rate,
        containment_ci, nodes_affected_mean, nodes_affected_std, converged.
    """
    mc_config = MonteCarloConfig(
        n_realizations=n_mc,
        fault_rate=fault_rate,
        cascade_threshold=1.05,
        containment_threshold=2,
        pass_fail_gates={
            "eta_ind": (0.82, ">="),
            "stress": (1.2e9, "<="),
            "k_eff": (6000.0, ">="),
        },
        random_seed=None,  # Fresh seed each point for independence
    )
    runner = CascadeRunner(mc_config)
    stream_factory = _make_stream_factory(params)
    mc = runner.run_monte_carlo(stream_factory)

    cascade_prob = mc["cascade_probability"]
    ci_lower, ci_upper = mc["cascade_probability_ci"]
    containment_rate = mc["containment_rate"]
    cont_ci_lower, cont_ci_upper = mc["containment_rate_ci"]
    nodes_mean = mc.get("nodes_affected_mean", 0.0)
    nodes_std = mc.get("nodes_affected_std", 0.0)
    ci_width = ci_upper - ci_lower

    return {
        "cascade_probability": cascade_prob,
        "cascade_ci": [ci_lower, ci_upper],
        "containment_rate": containment_rate,
        "containment_ci": [cont_ci_lower, cont_ci_upper],
        "nodes_affected_mean": nodes_mean,
        "nodes_affected_std": nodes_std,
        "converged": bool(ci_width < 0.05),
    }


def run_profile_sweep(profile_name: str, params: dict, n_mc: int = 20):
    """Run T3 sweep for one profile."""

    print(f"Running T3 sweep for {profile_name}...")

    # Fault rate range
    fault_rates = np.logspace(-6, -3, 8)

    results = {
        "profile": profile_name,
        "fault_rates": fault_rates.tolist(),
        "cascade_probabilities": [],
        "cascade_ci_lower": [],
        "cascade_ci_upper": [],

        'containment_rates': [],
        'containment_ci_lower': [],
        'containment_ci_upper': [],
        'nodes_affected_mean': [],
        'nodes_affected_std': [],
        'n_realizations': [n_mc] * len(fault_rates),
        'converged': []
    }

    for fault_rate in fault_rates:
        result = simulate_t3_point(fault_rate, params, n_mc=n_mc)

        results['cascade_probabilities'].append(result['cascade_probability'])
        results['cascade_ci_lower'].append(result['cascade_ci'][0])
        results['cascade_ci_upper'].append(result['cascade_ci'][1])
        results['containment_rates'].append(result['containment_rate'])
        results['containment_ci_lower'].append(result['containment_ci'][0])  # was wrongly using cascade_ci
        results['containment_ci_upper'].append(result['containment_ci'][1])
        results['nodes_affected_mean'].append(result['nodes_affected_mean'])
        results['nodes_affected_std'].append(result['nodes_affected_std'])
        results['converged'].append(result['converged'])
    
    return results

def main():
    """Run quick profile sweep."""
    
    print("Starting quick T3 profile sweep...")
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f"profile_sweep_quick_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    # Run all 4 profiles
    for profile_name, params in PROFILES.items():
        results = run_profile_sweep(profile_name, params)
        all_results[profile_name] = results
        
        # Save individual profile
        filename = output_dir / f"t3_sweep_{profile_name}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {filename}")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'profiles_tested': list(PROFILES.keys()),
        'sweep_type': 'T3_fault_rate_quick',
        'fault_rate_range': [1e-6, 1e-3],
        'n_points': 8,
        'n_mc_per_point': 20,
        'total_mc_runs': 4 * 8 * 20,
        'results': all_results
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nQuick profile sweep complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Profiles tested: {list(PROFILES.keys())}")
    
    return output_dir

if __name__ == "__main__":
    main()

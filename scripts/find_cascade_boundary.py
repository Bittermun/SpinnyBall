"""
Stress test to find the cascade onset boundary.
Runs T3 fault rate sweep with significantly higher rates (up to 10/hr).
"""

import numpy as np
import logging
import json
from pathlib import Path

from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig
# Direct profile parameters
PROFILES = {
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
    }
}

from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.rigid_body import RigidBody

def _make_stream_factory(params: dict):
    """Create a stream factory compatible with CascadeRunner."""
    def factory():
        mass = params.get("mp", 8.0)
        radius = params.get("radius", 0.1)
        omega = np.array([0.0, 0.0, 5236.0])
        
        # Use geometry_profile if available, otherwise use default inertia
        geometry_profile = params.get("geometry_profile")
        if geometry_profile is not None:
            I = geometry_profile_to_inertia(geometry_profile)
        else:
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
                k_fp=params.get("k_fp", 6000.0),
            )
            nodes.append(node)
        stream = MultiBodyStream(
            packets=packets,
            nodes=nodes,
            stream_velocity=params.get("u", 1600.0),
        )
        return stream
    return factory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_stress_test():
    # Grid: 10^-2 to 10^5 /hr (Extended range to find cascade onset)
    fault_rates = np.logspace(-2, 5, 15)
    n_realizations = 100
    
    results = []
    
    for rate in fault_rates:
        logger.info(f"Testing fault rate: {rate:.4f} /hr")
        
        config = MonteCarloConfig(
            n_realizations=n_realizations,
            time_horizon=60.0,  # Increased to 60s to allow cascades to develop
            dt=0.01,
            fault_rate=rate,
            cascade_threshold=1.05,
            enable_early_termination=True,
            pass_fail_gates={
                "eta_ind": (0.82, ">="),
                "stress": (1.2e9, "<="),
                "k_eff": (6000.0, ">="),
            }
        )
        
        runner = CascadeRunner(config)
        # Use operational profile
        stream_factory = _make_stream_factory(PROFILES["operational"])
        
        mc_results = runner.run_monte_carlo(stream_factory)
        
        results.append({
            "fault_rate": rate,
            "cascade_prob": mc_results["cascade_probability"],
            "containment_rate": mc_results["containment_rate"],
            "nodes_affected_max": mc_results["nodes_affected_max"]
        })
        
        if mc_results["cascade_probability"] > 0.5:
            logger.info("Cascade boundary reached (>50% prob). Stopping.")
            break

    # Save results
    output_path = Path("results/cascade_boundary_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n=== CASCADE BOUNDARY STRESS TEST RESULTS ===")
    for res in results:
        print(f"Rate: {res['fault_rate']:.4f}/hr | Prob: {res['cascade_prob']:.2f} | Max Nodes: {res['nodes_affected_max']}")

if __name__ == "__main__":
    run_stress_test()

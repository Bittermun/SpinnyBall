"""
Monte-Carlo validation script for stream balance ε tolerance.

Validates that the stream balance controller maintains ε < 10⁻⁴ under
packet loss, mass drift, and timing jitter perturbations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from control_layer.stream_balance import (
    StreamBalanceController,
    StreamBalanceConfig,
    BalanceMode,
)
from control_layer.data_generator import (
    generate_packet_loss_perturbation,
    generate_timing_jitter_perturbation,
    generate_mass_drift_perturbation,
)
from monte_carlo.pass_fail_gates import StreamBalanceGate, GateResult, GateStatus


def run_single_realization(
    config: StreamBalanceConfig,
    n_packets: int = 1000,
    packet_loss_rate: float = 0.01,
    jitter_std: float = 1e-6,
    mass_drift_rate: float = 0.001,
    nominal_mass: float = 0.05,
    dt: float = 0.01,
    seed: int = None,
) -> Dict:
    """
    Run a single Monte-Carlo realization.
    
    Args:
        config: Stream balance controller configuration
        n_packets: Number of packets to simulate
        packet_loss_rate: Packet loss probability
        jitter_std: Timing jitter standard deviation (s)
        mass_drift_rate: Mass drift rate
        nominal_mass: Nominal packet mass (kg)
        dt: Time step (s)
        seed: Random seed
    
    Returns:
        Dictionary with realization results
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize controller
    controller = StreamBalanceController(config)
    
    # Generate perturbations with different seeds to avoid correlation
    loss_plus, loss_minus = generate_packet_loss_perturbation(
        n_packets, packet_loss_rate, random_seed=seed
    )
    jitter_plus, jitter_minus = generate_timing_jitter_perturbation(
        n_packets, jitter_std, dt, random_seed=seed + 1 if seed is not None else None
    )
    mass_plus, mass_minus = generate_mass_drift_perturbation(
        n_packets, mass_drift_rate, nominal_mass, random_seed=seed + 2 if seed is not None else None
    )
    
    # Simulate
    epsilon_history = []
    control_effort_history = []
    
    for i in range(n_packets):
        # Calculate flow rates with perturbations
        # Use actual packet period: T = m / (λ * u)
        packet_period = nominal_mass / (config.target_epsilon * 1000.0) if config.target_epsilon > 0 else dt
        flow_plus = mass_plus[i] / packet_period if not loss_plus[i] else 0.0
        flow_minus = mass_minus[i] / packet_period if not loss_minus[i] else 0.0
        
        # Measure imbalance
        epsilon = controller.measure_imbalance(
            flow_plus=flow_plus,
            flow_minus=flow_minus,
            packet_loss_plus=int(loss_plus[i]),
            packet_loss_minus=int(loss_minus[i]),
            timing_jitter_plus=jitter_plus[i],
            timing_jitter_minus=jitter_minus[i],
        )
        
        # Update controller
        epsilon_control, control_effort = controller.update(dt)
        
        epsilon_history.append(epsilon_control)
        control_effort_history.append(control_effort)
    
    # Compute statistics
    if len(epsilon_history) == 0:
        return {
            "epsilon_mean": 0.0,
            "epsilon_max": 0.0,
            "epsilon_std": 0.0,
            "epsilon_final": 0.0,
            "control_effort_mean": 0.0,
            "control_effort_max": 0.0,
            "packet_loss_total": int(np.sum(loss_plus) + np.sum(loss_minus)),
            "jitter_rms": 0.0,
            "within_tolerance": False,
        }
    
    epsilon_array = np.array(epsilon_history)
    
    return {
        "epsilon_mean": float(np.mean(epsilon_array)),
        "epsilon_max": float(np.max(epsilon_array)),
        "epsilon_std": float(np.std(epsilon_array)),
        "epsilon_final": float(epsilon_array[-1]),
        "control_effort_mean": float(np.mean(control_effort_history)),
        "control_effort_max": float(np.max(control_effort_history)),
        "packet_loss_total": int(np.sum(loss_plus) + np.sum(loss_minus)),
        "jitter_rms": float(np.sqrt(np.mean(jitter_plus**2 + jitter_minus**2))),
        "within_tolerance": bool(np.max(epsilon_array) <= config.target_epsilon),
    }


def run_monte_carlo(
    n_realizations: int = 100,
    config: StreamBalanceConfig = None,
    perturbation_ranges: Dict = None,
    dt: float = 0.01,
    output_dir: str = "monte_carlo/results",
) -> Dict:
    """
    Run Monte-Carlo validation of stream balance controller.
    
    Args:
        n_realizations: Number of Monte-Carlo realizations
        config: Stream balance controller configuration
        perturbation_ranges: Ranges for perturbation parameters
        dt: Time step (s)
        output_dir: Output directory for results
    
    Returns:
        Dictionary with Monte-Carlo results
    """
    if config is None:
        config = StreamBalanceConfig()
    
    if perturbation_ranges is None:
        perturbation_ranges = {
            "packet_loss_rate": (0.0, 0.05),
            "jitter_std": (0.0, 5e-6),
            "mass_drift_rate": (0.0, 0.005),
        }
    
    # Storage for results
    results = []
    
    for i in range(n_realizations):
        # Sample perturbation parameters
        packet_loss_rate = np.random.uniform(*perturbation_ranges["packet_loss_rate"])
        jitter_std = np.random.uniform(*perturbation_ranges["jitter_std"])
        mass_drift_rate = np.random.uniform(*perturbation_ranges["mass_drift_rate"])
        
        # Run realization
        result = run_single_realization(
            config=config,
            n_packets=1000,
            packet_loss_rate=packet_loss_rate,
            jitter_std=jitter_std,
            mass_drift_rate=mass_drift_rate,
            dt=dt,
            seed=i,
        )
        
        # Add perturbation parameters to result
        result["packet_loss_rate"] = packet_loss_rate
        result["jitter_std"] = jitter_std
        result["mass_drift_rate"] = mass_drift_rate
        result["realization_id"] = i
        
        results.append(result)
    
    # Compute aggregate statistics
    epsilon_means = [r["epsilon_mean"] for r in results]
    epsilon_maxs = [r["epsilon_max"] for r in results]
    pass_rate = sum(1 for r in results if r["within_tolerance"]) / n_realizations
    
    aggregate = {
        "n_realizations": n_realizations,
        "pass_rate": pass_rate,
        "epsilon_mean_mean": float(np.mean(epsilon_means)),
        "epsilon_mean_std": float(np.std(epsilon_means)),
        "epsilon_max_mean": float(np.mean(epsilon_maxs)),
        "epsilon_max_std": float(np.std(epsilon_maxs)),
        "epsilon_max_worst": float(np.max(epsilon_maxs)),
        "results": results,
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "stream_balance_monte_carlo.json", "w") as f:
        # Save aggregate stats without full results
        save_dict = aggregate.copy()
        save_dict["results"] = [r for r in results[:10]]  # Save first 10 for inspection
        json.dump(save_dict, f, indent=2)
    
    return aggregate


def evaluate_gate(results: Dict) -> GateResult:
    """
    Evaluate stream balance gate on Monte-Carlo results.
    
    Args:
        results: Monte-Carlo results dictionary
    
    Returns:
        GateResult object
    """
    gate = StreamBalanceGate(max_epsilon=1e-4)
    
    # Use worst-case epsilon
    worst_epsilon = results["epsilon_max_worst"]
    
    return gate.evaluate(worst_epsilon)


def main():
    """Main entry point for Monte-Carlo validation."""
    parser = argparse.ArgumentParser(
        description="Monte-Carlo validation of stream balance controller"
    )
    parser.add_argument(
        "--n_realizations",
        type=int,
        default=100,
        help="Number of Monte-Carlo realizations",
    )
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=1e-4,
        help="Target epsilon tolerance",
    )
    parser.add_argument(
        "--control_mode",
        type=str,
        default="pi",
        choices=["proportional", "pi", "pid", "adaptive"],
        help="Control mode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="monte_carlo/results",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = StreamBalanceConfig(
        target_epsilon=args.target_epsilon,
        control_mode=BalanceMode(args.control_mode),
    )
    
    print(f"Running Monte-Carlo validation with {args.n_realizations} realizations...")
    print(f"Target epsilon: {args.target_epsilon}")
    print(f"Control mode: {args.control_mode}")
    
    # Run Monte-Carlo
    results = run_monte_carlo(
        n_realizations=args.n_realizations,
        config=config,
        output_dir=args.output_dir,
    )
    
    # Print summary
    print("\n=== Monte-Carlo Results ===")
    print(f"Pass rate: {results['pass_rate']:.2%}")
    print(f"Epsilon mean (mean): {results['epsilon_mean_mean']:.6e}")
    print(f"Epsilon mean (std): {results['epsilon_mean_std']:.6e}")
    print(f"Epsilon max (mean): {results['epsilon_max_mean']:.6e}")
    print(f"Epsilon max (std): {results['epsilon_max_std']:.6e}")
    print(f"Epsilon max (worst): {results['epsilon_max_worst']:.6e}")
    
    # Evaluate gate
    gate_result = evaluate_gate(results)
    print(f"\n=== Gate Evaluation ===")
    print(f"Status: {gate_result.status.value}")
    print(f"Message: {gate_result.message}")
    
    # Save gate result
    output_path = Path(args.output_dir)
    with open(output_path / "stream_balance_gate_result.json", "w") as f:
        json.dump(
            {
                "status": gate_result.status.value,
                "value": gate_result.value,
                "threshold": gate_result.threshold,
                "message": gate_result.message,
            },
            f,
            indent=2,
        )
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()

"""
Run Monte-Carlo validation for EDT module.

This script runs the Monte-Carlo validation with EDT perturbations
to verify the EDT module meets the 90% pass rate target.
"""

import argparse
import numpy as np
from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig, Perturbation, PerturbationType
from monte_carlo.pass_fail_gates import create_default_gate_set
from dynamics.multi_body import MultiBodyStream, Packet
from dynamics.rigid_body import RigidBody
from dynamics.edt_packet import EDTPacket


def create_stream_factory():
    """Create a stream factory for Monte-Carlo validation."""
    def factory():
        packets = []
        nodes = []
        
        # Add regular packets
        for i in range(10):
            mass = 0.05
            I = np.diag([0.0001, 0.00011, 0.00009])
            position = np.array([0.0, 0.0, 0.0])
            velocity = np.array([1600.0, 0.0, 0.0])
            body = RigidBody(mass, I, position=position, velocity=velocity)
            packet = Packet(id=i, body=body)
            packets.append(packet)
        
        # Add EDT packets (hybrid mode)
        for i in range(10, 15):
            mass = 0.05
            I = np.diag([0.0001, 0.00011, 0.00009])
            position = np.array([0.0, 0.0, 0.0])
            velocity = np.array([1600.0, 0.0, 0.0])
            body = RigidBody(mass, I, position=position, velocity=velocity)
            edt_packet = EDTPacket(
                id=i,
                body=body,
                current=5.0,
                voltage=100.0,
                tether_segment_id=i-10,
                resistance=0.01,
                temperature=300.0,
            )
            
            # Validate EDT parameters are within realistic ranges
            assert edt_packet.current >= 0.0, f"Current must be non-negative, got {edt_packet.current}"
            assert edt_packet.resistance >= 0.0, f"Resistance must be non-negative, got {edt_packet.resistance}"
            assert edt_packet.temperature >= 273.15, f"Temperature must be >= 273.15 K, got {edt_packet.temperature}"
            assert edt_packet.temperature < 450.0, f"Temperature must be < 450 K, got {edt_packet.temperature}"
            
            packets.append(edt_packet)
        
        stream = MultiBodyStream(packets=packets, nodes=nodes)
        return stream
    
    return factory


def main():
    """Run Monte-Carlo validation."""
    parser = argparse.ArgumentParser(description="Run EDT Monte-Carlo validation")
    parser.add_argument("--n-realizations", type=int, default=10, help="Number of Monte-Carlo realizations")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step for integration (s)")
    parser.add_argument("--target-pass-rate", type=float, default=0.90, help="Target pass rate threshold")
    args = parser.parse_args()
    
    print("=" * 60)
    print("EDT Monte-Carlo Validation")
    print("=" * 60)
    
    # Configure Monte-Carlo with EDT perturbations
    config = MonteCarloConfig(
        n_realizations=args.n_realizations,
        dt=args.dt,
        perturbations=[
            Perturbation(type=PerturbationType.EDT_CURRENT_NOISE, magnitude=0.3, probability=0.5),
            Perturbation(type=PerturbationType.EDT_PLASMA_DENSITY, magnitude=0.5, probability=0.3),
            Perturbation(type=PerturbationType.EDT_THERMAL_TRANSIENT, magnitude=1.0, probability=0.2),
        ],
    )
    
    print(f"\nConfiguration:")
    print(f"  Realizations: {config.n_realizations}")
    print(f"  Perturbations: {len(config.perturbations)}")
    for p in config.perturbations:
        print(f"    - {p.type.value}: magnitude={p.magnitude}, probability={p.probability}")
    
    # Create cascade runner
    stream_factory = create_stream_factory()
    runner = CascadeRunner(config)
    
    print(f"\nRunning Monte-Carlo validation...")
    try:
        results = runner.run_monte_carlo(stream_factory)
    except Exception as e:
        print(f"\n✗ ERROR: Monte-Carlo simulation failed: {e}")
        return
    
    print(f"\nResults:")
    print(f"  Total realizations: {results['n_realizations']}")
    print(f"  Passed: {results['n_passed']}")
    print(f"  Failed: {results['n_failed']}")
    print(f"  Pass rate: {results['pass_rate']:.1%}")
    
    # Evaluate gates
    gate_set = create_default_gate_set()
    gate_results = gate_set.evaluate_all(results)
    
    print(f"\nGate Results:")
    for result in gate_results:
        status_symbol = "✓" if result.status.value == "pass" else "✗" if result.status.value == "fail" else "⚠"
        print(f"  {status_symbol} {result.gate_name}: {result.status.value.upper()}")
        print(f"    Value: {result.value}, Threshold: {result.threshold}")
    
    # Check EDT-specific metrics
    print(f"\nEDT Metrics:")
    print(f"  Max libration angle: {results.get('edt_libration_angle_max', 0.0):.4f} rad")
    print(f"  Max temperature: {results.get('edt_temperature_max', 0.0):.2f} K")
    print(f"  Max current: {results.get('edt_current_max', 0.0):.2f} A")
    
    # Calculate EDT power (P = I * V)
    edt_current_max = results.get('edt_current_max', 0.0)
    edt_voltage_estimate = 100.0  # V (typical EDT voltage)
    edt_power_max = edt_current_max * edt_voltage_estimate
    print(f"  Max power: {edt_power_max:.2f} W")
    
    # Check if pass rate meets target
    meets_target = results['pass_rate'] >= args.target_pass_rate
    
    print(f"\n" + "=" * 60)
    if meets_target:
        print(f"✓ PASS RATE TARGET MET: {results['pass_rate']:.1%} >= {args.target_pass_rate:.0%}")
    else:
        print(f"✗ PASS RATE TARGET NOT MET: {results['pass_rate']:.1%} < {args.target_pass_rate:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()

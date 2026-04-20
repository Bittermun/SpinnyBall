"""
Test latency injection in Monte-Carlo framework.
"""

import numpy as np
import pytest
from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig, Perturbation, PerturbationType
from monte_carlo.pass_fail_gates import LatencyGate, create_default_gate_set
from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.rigid_body import RigidBody


def test_latency_perturbation_type():
    """Test that latency perturbation type is recognized."""
    assert PerturbationType.LATENCY_INJECTION.value == "latency_injection"


def test_latency_config_parameters():
    """Test that MonteCarloConfig accepts latency parameters."""
    config = MonteCarloConfig(
        latency_ms=20.0,
        latency_std_ms=5.0,
        track_per_packet_latency=True,
    )

    assert config.latency_ms == 20.0
    assert config.latency_std_ms == 5.0
    assert config.track_per_packet_latency == True


def test_latency_gate_evaluation():
    """Test that latency gate evaluates correctly."""
    gate = LatencyGate(max_latency_ms=30.0)

    # Should pass
    result = gate.evaluate(20.0)
    assert result.status.value == "pass"

    # Should fail
    result = gate.evaluate(40.0)
    assert result.status.value == "fail"

    # Should warn (at 90% threshold)
    result = gate.evaluate(28.0)
    assert result.status.value == "warning"


def test_delayed_feedback_mechanism():
    """Test that delayed feedback stores and applies states correctly."""
    config = MonteCarloConfig(
        n_realizations=1,
        time_horizon=1.0,
        dt=0.01,
        latency_ms=50.0,  # 50ms latency
        track_per_packet_latency=True,
    )

    runner = CascadeRunner(config)

    # Create simple stream
    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    packets = [Packet(id=0, body=RigidBody(mass, I))]
    stream = MultiBodyStream(packets=packets, nodes=[], stream_velocity=100.0)

    result = runner.run_realization(stream, 0)

    # Should have latency events
    assert result.latency_events > 0
    assert result.max_latency_ms > 0

    # Per-packet latency should be tracked
    if config.track_per_packet_latency:
        assert result.per_packet_latency is not None
        assert len(result.per_packet_latency) > 0


def test_latency_without_perturbation():
    """Test that system works normally with zero latency."""
    config = MonteCarloConfig(
        n_realizations=1,
        time_horizon=1.0,
        dt=0.01,
        latency_ms=0.0,
    )

    runner = CascadeRunner(config)

    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    packets = [Packet(id=0, body=RigidBody(mass, I))]
    stream = MultiBodyStream(packets=packets, nodes=[], stream_velocity=100.0)

    result = runner.run_realization(stream, 0)

    # Should have no latency events
    assert result.latency_events == 0
    assert result.max_latency_ms == 0.0


def test_latency_gate_in_default_set():
    """Test that latency gate is included in default gate set."""
    gate_set = create_default_gate_set()
    gate_names = [gate.name for gate in gate_set.gates]

    assert "max_latency_ms" in gate_names


def test_latency_perturbation_application():
    """Test that latency perturbation can be applied to packets."""
    config = MonteCarloConfig(
        n_realizations=1,
        time_horizon=0.1,
        dt=0.01,
    )

    runner = CascadeRunner(config)

    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    packets = [Packet(id=0, body=RigidBody(mass, I))]
    stream = MultiBodyStream(packets=packets, nodes=[], stream_velocity=100.0)

    # Apply latency perturbation
    perturbation = Perturbation(
        type=PerturbationType.LATENCY_INJECTION,
        magnitude=0.05,  # 50ms
        probability=1.0,
    )
    runner.apply_perturbation(packets[0], perturbation)

    # Check that latency buffer was created
    assert hasattr(packets[0], 'latency_buffer')
    assert len(packets[0].latency_buffer) > 0


def test_per_packet_latency_tracking_disabled():
    """Test that per-packet latency tracking can be disabled."""
    config = MonteCarloConfig(
        n_realizations=1,
        time_horizon=1.0,
        dt=0.01,
        latency_ms=50.0,
        track_per_packet_latency=False,  # Disabled
    )

    runner = CascadeRunner(config)

    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    packets = [Packet(id=0, body=RigidBody(mass, I))]
    stream = MultiBodyStream(packets=packets, nodes=[], stream_velocity=100.0)

    result = runner.run_realization(stream, 0)

    # Should have latency events but no per-packet tracking
    assert result.latency_events > 0
    assert result.per_packet_latency is None


def test_latency_std_deviation():
    """Test that latency standard deviation produces varied latencies."""
    config = MonteCarloConfig(
        n_realizations=10,
        time_horizon=0.5,
        dt=0.01,
        latency_ms=30.0,
        latency_std_ms=10.0,  # High std dev
        track_per_packet_latency=True,
    )

    runner = CascadeRunner(config)

    def stream_factory():
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        packets = [Packet(id=0, body=RigidBody(mass, I))]
        return MultiBodyStream(packets=packets, nodes=[], stream_velocity=100.0)

    results = runner.run_monte_carlo(stream_factory)

    # Should have some variation in max latency
    assert results['max_latency_ms'] > 0
    # Should have latency events
    assert results['latency_events'] > 0


def test_latency_release_time_format():
    """Test that latency buffer uses correct (release_time, state) format."""
    config = MonteCarloConfig(
        n_realizations=1,
        time_horizon=0.1,
        dt=0.01,
    )

    runner = CascadeRunner(config)

    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    packets = [Packet(id=0, body=RigidBody(mass, I))]
    stream = MultiBodyStream(packets=packets, nodes=[], stream_velocity=100.0)

    # Apply latency perturbation at specific time
    current_time = 0.5
    perturbation = Perturbation(
        type=PerturbationType.LATENCY_INJECTION,
        magnitude=0.02,  # 20ms
        probability=1.0,
    )
    runner.apply_perturbation(packets[0], perturbation, current_time=current_time)

    # Check latency buffer format
    assert hasattr(packets[0], 'latency_buffer')
    assert len(packets[0].latency_buffer) == 1

    release_time, delayed_state = packets[0].latency_buffer[0]
    # release_time should be current_time + latency
    expected_release_time = current_time + perturbation.magnitude
    assert abs(release_time - expected_release_time) < 1e-6

    # delayed_state should have the expected attributes
    assert hasattr(delayed_state, 'position')
    assert hasattr(delayed_state, 'velocity')
    assert hasattr(delayed_state, 'quaternion')
    assert hasattr(delayed_state, 'angular_velocity')


def test_latency_timing_accuracy():
    """Test that latency injection timing is accurate within ±1 ms (with no std dev)."""
    config = MonteCarloConfig(
        n_realizations=1,
        time_horizon=0.5,
        dt=0.01,
        latency_ms=20.0,  # 20ms latency
        latency_std_ms=0.0,  # No variation for accuracy test
        track_per_packet_latency=True,
    )

    runner = CascadeRunner(config)

    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    packets = [Packet(id=0, body=RigidBody(mass, I))]
    stream = MultiBodyStream(packets=packets, nodes=[], stream_velocity=100.0)

    result = runner.run_realization(stream, 0)

    # Check that max latency is close to configured latency (±1 ms tolerance)
    expected_latency_ms = config.latency_ms
    actual_max_latency_ms = result.max_latency_ms

    # Allow ±1 ms tolerance for timing accuracy
    assert abs(actual_max_latency_ms - expected_latency_ms) <= 1.0, \
        f"Expected latency ~{expected_latency_ms}ms, got {actual_max_latency_ms}ms"


def test_multiple_latency_injections():
    """Test that multiple latency injections are handled correctly."""
    config = MonteCarloConfig(
        n_realizations=1,
        time_horizon=0.5,
        dt=0.01,
    )

    runner = CascadeRunner(config)

    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    packets = [Packet(id=0, body=RigidBody(mass, I))]
    stream = MultiBodyStream(packets=packets, nodes=[], stream_velocity=100.0)

    # Apply multiple latency perturbations at different times
    perturbation1 = Perturbation(
        type=PerturbationType.LATENCY_INJECTION,
        magnitude=0.01,  # 10ms
        probability=1.0,
    )
    perturbation2 = Perturbation(
        type=PerturbationType.LATENCY_INJECTION,
        magnitude=0.02,  # 20ms
        probability=1.0,
    )

    runner.apply_perturbation(packets[0], perturbation1, current_time=0.0)
    runner.apply_perturbation(packets[0], perturbation2, current_time=0.1)

    # Check that both latency entries are in buffer
    assert len(packets[0].latency_buffer) == 2

    # Check that release times are correct
    release_time1, _ = packets[0].latency_buffer[0]
    release_time2, _ = packets[0].latency_buffer[1]

    assert abs(release_time1 - 0.01) < 1e-6  # 0.0 + 0.01
    assert abs(release_time2 - 0.12) < 1e-6  # 0.1 + 0.02


def test_latency_10_to_30_ms_range():
    """Test latency injection across 10-30 ms range (with no std dev)."""
    for latency_ms in [10.0, 15.0, 20.0, 25.0, 30.0]:
        config = MonteCarloConfig(
            n_realizations=1,
            time_horizon=0.5,
            dt=0.01,
            latency_ms=latency_ms,
            latency_std_ms=0.0,  # No variation for accuracy test
            track_per_packet_latency=True,
        )

        runner = CascadeRunner(config)

        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        packets = [Packet(id=0, body=RigidBody(mass, I))]
        stream = MultiBodyStream(packets=packets, nodes=[], stream_velocity=100.0)

        result = runner.run_realization(stream, 0)

        # Check that max latency is within ±1 ms of target
        assert abs(result.max_latency_ms - latency_ms) <= 1.0, \
            f"Latency {latency_ms}ms: expected ~{latency_ms}ms, got {result.max_latency_ms}ms"

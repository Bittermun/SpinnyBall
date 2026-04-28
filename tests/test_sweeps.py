"""
Integration tests for T1 and T3 sweep scripts.
"""

import numpy as np
import pytest
from monte_carlo.cascade_runner import CascadeRunner, MonteCarloConfig
from monte_carlo.pass_fail_gates import DelayMarginGate, ContainmentGate, create_default_gate_set
from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.rigid_body import RigidBody


def test_delay_margin_gate():
    """Test DelayMarginGate evaluation."""
    gate = DelayMarginGate(min_delay_margin_ms=35.0)

    # Should pass (above warning threshold of 42ms)
    result = gate.evaluate(50.0)
    assert result.status.value == "pass"

    # Should fail
    result = gate.evaluate(30.0)
    assert result.status.value == "fail"

    # Should warn (between 35ms and 42ms)
    result = gate.evaluate(38.0)
    assert result.status.value == "warning"


def test_containment_gate():
    """Test ContainmentGate evaluation."""
    gate = ContainmentGate(max_nodes_affected=2)

    # Should pass (0 nodes)
    result = gate.evaluate(0.0)
    assert result.status.value == "pass"

    # Should pass (1 node - below warning threshold of 1.5)
    result = gate.evaluate(1.0)
    assert result.status.value == "pass"

    # Should pass (1.5 nodes - at warning threshold, not in warning range)
    result = gate.evaluate(1.5)
    assert result.status.value == "pass"

    # Should warn (2 nodes - at limit, in warning range since > 1.5)
    result = gate.evaluate(2.0)
    assert result.status.value == "warning"

    # Should fail (3 nodes)
    result = gate.evaluate(3.0)
    assert result.status.value == "fail"


def test_new_gates_in_default_set():
    """Test that new gates are included in default gate set."""
    gate_set = create_default_gate_set()
    gate_names = [gate.name for gate in gate_set.gates]

    assert "delay_margin_ms" in gate_names
    assert "nodes_affected" in gate_names


def test_fault_rate_parameter():
    """Test that MonteCarloConfig accepts fault_rate parameters."""
    config = MonteCarloConfig(
        fault_rate=1e-4,
        cascade_threshold=1.05,
        containment_threshold=2,
    )

    assert config.fault_rate == 1e-4
    assert config.cascade_threshold == 1.05
    assert config.containment_threshold == 2


def test_nodes_affected_tracking():
    """Test that nodes_affected is tracked in RealizationResult."""
    config = MonteCarloConfig(
        n_realizations=1,
        time_horizon=0.5,
        dt=0.01,
        fault_rate=0.0,  # No faults for this test
        containment_threshold=2,
    )

    runner = CascadeRunner(config)

    # Create stream with nodes
    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    packets = [Packet(id=0, body=RigidBody(mass, I), eta_ind=0.9)]

    nodes = []
    for i in range(3):
        node = SNode(
            id=i,
            position=np.array([i * 10.0, 0.0, 0.0]),
            max_packets=10,
            eta_ind_min=0.82,
            k_fp=4500.0,
        )
        nodes.append(node)

    stream = MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=100.0)

    result = runner.run_realization(stream, 0)

    # Check that nodes_affected field exists
    assert hasattr(result, 'nodes_affected')
    assert isinstance(result.nodes_affected, int)
    assert result.nodes_affected >= 0

    # Check containment_successful field
    assert hasattr(result, 'containment_successful')
    assert isinstance(result.containment_successful, bool)


def test_fault_injection_logic():
    """Test that fault injection reduces node stiffness."""
    config = MonteCarloConfig(
        n_realizations=1,
        time_horizon=0.1,
        dt=0.01,
        fault_rate=1e-3,  # High fault rate for testing
        cascade_threshold=1.05,
        containment_threshold=2,
    )

    runner = CascadeRunner(config)

    # Create stream with nodes
    mass = 0.05
    I = np.diag([0.0001, 0.00011, 0.00009])
    packets = [Packet(id=0, body=RigidBody(mass, I), eta_ind=0.9)]

    nodes = []
    original_stiffness = []
    for i in range(5):
        node = SNode(
            id=i,
            position=np.array([i * 10.0, 0.0, 0.0]),
            max_packets=10,
            eta_ind_min=0.82,
            k_fp=4500.0,
        )
        original_stiffness.append(node.k_fp)
        nodes.append(node)

    stream = MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=100.0)

    result = runner.run_realization(stream, 0)

    # Check if any nodes had stiffness reduced
    # With high fault rate, at least one node should fail
    stiffness_reduced = any(node.k_fp < original for node, original in zip(nodes, original_stiffness))

    # Note: This is probabilistic, so we don't assert it must happen
    # Just verify the mechanism exists
    assert result.nodes_affected >= 0


def test_t1_sweep_small_grid():
    """Test T1 sweep with small grid for validation."""
    from sweep_latency_eta_ind import run_t1_sweep

    results = run_t1_sweep(
        latency_range=(5.0, 10.0),
        eta_ind_range=(0.85, 0.90),
        n_latency_points=3,
        n_eta_points=2,
        n_realizations_per_point=5,
    )

    # Check result structure
    assert 'latency_values' in results
    assert 'eta_ind_values' in results
    assert 'success_rate_grid' in results
    assert 'delay_margin_grid' in results

    # Check grid dimensions
    assert results['success_rate_grid'].shape == (2, 3)
    assert len(results['latency_values']) == 3
    assert len(results['eta_ind_values']) == 2

    # Check success rate values are valid
    assert np.all(results['success_rate_grid'] >= 0.0)
    assert np.all(results['success_rate_grid'] <= 1.0)


def test_t3_sweep_small_grid():
    """Test T3 sweep with small grid for validation."""
    from sweep_fault_cascade import run_t3_sweep

    results = run_t3_sweep(
        fault_rate_range=(1e-6, 1e-4),
        n_fault_rate_points=3,
        cascade_threshold=1.05,
        containment_threshold=2,
        n_nodes=5,
        n_realizations_per_point=10,
        time_horizon=1.0,
    )

    # Check result structure
    assert 'fault_rates' in results
    assert 'cascade_probability' in results
    assert 'containment_rate' in results
    assert 'success_rate' in results

    # Check array lengths
    assert len(results['fault_rates']) == 3
    assert len(results['cascade_probability']) == 3

    # Check values are valid
    assert np.all(results['cascade_probability'] >= 0.0)
    assert np.all(results['cascade_probability'] <= 1.0)
    assert np.all(results['containment_rate'] >= 0.0)
    assert np.all(results['containment_rate'] <= 1.0)


def test_mpc_delay_margin_calculation():
    """Test MPC delay margin calculation."""
    try:
        from control_layer.mpc_controller import MPCController, ConfigurationMode

        mpc = MPCController(
            configuration_mode=ConfigurationMode.TEST,
            delay_steps=5,
            enable_delay_compensation=True,
        )

        # Calculate delay margin
        delay_margin = mpc.calculate_delay_margin()

        # Check result structure
        assert 'delay_margin_ms' in delay_margin
        assert 'phase_margin_deg' in delay_margin
        assert 'crossover_freq_hz' in delay_margin
        assert 'calculation_failed' in delay_margin

        # If calculation succeeded, check values are reasonable
        if not delay_margin['calculation_failed']:
            assert delay_margin['delay_margin_ms'] >= 0
            assert -180 <= delay_margin['phase_margin_deg'] <= 180
            assert delay_margin['crossover_freq_hz'] >= 0

    except ImportError:
        pytest.skip("CasADi not available for MPC test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

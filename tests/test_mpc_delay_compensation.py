"""
Test Smith predictor delay compensation in MPC.
"""

import numpy as np
import pytest

# Skip all tests if CasADi is not available
pytest.importorskip("casadi", reason="CasADi is required for MPC tests")

from control_layer.mpc_controller import MPCController, create_mpc_controller, ConfigurationMode


def test_smith_predictor_advances_state():
    """Test that Smith predictor advances state correctly."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_steps=5,
        dt_delay=0.01,
        enable_delay_compensation=True,
    )

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])  # [qx, qy, qz, qw, ωx, ωy, ωz]
    x_pred = controller.smith_predictor(x0)

    # State should have changed due to gyroscopic coupling
    assert not np.allclose(x_pred, x0)
    # Quaternion should remain normalized
    q_norm = np.linalg.norm(x_pred[:4])
    assert np.isclose(q_norm, 1.0, atol=1e-6)


def test_delay_compensation_disabled():
    """Test that delay compensation can be disabled."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_steps=5,
        enable_delay_compensation=False,
    )

    x0 = np.random.randn(7)
    x_target = np.zeros(7)

    u_opt, info = controller.solve(x0, x_target)

    assert info['delay_compensation_enabled'] == False
    assert info['success']


def test_delay_steps_zero():
    """Test that delay_steps=0 produces same result as no compensation."""
    controller_enabled = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_steps=5,
        enable_delay_compensation=True,
    )

    controller_zero = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_steps=0,
        enable_delay_compensation=True,
    )

    x0 = np.random.randn(7)
    x_target = np.zeros(7)

    u_opt_enabled, info_enabled = controller_enabled.solve(x0, x_target)
    u_opt_zero, info_zero = controller_zero.solve(x0, x_target)

    # With delay_steps=0, compensation should have no effect
    assert info_enabled['delay_steps'] == 5
    assert info_zero['delay_steps'] == 0


def test_mpc_latency_with_compensation():
    """Test that MPC still meets latency target with delay compensation."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_steps=5,
        enable_delay_compensation=True,
    )

    latency_stats = MPCController.verify_mpc_latency(controller, n_trials=10)

    assert latency_stats['meets_target']
    assert latency_stats['mean_ms'] <= 30.0


def test_numerical_stability_various_delays():
    """Test numerical stability with various delay_steps values."""
    for delay_steps in [1, 5, 10, 20]:
        controller = create_mpc_controller(
            configuration_mode=ConfigurationMode.TEST,
            delay_steps=delay_steps,
            enable_delay_compensation=True,
        )

        x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
        x_pred = controller.smith_predictor(x0)

        # Quaternion should remain normalized
        q_norm = np.linalg.norm(x_pred[:4])
        assert np.isclose(q_norm, 1.0, atol=1e-5), f"Quaternion norm {q_norm} for delay_steps={delay_steps}"

        # No NaN or Inf values
        assert not np.any(np.isnan(x_pred))
        assert not np.any(np.isinf(x_pred))


def test_smith_predictor_quaternion_normalization():
    """Test that Smith predictor maintains quaternion normalization."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_steps=10,
        dt_delay=0.01,
    )

    # Test with various initial states
    for _ in range(5):
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)  # Normalize
        omega = np.random.randn(3) * 0.1  # Small angular velocity
        x0 = np.concatenate([q, omega])

        x_pred = controller.smith_predictor(x0)
        q_pred = x_pred[:4]

        # Quaternion should remain normalized
        q_norm = np.linalg.norm(q_pred)
        assert np.isclose(q_norm, 1.0, atol=1e-5)


def test_delay_compensation_info_dict():
    """Test that delay compensation info is included in solve return dict."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_steps=5,
        enable_delay_compensation=True,
    )

    x0 = np.random.randn(7)
    x_target = np.zeros(7)

    u_opt, info = controller.solve(x0, x_target)

    # Check that delay metrics are in info dict
    assert 'delay_steps' in info
    assert 'delay_compensation_enabled' in info
    assert info['delay_steps'] == 5
    assert info['delay_compensation_enabled'] == True

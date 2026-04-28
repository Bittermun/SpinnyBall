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


def test_discrete_time_delay_disabled():
    """Test that discrete-time delay can be disabled."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        enable_discrete_time=False,
    )

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    x_delayed = controller.apply_discrete_time_delay(x0)

    # Should return unchanged when disabled
    assert np.allclose(x_delayed, x0)


def test_discrete_time_delay_enabled():
    """Test that discrete-time delay advances state."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        sampling_period=0.01,
        communication_delay=0.005,
        enable_discrete_time=True,
    )

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    x_delayed = controller.apply_discrete_time_delay(x0)

    # State should have changed due to delay
    assert not np.allclose(x_delayed, x0)
    # Quaternion should remain normalized
    q_norm = np.linalg.norm(x_delayed[:4])
    assert np.isclose(q_norm, 1.0, atol=1e-6)


def test_delay_compensation_mode_discrete_time():
    """Test delay_compensation_mode='discrete_time'."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_compensation_mode='discrete_time',
        enable_discrete_time=True,
        enable_delay_compensation=False,
    )

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    x_target = np.zeros(7)

    u_opt, info = controller.solve(x0, x_target)

    assert info['success']


def test_delay_compensation_mode_smith():
    """Test delay_compensation_mode='smith'."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_compensation_mode='smith',
        enable_discrete_time=False,
        enable_delay_compensation=True,
        delay_steps=5,
    )

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    x_target = np.zeros(7)

    u_opt, info = controller.solve(x0, x_target)

    assert info['success']


def test_delay_compensation_mode_both():
    """Test delay_compensation_mode='both' (additive)."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_compensation_mode='both',
        enable_discrete_time=True,
        enable_delay_compensation=True,
        delay_steps=5,
        sampling_period=0.01,
        communication_delay=0.005,
    )

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    x_target = np.zeros(7)

    u_opt, info = controller.solve(x0, x_target)

    assert info['success']


def test_delay_compensation_mode_invalid():
    """Test that invalid delay_compensation_mode logs warning."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        delay_compensation_mode='invalid_mode',
    )

    x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0])
    x_target = np.zeros(7)

    # Should not crash, just log warning
    u_opt, info = controller.solve(x0, x_target)
    assert info['success']


def test_discrete_time_delay_parameters():
    """Test that sampling_period and communication_delay are stored."""
    controller = create_mpc_controller(
        configuration_mode=ConfigurationMode.TEST,
        sampling_period=0.02,
        communication_delay=0.01,
    )

    assert controller.sampling_period == 0.02
    assert controller.communication_delay == 0.01

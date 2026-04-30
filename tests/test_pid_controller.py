"""
Unit tests for PID controller implementation.
"""

import numpy as np
import pytest

from sgms_anchor_control import (
    PIDController,
    PIDMode,
    PIDParameters,
    manual_tuning,
    ziegler_nichols_tuning,
)


def test_pid_parameters():
    """Test PIDParameters dataclass."""
    params = PIDParameters(kp=1.0, ki=0.1, kd=0.01)
    assert params.kp == 1.0
    assert params.ki == 0.1
    assert params.kd == 0.01
    assert params.mode == PIDMode.POSITION
    assert params.tau_filter == 0.1


def test_pid_controller_step():
    """Test single step of PID controller."""
    params = PIDParameters(kp=1.0, ki=0.1, kd=0.01)
    pid = PIDController(params, dt=0.01)
    output = pid.update(1.0)
    assert output > 0  # Should respond to positive error


def test_pid_anti_windup():
    """Test anti-windup clamping."""
    params = PIDParameters(
        kp=1.0, ki=0.1, kd=0.0,
        integral_min=-10.0, integral_max=10.0
    )
    pid = PIDController(params, dt=0.01)
    # Force integral to saturate
    for _ in range(1000):
        pid.update(1.0)
    assert pid.integral <= 10.0


def test_pid_derivative_filtering():
    """Test derivative low-pass filtering."""
    params = PIDParameters(kp=0.0, ki=0.0, kd=1.0, tau_filter=0.1)
    pid = PIDController(params, dt=0.01)
    # Step input should produce filtered derivative
    output1 = pid.update(0.0)
    output2 = pid.update(1.0)
    assert abs(output2) < 100.0  # Should be filtered


def test_ziegler_nichols_tuning():
    """Test Ziegler-Nichols tuning method."""
    params = ziegler_nichols_tuning(ku=10.0, tu=5.0)
    assert params.kp == 6.0
    assert params.ki == 12.0 / 5.0
    assert params.kd == 0.75


def test_manual_tuning():
    """Test manual tuning."""
    params = manual_tuning(kp=5.0, ki=0.5, kd=0.05)
    assert params.kp == 5.0
    assert params.ki == 0.5
    assert params.kd == 0.05


def test_pid_reset():
    """Test controller reset."""
    params = PIDParameters(kp=1.0, ki=0.1, kd=0.01)
    pid = PIDController(params, dt=0.01)
    pid.update(1.0)
    pid.reset()
    assert pid.integral == 0.0
    assert pid.prev_error == 0.0
    assert pid.prev_derivative == 0.0


def test_pid_setpoint_tracking():
    """Test setpoint tracking with PID."""
    params = PIDParameters(kp=10.0, ki=1.0, kd=0.1)
    pid = PIDController(params, dt=0.01)
    errors = []
    setpoint = 1.0
    for i in range(100):
        measurement = 0.0  # Start at 0
        error = setpoint - measurement
        output = pid.update(error)
        errors.append(abs(error))
    # Errors should decrease over time
    assert errors[-1] < errors[0]


def test_pid_output_saturation():
    """Test output saturation."""
    params = PIDParameters(
        kp=10.0, ki=1.0, kd=0.1,
        output_min=-5.0, output_max=5.0
    )
    pid = PIDController(params, dt=0.01)
    # Large error should saturate output
    output = pid.update(10.0)
    assert output <= 5.0


def test_pid_delay_compensation():
    """Test delay compensation actually delays output."""
    params = PIDParameters(kp=1.0, ki=0.0, kd=0.0, delay_steps=3)
    pid = PIDController(params, dt=0.01)
    
    # Feed constant error of 1.0
    outputs = []
    for i in range(5):
        output = pid.update(1.0)
        outputs.append(output)
    
    # First delay_steps outputs should be 0 (buffered)
    for i in range(3):
        assert outputs[i] == 0.0, f"Output at step {i} should be 0 during buffering, got {outputs[i]}"
    
    # After delay_steps, output should appear
    assert outputs[3] == 1.0, f"Output at step 3 should be 1.0 after delay, got {outputs[3]}"
    assert outputs[4] == 1.0, f"Output at step 4 should be 1.0, got {outputs[4]}"


def test_pid_mode_enum():
    """Test PIDMode enum values."""
    assert PIDMode.POSITION.value == "position"
    assert PIDMode.VELOCITY.value == "velocity"
    assert PIDMode.TEMPERATURE.value == "temperature"


def test_pid_integral_only():
    """Test I-only controller."""
    params = PIDParameters(kp=0.0, ki=1.0, kd=0.0)
    pid = PIDController(params, dt=0.01)
    
    # Constant error should cause integral to grow
    for _ in range(10):
        output = pid.update(1.0)
        assert output > 0


def test_pid_derivative_only():
    """Test D-only controller."""
    params = PIDParameters(kp=0.0, ki=0.0, kd=1.0)
    pid = PIDController(params, dt=0.01)
    
    # Step change should produce derivative response
    output1 = pid.update(0.0)
    output2 = pid.update(1.0)
    assert output2 != 0  # Derivative should respond to change


def test_pid_proportional_only():
    """Test P-only controller."""
    params = PIDParameters(kp=5.0, ki=0.0, kd=0.0)
    pid = PIDController(params, dt=0.01)
    
    output = pid.update(1.0)
    assert output == 5.0  # P-only: output = kp * error


def test_pid_zero_error():
    """Test response to zero error."""
    params = PIDParameters(kp=1.0, ki=0.1, kd=0.01)
    pid = PIDController(params, dt=0.01)
    
    # Zero error should produce small output (only from derivative of previous error)
    output = pid.update(0.0)
    # First step: derivative of previous error (0 - 0) = 0
    assert output == 0.0

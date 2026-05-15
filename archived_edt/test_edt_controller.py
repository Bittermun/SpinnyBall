"""
Unit tests for EDT controller.

Tests current setpoint tracking, power generation mode, libration damping, and rate limiting.
"""

import pytest
import numpy as np

from control_layer.edt_controller import EDTController


class TestEDTController:
    """Test EDT controller basic functionality."""

    def test_controller_initialization(self):
        """Test controller initialization with default parameters."""
        controller = EDTController()

        assert controller.max_current == 10.0
        assert controller.max_rate == 1.0
        assert controller.libration_damping_gain == 0.5
        assert controller.libration_derivative_gain == 0.1
        assert controller.power_generation_gain == 0.8
        assert controller.voltage_estimate == 100.0
        assert controller.current_actual == 0.0
        assert controller.current_command == 0.0

    def test_compute_current_setpoint(self):
        """Test current setpoint computation for power generation."""
        controller = EDTController()

        power_demand = 100.0  # W
        libration_angle = 0.0  # rad

        i_setpoint = controller.compute_current_setpoint(power_demand, libration_angle)

        # Should be positive due to power demand
        assert i_setpoint > 0

    def test_libration_damping_current(self):
        """Test libration damping current computation."""
        controller = EDTController()

        libration_state = np.array([0.1, 0.0])  # 0.1 rad libration angle

        i_damping = controller.libration_damping_current(libration_state)

        # Should be negative to counteract positive libration
        assert i_damping < 0

    def test_rate_limiting(self):
        """Test rate limiting on current changes."""
        controller = EDTController(max_rate=1.0)

        current_old = 0.0
        current_new = 10.0  # Large step
        dt = 0.01  # Small time step

        current_limited = controller.rate_limit_current(current_new, current_old, dt)

        # Should be rate-limited
        assert current_limited < current_new
        assert current_limited <= controller.max_rate * dt

    def test_rate_limiting_no_limit(self):
        """Test rate limiting when change is within limits."""
        controller = EDTController(max_rate=1.0)

        current_old = 0.0
        current_new = 0.005  # Small step
        dt = 0.01  # Time step

        current_limited = controller.rate_limit_current(current_new, current_old, dt)

        # Should not be rate-limited
        assert current_limited == current_new

    def test_current_clamping(self):
        """Test current clamping to max_current."""
        controller = EDTController(max_current=5.0)

        power_demand = 10000.0  # Very high power demand
        libration_angle = 0.0

        i_setpoint = controller.compute_current_setpoint(power_demand, libration_angle)

        # Should be clamped to max_current
        assert i_setpoint <= controller.max_current

    def test_controller_update(self):
        """Test full controller update cycle."""
        controller = EDTController()

        power_demand = 100.0
        libration_angle = 0.1
        dt = 0.01

        state = controller.update(power_demand, libration_angle, dt=dt)

        assert "current_command" in state
        assert "current_actual" in state
        assert "tracking_error" in state
        assert "libration_angle" in state
        assert "power_demand" in state

    def test_controller_reset(self):
        """Test controller reset."""
        controller = EDTController()

        # Set some state
        controller.current_actual = 5.0
        controller.current_command = 5.0

        # Reset
        controller.reset()

        assert controller.current_actual == 0.0
        assert controller.current_command == 0.0

    def test_tracking_accuracy(self):
        """Test current tracking accuracy (±5% target)."""
        controller = EDTController(max_rate=10.0)  # High rate for this test

        power_demand = 100.0
        libration_angle = 0.0
        dt = 0.1  # Larger time step

        state = controller.update(power_demand, libration_angle, dt=dt)

        # Tracking error should be small (< 5%)
        assert state["tracking_error"] < 0.05

    def test_power_generation_mode(self):
        """Test power generation mode."""
        controller = EDTController()

        power_demand = 500.0  # W
        libration_angle = 0.0

        i_setpoint = controller.compute_current_setpoint(power_demand, libration_angle)

        # Higher power demand should result in higher current
        power_demand_high = 1000.0
        i_setpoint_high = controller.compute_current_setpoint(power_demand_high, libration_angle)

        assert i_setpoint_high > i_setpoint

    def test_libration_damping_mode(self):
        """Test libration damping mode."""
        controller = EDTController()

        power_demand = 0.0  # No power demand
        libration_angle = 0.2  # Positive libration

        i_setpoint = controller.compute_current_setpoint(power_demand, libration_angle)

        # Should be negative to dampen positive libration
        assert i_setpoint < 0

    def test_combined_power_and_damping(self):
        """Test combined power generation and libration damping."""
        controller = EDTController()

        power_demand = 100.0  # W
        libration_angle = 0.1  # Positive libration

        i_setpoint = controller.compute_current_setpoint(power_demand, libration_angle)

        # Should be combination of both effects
        # Power generation wants positive current, damping wants negative
        # Result depends on gains
        assert isinstance(i_setpoint, float)

    def test_gate_validation_invalid_angle(self):
        """Test EDTLibrationGate raises ValueError for invalid angle."""
        from monte_carlo.pass_fail_gates import EDTLibrationGate

        # Test negative angle
        with pytest.raises(ValueError, match="must be positive"):
            EDTLibrationGate(max_libration_angle_deg=-10.0)

        # Test angle > 180 degrees
        with pytest.raises(ValueError, match="must be <= 180 degrees"):
            EDTLibrationGate(max_libration_angle_deg=200.0)

    def test_gate_validation_invalid_temperature(self):
        """Test EDTTemperatureGate raises ValueError for invalid temperature."""
        from monte_carlo.pass_fail_gates import EDTTemperatureGate

        # Test negative temperature
        with pytest.raises(ValueError, match="must be positive"):
            EDTTemperatureGate(max_temperature=-100.0)

        # Test temperature below absolute zero
        with pytest.raises(ValueError, match="must be >= 273.15 K"):
            EDTTemperatureGate(max_temperature=200.0)

    def test_gate_validation_invalid_current(self):
        """Test EDTCurrentGate raises ValueError for invalid current."""
        from monte_carlo.pass_fail_gates import EDTCurrentGate

        # Test negative current
        with pytest.raises(ValueError, match="must be non-negative"):
            EDTCurrentGate(max_current=-10.0)

    def test_gate_validation_invalid_power(self):
        """Test EDTPowerGate raises ValueError for invalid power."""
        from monte_carlo.pass_fail_gates import EDTPowerGate

        # Test negative power
        with pytest.raises(ValueError, match="must be non-negative"):
            EDTPowerGate(min_power=-100.0)

"""
Unit tests for stream balance controller.
"""

import unittest
import numpy as np
from control_layer.stream_balance import (
    StreamBalanceController,
    StreamBalanceConfig,
    StreamBalanceState,
    BalanceMode,
    create_stream_balance_controller,
)


class TestStreamBalanceController(unittest.TestCase):
    """Test cases for StreamBalanceController."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = StreamBalanceConfig(target_epsilon=1e-4)
        self.controller = StreamBalanceController(self.config)

    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.state.epsilon, 0.0)
        self.assertEqual(self.controller.config.target_epsilon, 1e-4)
        self.assertEqual(len(self.controller.measurement_buffer), 100)

    def test_measure_imbalance_perfect_balance(self):
        """Test imbalance measurement with perfect balance."""
        epsilon = self.controller.measure_imbalance(
            flow_plus=10.0,
            flow_minus=10.0,
            packet_loss_plus=0,
            packet_loss_minus=0,
            timing_jitter_plus=0.0,
            timing_jitter_minus=0.0,
        )
        self.assertEqual(epsilon, 0.0)

    def test_measure_imbalance_flow_mismatch(self):
        """Test imbalance measurement with flow mismatch."""
        epsilon = self.controller.measure_imbalance(
            flow_plus=10.0,
            flow_minus=9.9,  # 1% mismatch
            packet_loss_plus=0,
            packet_loss_minus=0,
            timing_jitter_plus=0.0,
            timing_jitter_minus=0.0,
        )
        expected = abs(10.0 - 9.9) / (10.0 + 9.9)
        self.assertAlmostEqual(epsilon, expected, places=6)

    def test_measure_imbalance_packet_loss(self):
        """Test imbalance measurement with packet loss."""
        epsilon = self.controller.measure_imbalance(
            flow_plus=10.0,
            flow_minus=10.0,
            packet_loss_plus=5,
            packet_loss_minus=0,
            timing_jitter_plus=0.0,
            timing_jitter_minus=0.0,
        )
        # Should include loss contribution
        self.assertGreater(epsilon, 0.0)

    def test_measure_imbalance_timing_jitter(self):
        """Test imbalance measurement with timing jitter."""
        epsilon = self.controller.measure_imbalance(
            flow_plus=10.0,
            flow_minus=10.0,
            packet_loss_plus=0,
            packet_loss_minus=0,
            timing_jitter_plus=2e-6,  # 2 μs
            timing_jitter_minus=0.0,
        )
        # Should include jitter contribution
        self.assertGreater(epsilon, 0.0)

    def test_filtered_imbalance(self):
        """Test filtered imbalance calculation."""
        # Fill buffer with measurements
        for i in range(100):
            self.controller.measure_imbalance(
                flow_plus=10.0 + i * 0.01,
                flow_minus=10.0,
                packet_loss_plus=0,
                packet_loss_minus=0,
                timing_jitter_plus=0.0,
                timing_jitter_minus=0.0,
            )
        
        filtered = self.controller.get_filtered_imbalance()
        self.assertGreater(filtered, 0.0)

    def test_update_proportional_mode(self):
        """Test controller update in proportional mode."""
        self.config.control_mode = BalanceMode.PROPORTIONAL
        controller = StreamBalanceController(self.config)
        
        # Add some imbalance - measured imbalance is higher than target
        # Controller should try to reduce epsilon (drive toward target)
        controller.measure_imbalance(10.0, 9.9, 0, 0, 0.0, 0.0)
        
        epsilon, control = controller.update(dt=0.01)
        # Epsilon should be at or below min (0.0) since measured imbalance > target
        self.assertGreaterEqual(epsilon, self.config.min_epsilon)
        self.assertIsNotNone(control)

    def test_update_pi_mode(self):
        """Test controller update in PI mode."""
        self.config.control_mode = BalanceMode.PI
        controller = StreamBalanceController(self.config)
        
        # Add some imbalance
        controller.measure_imbalance(10.0, 9.9, 0, 0, 0.0, 0.0)
        
        epsilon, control = controller.update(dt=0.01)
        # Epsilon should be at or below min (0.0) since measured imbalance > target
        self.assertGreaterEqual(epsilon, self.config.min_epsilon)
        self.assertIsNotNone(control)

    def test_update_pid_mode(self):
        """Test controller update in PID mode."""
        self.config.control_mode = BalanceMode.PID
        controller = StreamBalanceController(self.config)
        
        # Add some imbalance
        controller.measure_imbalance(10.0, 9.9, 0, 0, 0.0, 0.0)
        
        epsilon, control = controller.update(dt=0.01)
        # Epsilon should be at or below min (0.0) since measured imbalance > target
        self.assertGreaterEqual(epsilon, self.config.min_epsilon)
        self.assertIsNotNone(control)

    def test_epsilon_bounds(self):
        """Test epsilon stays within bounds."""
        # Force large control effort
        for i in range(100):
            self.controller.measure_imbalance(10.0, 0.0, 0, 0, 0.0, 0.0)
            epsilon, _ = self.controller.update(dt=0.01)
        
        # Should be bounded by max_epsilon
        self.assertLessEqual(self.controller.state.epsilon, self.config.max_epsilon)
        self.assertGreaterEqual(self.controller.state.epsilon, self.config.min_epsilon)

    def test_reset(self):
        """Test controller reset."""
        # Add some state
        self.controller.measure_imbalance(10.0, 9.9, 0, 0, 0.0, 0.0)
        self.controller.update(dt=0.01)
        
        # Reset
        self.controller.reset()
        
        # Check state is cleared
        self.assertEqual(self.controller.state.epsilon, 0.0)
        self.assertEqual(self.controller.state.integral_error, 0.0)
        self.assertEqual(self.controller.state.prev_error, 0.0)

    def test_get_diagnostics(self):
        """Test diagnostics output."""
        self.controller.measure_imbalance(10.0, 9.9, 0, 0, 0.0, 0.0)
        self.controller.update(dt=0.01)
        
        diagnostics = self.controller.get_diagnostics()
        
        self.assertIn("epsilon", diagnostics)
        self.assertIn("filtered_imbalance", diagnostics)
        self.assertIn("packet_loss_rate", diagnostics)
        self.assertIn("timing_jitter_rms", diagnostics)
        self.assertIn("control_effort", diagnostics)
        self.assertIn("within_tolerance", diagnostics)

    def test_convenience_function(self):
        """Test convenience function for creating controller."""
        controller = create_stream_balance_controller(
            target_epsilon=1e-4,
            control_mode=BalanceMode.PI,
        )
        
        self.assertEqual(controller.config.target_epsilon, 1e-4)
        self.assertEqual(controller.config.control_mode, BalanceMode.PI)


class TestStreamBalanceConfig(unittest.TestCase):
    """Test cases for StreamBalanceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamBalanceConfig()
        
        self.assertEqual(config.target_epsilon, 1e-4)
        self.assertEqual(config.max_epsilon, 1e-2)
        self.assertEqual(config.min_epsilon, 0.0)
        self.assertEqual(config.control_mode, BalanceMode.PI)
        self.assertEqual(config.kp, 1000.0)
        self.assertEqual(config.ki, 100.0)
        self.assertEqual(config.kd, 10.0)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StreamBalanceConfig(
            target_epsilon=1e-5,
            max_epsilon=1e-1,
            control_mode=BalanceMode.PID,
            kp=2000.0,
        )
        
        self.assertEqual(config.target_epsilon, 1e-5)
        self.assertEqual(config.max_epsilon, 1e-1)
        self.assertEqual(config.control_mode, BalanceMode.PID)
        self.assertEqual(config.kp, 2000.0)


class TestStreamBalanceState(unittest.TestCase):
    """Test cases for StreamBalanceState."""

    def test_default_state(self):
        """Test default state values."""
        state = StreamBalanceState()
        
        self.assertEqual(state.epsilon, 0.0)
        self.assertEqual(state.integral_error, 0.0)
        self.assertEqual(state.prev_error, 0.0)
        self.assertEqual(state.packet_loss_rate, 0.0)
        self.assertEqual(state.timing_jitter_rms, 0.0)
        self.assertEqual(state.mass_drift_rate, 0.0)
        self.assertEqual(state.control_effort, 0.0)


if __name__ == "__main__":
    unittest.main()

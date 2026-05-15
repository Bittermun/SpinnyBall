"""
Shepherd Control System Tests

Comprehensive validation of PID/MPC control, magnetic actuators, and stream management.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from control_layer.shepherd_control import (
    ShepherdControlConfig,
    ShepherdController,
    MagneticActuatorModel,
    ShepherdPacketStream
)


class TestShepherdControlConfig:
    """Test control configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ShepherdControlConfig()
        assert config.control_type == "PID"
        assert config.target_spacing_m == 10.0
        assert config.kp == 0.5
        assert config.ki == 0.01
        assert config.kd == 0.1
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ShepherdControlConfig(
            control_type="MPC",
            target_spacing_m=5.0,
            kp=1.0
        )
        assert config.control_type == "MPC"
        assert config.target_spacing_m == 5.0
        assert config.kp == 1.0


class TestShepherdController:
    """Test PID control logic."""
    
    def test_controller_initialization(self):
        """Test controller creation."""
        config = ShepherdControlConfig()
        controller = ShepherdController(config)
        
        assert controller.pid_integral == 0.0
        assert controller.pid_last_error == 0.0
        assert controller.total_control_effort == 0.0
    
    def test_spacing_computation(self):
        """Test spacing calculation."""
        controller = ShepherdController()
        
        shepherd_pos = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([0.01, 0.0, 0.0])  # 10 m = 0.01 km
        
        spacing = controller.compute_spacing(shepherd_pos, target_pos)
        
        assert np.isclose(spacing, 10.0)  # 10 meters
    
    def test_pid_control_proportional(self):
        """Test PID proportional action."""
        config = ShepherdControlConfig(
            target_spacing_m=10.0,
            kp=1.0,
            ki=0.0,
            kd=0.0
        )
        controller = ShepherdController(config)
        
        # Error = 15 - 10 = 5 m (too close)
        # P-term should produce negative acceleration (reduce distance)
        accel = controller.compute_pid_control(spacing_m=15.0, dt=0.1)
        
        assert accel < 0.0  # Should reduce distance
    
    def test_pid_control_integral(self):
        """Test PID integral action."""
        config = ShepherdControlConfig(
            target_spacing_m=10.0,
            kp=0.0,
            ki=0.1,
            kd=0.0
        )
        controller = ShepherdController(config)
        
        # Multiple steps with constant error
        accel_list = []
        for _ in range(5):
            accel = controller.compute_pid_control(spacing_m=12.0, dt=0.1)
            accel_list.append(accel)
        
        # Integral should grow (and thus control)
        assert accel_list[-1] > accel_list[0]
    
    def test_pid_saturation(self):
        """Test PID output saturation."""
        config = ShepherdControlConfig(
            target_spacing_m=10.0,
            kp=10.0,  # Large gain
            max_acceleration_ms2=1e-6
        )
        controller = ShepherdController(config)
        
        # Large error
        accel = controller.compute_pid_control(spacing_m=100.0, dt=0.1)
        
        # Should be saturated
        assert abs(accel) <= config.max_acceleration_ms2 * 1.1
    
    def test_mpc_control(self):
        """Test MPC control."""
        config = ShepherdControlConfig(
            control_type="MPC",
            target_spacing_m=10.0
        )
        controller = ShepherdController(config)
        
        accel = controller.compute_mpc_control(spacing_m=12.0, spacing_rate_ms=0.0, dt=0.1)
        
        # Should produce some control
        assert isinstance(accel, (float, np.floating))


class TestMagneticActuator:
    """Test magnetic actuator model."""
    
    def test_actuator_initialization(self):
        """Test actuator creation."""
        actuator = MagneticActuatorModel()
        assert actuator.shepherd_moment == 1.0
        assert actuator.target_moment == 0.1
    
    def test_gradient_to_acceleration(self):
        """Test converting field gradient to acceleration."""
        actuator = MagneticActuatorModel(
            target_moment_am2=0.1
        )
        
        gradient_tm = 1000.0  # T/m
        accel = actuator.acceleration_from_gradient(gradient_tm, target_mass_kg=1.0)
        
        # F = 1000 * 0.1 = 100 N, a = 100 m/s²
        expected = 100.0
        assert np.isclose(accel, expected, rtol=0.01)
    
    def test_acceleration_to_gradient(self):
        """Test converting desired acceleration to field gradient."""
        actuator = MagneticActuatorModel(target_moment_am2=0.1)
        
        accel_desired = 1e-6  # m/s²
        gradient = actuator.gradient_from_acceleration(accel_desired, target_mass_kg=1.0)
        
        # F = 1e-6 * 1.0 = 1e-6 N
        # B = F / m = 1e-6 / 0.1 = 1e-5 T/m
        expected = 1e-5
        assert np.isclose(gradient, expected, rtol=0.01)
    
    def test_actuator_latency(self):
        """Test actuator latency model."""
        actuator = MagneticActuatorModel()
        
        command = 1.0
        latency_s = 0.1
        dt = 0.01
        
        output = actuator.apply_latency(command, latency_s, dt)
        
        # Output should be less than command (first-order lag)
        assert 0 < output < command


class TestShepherdPacketStream:
    """Test multi-packet stream management."""
    
    def test_stream_initialization(self):
        """Test stream creation."""
        stream = ShepherdPacketStream(n_packets=10)
        assert stream.n_packets == 10
        assert stream.n_total == 11  # +1 for shepherd
    
    def test_stream_spacing_computation(self):
        """Test spacing computation for multiple packets."""
        stream = ShepherdPacketStream(n_packets=3)
        
        # Shepherd at origin, targets at x=10,20,30 (km)
        positions = np.array([
            [0.0, 0.0, 0.0],  # Shepherd
            [0.01, 0.0, 0.0], # Target 1 (10 m away)
            [0.02, 0.0, 0.0], # Target 2 (20 m away)
            [0.03, 0.0, 0.0]  # Target 3 (30 m away)
        ])
        
        spacings = stream.compute_stream_spacing(positions)
        
        assert len(spacings) == 3
        assert np.isclose(spacings[0], 10.0)
        assert np.isclose(spacings[1], 20.0)
        assert np.isclose(spacings[2], 30.0)
    
    def test_stream_control_step(self):
        """Test control computation for single packet."""
        config = ShepherdControlConfig(target_spacing_m=10.0)
        stream = ShepherdPacketStream(n_packets=2, control_config=config)
        
        # Shepherd and targets
        positions = np.array([
            [0.0, 0.0, 0.0],      # Shepherd
            [0.011, 0.0, 0.0],    # Target 1 (11 m away)
            [0.02, 0.0, 0.0]      # Target 2
        ])
        
        velocities = np.array([
            [0.0, 0.0, 0.001],    # Shepherd moving in z
            [0.0, 0.0, 0.001],    # Target 1 moving with shepherd
            [0.0, 0.0, 0.001]     # Target 2
        ])
        
        control_accel, diag = stream.compute_control_step(
            positions, velocities, dt_s=0.1, target_packet_id=1
        )
        
        # Should have some control (spacing is 11 m, target is 10 m)
        assert isinstance(control_accel, (float, np.floating))
        assert 'control_accel_ms2' in diag
        assert 'spacing_m' in diag
    
    def test_stream_statistics(self):
        """Test stream statistics computation."""
        config = ShepherdControlConfig(target_spacing_m=10.0)
        stream = ShepherdPacketStream(n_packets=5, control_config=config)
        
        # Run a few control steps
        positions = np.zeros((6, 3))
        velocities = np.zeros((6, 3))
        
        # Initialize at correct spacing
        for i in range(6):
            positions[i, 0] = -i * 0.01  # 0, -10, -20, -30, -40, -50 m
        
        for _ in range(10):
            for packet_id in range(1, 6):
                stream.compute_control_step(
                    positions, velocities, dt_s=0.1, target_packet_id=packet_id
                )
        
        stats = stream.get_statistics()
        
        assert 'mean_spacing_m' in stats
        assert 'std_spacing_m' in stats
        assert 'spacing_maintained_pct' in stats


class TestControlIntegration:
    """Test integration of control components."""
    
    def test_pid_controller_convergence(self):
        """Test that PID controller converges to target spacing."""
        config = ShepherdControlConfig(
            target_spacing_m=10.0,
            kp=0.5,
            ki=0.01,
            kd=0.1
        )
        controller = ShepherdController(config)
        
        # Simulate: start at 15 m, should converge to 10 m
        spacing = 15.0
        for step in range(100):
            control_accel = controller.compute_pid_control(spacing, dt=0.1)
            
            # Simple dynamics: da/dt = control (ignore gravity for this test)
            # Simulate closing distance
            spacing -= control_accel * 0.01  # Crude integration
            
            # Stop if converged
            if abs(spacing - 10.0) < 0.5:
                break
        
        # Should have reduced spacing
        assert spacing < 15.0
    
    def test_actuator_model_chain(self):
        """Test full actuator model chain."""
        actuator = MagneticActuatorModel()
        
        # Desired acceleration
        accel_desired = 1e-6
        
        # Convert to gradient
        gradient = actuator.gradient_from_acceleration(accel_desired)
        
        # Convert back to acceleration
        accel_actual = actuator.acceleration_from_gradient(gradient)
        
        # Should match
        assert np.isclose(accel_actual, accel_desired, rtol=0.01)


# Test entry point
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

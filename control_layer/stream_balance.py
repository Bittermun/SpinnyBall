"""
Stream balance controller for counter-stream mass-flow regulation.

Implements dynamic control of stream balance parameter ε to maintain
<0.01% mismatch between counter-streams under packet loss, mass drift,
and timing jitter perturbations.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class BalanceMode(Enum):
    """Control mode for stream balance."""
    PROPORTIONAL = "proportional"
    PI = "pi"
    PID = "pid"
    ADAPTIVE = "adaptive"


@dataclass
class StreamBalanceConfig:
    """Configuration for stream balance controller."""
    target_epsilon: float = 1e-4  # Target balance ε < 10⁻⁴
    max_epsilon: float = 1e-2  # Maximum allowable ε (1%)
    min_epsilon: float = 0.0  # Minimum ε (perfect balance)
    control_mode: BalanceMode = BalanceMode.PI
    kp: float = 1000.0  # Proportional gain
    ki: float = 100.0  # Integral gain
    kd: float = 10.0  # Derivative gain
    integral_limit: float = 1e-3  # Anti-windup limit
    measurement_window: int = 100  # Number of samples for averaging
    packet_loss_threshold: float = 0.01  # 1% packet loss triggers corrective action
    timing_jitter_threshold: float = 1e-6  # 1 μs jitter threshold
    max_derivative: float = 1000.0  # Maximum derivative for PID
    max_adaptive_gain: float = 10.0  # Maximum adaptive gain factor
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive")
        if self.max_epsilon <= self.min_epsilon:
            raise ValueError("max_epsilon must be greater than min_epsilon")
        if self.kp < 0 or self.ki < 0 or self.kd < 0:
            raise ValueError("Control gains must be non-negative")
        if self.integral_limit <= 0:
            raise ValueError("integral_limit must be positive")
        if self.measurement_window <= 0:
            raise ValueError("measurement_window must be positive")
        if self.packet_loss_threshold < 0 or self.packet_loss_threshold > 1:
            raise ValueError("packet_loss_threshold must be in [0, 1]")
        if self.timing_jitter_threshold < 0:
            raise ValueError("timing_jitter_threshold must be non-negative")


@dataclass
class StreamBalanceState:
    """Current state of stream balance controller."""
    epsilon: float = 0.0
    integral_error: float = 0.0
    prev_error: float = 0.0
    packet_loss_rate: float = 0.0
    timing_jitter_rms: float = 0.0
    mass_drift_rate: float = 0.0
    control_effort: float = 0.0
    total_packets_processed: int = 0  # Track total packets for rate calculation
    total_packets_lost: int = 0  # Track total lost packets


class StreamBalanceController:
    """
    Dynamic stream balance controller for counter-stream regulation.
    
    Maintains ε < 10⁻⁴ by adjusting stream parameters in response to:
    - Packet loss
    - Mass drift
    - Timing jitter
    
    Control law:
        ε(t) = Kp * e(t) + Ki * ∫e(τ)dτ + Kd * de/dt
    where e(t) is the measured stream imbalance.
    """
    
    def __init__(self, config: StreamBalanceConfig = None):
        """
        Initialize stream balance controller.
        
        Args:
            config: StreamBalanceConfig instance
        """
        self.config = config or StreamBalanceConfig()
        self.state = StreamBalanceState()
        self.measurement_buffer = np.zeros(self.config.measurement_window)
        self.buffer_idx = 0
        self.valid_samples = 0  # Track number of valid samples in buffer
        
    def measure_imbalance(
        self,
        flow_plus: float,
        flow_minus: float,
        packet_loss_plus: int = 0,
        packet_loss_minus: int = 0,
        timing_jitter_plus: float = 0.0,
        timing_jitter_minus: float = 0.0,
    ) -> float:
        """
        Measure current stream imbalance.
        
        Args:
            flow_plus: Mass flow rate in plus stream (kg/s)
            flow_minus: Mass flow rate in minus stream (kg/s)
            packet_loss_plus: Number of lost packets in plus stream
            packet_loss_minus: Number of lost packets in minus stream
            timing_jitter_plus: RMS timing jitter in plus stream (s)
            timing_jitter_minus: RMS timing jitter in minus stream (s)
        
        Returns:
            Measured imbalance ε
        """
        # Flow imbalance
        total_flow = flow_plus + flow_minus
        if total_flow > 0:
            flow_imbalance = abs(flow_plus - flow_minus) / total_flow
        else:
            flow_imbalance = 0.0
        
        # Packet loss contribution
        total_packets = packet_loss_plus + packet_loss_minus
        if total_packets > 0:
            loss_imbalance = abs(packet_loss_plus - packet_loss_minus) / total_packets
        else:
            loss_imbalance = 0.0
        
        # Timing jitter contribution (normalized to packet period)
        jitter_rms = np.sqrt((timing_jitter_plus**2 + timing_jitter_minus**2) / 2)
        jitter_imbalance = jitter_rms / self.config.timing_jitter_threshold if self.config.timing_jitter_threshold > 0 else 0.0
        
        # Combined imbalance
        epsilon = flow_imbalance + 0.5 * loss_imbalance + 0.3 * jitter_imbalance
        
        # Update state - track packet loss rate properly
        # Track actual packets processed (2 streams * n_packets_per_stream)
        n_packets_per_stream = len(packet_loss_plus) if hasattr(packet_loss_plus, '__len__') else 1
        self.state.total_packets_processed += 2 * n_packets_per_stream  # Both streams
        self.state.total_packets_lost += total_packets
        if self.state.total_packets_processed > 0:
            self.state.packet_loss_rate = self.state.total_packets_lost / self.state.total_packets_processed
        self.state.timing_jitter_rms = jitter_rms
        
        # Add to measurement buffer
        self.measurement_buffer[self.buffer_idx] = epsilon
        self.buffer_idx = (self.buffer_idx + 1) % self.config.measurement_window
        if self.valid_samples < self.config.measurement_window:
            self.valid_samples += 1
        
        return epsilon
    
    def get_filtered_imbalance(self) -> float:
        """Get filtered (averaged) imbalance from measurement buffer."""
        if self.valid_samples == 0:
            return 0.0
        return np.mean(self.measurement_buffer[:self.valid_samples])
    
    def update(self, dt: float) -> Tuple[float, float]:
        """
        Update controller and compute control action.
        
        Args:
            dt: Time step (s)
        
        Returns:
            Tuple of (epsilon, control_effort)
        """
        # Get filtered measurement
        epsilon_filtered = self.get_filtered_imbalance()
        
        # Error from target
        error = self.config.target_epsilon - epsilon_filtered
        
        # Control based on mode
        if self.config.control_mode == BalanceMode.PROPORTIONAL:
            control = self.config.kp * error
        elif self.config.control_mode == BalanceMode.PI:
            # Integral with anti-windup
            self.state.integral_error += error * dt
            self.state.integral_error = np.clip(
                self.state.integral_error,
                -self.config.integral_limit,
                self.config.integral_limit
            )
            control = self.config.kp * error + self.config.ki * self.state.integral_error
        elif self.config.control_mode == BalanceMode.PID:
            # Derivative term with limiting
            derivative = (error - self.state.prev_error) / dt if dt > 0 else 0.0
            derivative = np.clip(derivative, -self.config.max_derivative, self.config.max_derivative)
            self.state.prev_error = error
            
            # Integral with anti-windup
            self.state.integral_error += error * dt
            self.state.integral_error = np.clip(
                self.state.integral_error,
                -self.config.integral_limit,
                self.config.integral_limit
            )
            
            control = (self.config.kp * error + 
                      self.config.ki * self.state.integral_error + 
                      self.config.kd * derivative)
        else:  # ADAPTIVE
            # Adaptive gain based on error magnitude with clamping
            gain_factor = 1.0 + 10.0 * abs(error)
            gain_factor = np.clip(gain_factor, 1.0, self.config.max_adaptive_gain)
            self.state.integral_error += error * dt
            self.state.integral_error = np.clip(
                self.state.integral_error,
                -self.config.integral_limit,
                self.config.integral_limit
            )
            control = gain_factor * self.config.kp * error + self.config.ki * self.state.integral_error
        
        # Apply control to epsilon (bounded)
        epsilon_new = self.state.epsilon + control * dt
        epsilon_new = np.clip(epsilon_new, self.config.min_epsilon, self.config.max_epsilon)
        
        self.state.epsilon = epsilon_new
        self.state.control_effort = control
        
        return epsilon_new, control
    
    def reset(self):
        """Reset controller state."""
        self.state = StreamBalanceState()
        self.measurement_buffer = np.zeros(self.config.measurement_window)
        self.buffer_idx = 0
        self.valid_samples = 0
    
    def get_diagnostics(self) -> dict:
        """Get controller diagnostics."""
        return {
            "epsilon": self.state.epsilon,
            "filtered_imbalance": self.get_filtered_imbalance(),
            "packet_loss_rate": self.state.packet_loss_rate,
            "timing_jitter_rms": self.state.timing_jitter_rms,
            "control_effort": self.state.control_effort,
            "integral_error": self.state.integral_error,
            "within_tolerance": self.state.epsilon <= self.config.target_epsilon,
        }


def create_stream_balance_controller(
    target_epsilon: float = 1e-4,
    control_mode: BalanceMode = BalanceMode.PI,
) -> StreamBalanceController:
    """
    Convenience function to create stream balance controller.
    
    Args:
        target_epsilon: Target balance ε
        control_mode: Control mode
    
    Returns:
        StreamBalanceController instance
    """
    config = StreamBalanceConfig(
        target_epsilon=target_epsilon,
        control_mode=control_mode,
    )
    return StreamBalanceController(config)

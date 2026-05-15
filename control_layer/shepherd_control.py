"""
Shepherd Control Loop for Cislunar Packet Swarms

Implements feedback control to maintain packet spacing in cislunar environment.

Features:
    - PID and MPC control laws
    - Magnetic force actuator model
    - Multi-packet stream support
    - Latency and discretization
    - Real-time state estimation
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from scipy.optimize import minimize


@dataclass
class ShepherdControlConfig:
    """Configuration for shepherd control loop."""
    
    control_type: str = "PID"
    """Control law: 'PID' or 'MPC'."""
    
    target_spacing_m: float = 10.0
    """Desired spacing between packets in meters."""
    
    spacing_tolerance_m: float = 1.0
    """Tolerance for spacing error (±) in meters."""
    
    kp: float = 0.5
    """PID proportional gain."""
    
    ki: float = 0.01
    """PID integral gain."""
    
    kd: float = 0.1
    """PID derivative gain."""
    
    max_acceleration_ms2: float = 1e-6
    """Maximum control acceleration in m/s²."""
    
    control_rate_hz: float = 1.0
    """Control update frequency in Hz."""
    
    measurement_noise_m: float = 0.01
    """Measurement noise std dev in meters."""
    
    actuator_latency_s: float = 0.1
    """Actuator response latency in seconds."""
    
    mpc_horizon_s: float = 100.0
    """MPC prediction horizon in seconds."""
    
    mpc_n_steps: int = 10
    """Number of MPC discretization steps."""


class ShepherdController:
    """
    Shepherd control system for maintaining packet spacing.
    
    Control objectives:
    1. Maintain desired spacing d_target between shepherd and target
    2. Minimize control effort (fuel/magnetic force)
    3. Handle measurement noise and latency
    4. Prevent collision (enforced spacing > d_min)
    """
    
    # Minimum safe spacing (collision buffer)
    MIN_SPACING_M = 1.0
    
    def __init__(self, config: Optional[ShepherdControlConfig] = None):
        """
        Initialize shepherd controller.
        
        Args:
            config: ShepherdControlConfig instance
        """
        self.config = config or ShepherdControlConfig()
        
        # PID state
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        
        # Measurement history
        self.measurement_history = []
        self.time_history = []
        
        # Control effort tracking
        self.total_control_effort = 0.0
        self.control_commands = []
    
    def compute_spacing(self, shepherd_pos_km: np.ndarray, 
                       target_pos_km: np.ndarray) -> float:
        """
        Compute spacing between shepherd and target.
        
        Args:
            shepherd_pos_km: Shepherd position in ECI km
            target_pos_km: Target position in ECI km
        
        Returns:
            Spacing in meters
        """
        delta_pos_km = target_pos_km - shepherd_pos_km
        spacing_m = np.linalg.norm(delta_pos_km) * 1000.0
        return spacing_m
    
    def compute_pid_control(self, spacing_m: float, dt: float, 
                           target_velocity: float = 0.0) -> float:
        """
        Compute PID control acceleration.
        
        Args:
            spacing_m: Current spacing in meters
            dt: Time step in seconds
            target_velocity: Target spacing velocity (for D term)
        
        Returns:
            Control acceleration in m/s² (magnitude)
        """
        # Error
        error = spacing_m - self.config.target_spacing_m
        
        # Proportional
        p_term = self.config.kp * error
        
        # Integral (with anti-windup)
        self.pid_integral += error * dt
        self.pid_integral = np.clip(
            self.pid_integral,
            -1.0 / self.config.ki,
            1.0 / self.config.ki
        )
        i_term = self.config.ki * self.pid_integral
        
        # Derivative
        if dt > 0:
            d_error = (error - self.pid_last_error) / dt - target_velocity
            d_term = self.config.kd * d_error
        else:
            d_term = 0.0
        
        self.pid_last_error = error
        
        # Total
        control_accel = p_term + i_term + d_term
        
        # Saturate
        control_accel = np.clip(
            control_accel,
            -self.config.max_acceleration_ms2,
            self.config.max_acceleration_ms2
        )
        
        return control_accel
    
    def compute_mpc_control(self, spacing_m: float, spacing_rate_ms: float,
                           dt: float) -> float:
        """
        Compute Model Predictive Control acceleration.
        
        Objective: minimize ||spacing - target||² + λ ||control||²
        
        Args:
            spacing_m: Current spacing
            spacing_rate_ms: Spacing rate (d(spacing)/dt) in m/s
            dt: Time step in seconds
        
        Returns:
            Optimal control acceleration in m/s²
        """
        # Simple linear model: d²s/dt² = a_control
        # Predict trajectory over horizon
        
        n_steps = self.config.mpc_n_steps
        dt_mpc = self.config.mpc_horizon_s / n_steps
        
        def objective(accel_seq):
            """Objective function: tracking error + control effort."""
            cost = 0.0
            s = spacing_m
            s_dot = spacing_rate_ms
            
            for a in accel_seq:
                # Integrate
                s = s + s_dot * dt_mpc
                s_dot = s_dot + a * dt_mpc
                
                # Tracking error
                cost += (s - self.config.target_spacing_m)**2
                
                # Control effort
                cost += 0.1 * a**2
            
            # Terminal cost
            cost += 10.0 * (s - self.config.target_spacing_m)**2
            
            return cost
        
        # Optimize
        a0 = np.zeros(n_steps)
        result = minimize(
            objective,
            a0,
            method='L-BFGS-B',
            bounds=[
                (-self.config.max_acceleration_ms2, self.config.max_acceleration_ms2)
                for _ in range(n_steps)
            ],
            options={'maxiter': 20}
        )
        
        # Return first control (receding horizon)
        control_accel = result.x[0]
        
        return float(control_accel)
    
    def compute_control_acceleration(self, spacing_m: float, 
                                     spacing_rate_ms: float,
                                     dt: float) -> Tuple[float, Dict]:
        """
        Compute control acceleration command.
        
        Args:
            spacing_m: Current spacing in meters
            spacing_rate_ms: Rate of spacing change in m/s
            dt: Time step in seconds
        
        Returns:
            (control_acceleration_ms2, diagnostics_dict)
        """
        # Check bounds
        if spacing_m <= self.MIN_SPACING_M:
            # Emergency: collision imminent
            control_accel = -self.config.max_acceleration_ms2 * 2.0
            status = "COLLISION_ALERT"
        elif spacing_m > self.config.target_spacing_m + 100:
            # Lost target
            control_accel = self.config.max_acceleration_ms2
            status = "PURSUING"
        else:
            # Normal control
            if self.config.control_type == "PID":
                control_accel = self.compute_pid_control(spacing_m, dt)
                status = "PID"
            elif self.config.control_type == "MPC":
                control_accel = self.compute_mpc_control(spacing_m, spacing_rate_ms, dt)
                status = "MPC"
            else:
                control_accel = 0.0
                status = "UNKNOWN"
        
        # Saturate
        control_accel = np.clip(
            control_accel,
            -self.config.max_acceleration_ms2 * 2.0,
            self.config.max_acceleration_ms2 * 2.0
        )
        
        # Track effort
        self.total_control_effort += abs(control_accel) * dt
        self.control_commands.append(control_accel)
        
        diagnostics = {
            'control_accel_ms2': control_accel,
            'spacing_m': spacing_m,
            'spacing_rate_ms': spacing_rate_ms,
            'status': status,
            'total_effort': self.total_control_effort
        }
        
        return control_accel, diagnostics
    
    def reset(self):
        """Reset controller state."""
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.measurement_history = []
        self.time_history = []
        self.total_control_effort = 0.0
        self.control_commands = []


class MagneticActuatorModel:
    """
    Model of magnetic force actuator.
    
    Converts control acceleration command to actual magnetic force.
    Includes saturation, latency, and noise.
    """
    
    def __init__(self, shepherd_moment_am2: float = 1.0,
                 target_moment_am2: float = 0.1,
                 max_gradient_tm: float = 1000.0):
        """
        Initialize magnetic actuator.
        
        Args:
            shepherd_moment_am2: Shepherd magnetic moment
            target_moment_am2: Target packet magnetic moment
            max_gradient_tm: Maximum field gradient in T/m
        """
        self.shepherd_moment = shepherd_moment_am2
        self.target_moment = target_moment_am2
        self.max_gradient = max_gradient_tm
    
    def acceleration_from_gradient(self, gradient_tm: float,
                                   target_mass_kg: float = 1.0) -> float:
        """
        Convert field gradient to acceleration on target.
        
        F = dB/dr × m  →  a = F/M
        
        Args:
            gradient_tm: Magnetic field gradient in T/m
            target_mass_kg: Target mass in kg
        
        Returns:
            Acceleration in m/s²
        """
        force_n = gradient_tm * self.target_moment
        accel_ms2 = force_n / target_mass_kg
        return accel_ms2
    
    def gradient_from_acceleration(self, accel_ms2: float,
                                   target_mass_kg: float = 1.0) -> float:
        """
        Convert desired acceleration to required field gradient.
        
        Args:
            accel_ms2: Desired acceleration in m/s²
            target_mass_kg: Target mass in kg
        
        Returns:
            Required gradient in T/m
        """
        force_n = accel_ms2 * target_mass_kg
        gradient_tm = force_n / self.target_moment
        
        # Saturate
        gradient_tm = np.clip(gradient_tm, -self.max_gradient, self.max_gradient)
        
        return gradient_tm
    
    def apply_latency(self, command: float, latency_s: float,
                     simulation_dt_s: float) -> float:
        """
        Apply actuator latency (simplified: first-order lag).
        
        Args:
            command: Desired control output
            latency_s: Time constant in seconds
            simulation_dt_s: Integration time step
        
        Returns:
            Actual output with latency
        """
        if latency_s <= 0:
            return command
        
        # First-order response: tau * dx/dt = -x + u
        tau = latency_s
        actual_accel = command * (1 - np.exp(-simulation_dt_s / tau))
        
        return actual_accel


class ShepherdPacketStream:
    """
    Multi-packet stream with shepherd control.
    
    Manages N target packets controlled by 1 shepherd packet.
    """
    
    def __init__(self, n_packets: int = 10,
                 control_config: Optional[ShepherdControlConfig] = None):
        """
        Initialize packet stream.
        
        Args:
            n_packets: Number of target packets
            control_config: Control configuration
        """
        self.n_packets = n_packets
        self.n_total = n_packets + 1  # +1 for shepherd
        
        self.controller = ShepherdController(control_config)
        self.actuator = MagneticActuatorModel()
        
        # State history
        self.spacing_history = []
        self.control_history = []
        self.time_history = []
        
        # Packet index: 0 = shepherd, 1...n_packets = targets
    
    def compute_stream_spacing(self, positions_km: np.ndarray) -> np.ndarray:
        """
        Compute spacing for all packets relative to shepherd.
        
        Args:
            positions_km: Positions shape (n_total, 3) in km
        
        Returns:
            Spacings shape (n_packets,) in meters
        """
        shepherd_pos = positions_km[0]
        target_spacings = []
        
        for i in range(1, self.n_total):
            spacing = self.controller.compute_spacing(
                shepherd_pos,
                positions_km[i]
            )
            target_spacings.append(spacing)
        
        return np.array(target_spacings)
    
    def compute_control_step(self, positions_km: np.ndarray,
                           velocities_kms: np.ndarray,
                           dt_s: float,
                           target_packet_id: int = 1) -> Tuple[float, Dict]:
        """
        Compute control for single target packet.
        
        Args:
            positions_km: All positions
            velocities_kms: All velocities
            dt_s: Time step
            target_packet_id: Which packet to control (1...n_packets)
        
        Returns:
            (control_acceleration, diagnostics)
        """
        shepherd_pos = positions_km[0]
        target_pos = positions_km[target_packet_id]
        
        shepherd_vel = velocities_kms[0]
        target_vel = velocities_kms[target_packet_id]
        
        # Spacing and spacing rate
        spacing_m = self.controller.compute_spacing(shepherd_pos, target_pos)
        
        # Spacing rate: d(spacing)/dt
        rel_pos = target_pos - shepherd_pos
        rel_vel = target_vel - shepherd_vel
        
        if np.linalg.norm(rel_pos) > 1e-10:
            spacing_rate_ms = np.dot(rel_vel * 1000, rel_pos) / np.linalg.norm(rel_pos)
        else:
            spacing_rate_ms = 0.0
        
        # Control command
        control_accel_ms2, diagnostics = self.controller.compute_control_acceleration(
            spacing_m,
            spacing_rate_ms,
            dt_s
        )
        
        # Store history
        self.spacing_history.append(spacing_m)
        self.control_history.append(control_accel_ms2)
        
        return control_accel_ms2, diagnostics
    
    def get_statistics(self) -> Dict:
        """
        Get stream control statistics.
        
        Returns:
            Statistics dictionary
        """
        if len(self.spacing_history) == 0:
            return {}
        
        spacings = np.array(self.spacing_history)
        controls = np.array(self.control_history)
        
        stats = {
            'mean_spacing_m': np.mean(spacings),
            'std_spacing_m': np.std(spacings),
            'min_spacing_m': np.min(spacings),
            'max_spacing_m': np.max(spacings),
            'mean_control_ms2': np.mean(np.abs(controls)),
            'max_control_ms2': np.max(np.abs(controls)),
            'total_control_effort': self.controller.total_control_effort,
            'spacing_maintained_pct': 100.0 * np.sum(
                np.abs(spacings - self.controller.config.target_spacing_m) 
                <= self.controller.config.spacing_tolerance_m
            ) / len(spacings)
        }
        
        return stats


# Export
__all__ = [
    'ShepherdControlConfig',
    'ShepherdController',
    'MagneticActuatorModel',
    'ShepherdPacketStream',
]

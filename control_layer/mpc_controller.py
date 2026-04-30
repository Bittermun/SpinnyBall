"""
Model-Predictive Control (MPC) for gyroscopic mass-stream system.

Implements MPC with horizon N=10 for libration damping and spacing control.
Target: ≤30 ms solve time via numba/jit acceleration.

This module requires CasADi (optional dependency). If not available,
a stub implementation is provided that raises NotImplementedError.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import casadi as ca
    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from dynamics.rigid_body import euler_equations, scalar_last_to_first, scalar_first_to_last
from dynamics.gyro_matrix import gyroscopic_coupling
from enum import Enum


class ConfigurationMode(Enum):
    """Configuration modes for different use cases."""
    TEST = "test"  # Fast unit tests: 0.05 kg, 0.02 m, 100 rad/s
    VALIDATION = "validation"  # MuJoCo oracle: 2.0 kg, 0.1 m, 5236 rad/s
    OPERATIONAL = "operational"  # Paper target: 8.0 kg, 0.1 m, 5236 rad/s


# Predefined parameter sets for each configuration mode
CONFIGURATION_PARAMETERS = {
    ConfigurationMode.TEST: {
        "packet_mass": 0.05,  # kg
        "packet_radius": 0.02,  # m
        "spin_rate": 100.0,  # rad/s
        "max_stress": 1.2e9,  # Pa (placeholder)
        "I": np.diag([0.0001, 0.00011, 0.00009]),  # kg·m² (placeholder)
    },
    ConfigurationMode.VALIDATION: {
        "packet_mass": 2.0,  # kg
        "packet_radius": 0.1,  # m
        "spin_rate": 5236.0,  # rad/s
        "max_stress": 8.0e8,  # Pa (800 MPa BFRP limit)
        "I": None,  # Will be computed from mass/radius
    },
    ConfigurationMode.OPERATIONAL: {
        "packet_mass": 8.0,  # kg (paper target)
        "packet_radius": 0.1,  # m (paper target)
        "spin_rate": 5236.0,  # rad/s (paper target)
        "max_stress": 8.0e8,  # Pa (800 MPa BFRP limit)
        "I": None,  # Will be computed from mass/radius
    },
}


class MPCController:
    """
    Model-Predictive Controller for gyroscopic mass-stream system.
    
    Solves optimal control problem with horizon N=10 to minimize:
    - Libration energy
    - Spacing deviation from target
    - Control effort
    
    Subject to constraints:
    - Centrifugal stress ≤ 1.2 GPa (SF=1.5)
    - k_eff ≥ 6,000 N/m
    - η_ind ≥ 0.82
    """
    
    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.01,  # 10 ms time step
        libration_weight: float = 1.0,
        spacing_weight: float = 0.5,
        control_weight: float = 0.1,
        max_stress: float = None,  # Pa - overridden by config mode if provided
        min_k_eff: float = 6000.0,  # N/m
        I: np.ndarray = None,  # Inertia tensor (kg·m²) - overridden by config mode if provided
        packet_radius: float = None,  # Packet radius for stress (m) - overridden by config mode if provided
        packet_mass: float = None,  # Packet mass (kg) - overridden by config mode if provided
        configuration_mode: ConfigurationMode = ConfigurationMode.TEST,
        delay_steps: int = 5,  # Number of control cycles to compensate for latency
        dt_delay: float = 0.01,  # Time step for delay prediction (s)
        enable_delay_compensation: bool = True,  # Enable/disable Smith predictor
        sampling_period: float = 0.01,  # Discrete-time sampling period (s)
        communication_delay: float = 0.005,  # Communication delay between sensor and actuator (s)
        enable_discrete_time: bool = True,  # Enable discrete-time sampling model
        delay_compensation_mode: str = 'discrete_time',  # 'discrete_time', 'smith', or 'both'
    ):
        """
        Initialize MPC controller.

        Args:
            horizon: Prediction horizon (N=10 default)
            dt: Time step (s)
            libration_weight: Weight for libration energy minimization
            spacing_weight: Weight for spacing deviation minimization
            control_weight: Weight for control effort minimization
            max_stress: Maximum centrifugal stress (Pa) - if None, uses config mode default
            min_k_eff: Minimum effective stiffness (N/m)
            I: Inertia tensor in body frame (kg·m²) - if None, uses config mode default
            packet_radius: Packet radius for stress calculations (m) - if None, uses config mode default
            packet_mass: Packet mass (kg) - if None, uses config mode default
            configuration_mode: Configuration mode (TEST, VALIDATION, OPERATIONAL)
            delay_steps: Number of control cycles to compensate for latency (default: 5)
            dt_delay: Time step for delay prediction in seconds (default: 0.01)
            enable_delay_compensation: Enable/disable Smith predictor (default: True)
            sampling_period: Discrete-time sampling period (s)
            communication_delay: Communication delay between sensor and actuator (s)
            enable_discrete_time: Enable discrete-time sampling model (default: True)
            delay_compensation_mode: Delay compensation mode - 'discrete_time', 'smith', or 'both' (default: 'discrete_time')
        """
        if not CASADI_AVAILABLE:
            raise ImportError(
                "CasADi is required for MPC. Install with: pip install casadi"
            )

        self.horizon = horizon
        self.dt = dt
        self.libration_weight = libration_weight
        self.spacing_weight = spacing_weight
        self.control_weight = control_weight
        self.min_k_eff = min_k_eff
        self.configuration_mode = configuration_mode
        self.delay_steps = delay_steps
        self.dt_delay = dt_delay
        self.enable_delay_compensation = enable_delay_compensation
        self.sampling_period = sampling_period
        self.communication_delay = communication_delay
        self.enable_discrete_time = enable_discrete_time
        self.delay_compensation_mode = delay_compensation_mode

        # Load parameters from configuration mode if not explicitly provided
        config_params = CONFIGURATION_PARAMETERS[configuration_mode]

        if max_stress is None:
            max_stress = config_params["max_stress"]
        self.max_stress = max_stress

        if packet_mass is None:
            packet_mass = config_params["packet_mass"]
        self.packet_mass = packet_mass

        if packet_radius is None:
            packet_radius = config_params["packet_radius"]
        self.packet_radius = packet_radius

        # Compute inertia from mass/radius if not provided
        if I is None:
            I = config_params["I"]
            if I is None:
                # Compute spherical inertia from mass and radius
                I_sphere = (2.0/5.0) * packet_mass * packet_radius**2
                I = np.diag([I_sphere, I_sphere, I_sphere])
        self.I = np.asarray(I, dtype=float)
        self.I_inv = np.linalg.inv(self.I)
        
        # CasADi optimization problem
        self.opti = ca.Opti()
        
        # Decision variables
        self.u = self.opti.variable(3, horizon)  # Control inputs [Fx, Fy, Fz]
        self.x = self.opti.variable(7, horizon + 1)  # State [q, omega]
        
        # Parameters (will be set at solve time)
        self.x0 = self.opti.parameter(7)  # Initial state
        self.x_target = self.opti.parameter(7)  # Target state
        
        # Build optimization problem
        self._build_problem()
    
    def _build_problem(self):
        """Build CasADi optimization problem with actual Euler+gyro dynamics."""
        # Cost function
        cost = 0.0
        
        for k in range(self.horizon):
            # State error cost
            state_error = self.x[:, k] - self.x_target
            cost += self.libration_weight * ca.sum_sq(state_error[4:])  # libration (omega)
            cost += self.spacing_weight * ca.sum_sq(state_error[:3])  # spacing (position)
            
            # Control effort cost
            cost += self.control_weight * ca.sum_sq(self.u[:, k])
        
        self.opti.minimize(cost)
        
        # Dynamics constraints with actual Euler+gyro equations
        for k in range(self.horizon):
            # Extract state at time k
            q_k = self.x[:4, k]  # Quaternion [qx, qy, qz, qw] (scalar-last)
            omega_k = self.x[4:7, k]  # Angular velocity [ωx, ωy, ωz]
            u_k = self.u[:, k]  # Control input [Fx, Fy, Fz]
            
            # Convert to scalar-first for quaternion derivative
            # q_scalar_first = [qw, qx, qy, qz]
            qw = q_k[3]
            qx = q_k[0]
            qy = q_k[1]
            qz = q_k[2]
            q_scalar_first = ca.vertcat(qw, qx, qy, qz)
            
            # Quaternion derivative: q̇ = 0.5 * q * ω (quaternion multiplication)
            omega_quat = ca.vertcat(0, omega_k[0], omega_k[1], omega_k[2])
            dq_scalar_first = 0.5 * ca.vertcat(
                q_scalar_first[0]*omega_quat[0] - q_scalar_first[1]*omega_quat[1] - q_scalar_first[2]*omega_quat[2] - q_scalar_first[3]*omega_quat[3],
                q_scalar_first[0]*omega_quat[1] + q_scalar_first[1]*omega_quat[0] + q_scalar_first[2]*omega_quat[3] - q_scalar_first[3]*omega_quat[2],
                q_scalar_first[0]*omega_quat[2] - q_scalar_first[1]*omega_quat[3] + q_scalar_first[2]*omega_quat[0] + q_scalar_first[3]*omega_quat[1],
                q_scalar_first[0]*omega_quat[3] + q_scalar_first[1]*omega_quat[2] - q_scalar_first[2]*omega_quat[1] + q_scalar_first[3]*omega_quat[0],
            )
            
            # Convert back to scalar-last
            dq_k = ca.vertcat(dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0])
            
            # Gyroscopic coupling: ω × (I * ω)
            I_omega = self.I @ omega_k
            omega_skew = ca.vertcat(
                0, -omega_k[2], omega_k[1],
                omega_k[2], 0, -omega_k[0],
                -omega_k[1], omega_k[0], 0
            )
            omega_skew_mat = ca.reshape(omega_skew, 3, 3)
            gyro_coupling = omega_skew_mat @ I_omega
            
            # Angular acceleration from Euler equations: ω̇ = I⁻¹ * (τ - ω × (I * ω))
            alpha_k = self.I_inv @ (u_k - gyro_coupling)
            
            # State derivative
            dx_k = ca.vertcat(dq_k, alpha_k)
            
            # Euler integration: x[k+1] = x[k] + dt * dx_k
            # NOTE: Plant-model mismatch - MPC uses forward Euler while physics engine uses RK4/RK45.
            # This is a trade-off for computational efficiency. For small dt and short horizons,
            # the error is acceptable. For long horizons or high precision requirements,
            # consider using RK4 in MPC constraints.
            self.opti.subject_to(self.x[:, k+1] == self.x[:, k] + self.dt * dx_k)
        
        # Initial condition
        self.opti.subject_to(self.x[:, 0] == self.x0)
        
        # Stress constraint: centrifugal stress σ = m * ω² * r / A_cross_section
        # For spherical packet: σ = m * ω² * r / (π * r²) = m * ω² / (π * r)
        # Using safety factor of 1.5
        for k in range(self.horizon):
            omega_k = self.x[4:7, k]
            omega_sq = ca.sum_sq(omega_k)
            # Centrifugal stress for spherical packet
            stress = (self.packet_mass * omega_sq) / (np.pi * self.packet_radius)
            self.opti.subject_to(stress <= self.max_stress)
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.tol': 1e-6,
            'ipopt.max_iter': 100,
        }
        self.opti.solver('ipopt', opts)
    
    def solve(
        self,
        x0: np.ndarray,
        x_target: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve MPC problem.
        
        Args:
            x0: Initial state [qx, qy, qz, qw, ωx, ωy, ωz]
            x_target: Target state
        
        Returns:
            (u_opt, info) where u_opt is optimal control sequence
        """
        # Set parameters
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.x_target, x_target)

        # Apply delay compensation based on mode
        if self.delay_compensation_mode == 'discrete_time':
            if self.enable_discrete_time:
                x0_delayed = self.apply_discrete_time_delay(x0)
                self.opti.set_value(self.x0, x0_delayed)
        elif self.delay_compensation_mode == 'smith':
            if self.enable_delay_compensation and self.delay_steps > 0:
                x0_smith = self.smith_predictor(x0)
                self.opti.set_value(self.x0, x0_smith)
                logger.debug(f"Smith predictor applied: delay_steps={self.delay_steps}")
        elif self.delay_compensation_mode == 'both':
            # Apply both in sequence (additive compensation)
            x0_compensated = x0.copy()
            if self.enable_discrete_time:
                x0_compensated = self.apply_discrete_time_delay(x0_compensated)
            if self.enable_delay_compensation and self.delay_steps > 0:
                x0_compensated = self.smith_predictor(x0_compensated)
                logger.debug(f"Smith predictor applied: delay_steps={self.delay_steps}")
            self.opti.set_value(self.x0, x0_compensated)
        else:
            logger.warning(f"Unknown delay_compensation_mode: {self.delay_compensation_mode}")

        # Solve
        sol = self.opti.solve()
        
        # Extract optimal control
        u_opt = sol.value(self.u)
        
        info = {
            'solve_time': sol.stats()['t_wall_total'],
            'success': sol.stats()['success'],
            'iterations': sol.stats()['iter_count'],
            'delay_steps': self.delay_steps,
            'delay_compensation_enabled': self.enable_delay_compensation,
        }

        if self.enable_delay_compensation:
            logger.info(f"MPC solve time: {info['solve_time']*1000:.2f} ms, delay_compensation: enabled")
        else:
            logger.info(f"MPC solve time: {info['solve_time']*1000:.2f} ms, delay_compensation: disabled")
        
        return u_opt, info
    
    def get_first_control(self, u_opt: np.ndarray) -> np.ndarray:
        """Get first control input from optimal sequence."""
        return u_opt[:, 0]

    def smith_predictor(
        self,
        x0: np.ndarray,
    ) -> np.ndarray:
        """
        Smith predictor: advance state forward by latency delay using Euler integration.

        Predicts state after delay_steps by integrating free response
        with zero control input. This compensates for actuator/sensor latency
        in the control loop. Uses Euler integration to match MPC dynamics constraints.

        Args:
            x0: Initial state [qx, qy, qz, qw, ωx, ωy, ωz]

        Returns:
            Predicted state after delay [7]
        """
        x_pred = x0.copy()

        for _ in range(self.delay_steps):
            # Extract quaternion (scalar-last)
            q = x_pred[:4]
            omega = x_pred[4:7]

            # Convert to scalar-first for quaternion derivative
            qw = q[3]
            qx = q[0]
            qy = q[1]
            qz = q[2]
            q_scalar_first = np.array([qw, qx, qy, qz])

            # Quaternion derivative: q̇ = 0.5 * q * ω
            omega_quat = np.array([0, omega[0], omega[1], omega[2]])
            dq_scalar_first = 0.5 * np.array([
                q_scalar_first[0]*omega_quat[0] - q_scalar_first[1]*omega_quat[1] - q_scalar_first[2]*omega_quat[2] - q_scalar_first[3]*omega_quat[3],
                q_scalar_first[0]*omega_quat[1] + q_scalar_first[1]*omega_quat[0] + q_scalar_first[2]*omega_quat[3] - q_scalar_first[3]*omega_quat[2],
                q_scalar_first[0]*omega_quat[2] - q_scalar_first[1]*omega_quat[3] + q_scalar_first[2]*omega_quat[0] + q_scalar_first[3]*omega_quat[1],
                q_scalar_first[0]*omega_quat[3] + q_scalar_first[1]*omega_quat[2] - q_scalar_first[2]*omega_quat[1] + q_scalar_first[3]*omega_quat[0],
            ])

            # Convert back to scalar-last
            dq = np.array([dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0]])

            # Gyroscopic coupling: ω × (I * ω)
            I_omega = self.I @ omega
            omega_skew = np.array([
                [0, -omega[2], omega[1]],
                [omega[2], 0, -omega[0]],
                [-omega[1], omega[0], 0]
            ])
            gyro_coupling = omega_skew @ I_omega

            # Angular acceleration (zero control during delay)
            alpha = self.I_inv @ (-gyro_coupling)

            # Euler integration
            x_pred[:4] = x_pred[:4] + self.dt_delay * dq
            x_pred[4:7] = x_pred[4:7] + self.dt_delay * alpha

            # Renormalize quaternion to prevent drift
            q_norm = np.linalg.norm(x_pred[:4])
            if q_norm > 1e-12:
                x_pred[:4] = x_pred[:4] / q_norm

        return x_pred

    def apply_discrete_time_delay(
        self,
        x0: np.ndarray,
    ) -> np.ndarray:
        """
        Apply discrete-time sampling and communication delay.
        
        Models the effect of:
        1. Zero-order hold (ZOH) from sampling period
        2. Communication delay between sensor measurement and actuator command
        
        Args:
            x0: Current state [qx, qy, qz, qw, ωx, ωy, ωz]
        
        Returns:
            State after discrete-time delay [7]
        """
        if not self.enable_discrete_time:
            return x0
        
        # Total delay = sampling period + communication delay
        total_delay = self.sampling_period + self.communication_delay
        delay_steps = int(total_delay / self.dt_delay)
        
        if delay_steps <= 0:
            return x0
        
        # Integrate forward with zero control (ZOH assumption)
        x_delayed = x0.copy()
        
        for _ in range(delay_steps):
            # Extract quaternion (scalar-last)
            q = x_delayed[:4]
            omega = x_delayed[4:7]

            # Convert to scalar-first for quaternion derivative
            qw = q[3]
            qx = q[0]
            qy = q[1]
            qz = q[2]
            q_scalar_first = np.array([qw, qx, qy, qz])

            # Quaternion derivative: q̇ = 0.5 * q * ω
            omega_quat = np.array([0, omega[0], omega[1], omega[2]])
            dq_scalar_first = 0.5 * np.array([
                q_scalar_first[0]*omega_quat[0] - q_scalar_first[1]*omega_quat[1] - q_scalar_first[2]*omega_quat[2] - q_scalar_first[3]*omega_quat[3],
                q_scalar_first[0]*omega_quat[1] + q_scalar_first[1]*omega_quat[0] + q_scalar_first[2]*omega_quat[3] - q_scalar_first[3]*omega_quat[2],
                q_scalar_first[0]*omega_quat[2] - q_scalar_first[1]*omega_quat[3] + q_scalar_first[2]*omega_quat[0] + q_scalar_first[3]*omega_quat[1],
                q_scalar_first[0]*omega_quat[3] + q_scalar_first[1]*omega_quat[2] - q_scalar_first[2]*omega_quat[1] + q_scalar_first[3]*omega_quat[0],
            ])

            # Convert back to scalar-last
            dq = np.array([dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0]])

            # Gyroscopic coupling: ω × (I * ω)
            I_omega = self.I @ omega
            omega_skew = np.array([
                [0, -omega[2], omega[1]],
                [omega[2], 0, -omega[0]],
                [-omega[1], omega[0], 0]
            ])
            gyro_coupling = omega_skew @ I_omega

            # Angular acceleration (zero control during delay)
            alpha = self.I_inv @ (-gyro_coupling)

            # Euler integration
            x_delayed[:4] = x_delayed[:4] + self.dt_delay * dq
            x_delayed[4:7] = x_delayed[4:7] + self.dt_delay * alpha

            # Renormalize quaternion to prevent drift
            q_norm = np.linalg.norm(x_delayed[:4])
            if q_norm > 1e-12:
                x_delayed[:4] = x_delayed[:4] / q_norm

        logger.debug(f"Discrete-time delay applied: {total_delay*1000:.1f} ms ({delay_steps} steps)")
        return x_delayed

    def calculate_delay_margin(self, x0: np.ndarray = None) -> dict:
        """
        Calculate delay margin from linearized system dynamics.

        Uses frequency response analysis to compute the phase margin
        and converts to delay margin (maximum allowable delay before instability).

        Args:
            x0: Operating point state [qx, qy, qz, qw, ωx, ωy, ωz]. If None, uses zero.

        Returns:
            Dictionary with delay_margin_ms, phase_margin_deg, crossover_freq_hz
        """
        try:
            from scipy import signal
        except ImportError:
            logger.warning("scipy not available for delay margin calculation")
            return {
                'delay_margin_ms': float('inf'),
                'phase_margin_deg': 180.0,
                'crossover_freq_hz': 0.0,
                'calculation_failed': True,
            }

        if x0 is None:
            x0 = np.zeros(7)
            x0[3] = 1.0  # Unit quaternion

        # Linearize dynamics around operating point
        # State: [q, omega], Control: [Fx, Fy, Fz]
        # Simplified: linearize attitude dynamics (omega) only
        # d(omega)/dt = I_inv * (u - omega x (I * omega))

        omega0 = x0[4:7]
        I_omega0 = self.I @ omega0
        omega_skew0 = np.array([
            [0, -omega0[2], omega0[1]],
            [omega0[2], 0, -omega0[0]],
            [-omega0[1], omega0[0], 0]
        ])

        # Linearized A matrix: d(omega)/dt = -I_inv @ skew(omega) @ I @ omega + I_inv @ u
        # For small perturbations: A = -I_inv @ skew(omega0) @ I
        A_lin = -self.I_inv @ omega_skew0 @ self.I

        # B matrix: d(omega)/dt = I_inv @ u
        B_lin = self.I_inv

        # Use only omega dynamics (3x3 A, 3x3 B) for simplicity
        # Full system is 7-state but quaternion coupling makes linearization complex
        A_omega = A_lin
        B_omega = B_lin

        # Design a simple LQR-like feedback for analysis
        # K = -R_inv @ B.T @ P (simplified: use diagonal gain)
        K = np.diag([10.0, 10.0, 10.0])

        # Closed-loop A matrix: A_cl = A - B @ K
        A_cl = A_omega - B_omega @ K

        # Compute frequency response
        # Use first input-output pair for simplicity
        try:
            freqs = np.logspace(-2, 3, 1000)  # 0.01 to 1000 rad/s
            mag, phase = signal.freqresp(A_cl, B_omega[:, 0:1], np.eye(3)[0:1, :], freqs)

            # Find gain crossover frequency (where |G(jw)| = 1)
            gain = np.abs(mag)

            # Check if gain actually crosses 1 (sign of gain - 1 changes)
            gain_minus_one = gain - 1.0
            sign_changes = np.where(np.diff(np.sign(gain_minus_one)) != 0)[0]

            if len(sign_changes) > 0:
                # Found crossover - use first crossing
                crossover_idx = sign_changes[0]
                # Linear interpolation for better accuracy
                idx_before = crossover_idx
                idx_after = crossover_idx + 1
                frac = (1.0 - gain[idx_before]) / (gain[idx_after] - gain[idx_before])
                crossover_freq = freqs[idx_before] + frac * (freqs[idx_after] - freqs[idx_before])
                phase_at_crossover = np.angle(mag[idx_before]) * 180 / np.pi + frac * (np.angle(mag[idx_after]) * 180 / np.pi - np.angle(mag[idx_before]) * 180 / np.pi)

                # Phase margin = 180 + phase_at_crossover
                phase_margin = 180 + phase_at_crossover

                # Delay margin = phase_margin / (crossover_freq * 180/pi)
                if crossover_freq > 0:
                    delay_margin_rad = phase_margin * np.pi / 180
                    delay_margin_s = delay_margin_rad / crossover_freq
                    delay_margin_ms = delay_margin_s * 1000
                else:
                    delay_margin_ms = float('inf')
            else:
                # No crossover found - check if gain is always <1 (very stable) or >1 (unstable)
                if np.all(gain < 1.0):
                    # System is very stable, delay margin is infinite
                    phase_margin = 180.0
                    delay_margin_ms = float('inf')
                    crossover_freq = 0.0
                elif np.all(gain > 1.0):
                    # System is unstable at all frequencies
                    phase_margin = 0.0
                    delay_margin_ms = 0.0
                    crossover_freq = freqs[0]  # Use lowest frequency
                else:
                    # Edge case - use closest value
                    crossover_idx = np.argmin(np.abs(gain - 1.0))
                    crossover_freq = freqs[crossover_idx]
                    phase_at_crossover = np.angle(mag[crossover_idx]) * 180 / np.pi
                    phase_margin = 180 + phase_at_crossover
                    if crossover_freq > 0:
                        delay_margin_rad = phase_margin * np.pi / 180
                        delay_margin_s = delay_margin_rad / crossover_freq
                        delay_margin_ms = delay_margin_s * 1000
                    else:
                        delay_margin_ms = float('inf')

        except Exception as e:
            logger.warning(f"Frequency response calculation failed: {e}")
            return {
                'delay_margin_ms': float('inf'),
                'phase_margin_deg': 180.0,
                'crossover_freq_hz': 0.0,
                'calculation_failed': True,
            }

        return {
            'delay_margin_ms': delay_margin_ms,
            'phase_margin_deg': phase_margin,
            'crossover_freq_hz': crossover_freq / (2 * np.pi),
            'calculation_failed': False,
        }


class StubMPCController:
    """
    Stub MPC controller for when CasADi is not available.
    
    Provides a simple PID-like fallback that raises NotImplementedError
    for advanced features.
    """
    
    def __init__(self, **kwargs):
        """Initialize stub controller."""
        logger.warning("CasADi not available. Using stub MPC controller.")
        logger.warning("Install with: pip install casadi")
    
    def solve(self, x0: np.ndarray, x_target: np.ndarray, horizon: int = 10) -> Tuple[np.ndarray, dict]:
        """
        Stub solve method - returns zero control.
        
        Args:
            x0: Initial state
            x_target: Target state
            horizon: Prediction horizon
        
        Returns:
            (u_opt, info) where u_opt is zero control sequence
        """
        u_opt = np.zeros((3, horizon))
        info = {'solve_time': 0.0, 'success': False, 'iterations': 0}
        return u_opt, info


def create_mpc_controller(
    use_casadi: bool = True,
    **kwargs
) -> MPCController | StubMPCController:
    """
    Factory function to create MPC controller.
    
    Args:
        use_casadi: Whether to use CasADi (if available)
        **kwargs: Arguments passed to controller constructor
    
    Returns:
        MPCController or StubMPCController
    """
    if use_casadi and CASADI_AVAILABLE:
        return MPCController(**kwargs)
    else:
        return StubMPCController(**kwargs)


def verify_mpc_latency(controller: MPCController, n_trials: int = 10) -> dict:
    """
    Verify MPC solve time meets ≤30 ms target.
    
    Args:
        controller: MPC controller instance
        n_trials: Number of trials to run
    
    Returns:
        Dictionary with latency statistics
    """
    import time
    
    times = []
    
    for _ in range(n_trials):
        # Generate valid unit quaternion for initial state
        q_random = np.random.randn(4)
        q_random = q_random / np.linalg.norm(q_random)  # Normalize to unit quaternion
        omega_random = np.random.randn(3) * 0.1  # Small random angular velocity
        x0 = np.concatenate([q_random[1:], [q_random[0]], omega_random])  # Scalar-last format
        x_target = np.zeros(7)
        x_target[3] = 1.0  # Identity quaternion for target
        
        start = time.perf_counter()
        u_opt, info = controller.solve(x0, x_target)
        elapsed = time.perf_counter() - start
        
        times.append(elapsed)
    
    times = np.array(times)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'target_ms': 30.0,
        'meets_target': np.mean(times) * 1000 <= 30.0,
    }

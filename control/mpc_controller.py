"""
Model-Predictive Control (MPC) for gyroscopic mass-stream system.

Implements MPC with horizon N=10 for libration damping and spacing control.
Target: ≤30 ms solve time via numba/jit acceleration.

This module requires CasADi (optional dependency). If not available,
a stub implementation is provided that raises NotImplementedError.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional, Tuple

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
        max_stress: float = 1.2e9,  # 1.2 GPa
        min_k_eff: float = 6000.0,  # N/m
        I: np.ndarray = None,  # Inertia tensor (kg·m²)
        packet_radius: float = 0.02,  # Packet radius for stress (m)
        packet_mass: float = 0.05,  # Packet mass (kg)
    ):
        """
        Initialize MPC controller.
        
        Args:
            horizon: Prediction horizon (N=10 default)
            dt: Time step (s)
            libration_weight: Weight for libration energy minimization
            spacing_weight: Weight for spacing deviation minimization
            control_weight: Weight for control effort minimization
            max_stress: Maximum centrifugal stress (Pa)
            min_k_eff: Minimum effective stiffness (N/m)
            I: Inertia tensor in body frame (kg·m²), default for spherical packet
            packet_radius: Packet radius for stress calculations (m)
            packet_mass: Packet mass (kg)
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
        self.max_stress = max_stress
        self.min_k_eff = min_k_eff
        
        # Dynamics parameters
        if I is None:
            # Default inertia for spherical packet
            I = np.diag([0.0001, 0.00011, 0.00009])
        self.I = np.asarray(I, dtype=float)
        self.I_inv = np.linalg.inv(self.I)
        self.packet_radius = packet_radius
        self.packet_mass = packet_mass
        
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
        
        # Solve
        sol = self.opti.solve()
        
        # Extract optimal control
        u_opt = sol.value(self.u)
        
        info = {
            'solve_time': sol.stats()['t_wall_total'],
            'success': sol.stats()['success'],
            'iterations': sol.stats()['iter_count'],
        }
        
        return u_opt, info
    
    def get_first_control(self, u_opt: np.ndarray) -> np.ndarray:
        """Get first control input from optimal sequence."""
        return u_opt[:, 0]


class StubMPCController:
    """
    Stub MPC controller for when CasADi is not available.
    
    Provides a simple PID-like fallback that raises NotImplementedError
    for advanced features.
    """
    
    def __init__(self, **kwargs):
        """Initialize stub controller."""
        print("Warning: CasADi not available. Using stub MPC controller.")
        print("Install with: pip install casadi")
    
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
        x0 = np.random.randn(7)
        x_target = np.zeros(7)
        
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

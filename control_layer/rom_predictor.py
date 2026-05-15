"""
Reduced-Order Model (ROM) predictor using linearized Jacobian.

Implements a linearized ROM around an operating point using symbolic
Jacobian computation with sympy. This provides fast forward prediction
for Monte-Carlo stability analysis before adding the full VMD-IRCNN cascade.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


@dataclass
class ROMParameters:
    """Parameters for linearized ROM."""
    I: np.ndarray  # Inertia tensor
    mass: float
    operating_point: np.ndarray  # State around which to linearize [q, omega]


class LinearizedROM:
    """
    Linearized reduced-order model using Jacobian matrix.

    Linearizes the Euler dynamics around an operating point:
        δẋ ≈ A * δx + B * δu

    where A = ∂f/∂x|_x0 and B = ∂f/∂u|_x0 are Jacobians computed symbolically.
    """

    def __init__(self, params: ROMParameters, use_vmd_ircnn: bool = False):
        """
        Initialize linearized ROM.

        Args:
            params: ROMParameters object
            use_vmd_ircnn: Whether to use VMD-IRCNN for nonlinear prediction
        """
        if not SYMPY_AVAILABLE:
            raise ImportError("sympy is required for ROM. Install with: pip install sympy")

        self.params = params
        self.A: Optional[np.ndarray] = None  # State Jacobian
        self.B: Optional[np.ndarray] = None  # Input Jacobian
        self._I_inv = np.linalg.inv(self.params.I)  # Cache inverse
        self.use_vmd_ircnn = use_vmd_ircnn

        # Optional VMD-IRCNN components
        self.vmd_ircnn = None

        self._compute_jacobians()
    
    def _quaternion_kinematics(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Full quaternion derivative from angular velocity.
        
        Implements the full quaternion kinematics matrix:
        q̇ = 0.5 * Ω(ω) * q where Ω(ω) is the quaternion kinematics matrix
        
        This is the exact nonlinear formulation, valid for all rotations.
        No small-angle approximation.
        
        Args:
            q: Quaternion [qw, qx, qy, qz] (scalar-first convention)
            omega: Angular velocity [ωx, ωy, ωz] (rad/s)
        
        Returns:
            Quaternion derivative [dqw, dqx, dqy, dqz] (scalar-first)
        """
        # Quaternion kinematics matrix Ω(ω)
        # q̇ = 0.5 * Ω(ω) * q
        # where Ω(ω) = [[0, -ωx, -ωy, -ωz],
        #                 [ωx,   0, -ωz,  ωy],
        #                 [ωy,  ωz,   0, -ωx],
        #                 [ωz, -ωy,  ωx,   0]]
        
        Omega = 0.5 * np.array([
            [0,       -omega[0], -omega[1], -omega[2]],
            [omega[0],  0,       -omega[2],  omega[1]],
            [omega[1],  omega[2],  0,       -omega[0]],
            [omega[2], -omega[1],  omega[0],  0      ],
        ])
        
        dq = Omega @ q
        return dq
    
    def _compute_jacobians(self):
        """Compute Jacobian matrices symbolically using sympy."""
        # Define symbolic variables
        qx, qy, qz, qw = sp.symbols('qx qy qz qw')
        wx, wy, wz = sp.symbols('wx wy wz')
        Fx, Fy, Fz = sp.symbols('Fx Fy Fz')
        
        # State vector: [qx, qy, qz, qw, wx, wy, wz]
        # Control input: [Fx, Fy, Fz]
        
        # Full quaternion kinematics matrix: q̇ = 0.5 * Ω(ω) * q
        # This is the exact nonlinear formulation, valid for all rotations
        
        # Angular acceleration from Euler equations
        # ω̇ = I⁻¹ * (τ - ω × (I * ω))
        
        omega0 = self.params.operating_point[4:]
        
        # Gyroscopic coupling at operating point
        I_omega0 = self.params.I @ omega0
        gyro0 = np.cross(omega0, I_omega0)
        
        # Linearized gyroscopic coupling (simplified)
        domega = self._I_inv @ np.array([Fx, Fy, Fz]) - self._I_inv @ gyro0
        
        # Compute Jacobians numerically (simpler than full symbolic)
        # A = ∂f/∂x, B = ∂f/∂u
        self.A = self._compute_jacobian_fd()
        self.B = self._compute_jacobian_input()
    
    def _compute_jacobian_fd(self, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute state Jacobian using finite differences.
        
        Args:
            epsilon: Perturbation size
        
        Returns:
            7×7 Jacobian matrix A
        """
        def dynamics(state, control):
            """Full dynamics for Jacobian computation."""
            # Convert from scalar-last [qx, qy, qz, qw] to scalar-first [qw, qx, qy, qz]
            q_scalar_last = state[:4]
            q_scalar_first = np.array([q_scalar_last[3], q_scalar_last[0], q_scalar_last[1], q_scalar_last[2]])
            omega = state[4:]
            
            # Quaternion derivative (full kinematics matrix)
            dq_scalar_first = self._quaternion_kinematics(q_scalar_first, omega)
            # Convert back to scalar-last
            dq = np.array([dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0]])
            
            # Angular acceleration with full gyroscopic coupling
            I_omega = self.params.I @ omega
            gyro = np.cross(omega, I_omega)
            domega = self._I_inv @ (control - gyro)
            
            return np.concatenate([dq, domega])
        
        x0 = self.params.operating_point
        u0 = np.zeros(3)  # Zero control for state Jacobian
        
        n = len(x0)
        A = np.zeros((n, n))
        
        for i in range(n):
            x_plus = x0.copy()
            x_plus[i] += epsilon
            f_plus = dynamics(x_plus, u0)
            
            x_minus = x0.copy()
            x_minus[i] -= epsilon
            f_minus = dynamics(x_minus, u0)
            
            A[:, i] = (f_plus - f_minus) / (2 * epsilon)
        
        return A
    
    def _compute_jacobian_input(self, epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute input Jacobian using finite differences.
        
        Args:
            epsilon: Perturbation size
        
        Returns:
            7×3 Jacobian matrix B
        """
        def dynamics(state, control):
            """Full dynamics for Jacobian computation."""
            # Convert from scalar-last [qx, qy, qz, qw] to scalar-first [qw, qx, qy, qz]
            q_scalar_last = state[:4]
            q_scalar_first = np.array([q_scalar_last[3], q_scalar_last[0], q_scalar_last[1], q_scalar_last[2]])
            omega = state[4:]
            
            # Quaternion derivative (full kinematics matrix)
            dq_scalar_first = self._quaternion_kinematics(q_scalar_first, omega)
            # Convert back to scalar-last
            dq = np.array([dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0]])
            
            # Angular acceleration with full gyroscopic coupling
            I_omega = self.params.I @ omega
            gyro = np.cross(omega, I_omega)
            domega = self._I_inv @ (control - gyro)
            
            return np.concatenate([dq, domega])
        
        x0 = self.params.operating_point
        u0 = np.zeros(3)
        
        n_x = len(x0)
        n_u = len(u0)
        B = np.zeros((n_x, n_u))
        
        for i in range(n_u):
            u_plus = u0.copy()
            u_plus[i] += epsilon
            f_plus = dynamics(x0, u_plus)
            
            u_minus = u0.copy()
            u_minus[i] -= epsilon
            f_minus = dynamics(x0, u_minus)
            
            B[:, i] = (f_plus - f_minus) / (2 * epsilon)
        
        return B
    
    def set_vmd_ircnn(self, vmd_ircnn_model):
        """
        Set VMD-IRCNN model for nonlinear prediction.

        Args:
            vmd_ircnn_model: Trained VMD-IRCNN model
        """
        self.vmd_ircnn = vmd_ircnn_model
        self.use_vmd_ircnn = True

    def predict(
        self,
        delta_x: np.ndarray,
        delta_u: np.ndarray,
        dt: float = 0.01,
    ) -> np.ndarray:
        """
        Predict state change using linearized ROM or VMD-IRCNN.

        If use_vmd_ircnn is True and model is available, uses VMD-IRCNN with state conversion.
        Otherwise uses linearized ROM: δẋ ≈ A * δx + B * δu

        Args:
            delta_x: State deviation from operating point [δqx, δqy, δqz, δqw, δωx, δωy, δωz]
            delta_u: Control deviation from operating point [δFx, δFy, δFz]
            dt: Time step (s)

        Returns:
            Predicted state deviation at t+dt
        """
        if self.use_vmd_ircnn and self.vmd_ircnn is not None:
            # Use VMD-IRCNN with state conversion
            from .state_converter import StateConverter

            # Convert ROM state to VMD-IRCNN state
            x0_vmd = StateConverter.rom_to_vmd(delta_x)

            # Predict with VMD-IRCNN (model expects full state)
            pred_vmd = self._predict_vmd_ircnn(x0_vmd, delta_u, dt)

            # Convert back to ROM state
            return StateConverter.vmd_to_rom(pred_vmd)
        else:
            # Use linear ROM
            if self.A is None or self.B is None:
                raise RuntimeError("Jacobians not computed")

            delta_x_dot = self.A @ delta_x + self.B @ delta_u
            delta_x_next = delta_x + dt * delta_x_dot

            return delta_x_next

    def _predict_vmd_ircnn(self, x0_vmd: np.ndarray, delta_u: np.ndarray, dt: float) -> np.ndarray:
        """
        Predict using VMD-IRCNN model.

        Args:
            x0_vmd: VMD-IRCNN state [qw, qx, qy, qz, ωx, ωy, ωz]
            delta_u: Control deviation [δFx, δFy, δFz]
            dt: Time step (s)

        Returns:
            Predicted VMD-IRCNN state
        """
        # For now, use linear ROM as fallback
        # This would be replaced with actual VMD-IRCNN prediction when model is trained
        delta_x_rom = self._predict_linear(x0_vmd, delta_u, dt)
        return delta_x_rom

    def _predict_linear(self, x: np.ndarray, delta_u: np.ndarray, dt: float) -> np.ndarray:
        """
        Linear prediction (used as fallback for VMD-IRCNN).

        Args:
            x: State [7]
            delta_u: Control deviation [3]
            dt: Time step (s)

        Returns:
            Predicted state [7]
        """
        if self.A is None or self.B is None:
            raise RuntimeError("Jacobians not computed")

        x_dot = self.A @ x + self.B @ delta_u
        x_next = x + dt * x_dot

        return x_next
    
    def predict_trajectory(
        self,
        delta_x0: np.ndarray,
        delta_u_sequence: np.ndarray,
        dt: float = 0.01,
    ) -> np.ndarray:
        """
        Predict trajectory over control sequence.
        
        Args:
            delta_x0: Initial state deviation
            delta_u_sequence: Control deviations [n_steps × 3]
            dt: Time step (s)
        
        Returns:
            State deviations at each time step [n_steps+1 × 7]
        """
        n_steps = delta_u_sequence.shape[0]
        trajectory = np.zeros((n_steps + 1, 7))
        trajectory[0, :] = delta_x0
        
        delta_x = delta_x0.copy()
        for i in range(n_steps):
            delta_u = delta_u_sequence[i, :]
            delta_x = self.predict(delta_x, delta_u, dt)
            trajectory[i + 1, :] = delta_x
        
        return trajectory


def create_rom(
    mass: float,
    I: np.ndarray,
    operating_point: np.ndarray = None,
) -> LinearizedROM:
    """
    Factory function to create linearized ROM.
    
    Args:
        mass: Mass (kg)
        I: Inertia tensor (kg·m²)
        operating_point: State around which to linearize, default is identity quaternion + zero spin
    
    Returns:
        LinearizedROM instance
    """
    if operating_point is None:
        # Identity quaternion, zero angular velocity
        operating_point = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    
    params = ROMParameters(
        I=I,
        mass=mass,
        operating_point=operating_point,
    )
    
    return LinearizedROM(params)

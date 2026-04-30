"""
Full 3D rigid-body dynamics with explicit gyroscopic coupling.

Implements the corrected Euler rotational dynamics equation:

    I * ω̇ + ω × (I * ω) = τ_mag + τ_grav + τ_solar + τ_tether + τ_flux_pin

where ω × (I * ω) is the skew-symmetric gyroscopic coupling term that
produces precession and libration. Uses quaternion attitudes to avoid gimbal lock.

Conventions:
- Quaternion: scalar-last [x, y, z, w] for scipy Rotation compatibility
- Angular velocity: [ωx, ωy, ωz] in body frame (rad/s)
- State vector: [qx, qy, qz, qw, ωx, ωy, ωz] (7 elements)
- Inertia tensor: 3×3 in body frame (kg·m²)

Reference: Goldstein, Classical Mechanics (3rd ed.), Chapter 4
"""

from __future__ import annotations

import numpy as np
import warnings

from typing import TYPE_CHECKING, Callable, Optional

from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R

from .gyro_matrix import skew_symmetric

# Numba JIT for performance
try:
    from numba import jit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    # Fallback decorators
    def jit(nopython=True, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

if TYPE_CHECKING:
    from .bean_london_model import BeanLondonModel

# Optional flux-pinning model
try:
    from .bean_london_model import BeanLondonModel
    BEAN_LONDON_AVAILABLE = True
except ImportError:
    BEAN_LONDON_AVAILABLE = False
    BeanLondonModel = None


def geometry_profile_to_inertia(geometry_profile: dict | None) -> np.ndarray:
    """Convert geometry profile dict to inertia tensor.
    
    Args:
        geometry_profile: Geometry profile dict from catalog, or None for defaults
        
    Returns:
        3x3 inertia tensor (kg·m²)
        
    Note:
        This function is defined but not yet integrated into the simulation pipeline.
        TODO: Integrate geometry_profile_to_inertia into sgms_anchor_pipeline.py and
        sgms_anchor_v1.py to use geometry profiles for setting inertia tensors in simulations.
    """
    if geometry_profile is None:
        # Default: small sphere
        mass = 0.05
        radius = 0.02
        I_sphere = (2.0/5.0) * mass * radius**2
        return np.diag([I_sphere, I_sphere, I_sphere])
    
    shape = geometry_profile.get("shape", "sphere")
    mass = geometry_profile.get("mass", 0.05)
    radius = geometry_profile.get("radius", 0.02)
    
    if shape == "sphere":
        I_sphere = (2.0/5.0) * mass * radius**2
        return np.diag([I_sphere, I_sphere, I_sphere])
    elif shape == "prolate_spheroid":
        aspect_ratio = geometry_profile.get("aspect_ratio", 1.2)
        # For prolate spheroid: I_transverse = (1/5) * m * (a² + c²), I_axial = (2/5) * m * a²
        # where a = radius (equatorial), c = aspect_ratio * radius (polar)
        # IMPORTANT: The x-axis is the polar (axial) axis, y/z are transverse axes
        # This matches the inertia tensor ordering: diag([I_axial, I_transverse, I_transverse])
        a = radius
        c = aspect_ratio * radius
        I_transverse = (1.0/5.0) * mass * (a**2 + c**2)
        I_axial = (2.0/5.0) * mass * a**2
        return np.diag([I_axial, I_transverse, I_transverse])
    else:
        raise ValueError(f"Unknown shape type: {shape}")


def validate_quaternion(q: np.ndarray) -> np.ndarray:
    """Validate quaternion shape and return as array.

    Args:
        q: Quaternion to validate

    Returns:
        Validated quaternion as numpy array

    Raises:
        ValueError: If quaternion is not 4-element vector
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must be 4-element vector, got shape {q.shape}")
    return q

def validate_state_vector(state: np.ndarray) -> np.ndarray:
    """Validate state vector shape and return as array.

    State vector format: [qx, qy, qz, qw, ωx, ωy, ωz]

    Args:
        state: State vector to validate

    Returns:
        Validated state vector as numpy array

    Raises:
        ValueError: If state is not 7-element vector
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (7,):
        raise ValueError(f"State must be 7-element vector [qx,qy,qz,qw,ωx,ωy,ωz], got shape {state.shape}")
    return state

def scalar_last_to_first(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from scipy scalar-last [x, y, z, w] to scalar-first [w, x, y, z].
    Uses explicit indexing for clarity and minor performance gain in hot paths."""
    q = validate_quaternion(q)
    return np.take(q, [3, 0, 1, 2])  # [w,x,y,z] from [x,y,z,w]


def scalar_first_to_last(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from scalar-first [w, x, y, z] to scipy scalar-last [x, y, z, w].
    Uses explicit indexing for clarity and minor performance gain in hot paths."""
    q = validate_quaternion(q)
    return np.take(q, [1, 2, 3, 0])  # [x,y,z,w] from [w,x,y,z]


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute quaternion derivative from angular velocity.

    q̇ = 0.5 * q * ω  (quaternion multiplication)

    Args:
        q: Quaternion [w, x, y, z] (scalar-first convention, for internal computation)
        omega: Angular velocity vector [ωx, ωy, ωz] in body frame

    Returns:
        Quaternion derivative [q̇_w, q̇_x, q̇_y, q̇_z] (scalar-first)
    """
    # Quaternion multiplication: q̇ = 0.5 * q * omega_quat
    # where omega_quat = [0, ωx, ωy, ωz]
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    
    # Quaternion multiplication formula
    qw, qx, qy, qz = q
    ow, ox, oy, oz = omega_quat
    
    dq = 0.5 * np.array([
        qw*ow - qx*ox - qy*oy - qz*oz,
        qw*ox + qx*ow + qy*oz - qz*oy,
        qw*oy - qx*oz + qy*ow + qz*ox,
        qw*oz + qx*oy - qy*ox + qz*ow,
    ])
    
    return dq


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit magnitude.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Normalized quaternion
    """
    q = validate_quaternion(q)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError(
            f"Quaternion norm too small for normalization: norm={norm:.2e}, "
            f"quaternion={q}. This may indicate numerical integration instability. "
            f"Consider smaller timestep or higher-order integrator."
        )
    return q / norm


@jit(nopython=True)
def _skew_symmetric_numba(omega: np.ndarray) -> np.ndarray:
    """Numba-compiled skew-symmetric matrix."""
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])


@jit(nopython=True)
def _quaternion_derivative_numba(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Numba-compiled quaternion derivative (scalar-first convention)."""
    qw, qx, qy, qz = q
    ow, ox, oy, oz = 0.0, omega[0], omega[1], omega[2]
    dq = 0.5 * np.array([
        qw*ow - qx*ox - qy*oy - qz*oz,
        qw*ox + qx*ow + qy*oz - qz*oy,
        qw*oy - qx*oz + qy*ow + qz*ox,
        qw*oz + qx*oy - qy*ox + qz*ow,
    ])
    return dq


@jit(nopython=True)
def _rk4_step(rhs, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Single RK4 step (Numba-compiled)."""
    k1 = dt * rhs(t, y)
    k2 = dt * rhs(t + dt/2, y + k1/2)
    k3 = dt * rhs(t + dt/2, y + k2/2)
    k4 = dt * rhs(t + dt, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


@jit(nopython=True)
def _euler_equations_zero_torque_numba(
    t: float,
    state: np.ndarray,
    I: np.ndarray,
    I_inv: np.ndarray,
) -> np.ndarray:
    """Numba-compiled Euler equations with zero torque (no function callback)."""
    # Extract quaternion (scalar-last) and omega
    q_scipy = state[:4]
    omega = state[4:]

    # Convert to scalar-first
    q_scalar_first = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

    # Compute quaternion derivative
    dq_scalar_first = _quaternion_derivative_numba(q_scalar_first, omega)
    # Convert back to scalar-last
    dq_scipy = np.array([dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0]])

    # Compute gyroscopic coupling
    I_omega = I @ omega
    gyro_coupling = _skew_symmetric_numba(omega) @ I_omega

    # Solve for angular acceleration (zero torque)
    alpha = I_inv @ (-gyro_coupling)

    return np.concatenate([dq_scipy, alpha])


@jit(nopython=True)
def _euler_equations_numba(
    t: float,
    state: np.ndarray,
    I: np.ndarray,
    I_inv: np.ndarray,
    torque_vector: np.ndarray,
) -> np.ndarray:
    """Numba-compiled Euler equations (no torque function call)."""
    # Extract quaternion (scalar-last) and omega
    q_scipy = state[:4]
    omega = state[4:]
    
    # Convert to scalar-first
    q_scalar_first = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
    
    # Compute quaternion derivative
    dq_scalar_first = _quaternion_derivative_numba(q_scalar_first, omega)
    # Convert back to scalar-last
    dq_scipy = np.array([dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0]])
    
    # Compute gyroscopic coupling
    I_omega = I @ omega
    gyro_coupling = _skew_symmetric_numba(omega) @ I_omega
    
    # Solve for angular acceleration
    alpha = I_inv @ (torque_vector - gyro_coupling)
    
    return np.concatenate([dq_scipy, alpha])


def euler_equations(
    t: float,
    state: np.ndarray,
    I: np.ndarray,
    torques: Callable[[float, np.ndarray], np.ndarray],
    I_inv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Full Euler rotational dynamics with explicit gyroscopic coupling.
    
    State vector: [qx, qy, qz, qw, ωx, ωy, ωz] (quaternion + angular velocity)
    Note: scipy Rotation uses scalar-last, but we use scalar-first internally
    for derivative computation.
    
    Governing equation:
        I * ω̇ + ω × (I * ω) = τ
    
    Rearranged:
        ω̇ = I⁻¹ * (τ - ω × (I * ω))
    
    Args:
        t: Time
        state: State vector [qx, qy, qz, qw, ωx, ωy, ωz] (scalar-last for scipy)
        I: 3×3 inertia tensor in body frame (kg·m²)
        torques: Function torques(t, state) returning total torque vector [τx, τy, τz]
        I_inv: Optional precomputed inertia tensor inverse
    
    Returns:
        State derivative [q̇x, q̇y, q̇z, q̇w, ω̇x, ω̇y, ω̇z]
    """
    # Validate state shape
    state = validate_state_vector(state)
    
    # Extract quaternion (scalar-last for scipy compatibility)
    q_scipy = state[:4]  # [qx, qy, qz, qw]
    omega = state[4:]    # [ωx, ωy, ωz]
    
    # Convert to scalar-first for derivative computation
    q_scalar_first = scalar_last_to_first(q_scipy)
    
    # Compute quaternion derivative
    dq_scalar_first = quaternion_derivative(q_scalar_first, omega)
    # Convert back to scalar-last
    dq_scipy = scalar_first_to_last(dq_scalar_first)
    
    # Compute gyroscopic coupling: ω × (I * ω)
    I_omega = I @ omega
    gyro_coupling = skew_symmetric(omega) @ I_omega
    
    # Get total torque
    tau = np.asarray(torques(t, state), dtype=float)
    if tau.ndim == 0:
        raise ValueError("torques must return 3-element vector, got scalar")
    if tau.ndim != 1:
        raise ValueError(f"torques must return 1D array, got ndim={tau.ndim}")
    if tau.shape != (3,):
        raise ValueError(f"torques must return 3-element vector, got shape {tau.shape}")

    # Solve for angular acceleration: ω̇ = I⁻¹ * (τ - ω × (I * ω))
    if I_inv is None:
        I_inv = np.linalg.inv(I)
    alpha = I_inv @ (tau - gyro_coupling)
    
    return np.concatenate([dq_scipy, alpha])


class RigidBody:
    """
    3D rigid body with quaternion attitude and full Euler dynamics.

    State vector format: [qx, qy, qz, qw, ωx, ωy, ωz]
    - Quaternion: [qx, qy, qz, qw] (scalar-last for scipy compatibility)
    - Angular velocity: [ωx, ωy, ωz] (rad/s) in body frame

    Attributes:
        mass: Mass (kg)
        I: Inertia tensor in body frame (kg·m²)
        state: Current state [qx, qy, qz, qw, ωx, ωy, ωz]
    """
    
    def __init__(
        self,
        mass: float,
        I: np.ndarray,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        quaternion: np.ndarray = None,
        angular_velocity: np.ndarray = None,
        I_inv: np.ndarray = None,
        flux_model: Optional[BeanLondonModel] = None,
    ):
        """
        Initialize rigid body.

        Args:
            mass: Mass (kg)
            I: 3×3 inertia tensor in body frame (kg·m²)
            position: Initial position [x, y, z] (m), default [0, 0, 0]
            velocity: Initial velocity [vx, vy, vz] (m/s), default [0, 0, 0]
            quaternion: Initial quaternion [qx, qy, qz, qw] (scalar-last), default [0, 0, 0, 1]
            angular_velocity: Initial angular velocity [ωx, ωy, ωz] (rad/s), default [0, 0, 0]
            I_inv: Precomputed inertia tensor inverse (optional, for performance)
            flux_model: Optional BeanLondonModel for flux-pinning force calculation
        """
        self.mass = mass
        self._I = np.asarray(I, dtype=float)

        if self._I.shape != (3, 3):
            raise ValueError("Inertia tensor must be 3×3")

        # Lazy cache: compute inverse only if needed or provided
        if I_inv is not None:
            I_inv = np.asarray(I_inv, dtype=float)
            if I_inv.shape != (3, 3):
                raise ValueError("I_inv must be 3×3")
            # Verify it's actually the inverse (use both absolute and relative tolerance)
            if not np.allclose(self._I @ I_inv, np.eye(3), rtol=1e-9, atol=1e-10):
                raise ValueError("Provided I_inv is not the inverse of I")
            self._I_inv = I_inv
        else:
            self._I_inv = None

        # Flux-pinning model (optional)
        self.flux_model = flux_model

        # Translational state
        self.position = np.zeros(3) if position is None else np.asarray(position, dtype=float)
        self.velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float)

        # Rotational state
        if quaternion is None:
            self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # Identity rotation
        else:
            self.quaternion = np.asarray(quaternion, dtype=float)
            self.quaternion = normalize_quaternion(self.quaternion)

        self.angular_velocity = np.zeros(3) if angular_velocity is None else np.asarray(angular_velocity, dtype=float)

        # Combined state vector for integrator
        self.state = np.concatenate([
            self.quaternion,
            self.angular_velocity,
        ])
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get current rotation matrix from quaternion."""
        return R.from_quat(self.quaternion).as_matrix()

    def state_copy(self):
        """
        Create a copy of the rigid body state for latency injection.

        Returns a simple object with copies of all state attributes
        (position, velocity, quaternion, angular_velocity) to store
        in the latency buffer for delayed application.

        Returns:
            Simple object with copied state attributes
        """
        from types import SimpleNamespace
        return SimpleNamespace(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )

    @property
    def I(self) -> np.ndarray:
        """Inertia tensor in body frame (kg·m²)."""
        return self._I
    
    @property
    def angular_momentum(self) -> np.ndarray:
        """Compute angular momentum L = I * ω in body frame."""
        return self._I @ self.angular_velocity
    
    @property
    def rotational_energy(self) -> float:
        """Compute rotational kinetic energy E = 0.5 * ωᵀ * I * ω."""
        return 0.5 * self.angular_velocity @ (self.I @ self.angular_velocity)

    @property
    def I_inv(self) -> np.ndarray:
        """Lazy-computed inertia tensor inverse."""
        if self._I_inv is None:
            self._I_inv = np.linalg.inv(self.I)
        return self._I_inv
    
    def integrate_numba_rk4_zero_torque(
        self,
        t_span: tuple[float, float],
        dt: float = 0.01,
    ) -> dict:
        """
        Integrate using Numba-compiled RK4 with zero torque (no function callback).

        Args:
            t_span: (t_start, t_end) integration interval
            dt: Fixed timestep

        Returns:
            Dictionary with 't', 'state', 'sol'
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        # Precompute inertia inverse and inertia tensor
        I_inv = self.I_inv
        I_local = self.I

        # Storage for trajectory
        t_values = np.zeros(n_steps + 1)
        state_values = np.zeros((7, n_steps + 1))

        t_values[0] = t_start
        state_values[:, 0] = self.state

        # Integration loop with inlined RHS (no function callback)
        current_state = self.state.copy()
        current_t = t_start

        for i in range(n_steps):
            # Inline RK4 step with zero-torque Euler equations
            y = current_state
            t = current_t

            # k1
            q_scipy = y[:4]
            omega = y[4:]
            q_scalar_first = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
            dq_scalar_first = _quaternion_derivative_numba(q_scalar_first, omega)
            dq_scipy = np.array([dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0]])
            I_omega = I_local @ omega
            gyro_coupling = _skew_symmetric_numba(omega) @ I_omega
            alpha = I_inv @ (-gyro_coupling)
            k1 = dt * np.concatenate([dq_scipy, alpha])

            # k2
            y2 = y + k1/2
            q_scipy = y2[:4]
            omega = y2[4:]
            q_scalar_first = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
            dq_scalar_first = _quaternion_derivative_numba(q_scalar_first, omega)
            dq_scipy = np.array([dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0]])
            I_omega = I_local @ omega
            gyro_coupling = _skew_symmetric_numba(omega) @ I_omega
            alpha = I_inv @ (-gyro_coupling)
            k2 = dt * np.concatenate([dq_scipy, alpha])

            # k3
            y3 = y + k2/2
            q_scipy = y3[:4]
            omega = y3[4:]
            q_scalar_first = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
            dq_scalar_first = _quaternion_derivative_numba(q_scalar_first, omega)
            dq_scipy = np.array([dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0]])
            I_omega = I_local @ omega
            gyro_coupling = _skew_symmetric_numba(omega) @ I_omega
            alpha = I_inv @ (-gyro_coupling)
            k3 = dt * np.concatenate([dq_scipy, alpha])

            # k4
            y4 = y + k3
            q_scipy = y4[:4]
            omega = y4[4:]
            q_scalar_first = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])
            dq_scalar_first = _quaternion_derivative_numba(q_scalar_first, omega)
            dq_scipy = np.array([dq_scalar_first[1], dq_scalar_first[2], dq_scalar_first[3], dq_scalar_first[0]])
            I_omega = I_local @ omega
            gyro_coupling = _skew_symmetric_numba(omega) @ I_omega
            alpha = I_inv @ (-gyro_coupling)
            k4 = dt * np.concatenate([dq_scipy, alpha])

            # RK4 update
            current_state = y + (k1 + 2*k2 + 2*k3 + k4) / 6
            current_t += dt

            t_values[i + 1] = current_t
            state_values[:, i + 1] = current_state

        # Update final state
        self.state = current_state
        self.quaternion = self.state[:4]
        self.angular_velocity = self.state[4:]
        self.quaternion = normalize_quaternion(self.quaternion)

        return {
            "t": t_values,
            "state": state_values,
            "sol": None,  # No solve_ivp solution object
        }

    def integrate_numba_rk4(
        self,
        t_span: tuple[float, float],
        torques: Callable[[float, np.ndarray], np.ndarray],
        dt: float = 0.01,
    ) -> dict:
        """
        Integrate using Numba-compiled RK4 (much faster than solve_ivp).

        Args:
            t_span: (t_start, t_end) integration interval
            torques: Function torques(t, state) returning torque vector
            dt: Fixed timestep

        Returns:
            Dictionary with 't', 'state', 'sol'
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)

        # Precompute inertia inverse
        I_inv = self.I_inv

        # Storage for trajectory
        t_values = np.zeros(n_steps + 1)
        state_values = np.zeros((7, n_steps + 1))

        t_values[0] = t_start
        state_values[:, 0] = self.state

        # Define RHS for Numba
        def rhs(t, y):
            tau = torques(t, y)
            if tau.ndim == 0:
                tau = np.array([tau])
            return _euler_equations_numba(t, y, self.I, I_inv, tau)

        # Integration loop
        current_state = self.state.copy()
        current_t = t_start

        for i in range(n_steps):
            current_state = _rk4_step(rhs, current_t, current_state, dt)
            current_t += dt

            t_values[i + 1] = current_t
            state_values[:, i + 1] = current_state

        # Update final state
        self.state = current_state
        self.quaternion = self.state[:4]
        self.angular_velocity = self.state[4:]
        self.quaternion = normalize_quaternion(self.quaternion)

        return {
            "t": t_values,
            "state": state_values,
            "sol": None,  # No solve_ivp solution object
        }

    def integrate(
        self,
        t_span: tuple[float, float],
        torques: Callable[[float, np.ndarray], np.ndarray],
        method: str = "RK45",
        rtol: float = 1e-10,  # HIGH-FIDELITY: Tighter relative tolerance
        atol: float = 1e-12,  # HIGH-FIDELITY: Tighter absolute tolerance
        max_step: float = 0.01,  # HIGH-FIDELITY: Smaller max step (10ms not 250ms)
        dense_output: bool = True,
        use_numba_rk4: bool = True,
        use_zero_torque_numba: bool = False,
    ) -> dict:
        """
        Integrate rotational dynamics over time span.

        Args:
            t_span: (t_start, t_end) integration interval
            torques: Function torques(t, state) returning torque vector [τx, τy, τz]
            method: Integration method ("RK45", "LSODA", "Radau", etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_step: Maximum step size
            dense_output: Whether to return dense output for interpolation
            use_numba_rk4: Use Numba-compiled RK4 (much faster, fixed dt)
            use_zero_torque_numba: Use zero-torque Numba RK4 (fastest, no function callback)

        Returns:
            Dictionary with:
                't': Time points
                'state': State trajectories (shape: [7, n_points])
                'sol': Scipy solve_ivp solution object (None if use_numba_rk4)
        """
        if use_zero_torque_numba and _NUMBA_AVAILABLE:
            return self.integrate_numba_rk4_zero_torque(t_span, dt=max_step if max_step < 0.1 else 0.01)
        if use_numba_rk4 and _NUMBA_AVAILABLE:
            return self.integrate_numba_rk4(t_span, torques, dt=max_step if max_step < 0.1 else 0.01)
        def rhs(t, y):
            return euler_equations(t, y, self.I, torques, I_inv=self.I_inv)
        
        sol = solve_ivp(
            rhs,
            t_span,
            self.state,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=dense_output,
        )
        
        # Update final state
        self.state = sol.y[:, -1]
        self.quaternion = self.state[:4]
        self.angular_velocity = self.state[4:]
        
        # Renormalize quaternion to prevent drift
        self.quaternion = normalize_quaternion(self.quaternion)
        
        return {
            "t": sol.t,
            "state": sol.y,
            "sol": sol,
        }
    
    def set_inertia(self, I: np.ndarray, I_inv: Optional[np.ndarray] = None):
        """
        Set new inertia tensor and invalidate cache.

        Use this method instead of direct assignment to ensure cache consistency.

        Args:
            I: New 3×3 inertia tensor in body frame (kg·m²)
            I_inv: Optional precomputed inverse (for performance)
        """
        I = np.asarray(I, dtype=float)
        if I.shape != (3, 3):
            raise ValueError("Inertia tensor must be 3×3")

        if I_inv is not None:
            I_inv = np.asarray(I_inv, dtype=float)
            if I_inv.shape != (3, 3):
                raise ValueError("I_inv must be 3×3")
            if not np.allclose(I @ I_inv, np.eye(3), rtol=1e-9, atol=1e-10):
                raise ValueError("Provided I_inv is not the inverse of I")
            self._I_inv = I_inv
        else:
            self._I_inv = None  # Invalidate cache

        self._I = I
    
    def reset_state(
        self,
        quaternion: np.ndarray = None,
        angular_velocity: np.ndarray = None,
    ):
        """Reset rotational state."""
        if quaternion is not None:
            self.quaternion = normalize_quaternion(np.asarray(quaternion, dtype=float))
        if angular_velocity is not None:
            self.angular_velocity = np.asarray(angular_velocity, dtype=float)

        self.state = np.concatenate([self.quaternion, self.angular_velocity])

    def compute_flux_pinning_force(
        self,
        B_field: np.ndarray,
        superconductor_temp: float,
        displacement: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute 6-DoF flux-pinning force via Bean-London model.

        Returns force [3] and torque [3] as concatenated array [6].
        Force and torque are in body frame.

        Args:
            B_field: Magnetic field vector at position [Bx, By, Bz] (T)
            superconductor_temp: Superconductor temperature (K)
            displacement: Optional displacement vector [dx, dy, dz] (m).
                          If None, uses current position.

        Returns:
            6-DoF force/torque vector [Fx, Fy, Fz, τx, τy, τz] in body frame
        """
        if self.flux_model is None:
            return np.zeros(6)

        # Use current position if displacement not provided
        if displacement is None:
            displacement = self.position

        # Compute B-field magnitude
        B_mag = np.linalg.norm(B_field)

        # Compute pinning force for each axis using Bean-London model
        force = np.zeros(3)
        for i in range(3):
            force[i] = self.flux_model.compute_pinning_force(
                displacement[i], B_mag, superconductor_temp
            )

        # Compute torque from force (τ = r × F)
        # For simplified model, assume force acts at center of mass
        # Torque arises from angular displacement (libration)
        # Use stiffness to compute restoring torque
        # Extract angular displacement from quaternion using full rotation
        qw = self.quaternion[3]
        q_vector = self.quaternion[:3]
        qw = np.clip(qw, -1.0, 1.0)  # Numerical stability
        angle = 2.0 * np.arccos(qw)
        sin_half_angle = np.sin(angle / 2.0)
        if sin_half_angle > 1e-10:
            axis = q_vector / sin_half_angle
            angular_disp = axis * angle
        else:
            angular_disp = np.zeros(3)  # No rotation
        torque = np.zeros(3)
        for i in range(3):
            stiffness = self.flux_model.get_stiffness(
                angular_disp[i], B_mag, superconductor_temp
            )
            torque[i] = -stiffness * angular_disp[i]

        return np.concatenate([force, torque])

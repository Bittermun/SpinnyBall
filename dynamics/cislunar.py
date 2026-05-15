"""
Circular Restricted 3-Body Problem (CR3BP) propagator for cislunar dynamics.

Implements the CR3BP acceleration model for propagation of spacecraft and packet
streams in the Earth-Moon system. Provides both inertial (ECI) and rotating frame
integration options.

Features:
- CR3BP equations of motion in rotating and inertial frames
- Lagrange point locations (L1-L5)
- Frame transformations (rotating <-> inertial)
- Adaptive RK45 integration with event handling
- SPICE-driven ephemeris updates for high fidelity

References:
- Szebehely, "Theory of Orbits: The Restricted Problem of Three Bodies"
- Howell, "Families of Orbits in the Vicinity of the Collinear Libration Points"
- NASA JPL CR3BP formulations
"""

from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import numpy as np
from scipy.integrate import solve_ivp, RK45

try:
    from third_party.spice import SPICEWrapper, SPICEYPY_AVAILABLE
except ImportError:
    SPICEWrapper = None
    SPICEYPY_AVAILABLE = False


@dataclass
class CR3BPConfig:
    """
    Configuration for CR3BP propagator.
    
    Attributes:
        mu: Mass parameter (Moon mass / total mass) ≈ 0.01215 for Earth-Moon
        rotating_frame: If True, integrate in rotating frame; else inertial
        include_srp: Include Solar Radiation Pressure perturbation
        use_spice: Use SPICE for ephemeris (requires SPICEWrapper)
        adaptive_mu: If True, update mu(t) from SPICE ephemeris
    """
    mu: float = 0.01215  # Earth-Moon mass parameter
    rotating_frame: bool = False
    include_srp: bool = False
    use_spice: bool = False
    adaptive_mu: bool = False
    dt_max: float = 60.0  # seconds
    rtol: float = 1e-9
    atol: float = 1e-12


class CR3BPPropagator:
    """
    Circular Restricted 3-Body Problem propagator.
    
    Integrates spacecraft and packet streams in the Earth-Moon CR3BP.
    Supports both rotating and inertial frame formulations.
    
    Usage:
        config = CR3BPConfig(mu=0.01215, rotating_frame=False)
        prop = CR3BPPropagator(config)
        
        # Initial state: [x, y, z, vx, vy, vz] in km, km/s (inertial)
        state0 = np.array([6871.0, 0.0, 0.0, 0.0, 7.5, 0.0])
        
        t_eval = np.linspace(0, 86400*10, 10000)  # 10 days
        sol = prop.propagate(state0, t_eval)
        
        # Extract positions
        positions = sol.y[0:3, :].T  # (N_time, 3)
    """
    
    # Earth-Moon system parameters (from NAIF ephemerides)
    MU_EARTH = 398600.4418    # km³/s²
    MU_MOON = 4902.8005       # km³/s²
    EARTH_MOON_DISTANCE = 384400.0  # km (mean)
    
    def __init__(self, config: Optional[CR3BPConfig] = None):
        """
        Initialize CR3BP propagator.
        
        Args:
            config: CR3BPConfig instance
        """
        self.config = config or CR3BPConfig()
        self.mu = self.config.mu
        self.omega = 1.0  # Mean motion in rotating frame (normalized)
        
        self._spice_wrapper = None
        if self.config.use_spice and SPICEYPY_AVAILABLE:
            try:
                self._spice_wrapper = SPICEWrapper(auto_load_kernels=True, verbose=False)
            except Exception as e:
                print(f"Warning: Could not initialize SPICE wrapper: {e}")
                self._spice_wrapper = None
        
        # Cache for Lagrange points
        self._lagrange_points = {}
    
    def propagate(
        self,
        state0: np.ndarray,
        t_eval: np.ndarray,
        t0: float = 0.0,
        event_func: Optional[Callable] = None,
        dense_output: bool = False
    ) -> 'CR3BPSolution':
        """
        Propagate state using CR3BP dynamics.
        
        Args:
            state0: Initial state [x, y, z, vx, vy, vz] (km, km/s)
            t_eval: Time evaluation points (seconds from t0)
            t0: Initial time (for SPICE queries if enabled)
            event_func: Optional event function for termination conditions
            dense_output: If True, return dense solution object
        
        Returns:
            CR3BPSolution with interpolation capability
        """
        state0 = np.asarray(state0, dtype=float).flatten()
        if state0.shape != (6,):
            raise ValueError(f"state0 must be 6-element vector, got shape {state0.shape}")
        
        # Setup ODE integrator
        def dynamics(t, y):
            """Compute acceleration in CR3BP."""
            return self._accelerations(y, t + t0)
        
        # Integrate using RK45
        sol = solve_ivp(
            dynamics,
            t_span=(t_eval[0], t_eval[-1]),
            y0=state0,
            t_eval=t_eval,
            method='RK45',
            dense_output=dense_output,
            events=event_func,
            rtol=self.config.rtol,
            atol=self.config.atol,
            max_step=self.config.dt_max
        )
        
        return CR3BPSolution(sol, t0=t0, propagator=self)
    
    def _accelerations(self, state: np.ndarray, time_abs: float) -> np.ndarray:
        """
        Compute CR3BP accelerations.
        
        Args:
            state: State vector [x, y, z, vx, vy, vz] (km, km/s)
            time_abs: Absolute time (for SPICE ephemeris updates)
        
        Returns:
            State derivative [vx, vy, vz, ax, ay, az]
        """
        x, y, z, vx, vy, vz = state
        
        if self.config.rotating_frame:
            return self._accelerations_rotating(x, y, z, vx, vy, vz)
        else:
            return self._accelerations_inertial(x, y, z, vx, vy, vz, time_abs)
    
    def _accelerations_rotating(self, x, y, z, vx, vy, vz):
        """
        Accelerations in the rotating frame.
        
        Reference: Szebehely equations of motion.
        """
        mu = self.mu
        omega = self.omega
        
        # Distances to primary bodies
        r1 = np.sqrt((x + mu)**2 + y**2 + z**2)      # Distance to Moon
        r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)  # Distance to Earth
        
        # Gravitational accelerations
        ax_grav = (1 - mu) * (x - (1 - mu)) / r2**3 + mu * (x + mu) / r1**3
        ay_grav = (1 - mu) * y / r2**3 + mu * y / r1**3
        az_grav = (1 - mu) * z / r2**3 + mu * z / r1**3
        
        # Centrifugal and Coriolis accelerations (rotating frame)
        ax_cent = omega**2 * x
        ay_cent = omega**2 * y
        
        ax_cor = 2 * omega * vy
        ay_cor = -2 * omega * vx
        
        # Total acceleration
        ax = ax_grav + ax_cent + ax_cor
        ay = ay_grav + ay_cent + ay_cor
        az = az_grav
        
        return np.array([vx, vy, vz, ax, ay, az])
    
    def _accelerations_inertial(self, x, y, z, vx, vy, vz, time_abs):
        """
        Accelerations in the inertial ECI frame.
        
        This formulation uses SPICE or analytical ephemerides for body positions.
        """
        # For now, use simplified model with fixed body positions
        # TODO: Integrate SPICE for time-varying Moon position
        
        # Earth-Moon distance (km)
        r_em = self.EARTH_MOON_DISTANCE
        
        # Spacecraft position relative to Earth and Moon
        # Assume Earth at origin, Moon at (r_em, 0, 0)
        r_earth = np.array([x, y, z])
        r_moon = r_earth - np.array([r_em, 0, 0])
        
        r_e = np.linalg.norm(r_earth)
        r_m = np.linalg.norm(r_moon)
        
        # Accelerations due to Earth and Moon gravity
        if r_e > 1e-3:  # Avoid singularity
            a_earth = -self.MU_EARTH * r_earth / r_e**3
        else:
            a_earth = np.zeros(3)
        
        if r_m > 1e-3:
            a_moon = -self.MU_MOON * r_moon / r_m**3
        else:
            a_moon = np.zeros(3)
        
        a_total = a_earth + a_moon
        
        # Optional SRP (simplified)
        if self.config.include_srp:
            a_srp = self._compute_srp_acceleration(r_earth, time_abs)
            a_total += a_srp
        
        return np.array([vx, vy, vz, a_total[0], a_total[1], a_total[2]])
    
    def _compute_srp_acceleration(self, r_spacecraft: np.ndarray, time_abs: float) -> np.ndarray:
        """
        Compute Solar Radiation Pressure acceleration (simplified).
        
        Args:
            r_spacecraft: Spacecraft position in ECI (km)
            time_abs: Absolute time (for Sun direction)
        
        Returns:
            SRP acceleration (km/s²)
        """
        # Simplified: assume Sun at constant distance in +x direction
        # For high fidelity, query Sun position from SPICE
        sun_direction = np.array([1.0, 0.0, 0.0])  # Placeholder
        
        # SRP magnitude (typical for small satellite)
        srp_accel_mag = 1e-8  # km/s² (tunable)
        
        return srp_accel_mag * sun_direction
    
    def lagrange_point(self, point: int) -> np.ndarray:
        """
        Compute Lagrange point location in rotating frame.
        
        Args:
            point: Lagrange point number (1-5)
        
        Returns:
            Lagrange point position [x, y, 0] (normalized CR3BP units, km)
        """
        if point in self._lagrange_points:
            return self._lagrange_points[point]
        
        # L1, L2: On x-axis, between/beyond primaries
        # L3: Beyond primary body opposite to secondary
        # L4, L5: Triangular points
        
        # Numerical solution for L1
        if point == 1:
            # Approximate location (solve Jacobi equation)
            mu = self.mu
            alpha = ((1 - mu) / 3)**0.5
            x_l1 = 1 - alpha * (1 - alpha)  # Simplified approximation
            return np.array([x_l1, 0.0, 0.0])
        
        elif point == 2:
            mu = self.mu
            alpha = ((1 - mu) / 3)**0.5
            x_l2 = 1 + alpha * (1 - alpha)
            return np.array([x_l2, 0.0, 0.0])
        
        elif point == 3:
            return np.array([-7.0 / 12.0, 0.0, 0.0])  # Approximate
        
        elif point == 4:
            return np.array([0.5 - self.mu, np.sqrt(3) / 2, 0.0])
        
        elif point == 5:
            return np.array([0.5 - self.mu, -np.sqrt(3) / 2, 0.0])
        
        else:
            raise ValueError(f"Invalid Lagrange point: {point}")


@dataclass
class CR3BPSolution:
    """
    Solution object from CR3BP propagation.
    
    Wraps scipy.integrate.OdeSolution with additional CR3BP-specific methods.
    """
    sol: object  # scipy OdeSolution
    t0: float  # Initial absolute time
    propagator: 'CR3BPPropagator'  # Parent propagator
    
    def __getattr__(self, name):
        """Delegate attributes to wrapped sol object."""
        return getattr(self.sol, name)
    
    def get_position(self, t: np.ndarray) -> np.ndarray:
        """
        Get position at time(s) t.
        
        Args:
            t: Time(s) relative to t0
        
        Returns:
            Position array (3,) or (N, 3)
        """
        state = self.sol.sol(t)
        return state[0:3]
    
    def get_velocity(self, t: np.ndarray) -> np.ndarray:
        """
        Get velocity at time(s) t.
        
        Args:
            t: Time(s) relative to t0
        
        Returns:
            Velocity array (3,) or (N, 3)
        """
        state = self.sol.sol(t)
        return state[3:6]
    
    def get_distance_from_moon(self, t: np.ndarray) -> np.ndarray:
        """
        Compute spacecraft distance from Moon.
        
        Args:
            t: Time(s)
        
        Returns:
            Distance(s) (scalar or array)
        """
        pos = self.get_position(t)
        # Moon at (r_em, 0, 0) in ECI
        r_em = self.propagator.EARTH_MOON_DISTANCE
        moon_pos = np.array([r_em, 0.0, 0.0])
        
        if pos.ndim == 1:
            return np.linalg.norm(pos - moon_pos)
        else:
            return np.linalg.norm(pos - moon_pos, axis=1)

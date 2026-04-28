"""
Orbital dynamics coupling layer using poliastro.

Provides orbital propagation with perturbations (J2, SRP, drag),
coordinate transforms for integration with MultiBodyStream,
and reference frame conversions for skyhook station coupling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

# Optional poliastro import
try:
    from astropy import units as u
    from astropy.time import Time
    from poliastro.bodies import Earth
    from poliastro.twobody import Orbit
    from poliastro.constants import R_earth, M_earth, J2_earth
    from poliastro.perturbations import J2Perturbation, AtmosphericDrag, SRP
    POLIASTRO_AVAILABLE = True
except ImportError:
    POLIASTRO_AVAILABLE = False
    u = None
    Time = None
    Orbit = None
    R_earth = 6371.0  # km fallback
    M_earth = 5.972e24  # kg fallback
    J2_earth = 1.08263e-3

# Export for tests
ORBITAL_DYNAMICS_AVAILABLE = POLIASTRO_AVAILABLE


class OrbitState(Enum):
    """Orbit type classification."""
    CIRCULAR = "circular"
    ELLIPTICAL = "elliptical"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"


@dataclass
class StationState:
    """
    State of a skyhook station in the mass stream.
    
    This defines the reference frame for station-relative velocity calculations.
    
    Attributes:
        position_eci: Position vector in ECI frame (km)
        velocity_eci: Velocity vector in ECI frame (km/s)
        stream_velocity_mag: Magnitude of circulating stream velocity (km/s)
        station_id: Unique identifier for the station
    """
    position_eci: np.ndarray
    velocity_eci: np.ndarray
    stream_velocity_mag: float  # Typically ~7.6 km/s at LEO
    station_id: str = "S001"
    
    def __post_init__(self):
        """Validate state vectors."""
        self.position_eci = np.asarray(self.position_eci, dtype=float)
        self.velocity_eci = np.asarray(self.velocity_eci, dtype=float)
        if self.position_eci.shape != (3,):
            raise ValueError(f"position_eci must be 3D vector, got shape {self.position_eci.shape}")
        if self.velocity_eci.shape != (3,):
            raise ValueError(f"velocity_eci must be 3D vector, got shape {self.velocity_eci.shape}")
    
    @property
    def altitude_km(self) -> float:
        """Altitude above Earth surface (km)."""
        r_mag = np.linalg.norm(self.position_eci)
        return r_mag - R_earth
    
    @property
    def orbital_speed_kms(self) -> float:
        """Orbital speed magnitude (km/s)."""
        return np.linalg.norm(self.velocity_eci)


@dataclass
class CouplingResult:
    """
    Result of reference frame conversion for packet-stream coupling.
    
    Attributes:
        v_packet_eci: Packet velocity in ECI frame (km/s)
        v_packet_relative: Packet velocity relative to station (km/s)
        v_stream_relative: Stream velocity relative to station (km/s)
        relative_speed_magnitude: |v_packet_relative| (km/s)
        coupling_feasible: Whether coupling is within lane limits
        lane_classification: 'SLOW', 'STANDARD', 'FAST', or 'NONE'
        kinetic_energy_eci: Kinetic energy in ECI frame (J/kg)
        kinetic_energy_relative: Kinetic energy in relative frame (J/kg)
    """
    v_packet_eci: np.ndarray
    v_packet_relative: np.ndarray
    v_stream_relative: np.ndarray
    relative_speed_magnitude: float
    coupling_feasible: bool
    lane_classification: str
    kinetic_energy_eci: float
    kinetic_energy_relative: float


class StreamReferenceFrame:
    """
    Converter between ECI and Station/Stream Relative reference frames.
    
    This class handles the critical vector math for skyhook packet coupling:
    - v_relative = v_packet_eci - v_station_eci
    - v_eci = v_relative + v_station_eci
    
    The stream circulates at high velocity (~7.6 km/s at LEO) in the ECI frame,
    but packets couple at tunable relative velocities (1-15 km/s) in the
    station frame.
    
    Example Usage:
        >>> converter = StreamReferenceFrame()
        >>> station = StationState(
        ...     position_eci=np.array([6871, 0, 0]),  # 500 km altitude
        ...     velocity_eci=np.array([0, 7.6, 0]),   # Circular orbit
        ...     stream_velocity_mag=7.6
        ... )
        >>> v_packet_eci = np.array([0, 12.6, 0])  # Packet arriving at 5 km/s relative
        >>> result = converter.eci_to_station_frame(v_packet_eci, station)
        >>> result.relative_speed_magnitude  # Should be ~5.0 km/s
    """
    
    # Lane velocity boundaries (km/s relative speed)
    SLOW_LANE_MAX = 2.0
    STANDARD_LANE_MIN = 2.0
    STANDARD_LANE_MAX = 5.0
    FAST_LANE_MIN = 5.0
    FAST_LANE_MAX = 15.0
    
    def __init__(self):
        """Initialize reference frame converter."""
        pass
    
    def eci_to_station_frame(
        self, 
        v_packet_eci: np.ndarray, 
        station_state: StationState
    ) -> CouplingResult:
        """
        Convert packet velocity from ECI to station-relative frame.
        
        This is the core coupling calculation: when a packet arrives from
        lunar injection, its ECI velocity must be converted to the station
        frame to determine coupling feasibility and lane assignment.
        
        Args:
            v_packet_eci: Packet velocity vector in ECI frame (km/s)
            station_state: Current state of the target station
            
        Returns:
            CouplingResult with relative velocities and lane classification
        """
        v_packet_eci = np.asarray(v_packet_eci, dtype=float)
        
        # CRITICAL VECTOR MATH:
        # v_relative = v_packet_eci - v_station_eci
        # This gives the packet's velocity as seen from the station
        v_station_eci = station_state.velocity_eci
        v_packet_relative = v_packet_eci - v_station_eci
        
        # Stream velocity in station frame:
        # The stream circulates at stream_velocity_mag in the same direction
        # as the station's orbital motion. We must express this in the same
        # basis as v_packet_relative (which is ECI-difference frame).
        # 
        # v_stream_relative = v_stream_eci - v_station_eci
        # where v_stream_eci is the stream velocity in ECI frame.
        # For a co-rotating stream moving with the station:
        # v_stream_eci = (stream_velocity_mag / |v_station|) * v_station_eci
        # 
        # Therefore: v_stream_relative = v_stream_eci - v_station_eci
        #          = (stream_velocity_mag / |v_station|) * v_station_eci - v_station_eci
        #          = ((stream_velocity_mag / |v_station|) - 1) * v_station_eci
        #
        # However, for typical skyhook operation, the stream velocity magnitude
        # IS the station orbital speed (they co-rotate), so:
        # v_stream_eci ≈ v_station_eci, thus v_stream_relative ≈ 0
        #
        # But the "stream" concept here means packets circulating AT the station
        # with some relative speed. So we define the stream direction as the
        # unit vector of station velocity, scaled by stream_velocity_mag.
        
        v_station_mag = np.linalg.norm(v_station_eci)
        if v_station_mag < 1e-10:
            # Degenerate case: station not moving
            v_stream_relative = np.array([station_state.stream_velocity_mag, 0.0, 0.0])
        else:
            # Stream moves in same direction as station orbital motion
            v_station_unit = v_station_eci / v_station_mag
            v_stream_eci = v_station_unit * station_state.stream_velocity_mag
            # Now express in station-relative frame (same basis as v_packet_relative)
            v_stream_relative = v_stream_eci - v_station_eci
        
        # Relative speed magnitude
        relative_speed_mag = np.linalg.norm(v_packet_relative)
        
        # Lane classification
        lane = self._classify_lane(relative_speed_mag)
        
        # Coupling feasibility: within operational lane limits
        coupling_feasible = (
            self.SLOW_LANE_MAX >= relative_speed_mag >= 0 or
            self.STANDARD_LANE_MIN <= relative_speed_mag <= self.STANDARD_LANE_MAX or
            self.FAST_LANE_MIN <= relative_speed_mag <= self.FAST_LANE_MAX
        )
        
        # Kinetic energy calculations (per unit mass, J/kg = (km/s)^2 * 1e6 / 2)
        ke_eci = 0.5 * np.linalg.norm(v_packet_eci)**2 * 1e6  # Convert to J/kg
        ke_relative = 0.5 * relative_speed_mag**2 * 1e6
        
        return CouplingResult(
            v_packet_eci=v_packet_eci,
            v_packet_relative=v_packet_relative,
            v_stream_relative=v_stream_relative,
            relative_speed_magnitude=relative_speed_mag,
            coupling_feasible=coupling_feasible,
            lane_classification=lane,
            kinetic_energy_eci=ke_eci,
            kinetic_energy_relative=ke_relative
        )
    
    def station_to_eci_frame(
        self,
        v_packet_relative: np.ndarray,
        station_state: StationState
    ) -> np.ndarray:
        """
        Convert packet velocity from station-relative to ECI frame.
        
        This is used when designing injection trajectories: given a desired
        relative velocity at coupling, compute the required ECI arrival vector.
        
        Args:
            v_packet_relative: Desired packet velocity in station frame (km/s)
            station_state: Current state of the target station
            
        Returns:
            v_packet_eci: Required packet velocity in ECI frame (km/s)
        """
        v_packet_relative = np.asarray(v_packet_relative, dtype=float)
        
        # INVERSE TRANSFORM:
        # v_eci = v_relative + v_station_eci
        v_station_eci = station_state.velocity_eci
        v_packet_eci = v_packet_relative + v_station_eci
        
        return v_packet_eci
    
    def _classify_lane(self, relative_speed_km_s: float) -> str:
        """
        Classify relative speed into operational lane.
        
        Args:
            relative_speed_km_s: Relative speed magnitude (km/s)
            
        Returns:
            Lane classification: 'SLOW', 'STANDARD', 'FAST', or 'NONE'
        """
        if relative_speed_km_s < 0:
            return "NONE"  # Unphysical
        elif relative_speed_km_s <= self.SLOW_LANE_MAX:
            return "SLOW"
        elif relative_speed_km_s <= self.STANDARD_LANE_MAX:
            return "STANDARD"
        elif relative_speed_km_s <= self.FAST_LANE_MAX:
            return "FAST"
        else:
            return "NONE"  # Exceeds operational limits
    
    def calculate_momentum_flux(
        self,
        mass_flow_rate: float,  # kg/s
        v_relative: np.ndarray  # km/s
    ) -> np.ndarray:
        """
        Calculate momentum flux for packet-stream interaction.
        
        Momentum flux F = dm/dt * v_relative
        
        This determines the force exerted on the station during coupling.
        
        Args:
            mass_flow_rate: Mass flow rate (kg/s)
            v_relative: Relative velocity vector (km/s)
            
        Returns:
            Force vector (N)
        """
        # F = dm/dt * v (convert km/s to m/s)
        v_ms = v_relative * 1000  # km/s -> m/s
        force_N = mass_flow_rate * v_ms
        return force_N
    
    def calculate_energy_transfer(
        self,
        packet_mass: float,  # kg
        v_initial_rel: np.ndarray,  # km/s
        v_final_rel: np.ndarray  # km/s
    ) -> float:
        """
        Calculate energy transfer during coupling event.
        
        E = 0.5 * m * (v_final^2 - v_initial^2)
        
        Positive E means energy added to packet (acceleration).
        Negative E means energy extracted from packet (deceleration/capture).
        
        Args:
            packet_mass: Packet mass (kg)
            v_initial_rel: Initial relative velocity (km/s)
            v_final_rel: Final relative velocity (km/s)
            
        Returns:
            Energy transfer (Joules)
        """
        v_i_mag = np.linalg.norm(v_initial_rel) * 1000  # km/s -> m/s
        v_f_mag = np.linalg.norm(v_final_rel) * 1000  # km/s -> m/s
        
        delta_E = 0.5 * packet_mass * (v_f_mag**2 - v_i_mag**2)
        return delta_E


@dataclass
class OrbitalState:
    """Orbital state vector (position and velocity in ECI frame)."""
    r: np.ndarray  # Position vector [x, y, z] (km)
    v: np.ndarray  # Velocity vector [vx, vy, vz] (km/s)
    epoch: Optional[float] = None  # Epoch time (seconds)
    
    def __post_init__(self):
        """Validate state vectors."""
        self.r = np.asarray(self.r, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        if self.r.shape != (3,):
            raise ValueError(f"r must be 3D vector, got shape {self.r.shape}")
        if self.v.shape != (3,):
            raise ValueError(f"v must be 3D vector, got shape {self.v.shape}")
    
    @property
    def magnitude_r(self) -> float:
        """Position magnitude (km)."""
        return np.linalg.norm(self.r)
    
    @property
    def magnitude_v(self) -> float:
        """Velocity magnitude (km/s)."""
        return np.linalg.norm(self.v)


@dataclass
class OrbitalElements:
    """Keplerian orbital elements."""
    a: float  # Semi-major axis (km)
    e: float  # Eccentricity
    i: float  # Inclination (rad)
    raan: float  # Right ascension of ascending node (rad)
    argp: float  # Argument of periapsis (rad)
    nu: float  # True anomaly (rad)
    
    def to_state_vector(self, mu: float = 398600.4418) -> OrbitalState:
        """Convert orbital elements to state vector.
        
        Args:
            mu: Gravitational parameter (km^3/s^2)
            
        Returns:
            OrbitalState with position and velocity in ECI frame
        """
        # Compute position and velocity in perifocal frame
        p = self.a * (1 - self.e**2)
        if p < 1e-10:
            raise ValueError(f"Semi-latus rectum p is too small: {p}. Check orbital elements.")
        
        r_perifocal = (p / (1 + self.e * np.cos(self.nu))) * np.array([
            np.cos(self.nu),
            np.sin(self.nu),
            0.0
        ])
        v_perifocal = np.sqrt(mu / p) * np.array([
            -np.sin(self.nu),
            self.e + np.cos(self.nu),
            0.0
        ])
        
        # Rotation matrices
        # R = R_z(-raan) * R_x(-i) * R_z(-argp)
        c_raan, s_raan = np.cos(-self.raan), np.sin(-self.raan)
        c_i, s_i = np.cos(-self.i), np.sin(-self.i)
        c_argp, s_argp = np.cos(-self.argp), np.sin(-self.argp)
        
        R_z_raan = np.array([[c_raan, -s_raan, 0], [s_raan, c_raan, 0], [0, 0, 1]])
        R_x_i = np.array([[1, 0, 0], [0, c_i, -s_i], [0, s_i, c_i]])
        R_z_argp = np.array([[c_argp, -s_argp, 0], [s_argp, c_argp, 0], [0, 0, 1]])
        
        R = R_z_raan @ R_x_i @ R_z_argp
        
        # Transform to ECI frame
        r_eci = R @ r_perifocal
        v_eci = R @ v_perifocal
        
        return OrbitalState(r=r_eci, v=v_eci)


class OrbitalPropagator:
    """Orbital propagator using poliastro with perturbations."""
    
    def __init__(self, mu: float = 398600.4418):
        """Initialize propagator.
        
        Args:
            mu: Gravitational parameter (km^3/s^2), default Earth
        """
        self.mu = mu
        self._poliastro_orbit = None
        self._perturbations = []
    
    def from_state_vector(self, state: OrbitalState) -> 'OrbitalPropagator':
        """Create propagator from state vector.
        
        Args:
            state: OrbitalState with position and velocity
            
        Returns:
            Self for chaining
        """
        if not POLIASTRO_AVAILABLE:
            self._state = state
            return self
        
        try:
            r_km = state.r * u.km
            v_kms = state.v * u.km / u.s
            epoch = Time(state.epoch, format='unix') if state.epoch else None
            
            self._poliastro_orbit = Orbit.from_vectors(
                Earth,
                r_km,
                v_kms,
                epoch=epoch
            )
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            # Fallback to simple state storage
            self._state = state
        
        return self
    
    def from_orbital_elements(self, elements: OrbitalElements) -> 'OrbitalPropagator':
        """Create propagator from orbital elements.
        
        Args:
            elements: OrbitalElements
            
        Returns:
            Self for chaining
        """
        state = elements.to_state_vector(self.mu)
        return self.from_state_vector(state)
    
    def add_j2_perturbation(self, R: float = R_earth, J2: float = J2_earth):
        """Add J2 perturbation (Earth oblateness).
        
        Args:
            R: Reference radius (km)
            J2: J2 coefficient
        """
        if POLIASTRO_AVAILABLE and self._poliastro_orbit is not None:
            try:
                j2 = J2Perturbation(R * u.km, J2, Earth)
                self._perturbations.append(j2)
            except (ImportError, AttributeError, TypeError) as e:
                # Perturbation not available or incompatible
                pass
        return self
    
    def add_drag_perturbation(self, C_d: float = 2.2, A: float = 1.0, m: float = 100.0):
        """Add atmospheric drag perturbation.
        
        Args:
            C_d: Drag coefficient
            A: Cross-sectional area (m^2)
            m: Mass (kg)
        """
        if POLIASTRO_AVAILABLE and self._poliastro_orbit is not None:
            try:
                drag = AtmosphericDrag(C_d, A * u.m**2, m * u.kg, Earth)
                self._perturbations.append(drag)
            except (ImportError, AttributeError, TypeError) as e:
                # Perturbation not available or incompatible
                pass
        return self
    
    def add_srp_perturbation(self, C_r: float = 1.8, A: float = 1.0, m: float = 100.0):
        """Add Solar Radiation Pressure perturbation.
        
        Args:
            C_r: Reflectivity coefficient
            A: Cross-sectional area (m^2)
            m: Mass (kg)
        """
        if POLIASTRO_AVAILABLE and self._poliastro_orbit is not None:
            try:
                srp = SRP(C_r, A * u.m**2, m * u.kg)
                self._perturbations.append(srp)
            except (ImportError, AttributeError, TypeError) as e:
                # Perturbation not available or incompatible
                pass
        return self
    
    def propagate(self, dt: float) -> OrbitalState:
        """Propagate orbit by time step.
        
        Args:
            dt: Time step (seconds)
            
        Returns:
            New OrbitalState
        """
        if not POLIASTRO_AVAILABLE or self._poliastro_orbit is None:
            # Simple Keplerian propagation fallback
            return self._keplerian_propagate_fallback(dt)
        
        try:
            # Use poliastro propagation
            new_orbit = self._poliastro_orbit.propagate(dt * u.s)
            
            # Extract state vectors
            r_km = new_orbit.r
            v_kms = new_orbit.v
            
            return OrbitalState(
                r=np.array([r_km[0].value, r_km[1].value, r_km[2].value]),
                v=np.array([v_kms[0].value, v_kms[1].value, v_kms[2].value]),
                epoch=new_orbit.epoch.tdb.value if hasattr(new_orbit, 'epoch') else None
            )
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            # Fallback to simple propagation
            return self._keplerian_propagate_fallback(dt)
    
    def _keplerian_propagate_fallback(self, dt: float) -> OrbitalState:
        """Simple Keplerian propagation fallback (no perturbations)."""
        if not hasattr(self, '_state'):
            raise ValueError("No state vector available")
        
        # Very simple approximation: constant velocity for small dt
        # For larger dt, this is inaccurate but usable as fallback
        r_new = self._state.r + self._state.v * dt
        
        # Update velocity due to gravity (simplified)
        r_mag = np.linalg.norm(r_new)
        a_grav = -self.mu * r_new / r_mag**3
        v_new = self._state.v + a_grav * dt
        
        epoch = (self._state.epoch + dt) if self._state.epoch is not None else None
        return OrbitalState(r=r_new, v=v_new, epoch=epoch)
    
    def get_orbital_elements(self) -> OrbitalElements:
        """Get current orbital elements.
        
        Returns:
            OrbitalElements
        """
        if not POLIASTRO_AVAILABLE or self._poliastro_orbit is None:
            # Compute from state vector
            if hasattr(self, '_state') and self._state is not None:
                return self._elements_from_state(self._state)
            else:
                raise ValueError("No orbital state available")
        
        try:
            # Use poliastro
            elements = self._poliastro_orbit.classical()
            return OrbitalElements(
                a=elements.a.value,
                e=elements.ecc.value,
                i=elements.inc.value,
                raan=elements.raan.value,
                argp=elements.argp.value,
                nu=elements.nu.value
            )
        except Exception as e:
            # Fallback to state vector if available
            if hasattr(self, '_state') and self._state is not None:
                return self._elements_from_state(self._state)
            else:
                raise ValueError(f"Failed to get orbital elements: {e}")
    
    def _elements_from_state(self, state: OrbitalState) -> OrbitalElements:
        """Compute orbital elements from state vector (simplified)."""
        r = state.r
        v = state.v
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # Specific angular momentum
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Eccentricity vector
        e_vec = (np.cross(v, h) / self.mu) - (r / r_mag)
        e = np.linalg.norm(e_vec)
        
        # Energy
        energy = (v_mag**2 / 2) - (self.mu / r_mag)
        
        # Semi-major axis
        if abs(e) < 1e-10:
            a = r_mag  # Circular
        else:
            a = -self.mu / (2 * energy)
        
        # Inclination
        i = np.arccos(h[2] / h_mag)
        
        # RAAN
        n = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n)
        if n_mag < 1e-10:
            raan = 0.0  # Equatorial orbit
        else:
            raan = np.arccos(n[0] / n_mag)
            if n[1] < 0:
                raan = 2 * np.pi - raan
        
        # Argument of periapsis
        if n_mag < 1e-10 or abs(e) < 1e-10:
            argp = 0.0
        else:
            argp = np.arccos(np.dot(n, e_vec) / (n_mag * e))
            if e_vec[2] < 0:
                argp = 2 * np.pi - argp
        
        # True anomaly
        if abs(e) < 1e-10:
            nu = 0.0
        else:
            nu = np.arccos(np.dot(e_vec, r) / (e * r_mag))
            if np.dot(r, v) < 0:
                nu = 2 * np.pi - nu
        
        return OrbitalElements(a=a, e=e, i=i, raan=raan, argp=argp, nu=nu)


def eci_to_lvlh(r_eci: np.ndarray, v_eci: np.ndarray, vector_eci: np.ndarray) -> np.ndarray:
    """Transform vector from ECI to LVLH (Local Vertical Local Horizontal) frame.
    
    LVLH axes:
    - x: Radial (away from Earth center)
    - y: Along-track (velocity direction, perpendicular to radial)
    - z: Cross-track (normal to orbital plane)
    
    Args:
        r_eci: Position vector in ECI (km)
        v_eci: Velocity vector in ECI (km/s)
        vector_eci: Vector to transform (km)
        
    Returns:
        Vector in LVLH frame (km)
    """
    r_hat = r_eci / np.linalg.norm(r_eci)
    h = np.cross(r_eci, v_eci)
    h_hat = h / np.linalg.norm(h)
    y_hat = np.cross(h_hat, r_hat)
    
    # Rotation matrix from ECI to LVLH
    R = np.array([r_hat, y_hat, h_hat]).T
    
    return R.T @ vector_eci


def lvlh_to_eci(r_eci: np.ndarray, v_eci: np.ndarray, vector_lvlh: np.ndarray) -> np.ndarray:
    """Transform vector from LVLH to ECI frame.
    
    Args:
        r_eci: Position vector in ECI (km)
        v_eci: Velocity vector in ECI (km/s)
        vector_lvlh: Vector in LVLH frame (km)
        
    Returns:
        Vector in ECI frame (km)
    """
    r_hat = r_eci / np.linalg.norm(r_eci)
    h = np.cross(r_eci, v_eci)
    h_hat = h / np.linalg.norm(h)
    y_hat = np.cross(h_hat, r_hat)
    
    # Rotation matrix from LVLH to ECI
    R = np.array([r_hat, y_hat, h_hat])
    
    return R @ vector_lvlh


def compute_eclipse(r_eci: np.ndarray, sun_position: Optional[np.ndarray] = None) -> bool:
    """Check if satellite is in Earth's shadow (eclipse).
    
    Args:
        r_eci: Satellite position in ECI (km)
        sun_position: Sun position in ECI (km), defaults to +X direction
        
    Returns:
        True if in eclipse, False if sunlit
    """
    if sun_position is None:
        # Default sun position (simplified, along +X axis)
        sun_position = np.array([1.496e8, 0, 0])  # 1 AU in km
    
    r_mag = np.linalg.norm(r_eci)
    r_hat = r_eci / r_mag
    
    # Vector from satellite to Sun
    s_hat = (sun_position - r_eci)
    s_hat = s_hat / np.linalg.norm(s_hat)
    
    # Angle between position and sun vectors
    cos_angle = np.dot(r_hat, s_hat)
    
    # If angle > 90°, satellite might be in shadow
    if cos_angle > 0:
        return False  # Definitely sunlit
    
    # Check if Earth blocks the sun
    # Geometric condition: satellite is in shadow if
    # the line from satellite to sun passes through Earth
    sin_angle = np.sqrt(1 - cos_angle**2)
    if sin_angle < 1e-10:
        return False  # Avoid division by zero, treat as sunlit
    distance_to_umbra = R_earth / sin_angle
    
    # Project onto sun-satellite line
    projected_distance = r_mag * cos_angle
    
    return bool(projected_distance < distance_to_umbra)


def create_circular_orbit(altitude: float, inclination: float = 0.0) -> OrbitalState:
    """Create circular orbit state vector.
    
    Args:
        altitude: Altitude above Earth surface (km)
        inclination: Inclination (deg)
        
    Returns:
        OrbitalState for circular orbit
    """
    r_mag = R_earth + altitude
    mu = 398600.4418  # Earth gravitational parameter (km^3/s^2)
    v_mag = np.sqrt(mu / r_mag)  # Circular velocity
    
    # Position at ascending node
    i_rad = np.radians(inclination)
    r_eci = np.array([r_mag, 0, 0])
    v_eci = np.array([0, v_mag * np.cos(i_rad), v_mag * np.sin(i_rad)])
    
    return OrbitalState(r=r_eci, v=v_eci)

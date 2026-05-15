"""
Shepherd station model for Halbach mass stream guidance.

Shepherd stations interact with the stream passively via Halbach quadrupole
magnetic lenses (for centering) and small active copper trim coils (for
occasional speed/steering corrections).

Physics:
- Halbach quadrupole lens: F_perp ≈ -k_lens * r_perp
- Trim coils: Small velocity/steering corrections
- Passive centering: Restoring force proportional to displacement

References:
- Halbach (1980) - Design of permanent multipole magnets
- Yonnet (1981) - Permanent magnet bearings
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .halbach_array import MU_0


class StationType(Enum):
    """Type of shepherd station."""
    PASSIVE = "passive"  # Passive quadrupole lens only
    ACTIVE = "active"    # Quadrupole + active trim coils
    ANCHOR = "anchor"    # Full anchor station with deflection capability


@dataclass
class QuadrupoleLens:
    """Halbach quadrupole lens configuration.
    
    A quadrupole lens provides linear restoring force in the transverse
    plane while being transparent to motion along the beam axis.
    
    Attributes:
        length: Length of quadrupole (m)
        bore_radius: Bore radius (m)
        gradient: Field gradient at bore (T/m)
        max_field: Maximum field at pole tip (T)
    """
    length: float = 0.5  # m
    bore_radius: float = 0.1  # m
    gradient: float = 10.0  # T/m
    max_field: float = 1.5  # T
    
    @property
    def lens_stiffness(self) -> float:
        """Effective lens stiffness for magnetic dipole.
        
        For a dipole m in field gradient G:
            F = m * G
            k_lens = m * G / r (for linear approximation)
        
        Returns:
            Lens stiffness (N/m)
        """
        # Typical dipole moment for Halbach sphere (5cm radius, NdFeB)
        # m ≈ 580 A·m²
        m_typical = 580.0  # A·m²
        
        # Stiffness ≈ m * G / r_bore
        if self.bore_radius <= 0:
            return 0.0
        
        return m_typical * self.gradient / self.bore_radius


@dataclass
class TrimCoil:
    """Active trim coil configuration for velocity/steering corrections.
    
    Trim coils provide small, precise adjustments to ball velocity
    and trajectory. They are used sparingly to correct accumulated errors.
    
    Attributes:
        n_turns: Number of turns
        radius: Coil radius (m)
        max_current: Maximum current (A)
        resistance: Coil resistance (Ohm)
        inductance: Coil inductance (H)
    """
    n_turns: int = 100
    radius: float = 0.15  # m
    max_current: float = 10.0  # A
    resistance: float = 1.0  # Ohm
    inductance: float = 1e-3  # H
    
    def compute_magnetic_field(
        self,
        current: float,
        axial_position: float
    ) -> float:
        """Compute axial magnetic field from trim coil.
        
        For a circular coil, the axial field is:
            B_z = (mu0 * n * I * R^2) / (2 * (R^2 + z^2)^(3/2))
        
        Args:
            current: Coil current (A)
            axial_position: Distance along coil axis (m)
        
        Returns:
            Axial magnetic field (T)
        """
        if current == 0:
            return 0.0
        
        R = self.radius
        z = axial_position
        
        numerator = MU_0 * self.n_turns * current * R**2
        denominator = 2.0 * (R**2 + z**2)**(3.0/2.0)
        
        if denominator <= 0:
            return 0.0
        
        return numerator / denominator
    
    def compute_force_on_dipole(
        self,
        dipole_moment: float,
        current: float,
        axial_velocity: float
    ) -> float:
        """Compute force on a moving dipole from trim coil.
        
        F = ∇(m · B) ≈ m * dB/dz
        
        Args:
            dipole_moment: Magnetic dipole moment (A·m²)
            current: Coil current (A)
            axial_velocity: Velocity along coil axis (m/s)
        
        Returns:
            Force on dipole (N)
        """
        # Approximate field gradient near coil center
        # dB/dz ≈ -3*mu0*n*I*R^2*z / (2*(R^2+z^2)^(5/2))
        # At z = R/2 (typical interaction region):
        z = self.radius / 2.0
        R = self.radius
        
        numerator = -3.0 * MU_0 * self.n_turns * current * R**2 * z
        denominator = 2.0 * (R**2 + z**2)**(5.0/2.0)
        
        if denominator == 0:
            return 0.0
        
        dB_dz = numerator / denominator
        
        # Force = m * dB/dz
        return dipole_moment * dB_dz
    
    @property
    def power_consumption(self) -> float:
        """Power consumption at max current (W)."""
        return self.resistance * self.max_current**2


@dataclass
class ShepherdConfig:
    """Configuration for shepherd station.
    
    Attributes:
        station_type: Type of station (passive/active/anchor)
        position: Station position [x, y, z] (m)
        capture_radius: Radius for ball capture (m)
        quadrupole: Quadrupole lens configuration
        trim_coil: Trim coil configuration (for active stations)
        max_deflection_angle: Maximum deflection angle for anchor stations (rad)
    """
    station_type: StationType = StationType.PASSIVE
    position: np.ndarray = None
    capture_radius: float = 10.0  # m
    quadrupole: QuadrupoleLens = None
    trim_coil: TrimCoil = None
    max_deflection_angle: float = np.pi / 6.0  # 30 degrees
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        else:
            self.position = np.asarray(self.position, dtype=float)
        
        if self.quadrupole is None:
            self.quadrupole = QuadrupoleLens()
        
        if self.trim_coil is None and self.station_type in (StationType.ACTIVE, StationType.ANCHOR):
            self.trim_coil = TrimCoil()


class ShepherdStation:
    """Shepherd station for Halbach mass stream guidance.
    
    Shepherd stations provide:
    1. Passive centering via Halbach quadrupole lens
    2. Active trim corrections via copper coils (optional)
    3. Anchor force generation via stream deflection (anchor stations)
    
    Attributes:
        config: ShepherdConfig with station parameters
        station_id: Unique identifier
    """
    
    def __init__(
        self,
        config: ShepherdConfig,
        station_id: int = 0
    ):
        """Initialize shepherd station.
        
        Args:
            config: ShepherdConfig with station parameters
            station_id: Unique identifier for this station
        """
        self.config = config
        self.station_id = station_id
        self._trim_current = 0.0  # Current trim coil current
    
    @property
    def position(self) -> np.ndarray:
        """Station position [x, y, z] (m)."""
        return self.config.position.copy()
    
    @property
    def lens_stiffness(self) -> float:
        """Quadrupole lens stiffness (N/m)."""
        return self.config.quadrupole.lens_stiffness
    
    def compute_passive_force(
        self,
        ball_position: np.ndarray,
        ball_velocity: np.ndarray,
        dipole_moment: float = 580.0
    ) -> np.ndarray:
        """Compute passive centering force from quadrupole lens.
        
        The quadrupole provides a linear restoring force in the transverse
        plane that tends to center the ball in the bore.
        
        F_perp = -k_lens * r_perp
        
        Args:
            ball_position: Ball position [x, y, z] (m)
            ball_velocity: Ball velocity [vx, vy, vz] (m/s)
            dipole_moment: Ball dipole moment (A·m²)
        
        Returns:
            Force vector [Fx, Fy, Fz] (N)
        """
        r_vec = np.asarray(ball_position) - self.config.position
        
        # Distance from station center
        r_perp = np.linalg.norm(r_vec[:2])  # Transverse plane (x-y)
        
        if r_perp > self.config.capture_radius:
            # Ball outside capture radius - no force
            return np.zeros(3)
        
        # Quadrupole lens force (linear restoring in transverse plane)
        k_lens = self.lens_stiffness
        
        # Force is toward station axis
        if r_perp > 1e-12:
            direction = -r_vec[:2] / r_perp  # Toward center
        else:
            direction = np.zeros(2)
        
        F_transverse = k_lens * r_perp * direction
        
        # No axial force from quadrupole (to first order)
        return np.array([F_transverse[0], F_transverse[1], 0.0])
    
    def compute_trim_force(
        self,
        ball_position: np.ndarray,
        ball_velocity: np.ndarray,
        dipole_moment: float = 580.0
    ) -> np.ndarray:
        """Compute active trim force from coil.
        
        Args:
            ball_position: Ball position [x, y, z] (m)
            ball_velocity: Ball velocity [vx, vy, vz] (m/s)
            dipole_moment: Ball dipole moment (A·m²)
        
        Returns:
            Force vector [Fx, Fy, Fz] (N)
        """
        if self.config.trim_coil is None or self._trim_current == 0:
            return np.zeros(3)
        
        # Compute axial force from trim coil
        F_axial = self.config.trim_coil.compute_force_on_dipole(
            dipole_moment, self._trim_current, ball_velocity[2]
        )
        
        # Force is primarily along the station axis (z)
        return np.array([0.0, 0.0, F_axial])
    
    def compute_total_force(
        self,
        ball_position: np.ndarray,
        ball_velocity: np.ndarray,
        dipole_moment: float = 580.0
    ) -> np.ndarray:
        """Compute total force from shepherd station.
        
        Args:
            ball_position: Ball position [x, y, z] (m)
            ball_velocity: Ball velocity [vx, vy, vz] (m/s)
            dipole_moment: Ball dipole moment (A·m²)
        
        Returns:
            Total force vector [Fx, Fy, Fz] (N)
        """
        F_passive = self.compute_passive_force(ball_position, ball_velocity, dipole_moment)
        F_trim = self.compute_trim_force(ball_position, ball_velocity, dipole_moment)
        
        return F_passive + F_trim
    
    def set_trim_current(self, current: float):
        """Set trim coil current.
        
        Args:
            current: Trim coil current (A), clamped to max_current
        """
        if self.config.trim_coil is not None:
            max_I = self.config.trim_coil.max_current
            self._trim_current = np.clip(current, -max_I, max_I)
        else:
            self._trim_current = 0.0
    
    def compute_anchor_force(
        self,
        stream_tension: float,
        deflection_angle: float
    ) -> float:
        """Compute anchor force from stream deflection.
        
        For an anchor station that deflects the stream by angle θ:
            F_anchor = T * sin(θ)
        
        where T is the stream hoop tension.
        
        Args:
            stream_tension: Stream hoop tension T = λ*u² (N)
            deflection_angle: Deflection angle θ (rad)
        
        Returns:
            Anchor force (N)
        """
        if self.config.station_type != StationType.ANCHOR:
            return 0.0
        
        # Clamp to max deflection angle
        theta = np.clip(deflection_angle, -self.config.max_deflection_angle, 
                       self.config.max_deflection_angle)
        
        return stream_tension * np.sin(theta)
    
    def can_capture(self, ball_position: np.ndarray) -> bool:
        """Check if ball is within capture radius.
        
        Args:
            ball_position: Ball position [x, y, z] (m)
        
        Returns:
            True if ball can be captured
        """
        r_vec = np.asarray(ball_position) - self.config.position
        distance = np.linalg.norm(r_vec)
        return distance <= self.config.capture_radius
    
    def get_status(self) -> dict:
        """Get station status.
        
        Returns:
            Dictionary with station status
        """
        return {
            'station_id': self.station_id,
            'station_type': self.config.station_type.value,
            'position': self.position.tolist(),
            'lens_stiffness': self.lens_stiffness,
            'trim_current': self._trim_current,
            'capture_radius': self.config.capture_radius,
        }


def create_passive_shepherd(
    position: np.ndarray,
    station_id: int = 0,
    capture_radius: float = 10.0
) -> ShepherdStation:
    """Create a passive shepherd station.
    
    Args:
        position: Station position [x, y, z] (m)
        station_id: Unique identifier
        capture_radius: Capture radius (m)
    
    Returns:
        Configured ShepherdStation
    """
    config = ShepherdConfig(
        station_type=StationType.PASSIVE,
        position=position,
        capture_radius=capture_radius
    )
    return ShepherdStation(config, station_id)


def create_active_shepherd(
    position: np.ndarray,
    station_id: int = 0,
    capture_radius: float = 10.0
) -> ShepherdStation:
    """Create an active shepherd station with trim coils.
    
    Args:
        position: Station position [x, y, z] (m)
        station_id: Unique identifier
        capture_radius: Capture radius (m)
    
    Returns:
        Configured ShepherdStation
    """
    config = ShepherdConfig(
        station_type=StationType.ACTIVE,
        position=position,
        capture_radius=capture_radius
    )
    return ShepherdStation(config, station_id)


def create_anchor_station(
    position: np.ndarray,
    station_id: int = 0,
    capture_radius: float = 10.0,
    max_deflection: float = 30.0  # degrees
) -> ShepherdStation:
    """Create an anchor station with deflection capability.
    
    Args:
        position: Station position [x, y, z] (m)
        station_id: Unique identifier
        capture_radius: Capture radius (m)
        max_deflection: Maximum deflection angle (degrees)
    
    Returns:
        Configured ShepherdStation
    """
    config = ShepherdConfig(
        station_type=StationType.ANCHOR,
        position=position,
        capture_radius=capture_radius,
        max_deflection_angle=np.radians(max_deflection)
    )
    return ShepherdStation(config, station_id)

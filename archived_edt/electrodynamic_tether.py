"""
Electrodynamic tether (EDT) dynamics with IGRF magnetic field.

Implements Lorentz force generation, motional EMF calculation, and basic
libration dynamics for electrodynamic tethers in planetary magnetic fields.

Phase 4A: Simplified dipole magnetic field
Phase 4B: IGRF integration (optional with dipole fallback)
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Try to import ppigrf for IGRF magnetic field
try:
    import ppigrf
    _IGRF_AVAILABLE = True
    logger.info("ppigrf available for IGRF magnetic field")
except ImportError:
    _IGRF_AVAILABLE = False
    logger.warning("ppigrf not available, using dipole magnetic field")


@dataclass
class TetherState:
    """State of electrodynamic tether."""
    position: np.ndarray  # [x, y, z] in ECI (m)
    velocity: np.ndarray  # [vx, vy, vz] in ECI (m/s)
    libration_angle: float  # In-plane libration angle (rad)
    libration_rate: float  # In-plane libration rate (rad/s)
    out_of_plane_angle: float  # Out-of-plane libration angle (rad)
    out_of_plane_rate: float  # Out-of-plane libration rate (rad/s)


class ElectrodynamicTether:
    """
    Electrodynamic tether with simplified dipole magnetic field.

    Implements Lorentz force generation and motional EMF calculation
    using a dipole magnetic field model (simplified from IGRF).
    """

    def __init__(
        self,
        length: float = 5000.0,  # m
        diameter: float = 0.001,  # m (1mm wire)
        conductivity: float = 5.8e7,  # S/m (copper)
        mass_per_length: float = 0.01,  # kg/m
        orbit_altitude: float = 400.0,  # km
        orbit_inclination: float = 51.6,  # degrees (ISS)
    ):
        """
        Initialize electrodynamic tether.

        Args:
            length: Tether length (m)
            diameter: Tether diameter (m)
            conductivity: Tether conductivity (S/m)
            mass_per_length: Mass per unit length (kg/m)
            orbit_altitude: Orbit altitude (km)
            orbit_inclination: Orbit inclination (degrees)
        """
        self.length = length
        self.diameter = diameter
        self.conductivity = conductivity
        self.mass_per_length = mass_per_length
        self.orbit_altitude = orbit_altitude * 1000.0  # Convert to m
        self.orbit_inclination = np.radians(orbit_inclination)

        # Earth parameters
        self.earth_radius = 6371.0e3  # m
        self.earth_mu = 3.986004418e14  # m³/s² (gravitational parameter)
        self.earth_magnetic_moment = 7.96e22  # A·m² (dipole moment)

        # Tether resistance
        cross_section_area = np.pi * (diameter / 2) ** 2
        self.resistance = length / (conductivity * cross_section_area)

        # Orbital parameters
        self.orbit_radius = self.earth_radius + self.orbit_altitude
        self.orbital_velocity = np.sqrt(self.earth_mu / self.orbit_radius)

        # Libration parameters (gravity gradient)
        # Natural libration frequency: ω_lib = √(3μ/R³)
        self.libration_frequency = np.sqrt(3 * self.earth_mu / self.orbit_radius ** 3)
        self.libration_period = 2 * np.pi / self.libration_frequency

        # IGRF parameters
        self.use_igrf = _IGRF_AVAILABLE
        if self.use_igrf:
            logger.info("EDT will use IGRF magnetic field")
        else:
            logger.info("EDT will use dipole magnetic field (IGRF unavailable)")

        logger.info(
            f"EDT initialized: length={length}m, altitude={orbit_altitude}km, "
            f"libration_period={self.libration_period:.2f}s"
        )

    def dipole_magnetic_field(
        self,
        position: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Earth's dipole magnetic field at position.

        Uses simplified dipole model: B = (μ₀/4π) * (3(m·r̂)r̂ - m) / r³

        Args:
            position: Position in ECI coordinates [x, y, z] (m)

        Returns:
            Magnetic field vector [Bx, By, Bz] (T)
        """
        r = np.linalg.norm(position)
        r_hat = position / r

        # Earth's magnetic dipole moment (tilted ~11 degrees from rotation axis)
        # Simplified: aligned with z-axis for Phase 4A
        # Dipole points south (negative z), so field at equator points north
        m = np.array([0, 0, self.earth_magnetic_moment])  # North-pointing dipole moment

        mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability

        # Dipole field equation
        B = (mu_0 / (4 * np.pi)) * (3 * np.dot(m, r_hat) * r_hat - m) / r ** 3

        return B

    def igrf_magnetic_field(
        self,
        position: np.ndarray,
        date: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute Earth's IGRF magnetic field at position.

        Uses ppigrf library if available, otherwise falls back to dipole.

        Args:
            position: Position in ECI coordinates [x, y, z] (m)
            date: Date in decimal years (defaults to current date)

        Returns:
            Magnetic field vector [Bx, By, Bz] (T)
        """
        if not self.use_igrf:
            logger.debug("IGRF not available, using dipole field")
            return self.dipole_magnetic_field(position)

        try:
            # Convert ECI to geodetic coordinates for IGRF
            # Simplified: assume spherical Earth for Phase 4B
            r = np.linalg.norm(position)
            latitude = np.arcsin(position[2] / r)  # radians
            longitude = np.arctan2(position[1], position[0])  # radians
            altitude = r - self.earth_radius  # m

            # Convert to degrees for ppigrf
            lat_deg = np.degrees(latitude)
            lon_deg = np.degrees(longitude)
            alt_km = altitude / 1000.0

            # Current date in decimal years
            if date is None:
                import datetime
                now = datetime.datetime.now()
                date = now.year + (now.timetuple().tm_yday - 1) / 365.0

            # Call ppigrf
            # ppigrf expects: latitude (deg), longitude (deg), altitude (km), date (decimal year)
            B_north, B_east, B_down = ppigrf.igrf(lat_deg, lon_deg, alt_km, date)

            # Convert to ECI (simplified - ignore Earth rotation for Phase 4B)
            B = np.array([B_east, B_north, -B_down])  # Convert to ECI frame

            return B

        except Exception as e:
            logger.warning(f"IGRF calculation failed: {e}, falling back to dipole")
            return self.dipole_magnetic_field(position)

    def magnetic_field(
        self,
        position: np.ndarray,
        date: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute magnetic field at position (IGRF if available, else dipole).

        Args:
            position: Position in ECI coordinates [x, y, z] (m)
            date: Date in decimal years (for IGRF)

        Returns:
            Magnetic field vector [Bx, By, Bz] (T)
        """
        if self.use_igrf:
            return self.igrf_magnetic_field(position, date)
        else:
            return self.dipole_magnetic_field(position)

    def compute_emf(
        self,
        velocity: np.ndarray,
        position: np.ndarray,
        date: Optional[float] = None,
    ) -> float:
        """
        Compute motional EMF: V_emf = (v × B) · L

        Args:
            velocity: Velocity vector [vx, vy, vz] (m/s)
            position: Position vector [x, y, z] (m)
            date: Date in decimal years (for IGRF)

        Returns:
            Motional EMF (V)
        """
        B = self.magnetic_field(position, date)

        # Cross product: v × B
        v_cross_B = np.cross(velocity, B)

        # EMF = |v × B| × L (maximum when tether perpendicular to v and B)
        # This is simplified for Phase 4A - assumes optimal tether orientation
        emf = np.linalg.norm(v_cross_B) * self.length

        return emf

    def compute_lorentz_force(
        self,
        current: float,
        position: np.ndarray,
        date: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute Lorentz force: F = I × L × B

        Args:
            current: Tether current (A)
            position: Position vector [x, y, z] (m)
            date: Date in decimal years (for IGRF)

        Returns:
            Lorentz force vector [Fx, Fy, Fz] (N)
        """
        B = self.magnetic_field(position, date)

        # Lorentz force: F = I * L × B
        # Assuming tether aligned with velocity for simplicity
        # In reality, tether orientation matters - this is simplified for Phase 4A
        L_hat = position / np.linalg.norm(position)

        force = current * self.length * np.cross(L_hat, B)

        return force

    def libration_dynamics(
        self,
        state: TetherState,
        dt: float = 0.01,
        steps: int = 100,
    ) -> Tuple[np.ndarray, TetherState]:
        """
        Integrate libration dynamics using gravity gradient.

        Libration equation: θ̈ + ω_lib² sin(θ) = 0
        Linearized: θ̈ + ω_lib² θ = 0 (for small angles)

        Args:
            state: Initial tether state
            dt: Time step (s)
            steps: Number of integration steps

        Returns:
            Tuple of (libration_history, final_state)
        """
        libration_history = np.zeros((steps, 2))  # [angle, rate] over time

        current_state = state

        for i in range(steps):
            # Store current state
            libration_history[i, 0] = current_state.libration_angle
            libration_history[i, 1] = current_state.libration_rate

            # Libration equation: θ̈ = -ω_lib² sin(θ)
            libration_acceleration = -self.libration_frequency ** 2 * np.sin(current_state.libration_angle)

            # Euler integration
            current_state.libration_rate += libration_acceleration * dt
            current_state.libration_angle += current_state.libration_rate * dt

        return libration_history, current_state

    def compute_tether_dynamics(
        self,
        current: float,
        dt: float = 0.01,
        steps: int = 1000,
    ) -> dict:
        """
        Integrate full tether dynamics with libration.

        Args:
            current: Tether current (A)
            dt: Time step (s)
            steps: Number of integration steps

        Returns:
            Dictionary with dynamics results
        """
        # Initial state (simplified circular orbit)
        position = np.array([self.orbit_radius, 0, 0])
        velocity = np.array([0, self.orbital_velocity, 0])

        initial_state = TetherState(
            position=position,
            velocity=velocity,
            libration_angle=0.1,  # Small initial libration (rad)
            libration_rate=0.0,
            out_of_plane_angle=0.0,
            out_of_plane_rate=0.0,
        )

        # Integrate libration
        libration_history, final_state = self.libration_dynamics(initial_state, dt, steps)

        # Compute EMF and Lorentz force
        emf = self.compute_emf(velocity, position)
        lorentz_force = self.compute_lorentz_force(current, position)

        return {
            "emf": emf,
            "lorentz_force": lorentz_force,
            "libration_history": libration_history,
            "final_state": final_state,
            "libration_period": self.libration_period,
        }


def analytical_lorentz_force(
    current: float,
    length: float,
    magnetic_field: float,
) -> float:
    """
    Analytical Lorentz force for validation: F = I × L × B (perpendicular case)

    Args:
        current: Current (A)
        length: Length (m)
        magnetic_field: Magnetic field strength (T)

    Returns:
        Lorentz force magnitude (N)
    """
    return current * length * magnetic_field


def analytical_emf(
    velocity: float,
    magnetic_field: float,
    length: float,
) -> float:
    """
    Analytical EMF for validation: V = v × B × L (perpendicular case)

    Args:
        velocity: Velocity (m/s)
        magnetic_field: Magnetic field strength (T)
        length: Length (m)

    Returns:
        EMF (V)
    """
    return velocity * magnetic_field * length


def analytical_libration_period(
    orbit_radius: float,
    earth_mu: float = 3.986004418e14,
) -> float:
    """
    Analytical libration period: T = 2π/ω where ω = √(3μ/R³)

    Args:
        orbit_radius: Orbit radius (m)
        earth_mu: Earth gravitational parameter (m³/s²)

    Returns:
        Libration period (s)
    """
    omega = np.sqrt(3 * earth_mu / orbit_radius ** 3)
    return 2 * np.pi / omega

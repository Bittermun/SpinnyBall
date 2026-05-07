"""
Spherical Halbach array ("magic sphere") physics model.

A Halbach array is a special arrangement of permanent magnets that augments
the magnetic field on one side while cancelling it on the other. A spherical
Halbach array produces a pure dipole field with minimal higher-order multipoles.

References:
- Leupold & Potenziani (1988) - Halbach cylinders & spherical extensions
- Yonnet (1981) - Permanent magnet bearings
- Jackson, Classical Electrodynamics (3rd ed.) - Dipole fields

Equations:
- Magnetization: M_r = 2*M_0*cos(theta), M_theta = M_0*sin(theta)
- Dipole moment: m = (4*pi/3) * R^3 * M_eff
- External field: B = (mu0/4*pi*r^3) * [3*(m·r̂)*r̂ - m]
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)


@dataclass
class HalbachConfig:
    """Configuration for spherical Halbach array.
    
    Attributes:
        radius: Sphere radius (m)
        remanence: Remanent flux density B_r (T)
        material: Magnet material ('NdFeB', 'SmCo')
        temperature: Operating temperature (K)
        magnetization_angle: Angle of magnetization axis (rad), default 0 (aligned with z)
    """
    radius: float = 0.05  # m (5 cm default)
    remanence: float = 1.4  # T (NdFeB default)
    material: str = 'NdFeB'
    temperature: float = 293.0  # K (room temperature)
    magnetization_angle: float = 0.0  # rad
    
    # Temperature coefficients (%/K)
    TEMP_COEFF_NDFEB = -0.0012  # -0.12%/K
    TEMP_COEFF_SMCO = -0.0003   # -0.03%/K
    
    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError(f"Radius must be > 0, got {self.radius}")
        if self.remanence <= 0:
            raise ValueError(f"Remanence must be > 0, got {self.remanence}")
    
    @property
    def effective_magnetization(self) -> float:
        """Effective magnetization M_eff = B_r / mu_0 (A/m)."""
        return self.remanence / MU_0
    
    @property
    def temperature_corrected_remanence(self) -> float:
        """Remanence corrected for temperature.
        
        B_r(T) = B_r(293K) * [1 + alpha * (T - 293)]
        """
        if self.material == 'NdFeB':
            alpha = self.TEMP_COEFF_NDFEB
        elif self.material == 'SmCo':
            alpha = self.TEMP_COEFF_SMCO
        else:
            alpha = 0.0
        
        delta_T = self.temperature - 293.0
        return self.remanence * (1.0 + alpha * delta_T)
    
    @property
    def volume(self) -> float:
        """Sphere volume (m^3)."""
        return (4.0 / 3.0) * np.pi * self.radius**3
    
    @property
    def mass(self) -> float:
        """Approximate mass based on material density.
        
        NdFeB: ~7500 kg/m^3
        SmCo: ~8400 kg/m^3
        """
        densities = {'NdFeB': 7500.0, 'SmCo': 8400.0}
        density = densities.get(self.material, 7500.0)
        return self.volume * density


class HalbachArray:
    """Spherical Halbach array ("magic sphere") model.
    
    Implements the spherical Halbach array that produces a pure dipole field.
    The magnetization distribution is:
        M_r = 2*M_0*cos(theta)
        M_theta = M_0*sin(theta)
    
    This creates a dipole moment aligned with the z-axis (by default).
    
    Attributes:
        config: HalbachConfig with physical parameters
        dipole_moment: Magnetic dipole moment vector (A·m²)
    """
    
    def __init__(self, config: HalbachConfig = None):
        """Initialize Halbach array.
        
        Args:
            config: HalbachConfig with physical parameters, or None for defaults
        """
        self.config = config if config is not None else HalbachConfig()
        self._dipole_moment = self._compute_dipole_moment()
    
    def _compute_dipole_moment(self) -> np.ndarray:
        """Compute dipole moment vector.
        
        For spherical Halbach array:
            m = (4*pi/3) * R^3 * M_eff * ẑ
        
        Returns:
            Dipole moment vector [mx, my, mz] (A·m²)
        """
        M_eff = self.config.effective_magnetization
        volume_factor = (4.0 * np.pi / 3.0) * self.config.radius**3
        m_magnitude = volume_factor * M_eff
        
        # Dipole aligned with z-axis (can be rotated via magnetization_angle)
        angle = self.config.magnetization_angle
        return np.array([
            m_magnitude * np.sin(angle),
            0.0,
            m_magnitude * np.cos(angle)
        ])
    
    @property
    def dipole_moment(self) -> np.ndarray:
        """Magnetic dipole moment vector (A·m²)."""
        return self._dipole_moment.copy()
    
    @property
    def dipole_magnitude(self) -> float:
        """Magnitude of dipole moment (A·m²)."""
        return float(np.linalg.norm(self._dipole_moment))
    
    def magnetic_field(self, position: np.ndarray) -> np.ndarray:
        """Compute magnetic field at a point due to this Halbach array.
        
        For a dipole, the field is:
            B = (mu0/4*pi*r^3) * [3*(m·r̂)*r̂ - m]
        
        Args:
            position: Position vector [x, y, z] relative to sphere center (m)
        
        Returns:
            Magnetic field vector [Bx, By, Bz] (T)
        """
        r_vec = np.asarray(position, dtype=float)
        r = np.linalg.norm(r_vec)
        
        if r < self.config.radius:
            # Inside the sphere - simplified model (internal field)
            # For Halbach sphere, internal field is uniform
            return self._internal_field()
        
        if r < 1e-12:
            raise ValueError("Position too close to origin")
        
        r_hat = r_vec / r
        m = self._dipole_moment
        
        # Dipole field formula
        # B = (mu0/4*pi*r^3) * [3*(m·r̂)*r̂ - m]
        m_dot_r_hat = np.dot(m, r_hat)
        prefactor = MU_0 / (4.0 * np.pi * r**3)
        
        B = prefactor * (3.0 * m_dot_r_hat * r_hat - m)
        return B
    
    def _internal_field(self) -> np.ndarray:
        """Internal magnetic field (simplified uniform field model).
        
        For a Halbach sphere, the internal field is approximately uniform
        and related to the magnetization.
        
        Returns:
            Internal field vector [Bx, By, Bz] (T)
        """
        # Simplified: internal field is approximately (2/3)*mu0*M_eff in z-direction
        M_eff = self.config.effective_magnetization
        B_internal = (2.0 / 3.0) * MU_0 * M_eff
        
        angle = self.config.magnetization_angle
        return np.array([
            B_internal * np.sin(angle),
            0.0,
            B_internal * np.cos(angle)
        ])
    
    def field_on_strong_side(self, distance: float) -> np.ndarray:
        """Compute field on the strong side of the Halbach array.
        
        The Halbach array concentrates field on one side ("strong side")
        while cancelling it on the other ("weak side").
        
        Args:
            distance: Distance from sphere center along dipole axis (m)
        
        Returns:
            Magnetic field vector [Bx, By, Bz] (T)
        """
        # Strong side is along positive z (or magnetization angle direction)
        angle = self.config.magnetization_angle
        position = np.array([
            distance * np.sin(angle),
            0.0,
            distance * np.cos(angle)
        ])
        return self.magnetic_field(position)
    
    def field_on_weak_side(self, distance: float) -> np.ndarray:
        """Compute field on the weak side of the Halbach array.
        
        Args:
            distance: Distance from sphere center opposite to dipole axis (m)
        
        Returns:
            Magnetic field vector [Bx, By, Bz] (T)
        """
        # Weak side is opposite to dipole direction
        angle = self.config.magnetization_angle
        position = np.array([
            -distance * np.sin(angle),
            0.0,
            -distance * np.cos(angle)
        ])
        return self.magnetic_field(position)
    
    def force_on_dipole(self, external_dipole: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Compute force on an external dipole in the field of this array.
        
        Force on dipole m2 in field B1 from dipole m1:
            F = ∇(m2 · B1)
        
        For two dipoles, this reduces to the dipole-dipole force formula.
        
        Args:
            external_dipole: External dipole moment [mx, my, mz] (A·m²)
            position: Position of external dipole relative to this array (m)
        
        Returns:
            Force vector [Fx, Fy, Fz] (N)
        """
        # Use the interball_magnetic module for the full calculation
        from .interball_magnetic import dipole_dipole_force
        return dipole_dipole_force(self._dipole_moment, external_dipole, position)
    
    def torque_on_dipole(self, external_dipole: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Compute torque on an external dipole in the field of this array.
        
        Torque on dipole m2 in field B1:
            τ = m2 × B1
        
        Args:
            external_dipole: External dipole moment [mx, my, mz] (A·m²)
            position: Position of external dipole relative to this array (m)
        
        Returns:
            Torque vector [τx, τy, τz] (N·m)
        """
        B = self.magnetic_field(position)
        return np.cross(external_dipole, B)
    
    def potential_energy(self, external_dipole: np.ndarray, position: np.ndarray) -> float:
        """Compute potential energy of an external dipole in this array's field.
        
        U = -m2 · B1
        
        Args:
            external_dipole: External dipole moment [mx, my, mz] (A·m²)
            position: Position of external dipole relative to this array (m)
        
        Returns:
            Potential energy (J)
        """
        B = self.magnetic_field(position)
        return -np.dot(external_dipole, B)
    
    def rotate_dipole(self, rotation_matrix: np.ndarray) -> 'HalbachArray':
        """Return a new HalbachArray with rotated dipole moment.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
        
        Returns:
            New HalbachArray with rotated dipole
        """
        new_config = HalbachConfig(
            radius=self.config.radius,
            remanence=self.config.remanence,
            material=self.config.material,
            temperature=self.config.temperature
        )
        new_array = HalbachArray(new_config)
        new_array._dipole_moment = rotation_matrix @ self._dipole_moment
        return new_array
    
    def get_characteristic_length(self) -> float:
        """Get characteristic length scale for spacing calculations.
        
        Returns:
            Characteristic length (m), typically 2*radius
        """
        return 2.0 * self.config.radius


def create_standard_halbach(
    radius: float = 0.05,
    material: str = 'NdFeB',
    temperature: float = 293.0
) -> HalbachArray:
    """Create a standard Halbach array with canonical parameters.
    
    Args:
        radius: Sphere radius (m), default 5 cm
        material: 'NdFeB' or 'SmCo'
        temperature: Operating temperature (K)
    
    Returns:
        Configured HalbachArray
    """
    remanences = {'NdFeB': 1.4, 'SmCo': 1.1}
    B_r = remanences.get(material, 1.4)
    
    config = HalbachConfig(
        radius=radius,
        remanence=B_r,
        material=material,
        temperature=temperature
    )
    return HalbachArray(config)


def compute_halbach_spacing_equilibrium(
    halbach: HalbachArray,
    stream_velocity: float,
    linear_density: float
) -> float:
    """Compute equilibrium spacing for a stream of Halbach arrays.
    
    The equilibrium balances magnetic repulsion against hoop tension:
        k_mag * δx ≈ T * curvature
    where T = λ * u^2 is the hoop tension.
    
    Args:
        halbach: HalbachArray configuration
        stream_velocity: Stream velocity (m/s)
        linear_density: Linear mass density (kg/m)
    
    Returns:
        Equilibrium spacing (m)
    """
    # Hoop tension
    T = linear_density * stream_velocity**2
    
    # Magnetic stiffness scale (approximate)
    m = halbach.dipole_magnitude
    # Characteristic spacing where magnetic force balances tension
    # F_mag ~ mu0*m^2/(4*pi*d^4), set F_mag * d ~ T
    # => d^3 ~ mu0*m^2/(4*pi*T)
    
    if T < 1e-12:
        return halbach.get_characteristic_length()
    
    d_eq = ((MU_0 * m**2) / (4.0 * np.pi * T))**(1.0 / 3.0)
    
    # Ensure spacing is at least 2*radius (balls don't overlap)
    return max(d_eq, 2.0 * halbach.config.radius)

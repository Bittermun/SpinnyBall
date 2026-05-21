"""
Cislunar Propagator with Halbach Magnetic Perturbations

Integrates Halbach array multipole expansion into CR3BP + mascon dynamics.
Enables calculation of magnetic forces on charged packets and spacecraft.

Features:
    - Magnetic field and gradient computation in propagator context
    - Force on magnetic dipoles (packets with magnetization)
    - Lorentz force on charged particles (future extension)
    - Integration with existing CR3BP + mascon framework
    - Optional Halbach perturbations via config flag
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from dynamics.cislunar_mascon import (
    CR3BPMasconPropagator,
    CR3BPMasconConfig,
    CR3BPMasconSolution
)
from dynamics.halbach_multipole import (
    HalbachSphericalHarmonic,
    HalbachSphericalHarmonicConfig
)


@dataclass
class CR3BPHalbachConfig(CR3BPMasconConfig):
    """Configuration for CR3BP + Mascon + Halbach propagator."""
    
    use_halbach: bool = True
    """Enable Halbach magnetic perturbations."""
    
    halbach_degree_max: int = 4
    """Halbach multipole degree."""
    
    halbach_dipole_moment_am2: float = 1.0
    """Halbach dipole moment in A⋅m²."""
    
    halbach_radius_m: float = 0.05
    """Halbach array reference radius in meters."""
    
    packet_magnetic_moment_am2: float = 0.1
    """Packet magnetic moment in A⋅m²."""
    
    packet_mass_kg: float = 1.0
    """Packet mass in kg (for acceleration calculation)."""
    
    halbach_position_earth_relative: Optional[np.ndarray] = None
    """Position of Halbach shepherd relative to Earth in ECI km.
    If None, uses position from propagator state."""
    
    fixed_halbach_position: bool = False
    """Keep Halbach at fixed position (or follow dynamics)."""


class CR3BPHalbachPropagator(CR3BPMasconPropagator):
    """
    CR3BP + Mascon + Halbach multipole propagator.
    
    Extends CR3BPMasconPropagator with magnetic Halbach field effects.
    """
    
    # Conversion factors
    KM_TO_M = 1000.0
    T_TO_GAUSS = 1e4
    
    def __init__(self, config: Optional[CR3BPHalbachConfig] = None):
        """
        Initialize Halbach-enhanced cislunar propagator.
        
        Args:
            config: CR3BPHalbachConfig instance
        """
        if config is None:
            config = CR3BPHalbachConfig()
        
        # Call parent (CR3BPMasconPropagator) constructor
        super().__init__(config)
        
        self.halbach_config = config
        
        # Initialize Halbach model if enabled
        if config.use_halbach:
            halbach_config = HalbachSphericalHarmonicConfig(
                degree_max=config.halbach_degree_max,
                moment_magnitude_am2=config.halbach_dipole_moment_am2,
                radius_m=config.halbach_radius_m
            )
            self.halbach = HalbachSphericalHarmonic(halbach_config)
        else:
            self.halbach = None
    
    def _halbach_acceleration(self, state: np.ndarray, time_abs: float) -> np.ndarray:
        """
        Compute acceleration on packet due to Halbach magnetic field.
        
        Assumes:
            - State contains position and velocity of target packet
            - Halbach is at fixed or specified location
            - Packet has magnetic moment m_packet
            - Force: F = ∇(m · B) where B is Halbach field
        
        Args:
            state: State vector [x, y, z, vx, vy, vz] in ECI frame (km, km/s)
            time_abs: Absolute time (not used for fixed Halbach)
        
        Returns:
            Acceleration (a_x, a_y, a_z) in km/s²
        """
        if not self.halbach_config.use_halbach or self.halbach is None:
            return np.array([0.0, 0.0, 0.0])
        
        # Extract positions
        packet_pos_km = state[0:3]
        
        # Halbach position
        if self.halbach_config.halbach_position_earth_relative is not None:
            halbach_pos_km = self.halbach_config.halbach_position_earth_relative
        else:
            # Default: Halbach at Earth (shepherd at Earth initially)
            halbach_pos_km = np.array([0.0, 0.0, 0.0])
        
        # Relative position (packet relative to Halbach) in meters
        r_rel_km = packet_pos_km - halbach_pos_km
        r_rel_m = r_rel_km * self.KM_TO_M
        
        # Distance threshold optimization: beyond 10 km, the magnetic force decays to less than 10^-20 N.
        # This is 8+ orders of magnitude below machine double-precision compared to typical ECI/lunar gravity,
        # so we short-circuit Legendre polynomial evaluations to avoid performance bottlenecks.
        if np.linalg.norm(r_rel_m) > 10000.0:
            return np.array([0.0, 0.0, 0.0])
        
        # Compute gradient (d B/dx tensor)
        try:
            grad_B = self.halbach.gradient(r_rel_m, degree=self.halbach_config.halbach_degree_max)
        except:
            return np.array([0.0, 0.0, 0.0])
        
        # Packet magnetic moment (assumed aligned with z-axis initially)
        m_packet = np.array([0.0, 0.0, self.halbach_config.packet_magnetic_moment_am2])
        
        # Force: F = grad(B) · m (Tesla/meter × A⋅m² = Tesla⋅A⋅m)
        # But grad_B is in Tesla/meter, so F is in Newtons
        force_N = grad_B @ m_packet
        
        # Acceleration in m/s²
        accel_ms2 = force_N / self.halbach_config.packet_mass_kg
        
        # Convert to km/s²
        accel_kms2 = accel_ms2 / self.KM_TO_M
        
        return accel_kms2
    
    def _accelerations(self, state: np.ndarray, time_abs: float) -> np.ndarray:
        """
        Compute total acceleration: CR3BP + Mascon + Halbach.
        
        Args:
            state: State vector
            time_abs: Absolute time (for SPICE updates)
        
        Returns:
            Acceleration vector [a_x, a_y, a_z]
        """
        # CR3BP + Mascon accelerations (from parent)
        accel = super()._accelerations(state, time_abs)
        
        # Add Halbach perturbations
        if self.halbach_config.use_halbach:
            accel_halbach = self._halbach_acceleration(state, time_abs)
            accel[3:6] += accel_halbach
        
        return accel
    
    def compute_magnetic_field_at_position(self, position_km: np.ndarray,
                                           halbach_pos_km: Optional[np.ndarray] = None,
                                           degree: Optional[int] = None) -> np.ndarray:
        """
        Compute Halbach magnetic field at given position.
        
        Utility method for analysis.
        
        Args:
            position_km: Position in ECI frame (km)
            halbach_pos_km: Halbach position (default: Earth)
            degree: Multipole degree (default: config value)
        
        Returns:
            Magnetic field (B_x, B_y, B_z) in Tesla
        """
        if not self.halbach_config.use_halbach or self.halbach is None:
            return np.array([0.0, 0.0, 0.0])
        
        if halbach_pos_km is None:
            halbach_pos_km = np.array([0.0, 0.0, 0.0])
        
        r_rel_km = position_km - halbach_pos_km
        r_rel_m = r_rel_km * self.KM_TO_M
        
        degree = degree or self.halbach_config.halbach_degree_max
        
        return self.halbach.field(r_rel_m, degree=degree)
    
    def compute_magnetic_force_on_packet(self, position_km: np.ndarray,
                                         moment_am2: Optional[np.ndarray] = None,
                                         halbach_pos_km: Optional[np.ndarray] = None,
                                         degree: Optional[int] = None) -> np.ndarray:
        """
        Compute magnetic force on a packet at given position.
        
        Args:
            position_km: Packet position (km)
            moment_am2: Magnetic moment vector (default: config value along z)
            halbach_pos_km: Halbach position (default: Earth)
            degree: Multipole degree (default: config value)
        
        Returns:
            Force vector (F_x, F_y, F_z) in Newtons
        """
        if not self.halbach_config.use_halbach or self.halbach is None:
            return np.array([0.0, 0.0, 0.0])
        
        if halbach_pos_km is None:
            halbach_pos_km = np.array([0.0, 0.0, 0.0])
        
        if moment_am2 is None:
            moment_am2 = np.array([0.0, 0.0, self.halbach_config.packet_magnetic_moment_am2])
        
        r_rel_km = position_km - halbach_pos_km
        r_rel_m = r_rel_km * self.KM_TO_M
        
        degree = degree or self.halbach_config.halbach_degree_max
        
        force_N = self.halbach.dipole_force(r_rel_m, moment_am2, degree=degree)
        
        return force_N
    
    def propagate_with_halbach_analysis(self, state0: np.ndarray, t_eval: np.ndarray,
                                        t0: float = 0.0) -> Tuple[CR3BPMasconSolution, Dict]:
        """
        Propagate with Halbach field analysis.
        
        Args:
            state0: Initial state
            t_eval: Time evaluation points
            t0: Initial time
        
        Returns:
            (solution, diagnostics_dict)
        """
        # Propagate
        sol = self.propagate(state0, t_eval, t0)
        
        # Compute Halbach diagnostics
        diagnostics = {
            'times': t_eval,
            'halbach_enabled': self.halbach_config.use_halbach,
            'halbach_dipole_moment': self.halbach_config.halbach_dipole_moment_am2,
            'packet_magnetic_moment': self.halbach_config.packet_magnetic_moment_am2,
            'halbach_forces': [],
            'magnetic_field_magnitudes': []
        }
        
        if self.halbach_config.use_halbach:
            for i, t in enumerate(t_eval):
                state = sol.y[:, i]
                pos = state[0:3]
                
                # Magnetic field
                B = self.compute_magnetic_field_at_position(
                    pos,
                    degree=self.halbach_config.halbach_degree_max
                )
                diagnostics['magnetic_field_magnitudes'].append(np.linalg.norm(B))
                
                # Force
                F = self.compute_magnetic_force_on_packet(
                    pos,
                    degree=self.halbach_config.halbach_degree_max
                )
                diagnostics['halbach_forces'].append(F)
        
        return sol, diagnostics


class CR3BPHalbachSolution(CR3BPMasconSolution):
    """Solution wrapper for Halbach-enhanced propagation."""
    pass


# Export
__all__ = [
    'CR3BPHalbachConfig',
    'CR3BPHalbachPropagator',
    'CR3BPHalbachSolution',
]

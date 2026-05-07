"""
Hoop tension dynamics for self-confined mass streams.

The stream's own momentum flux creates hoop tension that provides
restoring forces when the stream is perturbed from its equilibrium path.
This is the second mechanism (along with magnetic repulsion) that
maintains stream coherence.

Physics:
- Hoop tension: T = λ * u^2
- Radial restoring force: f_r = -(T/R^2) * (δr + d²δr/dθ²)
- Wave equation for stream perturbations

References:
- Hoyt & Forward (1999) - Momentum-exchange tethers
- Hughes & D'Eleuterio (1986) - Gyroelastic continua
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class StreamGeometry:
    """Geometric parameters for a closed-loop stream.
    
    Attributes:
        radius: Nominal radius of circular stream (m)
        n_balls: Number of discrete balls in the stream
        ball_mass: Mass of each ball (kg)
        stream_velocity: Tangential velocity (m/s)
    """
    radius: float = 1000.0  # m (1 km default)
    n_balls: int = 100
    ball_mass: float = 1.0  # kg
    stream_velocity: float = 1600.0  # m/s
    
    @property
    def circumference(self) -> float:
        """Stream circumference (m)."""
        return 2.0 * np.pi * self.radius
    
    @property
    def linear_density(self) -> float:
        """Linear mass density λ = mass per unit length (kg/m)."""
        if self.circumference <= 0:
            return 0.0
        return (self.n_balls * self.ball_mass) / self.circumference
    
    @property
    def hoop_tension(self) -> float:
        """Hoop tension T = λ * u^2 (N)."""
        return self.linear_density * self.stream_velocity**2
    
    @property
    def ball_spacing(self) -> float:
        """Equilibrium spacing between ball centers (m)."""
        if self.n_balls <= 0:
            return 0.0
        return self.circumference / self.n_balls
    
    @property
    def angular_velocity(self) -> float:
        """Angular velocity of stream rotation (rad/s)."""
        if self.radius <= 0:
            return 0.0
        return self.stream_velocity / self.radius


class HoopTensionModel:
    """Hoop tension restoring force model for mass streams.
    
    The hoop tension arises from the momentum flux of the circulating
    stream. When perturbed from circular equilibrium, the tension
    provides restoring forces that tend to return the stream to its
    nominal path.
    
    For a circular stream with perturbation δr(θ), the radial force
    per unit length is:
        f_r = -(T/R^2) * (δr + ∂²δr/∂θ²)
    
    Attributes:
        geometry: StreamGeometry with physical parameters
    """
    
    def __init__(self, geometry: StreamGeometry):
        """Initialize hoop tension model.
        
        Args:
            geometry: StreamGeometry with stream parameters
        """
        self.geometry = geometry
    
    def compute_radial_restoring_force(
        self,
        radial_displacement: float,
        azimuthal_position: float,
        curvature_perturbation: float = 0.0
    ) -> float:
        """Compute radial restoring force at a point on the stream.
        
        For a discrete ball at position θ with radial displacement δr:
            F_r = -(T/R) * (δr/R + ∂²δr/∂θ²)
        
        Args:
            radial_displacement: Radial perturbation δr (m)
            azimuthal_position: Angular position θ (rad)
            curvature_perturbation: Second derivative ∂²δr/∂θ² (m)
        
        Returns:
            Radial restoring force (N), positive = outward
        """
        T = self.geometry.hoop_tension
        R = self.geometry.radius
        
        if R <= 0:
            return 0.0
        
        # Hoop tension restoring force
        # f_r = -(T/R²) * (δr + ∂²δr/∂θ²)
        force_density = -(T / R**2) * (radial_displacement + curvature_perturbation)
        
        # Convert to force on a single ball (multiply by arc length per ball)
        arc_length = self.geometry.circumference / self.geometry.n_balls
        
        return force_density * arc_length
    
    def compute_transverse_stiffness(self) -> float:
        """Compute effective transverse stiffness from hoop tension.
        
        For small perturbations, the hoop tension provides a linear
        restoring force proportional to displacement.
        
        k_hoop ≈ T / R = λ * u² / R
        
        Returns:
            Transverse stiffness (N/m)
        """
        T = self.geometry.hoop_tension
        R = self.geometry.radius
        
        if R <= 0:
            return 0.0
        
        # Stiffness per unit length
        k_per_length = T / R**2
        
        # Convert to stiffness per ball
        arc_length = self.geometry.circumference / self.geometry.n_balls
        
        return k_per_length * arc_length
    
    def compute_stream_modes(self, n_modes: int = 5) -> List[Tuple[int, float]]:
        """Compute eigenfrequencies of stream perturbation modes.
        
        For a circular string with tension T and linear density λ,
        the eigenfrequencies are:
            ω_n = n * √(T/λ) / R = n * u / R
        
        where n is the mode number (n=1 is the "rigid body" mode,
        n≥2 are elastic deformation modes).
        
        Args:
            n_modes: Number of modes to compute
        
        Returns:
            List of (mode_number, frequency_hz) tuples
        """
        u = self.geometry.stream_velocity
        R = self.geometry.radius
        
        if R <= 0 or u <= 0:
            return []
        
        modes = []
        for n in range(1, n_modes + 1):
            omega_n = n * u / R  # rad/s
            f_n = omega_n / (2.0 * np.pi)  # Hz
            modes.append((n, f_n))
        
        return modes
    
    def compute_perturbation_dynamics(
        self,
        radial_displacements: np.ndarray,
        radial_velocities: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate radial perturbation dynamics for one timestep.
        
        Models the stream as a set of coupled oscillators with
        hoop tension providing the restoring force.
        
        Args:
            radial_displacements: Current radial displacements (m)
            radial_velocities: Current radial velocities (m/s)
            dt: Time step (s)
        
        Returns:
            Tuple of (new_displacements, new_velocities)
        """
        n = len(radial_displacements)
        if n != self.geometry.n_balls:
            raise ValueError(f"Expected {self.geometry.n_balls} balls, got {n}")
        
        # Compute second derivative (curvature) using finite differences
        # Periodic boundary conditions for closed loop
        d2r_dtheta2 = np.zeros(n)
        for i in range(n):
            i_prev = (i - 1) % n
            i_next = (i + 1) % n
            # Central difference for second derivative
            d2r_dtheta2[i] = radial_displacements[i_prev] - 2*radial_displacements[i] + radial_displacements[i_next]
        
        # Compute restoring accelerations
        T = self.geometry.hoop_tension
        R = self.geometry.radius
        m = self.geometry.ball_mass
        
        if R <= 0 or m <= 0:
            return radial_displacements, radial_velocities
        
        # Acceleration from hoop tension
        # a_r = -(T/mR) * (δr + ∂²δr/∂θ²)
        accelerations = -(T / (m * R)) * (radial_displacements + d2r_dtheta2)
        
        # Simple Euler integration
        new_velocities = radial_velocities + accelerations * dt
        new_displacements = radial_displacements + new_velocities * dt
        
        return new_displacements, new_velocities
    
    def compute_anchor_force(
        self,
        deflection_angle: float
    ) -> float:
        """Compute anchor force from stream deflection.
        
        When the stream is deflected by angle θ at a shepherd station,
        the anchor force is:
            F_anchor = T * sin(θ) = λ * u² * sin(θ)
        
        Args:
            deflection_angle: Deflection angle θ (rad)
        
        Returns:
            Anchor force (N)
        """
        T = self.geometry.hoop_tension
        return T * np.sin(deflection_angle)
    
    def compute_equilibrium_spacing(
        self,
        magnetic_stiffness: float
    ) -> float:
        """Compute equilibrium spacing balancing magnetic and tension forces.
        
        At equilibrium, magnetic repulsion balances hoop tension:
            k_mag * δr ≈ k_hoop * δr
        
        This determines the natural spacing of balls in the stream.
        
        Args:
            magnetic_stiffness: Effective magnetic stiffness (N/m)
        
        Returns:
            Equilibrium spacing (m)
        """
        k_hoop = self.compute_transverse_stiffness()
        
        # Total stiffness
        k_total = k_hoop + magnetic_stiffness
        
        # Characteristic spacing from stiffness balance
        if k_total <= 0:
            return self.geometry.ball_spacing
        
        # Spacing scales with √(k_mag/k_total)
        spacing_factor = np.sqrt(magnetic_stiffness / k_total) if k_total > 0 else 1.0
        
        return self.geometry.ball_spacing * spacing_factor


def create_stream_geometry_from_params(
    orbital_radius: float,
    stream_velocity: float,
    ball_mass: float,
    n_balls: int
) -> StreamGeometry:
    """Create StreamGeometry from canonical parameters.
    
    Args:
        orbital_radius: Nominal stream radius (m)
        stream_velocity: Stream velocity (m/s)
        ball_mass: Mass of each ball (kg)
        n_balls: Number of balls in stream
    
    Returns:
        Configured StreamGeometry
    """
    return StreamGeometry(
        radius=orbital_radius,
        n_balls=n_balls,
        ball_mass=ball_mass,
        stream_velocity=stream_velocity
    )


def compute_combined_stiffness(
    hoop_model: HoopTensionModel,
    magnetic_stiffness: float
) -> dict:
    """Compute combined stiffness from hoop tension and magnetic interactions.
    
    Args:
        hoop_model: HoopTensionModel instance
        magnetic_stiffness: Effective magnetic stiffness (N/m)
    
    Returns:
        Dictionary with stiffness breakdown
    """
    k_hoop = hoop_model.compute_transverse_stiffness()
    k_total = k_hoop + magnetic_stiffness
    
    return {
        'k_hoop': k_hoop,
        'k_magnetic': magnetic_stiffness,
        'k_total': k_total,
        'hoop_fraction': k_hoop / k_total if k_total > 0 else 0.0,
        'magnetic_fraction': magnetic_stiffness / k_total if k_total > 0 else 0.0,
    }

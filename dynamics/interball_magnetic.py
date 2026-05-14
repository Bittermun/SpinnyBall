"""
Inter-ball magnetic interactions for Halbach array mass streams.

Models the dipole-dipole interactions between neighboring spherical Halbach
arrays in a circulating mass stream. The self-confinement mechanism relies
on magnetic repulsion between balls to maintain stream coherence.

Physics:
- Dipole-dipole potential energy
- Force between magnetic dipoles
- Stiffness matrix for linearized stability analysis

References:
- Jackson, Classical Electrodynamics (3rd ed.), Chapter 5
- Yonnet (1981) - Permanent magnet bearings and dipole interactions
"""

from typing import List, Tuple, Optional

import numpy as np

from .halbach_array import HalbachArray, MU_0


def dipole_dipole_potential(
    m1: np.ndarray,
    m2: np.ndarray,
    r_vec: np.ndarray
) -> float:
    """Compute dipole-dipole interaction potential energy.
    
    U = (mu0/4*pi*r^3) * [m1·m2 - 3*(m1·r̂)*(m2·r̂)]
    
    Args:
        m1: Dipole moment of first ball [mx, my, mz] (A·m²)
        m2: Dipole moment of second ball [mx, my, mz] (A·m²)
        r_vec: Position vector from ball 1 to ball 2 [x, y, z] (m)
    
    Returns:
        Potential energy (J)
    """
    r = np.linalg.norm(r_vec)
    if r < 1e-12:
        raise ValueError("Distance too small for dipole approximation")
    
    r_hat = r_vec / r
    
    # Dipole-dipole interaction formula
    prefactor = MU_0 / (4.0 * np.pi * r**3)
    dot_m1_m2 = np.dot(m1, m2)
    dot_m1_r = np.dot(m1, r_hat)
    dot_m2_r = np.dot(m2, r_hat)
    
    U = prefactor * (dot_m1_m2 - 3.0 * dot_m1_r * dot_m2_r)
    return float(U)


def dipole_dipole_force(
    m1: np.ndarray,
    m2: np.ndarray,
    r_vec: np.ndarray
) -> np.ndarray:
    """Compute force on dipole 2 due to dipole 1 (dipole-dipole interaction).
    
    F_2 = -∇_2 U = (mu0/4*pi*r^4) * [
        3*(m1·m2)*r̂ - 15*(m1·r̂)*(m2·r̂)*r̂ + 3*(m2·r̂)*m1 + 3*(m1·r̂)*m2
    ]
    
    Args:
        m1: Dipole moment of dipole 1 [mx, my, mz] (A·m²)
        m2: Dipole moment of dipole 2 [mx, my, mz] (A·m²)
        r_vec: Position vector from dipole 1 to dipole 2 [x, y, z] (m)
    
    Returns:
        Force vector on dipole 2 [Fx, Fy, Fz] (N)
    """
    r = np.linalg.norm(r_vec)
    if r < 1e-12:
        raise ValueError("Distance too small for dipole approximation")
    
    r_hat = r_vec / r
    
    # Precompute dot products
    dot_m1_m2 = np.dot(m1, m2)
    dot_m1_r = np.dot(m1, r_hat)
    dot_m2_r = np.dot(m2, r_hat)
    
    # Dipole-dipole force formula
    prefactor = (3.0 * MU_0) / (4.0 * np.pi * r**4)
    
    F = prefactor * (
        dot_m1_m2 * r_hat
        - 5.0 * dot_m1_r * dot_m2_r * r_hat
        + dot_m2_r * m1
        + dot_m1_r * m2
    )
    
    return F


def dipole_dipole_torque(
    m1: np.ndarray,
    m2: np.ndarray,
    r_vec: np.ndarray
) -> np.ndarray:
    """Compute torque on ball 1 due to ball 2's magnetic field.
    
    τ = m1 × B2
    
    where B2 is the field from dipole m2 at the position of m1.
    
    Args:
        m1: Dipole moment of first ball [mx, my, mz] (A·m²)
        m2: Dipole moment of second ball [mx, my, mz] (A·m²)
        r_vec: Position vector from ball 1 to ball 2 [x, y, z] (m)
    
    Returns:
        Torque vector on ball 1 [τx, τy, τz] (N·m)
    """
    r = np.linalg.norm(r_vec)
    if r < 1e-12:
        raise ValueError("Distance too small for dipole approximation")
    
    r_hat = r_vec / r
    
    # Field from m2 at position of m1 (note: r_vec points from 1 to 2,
    # so field at 1 due to 2 uses -r_vec)
    prefactor = MU_0 / (4.0 * np.pi * r**3)
    dot_m2_r = np.dot(m2, r_hat)
    B2 = prefactor * (3.0 * dot_m2_r * r_hat - m2)
    
    # Torque = m1 × B2
    return np.cross(m1, B2)


def compute_linear_stiffness(
    m: float,
    d0: float,
    alignment: str = 'repulsive'
) -> float:
    """Compute linearized transverse magnetic stiffness.
    
    For two dipoles of magnitude m separated by distance d0:
    - If aligned head-to-tail (attractive): k ≈ -6*mu0*m^2/(4*pi*d0^5)
    - If aligned side-by-side (repulsive): k ≈ +3*mu0*m^2/(4*pi*d0^5)
    
    For SGMS stream, we want repulsive alignment to maintain spacing.
    
    Args:
        m: Dipole moment magnitude (A·m²)
        d0: Equilibrium separation (m)
        alignment: 'repulsive' or 'attractive'
    
    Returns:
        Linear stiffness (N/m)
    """
    if d0 <= 0:
        raise ValueError(f"Separation must be > 0, got {d0}")
    
    if alignment == 'repulsive':
        # Side-by-side alignment (dipoles parallel, perpendicular to separation)
        k = (3.0 * MU_0 * m**2) / (4.0 * np.pi * d0**5)
    elif alignment == 'attractive':
        # Head-to-tail alignment (dipoles parallel to separation)
        k = -(6.0 * MU_0 * m**2) / (4.0 * np.pi * d0**5)
    else:
        raise ValueError(f"Unknown alignment: {alignment}")
    
    return float(k)


class InterBallMagneticInteraction:
    """Manages magnetic interactions between balls in a stream.
    
    This class computes the net magnetic forces and torques on each ball
    due to its neighbors in the stream.
    
    Attributes:
        halbach_arrays: List of HalbachArray objects for each ball
        neighbor_range: Number of neighbors on each side to consider
    """
    
    def __init__(
        self,
        halbach_arrays: List[HalbachArray],
        neighbor_range: int = 2
    ):
        """Initialize inter-ball interaction model.
        
        Args:
            halbach_arrays: List of HalbachArray objects (one per ball)
            neighbor_range: Number of neighbors on each side to include
                           (default 2 = 4 nearest neighbors total)
        """
        self.halbach_arrays = halbach_arrays
        self.neighbor_range = neighbor_range
        self.n_balls = len(halbach_arrays)
    
    def compute_forces(
        self,
        positions: List[np.ndarray],
        orientations: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """Compute magnetic forces on all balls.
        
        Args:
            positions: List of position vectors [x, y, z] for each ball (m)
            orientations: Optional list of rotation matrices for each ball.
                         If None, assumes all dipoles aligned with z-axis.
        
        Returns:
            List of force vectors [Fx, Fy, Fz] for each ball (N)
        """
        forces = [np.zeros(3) for _ in range(self.n_balls)]
        
        for i in range(self.n_balls):
            # Get dipole moment for ball i (possibly rotated)
            if orientations is not None:
                m_i = orientations[i] @ self.halbach_arrays[i].dipole_moment
            else:
                m_i = self.halbach_arrays[i].dipole_moment
            
            # Sum forces from neighbors
            for j in range(
                max(0, i - self.neighbor_range),
                min(self.n_balls, i + self.neighbor_range + 1)
            ):
                if i == j:
                    continue
                
                # Get dipole moment for ball j
                if orientations is not None:
                    m_j = orientations[j] @ self.halbach_arrays[j].dipole_moment
                else:
                    m_j = self.halbach_arrays[j].dipole_moment
                
                # Position vector from i to j
                r_vec = positions[j] - positions[i]
                
                # Force on i due to j
                # dipole_dipole_force returns force on second dipole due to first.
                F_ij = dipole_dipole_force(m_j, m_i, -r_vec)
                forces[i] += F_ij
        
        return forces
    
    def compute_torques(
        self,
        positions: List[np.ndarray],
        orientations: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """Compute magnetic torques on all balls.
        
        Args:
            positions: List of position vectors [x, y, z] for each ball (m)
            orientations: Optional list of rotation matrices for each ball
        
        Returns:
            List of torque vectors [τx, τy, τz] for each ball (N·m)
        """
        torques = [np.zeros(3) for _ in range(self.n_balls)]
        
        for i in range(self.n_balls):
            # Get dipole moment for ball i
            if orientations is not None:
                m_i = orientations[i] @ self.halbach_arrays[i].dipole_moment
            else:
                m_i = self.halbach_arrays[i].dipole_moment
            
            # Sum torques from neighbors
            for j in range(
                max(0, i - self.neighbor_range),
                min(self.n_balls, i + self.neighbor_range + 1)
            ):
                if i == j:
                    continue
                
                # Get dipole moment for ball j
                if orientations is not None:
                    m_j = orientations[j] @ self.halbach_arrays[j].dipole_moment
                else:
                    m_j = self.halbach_arrays[j].dipole_moment
                
                # Position vector from i to j
                r_vec = positions[j] - positions[i]
                
                # Torque on i due to j
                tau_ij = dipole_dipole_torque(m_i, m_j, r_vec)
                torques[i] += tau_ij
        
        return torques
    
    def compute_stiffness_matrix(
        self,
        equilibrium_positions: List[np.ndarray],
        equilibrium_orientations: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """Compute linearized stiffness matrix for stability analysis.
        
        Computes the 3N × 3N stiffness matrix K where F = -K·δx for
        small displacements from equilibrium.
        
        Args:
            equilibrium_positions: List of equilibrium positions for each ball
            equilibrium_orientations: Optional list of equilibrium orientations
        
        Returns:
            Stiffness matrix (3N × 3N) in N/m
        """
        n_dof = 3 * self.n_balls
        K = np.zeros((n_dof, n_dof))
        
        # Finite difference for stiffness
        delta = 1e-6  # Small displacement for numerical derivative
        
        for i in range(self.n_balls):
            for alpha in range(3):  # x, y, z directions
                # Perturb ball i in direction alpha
                positions_plus = [p.copy() for p in equilibrium_positions]
                positions_minus = [p.copy() for p in equilibrium_positions]
                
                positions_plus[i][alpha] += delta
                positions_minus[i][alpha] -= delta
                
                # Compute forces
                F_plus = self.compute_forces(positions_plus, equilibrium_orientations)
                F_minus = self.compute_forces(positions_minus, equilibrium_orientations)
                
                # Numerical derivative: dF/dx ≈ (F_plus - F_minus) / (2*delta)
                for j in range(self.n_balls):
                    for beta in range(3):
                        row = 3 * j + beta
                        col = 3 * i + alpha
                        dF = F_plus[j][beta] - F_minus[j][beta]
                        K[row, col] = -dF / (2.0 * delta)  # Stiffness = -dF/dx
        
        return K
    
    def compute_total_potential_energy(
        self,
        positions: List[np.ndarray],
        orientations: Optional[List[np.ndarray]] = None
    ) -> float:
        """Compute total magnetic potential energy of the system.
        
        Args:
            positions: List of position vectors for each ball
            orientations: Optional list of rotation matrices
        
        Returns:
            Total potential energy (J)
        """
        U_total = 0.0
        
        for i in range(self.n_balls):
            # Get dipole moment for ball i
            if orientations is not None:
                m_i = orientations[i] @ self.halbach_arrays[i].dipole_moment
            else:
                m_i = self.halbach_arrays[i].dipole_moment
            
            for j in range(i + 1, self.n_balls):
                # Get dipole moment for ball j
                if orientations is not None:
                    m_j = orientations[j] @ self.halbach_arrays[j].dipole_moment
                else:
                    m_j = self.halbach_arrays[j].dipole_moment
                
                # Position vector from i to j
                r_vec = positions[j] - positions[i]
                
                # Potential energy of pair
                U_ij = dipole_dipole_potential(m_i, m_j, r_vec)
                U_total += U_ij
        
        return U_total


def compute_stream_magnetic_stiffness(
    halbach: HalbachArray,
    spacing: float,
    n_neighbors: int = 2
) -> float:
    """Compute effective magnetic stiffness for a uniform stream.
    
    For a stream of identical Halbach arrays with given spacing,
    compute the effective linear stiffness per ball from magnetic
    interactions with neighbors.
    
    Args:
        halbach: HalbachArray configuration for each ball
        spacing: Center-to-center spacing between balls (m)
        n_neighbors: Number of neighbors to include on each side
    
    Returns:
        Effective magnetic stiffness (N/m)
    """
    m = halbach.dipole_magnitude
    
    # For a linear stream with dipoles aligned perpendicular to stream
    # (repulsive configuration), sum contributions from neighbors
    k_total = 0.0
    
    for n in range(1, n_neighbors + 1):
        d = n * spacing
        k_n = compute_linear_stiffness(m, d, alignment='repulsive')
        k_total += 2 * k_n  # Both sides
    
    return k_total

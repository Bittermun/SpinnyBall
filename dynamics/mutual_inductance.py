"""
Mutual inductance model for multi-coil systems.

Implements analytical Neumann integral approximations for mutual inductance
between circular coils in pulsed magnetic systems.
"""

from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional
from scipy.special import ellipk, ellipe


@dataclass
class CoilGeometry:
    """Geometry of a circular coil."""
    radius: float  # m - coil radius
    position: np.ndarray  # m - center position [x, y, z]
    normal: np.ndarray  # unit normal vector [nx, ny, nz]
    turns: int  # number of turns


@dataclass
class MutualInductanceResult:
    """Result of mutual inductance calculation."""
    mutual_inductance: float  # H - mutual inductance
    coupling_coefficient: float  # dimensionless - coupling coefficient k = M / sqrt(L1*L2)
    distance: float  # m - distance between coil centers
    alignment_factor: float  # dimensionless - alignment of coil normals


class MutualInductanceModel:
    """Model for mutual inductance between circular coils."""
    
    def __init__(self, mu0: float = 4*np.pi*1e-7):
        """Initialize mutual inductance model.
        
        Args:
            mu0: Permeability of free space (H/m)
        """
        self.mu0 = mu0
    
    def neumann_integral_circular(
        self,
        coil1: CoilGeometry,
        coil2: CoilGeometry,
        num_points: int = 100
    ) -> float:
        """Compute mutual inductance between two circular coils using Neumann integral.
        
        M = (mu0 / 4pi) * ∮∮ (dl1 · dl2) / |r1 - r2|
        
        For circular coils, this can be expressed in terms of elliptic integrals.
        
        Args:
            coil1: First coil geometry
            coil2: Second coil geometry
            num_points: Number of points for numerical integration (if needed)
        
        Returns:
            Mutual inductance (H)
        """
        # Distance between coil centers
        r1 = coil1.position
        r2 = coil2.position
        d = np.linalg.norm(r2 - r1)
        
        # Coil radii
        R1 = coil1.radius
        R2 = coil2.radius
        
        # Check for coaxial alignment (simplest case)
        if self._is_coaxial(coil1, coil2):
            return self._coaxial_mutual_inductance(R1, R2, d)
        
        # General case: use filament approximation with numerical integration
        return self._filament_neumann_integral(coil1, coil2, num_points)
    
    def _is_coaxial(self, coil1: CoilGeometry, coil2: CoilGeometry, tol: float = 1e-6) -> bool:
        """Check if two coils are coaxial."""
        # Coils are coaxial if their normals are parallel (or anti-parallel)
        # and their centers lie on the same line parallel to the normals
        n1 = coil1.normal / np.linalg.norm(coil1.normal)
        n2 = coil2.normal / np.linalg.norm(coil2.normal)
        
        # Check if normals are parallel
        parallel = np.abs(np.dot(n1, n2)) > 1 - tol
        
        if not parallel:
            return False
        
        # Check if centers lie on the same line
        r_diff = coil2.position - coil1.position
        if np.linalg.norm(r_diff) < tol:
            return True  # Same center
        
        # Check if r_diff is parallel to normals
        r_diff_norm = r_diff / np.linalg.norm(r_diff)
        coaxial = np.abs(np.dot(r_diff_norm, n1)) > 1 - tol
        
        return coaxial
    
    def _coaxial_mutual_inductance(self, R1: float, R2: float, d: float) -> float:
        """Compute mutual inductance for coaxial circular coils.
        
        Uses elliptic integral formula:
        M = mu0 * sqrt(R1*R2) * [(2/k - k) * K(k) - (2/k) * E(k)]
        where k^2 = 4*R1*R2 / [(R1+R2)^2 + d^2]
        
        Args:
            R1: Radius of first coil (m)
            R2: Radius of second coil (m)
            d: Distance between coil centers (m)
        
        Returns:
            Mutual inductance (H)
        """
        # Compute k parameter
        numerator = 4 * R1 * R2
        denominator = (R1 + R2)**2 + d**2
        k_squared = numerator / denominator
        
        # Handle edge cases
        if k_squared >= 1:
            k_squared = 1 - 1e-10  # Avoid singularity
        if k_squared <= 0:
            return 0.0
        
        k = np.sqrt(k_squared)
        
        # Elliptic integrals
        K = ellipk(k_squared)
        E = ellipe(k_squared)
        
        # Mutual inductance formula
        M = self.mu0 * np.sqrt(R1 * R2) * ((2/k - k) * K - (2/k) * E)
        
        return M
    
    def _filament_neumann_integral(
        self,
        coil1: CoilGeometry,
        coil2: CoilGeometry,
        num_points: int
    ) -> float:
        """Compute mutual inductance using filament approximation.
        
        Discretizes coils into filaments and computes Neumann integral numerically.
        
        Args:
            coil1: First coil geometry
            coil2: Second coil geometry
            num_points: Number of points per coil
        
        Returns:
            Mutual inductance (H)
        """
        # Generate points on coil 1
        theta1 = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        points1 = self._generate_coil_points(coil1, theta1)
        
        # Generate points on coil 2
        theta2 = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        points2 = self._generate_coil_points(coil2, theta2)
        
        # Compute tangent vectors
        dl1 = self._compute_tangent_vectors(coil1, theta1)
        dl2 = self._compute_tangent_vectors(coil2, theta2)
        
        # Numerical integration
        M = 0.0
        dtheta = 2*np.pi / num_points
        
        for i in range(num_points):
            for j in range(num_points):
                r_diff = points1[i] - points2[j]
                r_mag = np.linalg.norm(r_diff)
                
                if r_mag < 1e-10:
                    continue  # Avoid singularity
                
                # Dot product of differential length elements
                dl_dot = np.dot(dl1[i], dl2[j])
                
                # Neumann integrand
                integrand = dl_dot / r_mag
                
                M += integrand
        
        # Scale by mu0/4pi and integration step
        M *= (self.mu0 / (4*np.pi)) * (dtheta**2)
        
        return M
    
    def _generate_coil_points(self, coil: CoilGeometry, theta: np.ndarray) -> np.ndarray:
        """Generate points on a circular coil.
        
        Args:
            coil: Coil geometry
            theta: Angular positions
        
        Returns:
            Array of points [num_points, 3]
        """
        # Create local coordinate system
        normal = coil.normal / np.linalg.norm(coil.normal)
        
        # Find two orthogonal vectors in the plane
        if np.abs(normal[0]) < 0.9:
            u = np.cross(normal, np.array([1, 0, 0]))
        else:
            u = np.cross(normal, np.array([0, 1, 0]))
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # Generate points
        points = np.zeros((len(theta), 3))
        for i, th in enumerate(theta):
            points[i] = coil.position + coil.radius * (np.cos(th) * u + np.sin(th) * v)
        
        return points
    
    def _compute_tangent_vectors(self, coil: CoilGeometry, theta: np.ndarray) -> np.ndarray:
        """Compute tangent vectors at points on a circular coil.
        
        Args:
            coil: Coil geometry
            theta: Angular positions
        
        Returns:
            Array of tangent vectors [num_points, 3]
        """
        # Create local coordinate system
        normal = coil.normal / np.linalg.norm(coil.normal)
        
        # Find two orthogonal vectors in the plane
        if np.abs(normal[0]) < 0.9:
            u = np.cross(normal, np.array([1, 0, 0]))
        else:
            u = np.cross(normal, np.array([0, 1, 0]))
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # Tangent vectors are in the direction of increasing theta
        tangents = np.zeros((len(theta), 3))
        for i, th in enumerate(theta):
            tangents[i] = coil.radius * (-np.sin(th) * u + np.cos(th) * v)
        
        return tangents
    
    def compute_coupling_coefficient(
        self,
        M: float,
        L1: float,
        L2: float
    ) -> float:
        """Compute coupling coefficient.
        
        k = M / sqrt(L1 * L2)
        
        Args:
            M: Mutual inductance (H)
            L1: Self-inductance of coil 1 (H)
            L2: Self-inductance of coil 2 (H)
        
        Returns:
            Coupling coefficient (dimensionless)
        """
        if L1 <= 0 or L2 <= 0:
            return 0.0
        
        k = M / np.sqrt(L1 * L2)
        
        # Clamp to valid range [-1, 1]
        k = max(-1.0, min(1.0, k))
        
        return k
    
    def compute_alignment_factor(
        self,
        coil1: CoilGeometry,
        coil2: CoilGeometry
    ) -> float:
        """Compute alignment factor between coil normals.
        
        Returns 1.0 for perfectly aligned, 0.0 for orthogonal.
        
        Args:
            coil1: First coil geometry
            coil2: Second coil geometry
        
        Returns:
            Alignment factor (dimensionless)
        """
        n1 = coil1.normal / np.linalg.norm(coil1.normal)
        n2 = coil2.normal / np.linalg.norm(coil2.normal)
        
        return np.abs(np.dot(n1, n2))
    
    def full_analysis(
        self,
        coil1: CoilGeometry,
        coil2: CoilGeometry,
        L1: float,
        L2: float,
        num_points: int = 100
    ) -> MutualInductanceResult:
        """Perform full mutual inductance analysis.
        
        Args:
            coil1: First coil geometry
            coil2: Second coil geometry
            L1: Self-inductance of coil 1 (H)
            L2: Self-inductance of coil 2 (H)
            num_points: Number of points for numerical integration
        
        Returns:
            MutualInductanceResult with all computed quantities
        """
        # Compute mutual inductance
        M = self.neumann_integral_circular(coil1, coil2, num_points)
        
        # Compute coupling coefficient
        k = self.compute_coupling_coefficient(M, L1, L2)
        
        # Compute distance
        distance = np.linalg.norm(coil2.position - coil1.position)
        
        # Compute alignment factor
        alignment = self.compute_alignment_factor(coil1, coil2)
        
        return MutualInductanceResult(
            mutual_inductance=M,
            coupling_coefficient=k,
            distance=distance,
            alignment_factor=alignment,
        )


def create_circular_coil(
    radius: float,
    center: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    turns: int = 1
) -> CoilGeometry:
    """Create a circular coil geometry.
    
    Args:
        radius: Coil radius (m)
        center: Center position [x, y, z] (m)
        normal: Normal vector [nx, ny, nz]
        turns: Number of turns
    
    Returns:
        CoilGeometry
    """
    return CoilGeometry(
        radius=radius,
        position=np.array(center),
        normal=np.array(normal),
        turns=turns,
    )

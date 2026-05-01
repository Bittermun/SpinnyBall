"""
Permanent magnet model for SmCo/NdFeB magnetic spring interactions.

Unlike superconductor flux-pinning (Bean-London), permanent magnets
provide restoring force via dipole-dipole interaction:

F(x) = -(3*mu0*m1*m2)/(2*pi*d^4) * f(x/d)

where m1, m2 are magnetic moments, d is equilibrium gap,
and f(x/d) is a geometry-dependent function.

For a Halbach array configuration:
k_eff = B_r^2 * A / (mu0 * d)

where B_r is remanence, A is pole face area, d is gap.
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional


@dataclass
class PermanentMagnetGeometry:
    """Geometry parameters for permanent magnet configuration."""
    # Pole face area (m^2)
    pole_face_area: float
    
    # Equilibrium gap between magnets (m)
    equilibrium_gap: float
    
    # Magnet thickness (m)
    thickness: Optional[float] = None
    
    # Configuration type: 'dipole', 'halbach', 'axial'
    config_type: str = 'axial'


class PermanentMagnetModel:
    """Magnetic spring model for permanent magnet (SmCo/NdFeB) interactions.
    
    Unlike superconductor flux-pinning (Bean-London), permanent magnets
    provide restoring force via dipole-dipole interaction:
    
    F(x) = -(3*mu0*m1*m2)/(2*pi*d^4) * f(x/d)
    
    where m1, m2 are magnetic moments, d is equilibrium gap,
    and f(x/d) is a geometry-dependent function.
    
    For a Halbach array configuration:
    k_eff = B_r^2 * A / (mu0 * d)
    
    where B_r is remanence, A is pole face area, d is gap.
    """
    
    def __init__(self, material_props: Dict[str, Any], geometry: PermanentMagnetGeometry):
        """Initialize permanent magnet model.
        
        Args:
            material_props: Material properties dictionary with:
                - remanence: Remanent flux density B_r (T)
                - coercivity: Coercive field H_c (A/m)
                - alpha_Br: Thermal coefficient of remanence (/K)
            geometry: PermanentMagnetGeometry with physical dimensions
        """
        self.B_r = material_props.get('remanence', 1.1)  # T
        self.H_c = material_props.get('coercivity', 700e3)  # A/m
        self.alpha_Br = material_props.get('alpha_Br', -0.0003)  # /K
        
        self.area = geometry.pole_face_area  # m^2
        self.gap = geometry.equilibrium_gap  # m
        self.thickness = geometry.thickness or 0.01  # m
        self.config_type = geometry.config_type
        
        # Vacuum permeability
        self.mu0 = 1.25663706212e-6  # H/m (N/A²)
        
        # Reference temperature (20°C = 293K)
        self.T_ref = 293.0  # K
        
        # Compute baseline stiffness at reference temperature
        self.k_0 = self._compute_baseline_stiffness()
    
    def _compute_baseline_stiffness(self) -> float:
        """Compute baseline magnetic spring stiffness at reference temperature.
        
        For axial magnet configuration:
        k_eff ≈ (B_r^2 * A) / (mu0 * d)
        
        Returns:
            Stiffness (N/m)
        """
        if self.gap <= 0:
            raise ValueError(f"Gap must be > 0, got {self.gap}")
        
        if self.config_type == 'halbach':
            # Halbach array focusing
            # k_eff = B_r^2 * A / (mu0 * d)
            k_eff = (self.B_r ** 2) * self.area / (self.mu0 * self.gap)
        elif self.config_type == 'dipole':
            # Dipole-dipole interaction
            # Magnetic moment m = B_r * V / mu0
            volume = self.area * self.thickness
            m = self.B_r * volume / self.mu0
            
            # Stiffness from dipole interaction
            # k = dF/dx ≈ (3 * mu0 * m^2) / (pi * d^5)
            if self.gap <= 0:
                return 0.0
            k_eff = (3 * self.mu0 * m ** 2) / (np.pi * self.gap ** 5)
        else:  # 'axial'
            # Axial magnet configuration (most common)
            # Simplified model: k_eff ≈ (B_r^2 * A) / (2 * mu0 * d)
            k_eff = (self.B_r ** 2) * self.area / (2 * self.mu0 * self.gap)
        
        return k_eff
    
    def compute_stiffness(self, displacement: float, temperature: float) -> float:
        """Temperature-dependent stiffness for permanent magnets.
        
        B_r(T) = B_r(20C) * [1 + alpha_Br * (T - 293)]
        
        where alpha_Br = -0.03%/K for SmCo, -0.12%/K for NdFeB
        
        Args:
            displacement: Displacement from equilibrium (m)
            temperature: Operating temperature (K)
        
        Returns:
            Temperature-dependent stiffness (N/m)
        """
        # Temperature correction for remanence
        delta_T = temperature - self.T_ref
        B_r_T = self.B_r * (1.0 + self.alpha_Br * delta_T)
        
        # Ensure B_r doesn't go negative
        B_r_T = max(B_r_T, 0.0)
        
        # Recompute stiffness with temperature-corrected B_r
        if self.config_type == 'halbach':
            k_eff = (B_r_T ** 2) * self.area / (self.mu0 * self.gap)
        elif self.config_type == 'dipole':
            volume = self.area * self.thickness
            m = B_r_T * volume / self.mu0
            if self.gap <= 0:
                return 0.0
            k_eff = (3 * self.mu0 * m ** 2) / (np.pi * self.gap ** 5)
        else:  # 'axial'
            k_eff = (B_r_T ** 2) * self.area / (2 * self.mu0 * self.gap)
        
        # Apply displacement-dependent saturation for large displacements
        # Stiffness decreases as displacement approaches gap size
        if abs(displacement) > 0.1 * self.gap:
            saturation_factor = 1.0 / (1.0 + (abs(displacement) / self.gap) ** 2)
            k_eff *= saturation_factor
        
        return k_eff
    
    def compute_force(self, displacement: float, temperature: float) -> float:
        """Restoring force as function of displacement from equilibrium.
        
        For small displacements: F(x) ≈ -k(T) * x
        
        For larger displacements, includes nonlinear terms.
        
        Args:
            displacement: Displacement from equilibrium (m)
            temperature: Operating temperature (K)
        
        Returns:
            Restoring force (N), negative sign indicates restoring direction
        """
        # Get temperature-dependent stiffness
        k_T = self.compute_stiffness(displacement, temperature)
        
        # Linear restoring force for small displacements
        F_linear = -k_T * displacement
        
        # Add nonlinear correction for larger displacements
        # Duffing-type nonlinearity: F = -k*x - k3*x^3
        if self.config_type == 'axial':
            # Hardening spring behavior for axial magnets
            k3 = k_T / (self.gap ** 2)  # Cubic stiffness coefficient
            F_nonlinear = -k3 * displacement ** 3
            return F_linear + F_nonlinear
        else:
            return F_linear
    
    def get_temperature_sensitivity(self) -> float:
        """Get fractional change in stiffness per degree K.
        
        Since k ∝ B_r^2 and B_r(T) = B_r0 * (1 + alpha_Br * ΔT):
        dk/k ≈ 2 * alpha_Br * ΔT
        
        Returns:
            Fractional sensitivity (%/K)
        """
        return 2 * self.alpha_Br * 100  # Convert to %/K
    
    def compare_to_bean_london(self, temperature: float) -> Dict[str, Any]:
        """Compare PM behavior to Bean-London superconductor model.
        
        Key differences:
        - PM stiffness DECREASES with temperature (alpha_Br < 0)
        - SC stiffness INCREASES as T decreases below Tc
        
        Args:
            temperature: Operating temperature (K)
        
        Returns:
            Dictionary with comparison metrics
        """
        k_pm = self.compute_stiffness(0.0, temperature)
        k_pm_ref = self.k_0
        
        # Fractional change from reference
        delta_k_pm = (k_pm - k_pm_ref) / k_pm_ref if k_pm_ref > 0 else 0.0
        
        # Temperature sensitivity
        temp_sensitivity = self.get_temperature_sensitivity()
        
        return {
            'stiffness_N_m': k_pm,
            'stiffness_reference_N_m': k_pm_ref,
            'fractional_change': delta_k_pm,
            'temp_sensitivity_pct_per_K': temp_sensitivity,
            'behavior': 'decreases_with_temperature',
            'comparison_note': 'PM stiffness decreases with T (alpha_Br < 0), '
                              'unlike SC which increases as T decreases below Tc',
        }


def create_pm_model_from_material_name(
    material_name: str,
    geometry: PermanentMagnetGeometry
) -> PermanentMagnetModel:
    """Create a PM model from canonical material name.
    
    Args:
        material_name: One of 'SmCo', 'NdFeB'
        geometry: Magnet geometry
    
    Returns:
        Configured PermanentMagnetModel
    
    Raises:
        ValueError: If material not found
    """
    from params.canonical_values import MATERIAL_PROPERTIES
    
    if material_name not in MATERIAL_PROPERTIES:
        raise ValueError(f"Unknown material: {material_name}. "
                        f"Available: {list(MATERIAL_PROPERTIES.keys())}")
    
    mat_props = MATERIAL_PROPERTIES[material_name]
    
    # Extract values from nested dict format
    props_dict = {}
    for key, value in mat_props.items():
        if isinstance(value, dict):
            props_dict[key] = value.get('value')
        else:
            props_dict[key] = value
    
    return PermanentMagnetModel(props_dict, geometry)

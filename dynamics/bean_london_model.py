"""
Bean-London critical-state model for flux-pinning.

Models the critical state where current density equals J_c everywhere
in the superconductor, creating a magnetization that opposes field changes.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional

from dynamics.gdBCO_material import GdBCOMaterial


@dataclass
class BeanLondonState:
    """State for Bean-London model (history-dependent)."""
    magnetization: np.ndarray  # Magnetization history
    previous_field: np.ndarray  # Previous magnetic field values
    penetration_depth: float  # Current flux penetration depth


class BeanLondonModel:
    """Bean-London critical-state model for flux-pinning.
    
    Models the critical state where current density equals J_c everywhere
    in the superconductor, creating a magnetization that opposes field changes.
    """
    
    def __init__(self, material: GdBCOMaterial, geometry: dict):
        """Initialize Bean-London model.
        
        Args:
            material: GdBCO material properties
            geometry: Dictionary with geometric parameters
                - thickness: Superconductor thickness (m)
                - width: Tape width (m)
                - length: Tape length (m)
        
        Raises:
            ValueError: If geometry parameters are invalid
        """
        # Validate geometry
        if geometry.get('thickness', 0) <= 0:
            raise ValueError(f"thickness must be > 0, got {geometry.get('thickness')}")
        if geometry.get('width', 0) <= 0:
            raise ValueError(f"width must be > 0, got {geometry.get('width')}")
        if geometry.get('length', 0) <= 0:
            raise ValueError(f"length must be > 0, got {geometry.get('length')}")
        
        self.material = material
        self.geometry = geometry
        self.state = BeanLondonState(
            magnetization=np.array([0.0]),
            previous_field=np.array([0.0]),
            penetration_depth=0.0,
        )
        
    def compute_pinning_force(self, displacement: float, B_field: float, 
                            temperature: float) -> float:
        """Compute flux-pinning force from Bean-London model.
        
        F_pin = ∫(J × B) dV
        
        For simplified geometry:
        F_pin = J_c(B, T) * B_field * volume * f(displacement)
        
        where f(displacement) models force saturation at large displacements.
        
        Args:
            displacement: Relative displacement (m)
            B_field: Magnetic flux density (T)
            temperature: Temperature (K)
        
        Returns:
            Pinning force (N)
        """
        # Get critical current density
        Jc = self.material.critical_current_density(B_field, temperature)
        
        # Compute penetration depth (increases with displacement)
        max_penetration = self.geometry["thickness"] / 2.0
        if max_penetration <= 0:
            raise ValueError("max_penetration must be > 0")
        
        penetration_depth = min(
            abs(displacement) / max_penetration * max_penetration,
            max_penetration
        )
        self.state.penetration_depth = penetration_depth  # Update state
        
        # Effective volume with critical current
        volume = self.geometry["thickness"] * self.geometry["width"] * \
                 self.geometry["length"]
        if max_penetration > 0:
            effective_volume = volume * (penetration_depth / max_penetration)
        else:
            effective_volume = 0.0
        
        # Pinning force density: f_p = J_c × B
        force_density = Jc * B_field
        
        # Total pinning force (with saturation)
        F_pin = force_density * effective_volume
        
        # Saturation factor (force doesn't increase indefinitely)
        if max_penetration > 0:
            saturation_factor = np.tanh(abs(displacement) / (max_penetration * 0.1))
            F_pin *= saturation_factor
        
        # Direction opposes displacement
        F_pin *= -np.sign(displacement)
        
        return F_pin
    
    def update_magnetization(self, B_field: float, temperature: float):
        """Update magnetization history (hysteresis).
        
        The magnetization changes when the external field changes,
        with the rate limited by flux creep.
        
        Args:
            B_field: Current magnetic flux density (T)
            temperature: Temperature (K)
        """
        # Simplified hysteresis model
        delta_B = B_field - self.state.previous_field[-1]
        
        # Magnetization change proportional to field change
        Jc = self.material.critical_current_density(B_field, temperature)
        delta_M = -Jc * delta_B  # Opposes field change
        
        # Update history
        self.state.magnetization = np.append(self.state.magnetization, 
                                            self.state.magnetization[-1] + delta_M)
        self.state.previous_field = np.append(self.state.previous_field, B_field)
        
        # Keep history manageable
        if len(self.state.magnetization) > 100:
            self.state.magnetization = self.state.magnetization[-100:]
            self.state.previous_field = self.state.previous_field[-100:]
    
    def get_stiffness(self, displacement: float, B_field: float, 
                     temperature: float) -> float:
        """Compute effective stiffness k_fp = -dF_pin/dx using analytical derivative.
        
        For x > 0: F_pin = -a * x * tanh(b*x)
        where a = Jc * B * volume / max_penetration
              b = 1/(max_penetration * 0.1)
        
        Derivative: dF/dx = -a * [tanh(b*x) + b*x * sech²(b*x)]
        Stiffness: k = -dF/dx = a * [tanh(b*x) + b*x * sech²(b*x)]
        
        Args:
            displacement: Relative displacement (m)
            B_field: Magnetic flux density (T)
            temperature: Temperature (K)
        
        Returns:
            Effective stiffness (N/m)
        """
        # Get critical current density
        Jc = self.material.critical_current_density(B_field, temperature)
        
        # Geometry parameters
        volume = self.geometry["thickness"] * self.geometry["width"] * self.geometry["length"]
        max_penetration = self.geometry["thickness"] / 2.0
        
        # Analytical derivative parameters
        a = Jc * B_field * volume / max_penetration
        b = 1.0 / (max_penetration * 0.1)
        x = abs(displacement)
        
        # Handle edge case for very small displacements
        if x < 1e-15:
            # For x → 0, tanh(b*x) → b*x, sech(b*x) → 1
            # Stiffness → a * [b*x + b*x * 1] = 2*a*b*x
            stiffness = 2.0 * a * b * x
        elif b * x > 20:
            # For large b*x, tanh(b*x) → 1, sech(b*x) → 0
            # Stiffness → a * [1 + 0] = a
            stiffness = a
        else:
            # Analytical derivative: k = a * [tanh(b*x) + b*x * sech²(b*x)]
            tanh_bx = np.tanh(b * x)
            sech_bx = 1.0 / np.cosh(b * x)
            stiffness = a * (tanh_bx + b * x * sech_bx**2)
        
        return stiffness

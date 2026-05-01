"""
GdBCO superconductor material properties.

Implements critical current density J_c(B, T) using Bean-London model
for flux-pinning stiffness calculations.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class GdBCOProperties:
    """Material properties for GdBCO superconductor."""
    # Critical parameters
    Tc: float = 92.0  # K (critical temperature)
    Jc0: float = 3e10  # A/m² (critical current density at 0K, 0T)
    n_exponent: float = 1.5  # Temperature dependence exponent
    
    # Magnetic field dependence parameters
    B0: float = 0.1  # T (characteristic field)
    alpha: float = 0.5  # Field dependence exponent
    
    # Physical properties
    density: float = 6380.0  # kg/m³
    specific_heat: float = 180.0  # J/kg/K at 77K (NOT room temperature)
    thermal_conductivity: float = 3.0  # W/m/K at 77K
    
    # Geometry (for coated conductor)
    thickness: float = 1e-6  # m (1 μm superconducting layer)
    width: float = 0.012  # m (12 mm wide tape)
    
    # Safety limits
    max_field_gradient: float = 50.0  # T/m - maximum safe field gradient
    max_current_density: float = 5e10  # A/m² - absolute maximum current density
    safety_margin: float = 0.8  # Safety margin (80% of theoretical limits)


class GdBCOMaterial:
    """GdBCO superconductor material model."""
    
    def __init__(self, props: GdBCOProperties):
        """Initialize material model.
        
        Args:
            props: GdBCOProperties with material parameters
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if props.Tc <= 0:
            raise ValueError(f"Tc must be > 0, got {props.Tc}")
        if props.Jc0 <= 0:
            raise ValueError(f"Jc0 must be > 0, got {props.Jc0}")
        if props.B0 <= 0:
            raise ValueError(f"B0 must be > 0, got {props.B0}")
        if props.thickness <= 0:
            raise ValueError(f"thickness must be > 0, got {props.thickness}")
        if props.width <= 0:
            raise ValueError(f"width must be > 0, got {props.width}")
        
        self.props = props
        
    def critical_current_density(self, B: float, T: float) -> float:
        """Compute critical current density J_c(B, T) using Bean-London model.
        
        J_c(B, T) = J_c0 * (1 - T/T_c)^n * f(B)
        
        where f(B) = 1 / (1 + (B/B0)^α)
        
        Args:
            B: Magnetic flux density (T)
            T: Temperature (K)
        
        Returns:
            Critical current density (A/m²)
        """
        # Temperature dependence
        if T >= self.props.Tc:
            return 0.0  # Normal state
        
        if self.props.Tc <= 0:
            raise ValueError("Tc must be > 0")
        temp_factor = (1.0 - T / self.props.Tc) ** self.props.n_exponent
        
        # Magnetic field dependence
        if self.props.B0 <= 0:
            raise ValueError("B0 must be > 0")
        field_factor = 1.0 / (1.0 + (B / self.props.B0) ** self.props.alpha)
        
        return self.props.Jc0 * temp_factor * field_factor
    
    def critical_current(self, B: float, T: float) -> float:
        """Compute critical current I_c(B, T).
        
        I_c = J_c * cross_sectional_area
        
        Args:
            B: Magnetic flux density (T)
            T: Temperature (K)
        
        Returns:
            Critical current (A)
        """
        Jc = self.critical_current_density(B, T)
        area = self.props.thickness * self.props.width
        return Jc * area
    
    def check_field_gradient(self, field_gradient: float) -> tuple[bool, str]:
        """Check if field gradient is within safe limits.
        
        Args:
            field_gradient: Field gradient (T/m)
        
        Returns:
            Tuple of (is_safe, message)
        """
        safe_limit = self.props.max_field_gradient * self.props.safety_margin
        
        if field_gradient > self.props.max_field_gradient:
            return False, f"Field gradient {field_gradient:.1f} T/m exceeds absolute limit {self.props.max_field_gradient:.1f} T/m"
        elif field_gradient > safe_limit:
            return False, f"Field gradient {field_gradient:.1f} T/m exceeds safety limit {safe_limit:.1f} T/m (80% of max)"
        else:
            return True, f"Field gradient {field_gradient:.1f} T/m within safe limits (< {safe_limit:.1f} T/m)"
    
    def check_current_density(self, current_density: float) -> tuple[bool, str]:
        """Check if current density is within safe limits.
        
        Args:
            current_density: Current density (A/m²)
        
        Returns:
            Tuple of (is_safe, message)
        """
        safe_limit = self.props.max_current_density * self.props.safety_margin
        
        if current_density > self.props.max_current_density:
            return False, f"Current density {current_density:.2e} A/m² exceeds absolute limit {self.props.max_current_density:.2e} A/m²"
        elif current_density > safe_limit:
            return False, f"Current density {current_density:.2e} A/m² exceeds safety limit {safe_limit:.2e} A/m² (80% of max)"
        else:
            return True, f"Current density {current_density:.2e} A/m² within safe limits (< {safe_limit:.2e} A/m²)"
    
    def apply_fringe_correction(self, B_field: float, distance_from_coil: float) -> float:
        """Apply fringe field correction factor.
        
        Fringe fields decay with distance from the coil center.
        Simple exponential decay model: B_corrected = B * exp(-distance/characteristic_length)
        
        Args:
            B_field: Nominal field at coil center (T)
            distance_from_coil: Distance from coil center (m)
        
        Returns:
            Corrected field (T)
        """
        # Characteristic length approximated as coil radius
        characteristic_length = self.props.width / 2
        
        # Fringe correction factor
        correction = np.exp(-distance_from_coil / characteristic_length)
        
        return B_field * correction
    
    def compute_thermal_degradation_factor(self, temperature: float) -> float:
        """Compute thermal degradation factor for J_c due to temperature rise.
        
        Models the additional degradation beyond the standard J_c(B,T) model
        when temperature approaches or exceeds the critical temperature.
        
        Args:
            temperature: Current temperature (K)
        
        Returns:
            Degradation factor (0-1), where 1 = no degradation, 0 = complete loss
        """
        if temperature >= self.props.Tc:
            return 0.0  # Complete loss above Tc
        
        # Temperature margin from critical temperature
        T_margin = self.props.Tc - temperature
        
        # Degradation increases as temperature approaches Tc
        # Use 10% margin for degradation onset (more conservative)
        critical_margin = 0.1 * self.props.Tc
        
        if T_margin >= critical_margin:
            return 1.0  # No degradation in safe operating range
        
        degradation_factor = (T_margin / critical_margin)**2
        return max(0.0, min(1.0, degradation_factor))
    
    def critical_current_with_thermal_feedback(
        self,
        B: float,
        T: float,
        switching_power: float = 0.0,
        thermal_mass: float = 10.0,
        dt: float = 0.01
    ) -> tuple[float, float, dict]:
        """Compute critical current with thermal feedback from switching losses.
        
        Args:
            B: Magnetic flux density (T)
            T: Initial temperature (K)
            switching_power: Switching power dissipation (W)
            thermal_mass: Thermal mass (J/K)
            dt: Time step (s)
        
        Returns:
            Tuple of (I_c, T_updated, feedback_dict)
        """
        # Base critical current
        I_c_base = self.critical_current(B, T)
        
        # Apply thermal degradation
        degradation = self.compute_thermal_degradation_factor(T)
        I_c_degraded = I_c_base * degradation
        
        # Update temperature due to switching losses
        if switching_power > 0 and thermal_mass > 0:
            dT = switching_power * dt / thermal_mass
            T_updated = T + dT
        else:
            T_updated = T
        
        # Recompute degradation at updated temperature
        degradation_updated = self.compute_thermal_degradation_factor(T_updated)
        I_c_final = I_c_base * degradation_updated
        
        feedback_dict = {
            'I_c_base_A': I_c_base,
            'degradation_initial': degradation,
            'I_c_degraded_A': I_c_degraded,
            'temperature_rise_K': T_updated - T,
            'T_updated_K': T_updated,
            'degradation_updated': degradation_updated,
            'I_c_final_A': I_c_final,
        }
        
        return I_c_final, T_updated, feedback_dict

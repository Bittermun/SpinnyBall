"""
GdBCO superconductor material properties.

Implements critical current density J_c(B, T) using Bean-London model
for flux-pinning stiffness calculations.
"""

from dataclasses import dataclass


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
    density: float = 6300.0  # kg/m³
    specific_heat: float = 500.0  # J/kg/K at 77K
    thermal_conductivity: float = 10.0  # W/m/K at 77K
    
    # Geometry (for coated conductor)
    thickness: float = 1e-6  # m (1 μm superconducting layer)
    width: float = 0.012  # m (12 mm wide tape)


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

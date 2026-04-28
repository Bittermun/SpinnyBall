"""
Cryocooler performance model for thermal management.

Implements temperature-dependent cooling power curves for cryogenic cooling
of GdBCO superconductors in the anchor system.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CryocoolerSpecs:
    """Cryocooler performance specifications."""
    # Cooling power curve parameters (W vs temperature)
    cooling_power_at_70k: float  # W
    cooling_power_at_80k: float  # W
    cooling_power_at_90k: float  # W
    
    # Input power parameters
    input_power_at_70k: float  # W
    input_power_at_80k: float  # W
    input_power_at_90k: float  # W
    
    # Thermal properties
    cooldown_time: float  # s (from 300K to 77K)
    warmup_time: float  # s (from 77K to 300K during quench)
    
    # Physical properties
    mass: float  # kg
    volume: float  # m³
    vibration_amplitude: float  # m (microphonics)


class CryocoolerModel:
    """Cryocooler performance model with temperature-dependent cooling power."""
    
    def __init__(self, specs: CryocoolerSpecs):
        """Initialize cryocooler model.
        
        Args:
            specs: CryocoolerSpecs with performance parameters
        """
        self.specs = specs
        # Fit cooling power curve: P_cool(T) = a*T² + b*T + c
        self._fit_cooling_curve()
        
    def _fit_cooling_curve(self):
        """Fit quadratic curve to cooling power data."""
        T = np.array([70.0, 80.0, 90.0])
        P = np.array([
            self.specs.cooling_power_at_70k,
            self.specs.cooling_power_at_80k,
            self.specs.cooling_power_at_90k,
        ])
        # Quadratic fit: P = a*T² + b*T + c
        coeffs = np.polyfit(T, P, 2)
        self.cooling_coeffs = coeffs
        
    def cooling_power(self, temperature: float) -> float:
        """Compute cooling power at given temperature.
        
        Args:
            temperature: Current temperature (K)
        
        Returns:
            Cooling power (W)
        """
        if temperature < 70.0:
            return self.specs.cooling_power_at_70k
        elif temperature > 90.0:
            return 0.0  # No cooling above 90K (quench range)
        else:
            T = temperature
            a, b, c = self.cooling_coeffs
            return a * T**2 + b * T + c
    
    def input_power(self, temperature: float) -> float:
        """Compute input power at given temperature.
        
        Uses piecewise linear interpolation between known data points.
        
        Args:
            temperature: Current temperature (K)
        
        Returns:
            Input power (W)
        """
        T = temperature
        if T <= 70.0:
            return self.specs.input_power_at_70k
        elif T >= 90.0:
            return self.specs.input_power_at_90k
        elif T <= 80.0:
            # Linear interpolation between 70K and 80K
            t = (T - 70.0) / (80.0 - 70.0)
            return (1 - t) * self.specs.input_power_at_70k + \
                   t * self.specs.input_power_at_80k
        else:
            # Linear interpolation between 80K and 90K
            t = (T - 80.0) / (90.0 - 80.0)
            return (1 - t) * self.specs.input_power_at_80k + \
                   t * self.specs.input_power_at_90k
    
    def cop(self, temperature: float) -> float:
        """Compute coefficient of performance (cooling power / input power).
        
        Args:
            temperature: Current temperature (K)
        
        Returns:
            Coefficient of performance (dimensionless)
        """
        p_cool = self.cooling_power(temperature)
        p_in = self.input_power(temperature)
        return p_cool / p_in if p_in > 0 else 0.0


# Default specifications (Thales LPT9310 series - placeholder values)
DEFAULT_CRYOCOOLER_SPECS = CryocoolerSpecs(
    cooling_power_at_70k=5.0,  # W
    cooling_power_at_80k=8.0,  # W
    cooling_power_at_90k=12.0,  # W
    input_power_at_70k=50.0,  # W
    input_power_at_80k=60.0,  # W
    input_power_at_90k=80.0,  # W
    cooldown_time=3600.0,  # 1 hour
    warmup_time=60.0,  # 1 minute (quench is fast)
    mass=5.0,  # kg
    volume=0.01,  # m³
    vibration_amplitude=1e-6,  # m
)

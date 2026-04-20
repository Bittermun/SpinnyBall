"""
Lumped-parameter thermal model for anchor system.

Simplified 2-node model (stator + rotor) using explicit Euler integration.
Models radiative cooling and conductive heat transfer between components.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class LumpedThermalParams:
    """Parameters for lumped-parameter thermal model."""
    # Stator (GdBCO superconductor)
    stator_mass: float = 10.0  # kg
    stator_specific_heat: float = 500.0  # J/kg/K at 77K
    stator_surface_area: float = 0.1  # m²
    stator_emissivity: float = 0.1  # Low emissivity for superconductor

    # Rotor (magnetic bearing)
    rotor_mass: float = 5.0  # kg
    rotor_specific_heat: float = 400.0  # J/kg/K (aluminum at 77K)
    rotor_surface_area: float = 0.05  # m²
    rotor_emissivity: float = 0.2

    # Thermal coupling
    shaft_conductance: float = 10.0  # W/K (conductive path through shaft)

    # Operating conditions
    ambient_temp: float = 4.0  # K (deep space)
    initial_temp: float = 77.0  # K (operating temperature)


class LumpedThermalModel:
    """Lumped-parameter thermal model with 2 nodes (stator + rotor)."""

    def __init__(self, params: LumpedThermalParams, dt: float = 0.01):
        """Initialize lumped thermal model.

        Args:
            params: LumpedThermalParams
            dt: Time step (s)
        
        Raises:
            ValueError: If dt <= 0 or any parameter is invalid
        """
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        
        # Validate parameters
        if params.stator_mass <= 0:
            raise ValueError(f"stator_mass must be > 0, got {params.stator_mass}")
        if params.stator_specific_heat <= 0:
            raise ValueError(f"stator_specific_heat must be > 0, got {params.stator_specific_heat}")
        if params.rotor_mass <= 0:
            raise ValueError(f"rotor_mass must be > 0, got {params.rotor_mass}")
        if params.rotor_specific_heat <= 0:
            raise ValueError(f"rotor_specific_heat must be > 0, got {params.rotor_specific_heat}")
        
        self.params = params
        self.dt = dt

        # State: [T_stator, T_rotor]
        self.T_stator = params.initial_temp
        self.T_rotor = params.initial_temp

    def step(self, heat_sources: dict[str, float]) -> dict:
        """Step thermal model using explicit Euler integration.

        Args:
            heat_sources: Dictionary with 'stator' and 'rotor' heat input (W)

        Returns:
            Dictionary with updated temperatures and heat flows
        """
        # Extract heat inputs
        Q_stator = heat_sources.get('stator', 0.0)
        Q_rotor = heat_sources.get('rotor', 0.0)

        # Compute radiative loss (Stefan-Boltzmann)
        sigma = 5.67e-8
        P_rad_stator = self.params.stator_emissivity * sigma * \
                       self.params.stator_surface_area * \
                       (self.T_stator**4 - self.params.ambient_temp**4)
        P_rad_rotor = self.params.rotor_emissivity * sigma * \
                      self.params.rotor_surface_area * \
                      (self.T_rotor**4 - self.params.ambient_temp**4)

        # Compute conductive heat flow between stator and rotor
        P_cond = self.params.shaft_conductance * (self.T_rotor - self.T_stator)

        # Net heat flow
        Q_net_stator = Q_stator - P_rad_stator + P_cond
        Q_net_rotor = Q_rotor - P_rad_rotor - P_cond

        # Temperature change (Euler integration)
        stator_thermal_mass = self.params.stator_mass * self.params.stator_specific_heat
        rotor_thermal_mass = self.params.rotor_mass * self.params.rotor_specific_heat
        
        if stator_thermal_mass <= 0:
            raise ValueError("stator thermal mass must be > 0")
        if rotor_thermal_mass <= 0:
            raise ValueError("rotor thermal mass must be > 0")
        
        dT_stator = Q_net_stator * self.dt / stator_thermal_mass
        dT_rotor = Q_net_rotor * self.dt / rotor_thermal_mass

        # Update temperatures
        self.T_stator += dT_stator
        self.T_rotor += dT_rotor

        return {
            'T_stator': self.T_stator,
            'T_rotor': self.T_rotor,
            'P_rad_stator': P_rad_stator,
            'P_rad_rotor': P_rad_rotor,
            'P_cond': P_cond,
            'Q_net_stator': Q_net_stator,
            'Q_net_rotor': Q_net_rotor,
        }

    def get_temperatures(self) -> np.ndarray:
        """Get current temperatures as array."""
        return np.array([self.T_stator, self.T_rotor])

    def reset(self):
        """Reset to initial conditions."""
        self.T_stator = self.params.initial_temp
        self.T_rotor = self.params.initial_temp

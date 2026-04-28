"""
Thermal model for packet temperature dynamics.

Implements radiative cooling and thermal limit checking for packets
in the mass-stream system.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from dynamics.lumped_thermal import LumpedThermalModel, LumpedThermalParams


@dataclass
class ThermalLimits:
    """Thermal safety limits for packets and nodes."""
    max_packet_temp: float = 450.0  # K - maximum packet temperature
    max_node_temp: float = 400.0  # K - maximum node temperature
    min_temp: float = 70.0  # K - minimum operating temperature (cryogenic range)


def update_temperature_euler(
    temperature: float,
    mass: float,
    radius: float,
    emissivity: float,
    specific_heat: float,
    dt: float,
    ambient_temp: float = 4.0,  # K - deep space temperature
    stefan_boltzmann: float = 5.67e-8,  # W/m²/K⁴
    solar_flux: float = 0.0,  # W/m² - solar heating flux
    eddy_heating_power: float = 0.0,  # W - eddy-current heating from drag
) -> float:
    """
    Update packet temperature using Euler integration with radiative cooling.
    
    Models radiative heat transfer: P = εσA(T⁴ - T_ambient⁴)
    Temperature change: dT/dt = (P_solar - P_rad)/(mc)
    
    Args:
        temperature: Current temperature (K)
        mass: Packet mass (kg)
        radius: Packet radius (m)
        emissivity: Surface emissivity (0-1)
        specific_heat: Specific heat capacity (J/kg/K)
        dt: Time step (s)
        ambient_temp: Ambient temperature (K), default 4K for deep space
        stefan_boltzmann: Stefan-Boltzmann constant (W/m²/K⁴)
        solar_flux: Solar heating flux (W/m²), default 0
        eddy_heating_power: Eddy-current heating power from drag (W), default 0
    
    Returns:
        Updated temperature (K)
    """
    # Surface area (assuming spherical packet)
    surface_area = 4 * np.pi * radius**2
    
    # Validate eddy_heating_power
    if eddy_heating_power < 0:
        raise ValueError(f"eddy_heating_power must be >= 0, got {eddy_heating_power}")
    
    # Radiative cooling power (W)
    power_out = emissivity * stefan_boltzmann * surface_area * (temperature**4 - ambient_temp**4)
    
    # Solar heating power (W)
    power_in = solar_flux * surface_area
    
    # Add eddy-current heating
    power_in += eddy_heating_power
    
    # Net power (heating - cooling)
    power_net = power_in - power_out
    
    # Temperature change
    temp_change = power_net * dt / (mass * specific_heat)
    
    # Update temperature
    new_temp = temperature + temp_change
    
    # Prevent temperature from going below ambient (physical limit)
    new_temp = max(new_temp, ambient_temp)
    
    return new_temp


def check_thermal_limits(
    temperature: float,
    limits: ThermalLimits,
) -> Tuple[bool, Optional[str]]:
    """
    Check if temperature is within safe limits.
    
    Args:
        temperature: Current temperature (K)
        limits: ThermalLimits object with max/min temperatures
    
    Returns:
        Tuple of (within_limits: bool, violation_type: Optional[str])
        violation_type is None if within limits, otherwise describes the violation
    """
    if temperature > limits.max_packet_temp:
        return False, f"exceeds max_packet_temp ({limits.max_packet_temp} K)"
    elif temperature < limits.min_temp:
        return False, f"below min_temp ({limits.min_temp} K)"
    else:
        return True, None


def eddy_heating_power(
    velocity: float,
    k_drag: float,
    radius: float,
) -> float:
    """
    Compute eddy-current heating power from velocity-dependent drag.
    
    Eddy-current drag force: F_drag = k_drag * v
    Heating power: P_eddy = F_drag * v = k_drag * v^2
    
    Args:
        velocity: Packet velocity (m/s)
        k_drag: Drag coefficient (N·s/m)
        radius: Packet radius (m) - for skin depth correction
    
    Returns:
        Eddy-current heating power (W)
    """
    # Quadratic drag: P = k_drag * v^2
    power = k_drag * velocity**2
    return power


def steady_state_temperature(
    power_in: float,
    mass: float,
    radius: float,
    emissivity: float,
    specific_heat: float,
    ambient_temp: float = 4.0,
    stefan_boltzmann: float = 5.67e-8,
) -> float:
    """
    Calculate steady-state temperature given constant power input.
    
    At steady state: P_in = P_out = εσA(T⁴ - T_ambient⁴)
    Solves for T: T = (P_in/(εσA) + T_ambient⁴)^(1/4)
    
    Args:
        power_in: Constant power input (W)
        mass: Packet mass (kg)
        radius: Packet radius (m)
        emissivity: Surface emissivity (0-1)
        specific_heat: Specific heat capacity (J/kg/K)
        ambient_temp: Ambient temperature (K)
        stefan_boltzmann: Stefan-Boltzmann constant (W/m²/K⁴)
    
    Returns:
        Steady-state temperature (K)
    """
    surface_area = 4 * np.pi * radius**2
    
    # Solve radiative balance: P_in = εσA(T⁴ - T_ambient⁴)
    temp_fourth = power_in / (emissivity * stefan_boltzmann * surface_area) + ambient_temp**4
    steady_temp = temp_fourth**0.25
    
    return steady_temp


def create_anchor_lumped_thermal(
    stator_mass: float = 10.0,
    stator_specific_heat: float = 500.0,  # J/kg/K (GdBCO at 77K)
    rotor_mass: float = 5.0,
    rotor_specific_heat: float = 400.0,  # J/kg/K (aluminum at 77K)
    shaft_conductance: float = 10.0,  # W/K
    dt: float = 0.01,
) -> LumpedThermalModel:
    """Create lumped-parameter thermal model for anchor system.

    Args:
        stator_mass: Mass of GdBCO stator (kg)
        stator_specific_heat: Specific heat of stator (J/kg/K)
        rotor_mass: Mass of rotor (kg)
        rotor_specific_heat: Specific heat of rotor (J/kg/K)
        shaft_conductance: Thermal conductance of shaft (W/K)
        dt: Time step (s)

    Returns:
        LumpedThermalModel with stator and rotor nodes
    """
    params = LumpedThermalParams(
        stator_mass=stator_mass,
        stator_specific_heat=stator_specific_heat,
        rotor_mass=rotor_mass,
        rotor_specific_heat=rotor_specific_heat,
        shaft_conductance=shaft_conductance,
    )

    return LumpedThermalModel(params, dt=dt)

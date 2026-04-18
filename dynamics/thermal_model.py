"""
Thermal model for radiation cooling of Sovereign Beans.

Implements Stefan-Boltzmann radiation cooling for decoupled thermal management
in vacuum environment. Packets cool via radiation only (no convection in vacuum).

Governing equation:
    P_rad = ε σ A T⁴
    dT/dt = -P_rad / (m · c_p)

where:
    ε: emissivity (0.8 for Al/BFRP)
    σ: Stefan-Boltzmann constant (5.67×10⁻⁸ W/m²·K⁴)
    A: surface area (m²)
    T: temperature (K)
    m: mass (kg)
    c_p: specific heat capacity (J/kg·K)

Temperature constraints:
    T_packet ≤ 450 K (packet limit)
    T_node ≤ 400 K (node limit)

Reference: Incropera & DeWitt, Fundamentals of Heat and Mass Transfer
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThermalLimits:
    """Temperature limits for thermal safety."""
    max_packet_temp: float = 450.0  # K
    max_node_temp: float = 400.0  # K


@dataclass
class ThermalProperties:
    """Thermal properties for materials."""
    emissivity: float = 0.8  # Al/BFRP
    specific_heat: float = 900.0  # J/kg·K for Al
    stefan_boltzmann: float = 5.67e-8  # W/m²·K⁴


def stefan_boltzmann_power(
    temperature: float,
    radius: float,
    emissivity: float,
    sigma: float = 5.67e-8,
) -> float:
    """
    Compute radiated power via Stefan-Boltzmann law.
    
    P = ε σ A T⁴
    where A = 4πr² for spherical packet
    
    Args:
        temperature: Temperature (K)
        radius: Packet radius (m)
        emissivity: Emissivity (dimensionless)
        sigma: Stefan-Boltzmann constant (W/m²·K⁴)
    
    Returns:
        Radiated power (W)
    """
    surface_area = 4 * np.pi * radius**2
    power = emissivity * sigma * surface_area * temperature**4
    return power


def temperature_derivative(
    temperature: float,
    mass: float,
    radius: float,
    emissivity: float,
    specific_heat: float,
    sigma: float = 5.67e-8,
) -> float:
    """
    Compute temperature derivative dT/dt from radiation cooling.
    
    dT/dt = -P_rad / (m · c_p)
    
    Args:
        temperature: Temperature (K)
        mass: Packet mass (kg)
        radius: Packet radius (m)
        emissivity: Emissivity (dimensionless)
        specific_heat: Specific heat capacity (J/kg·K)
        sigma: Stefan-Boltzmann constant (W/m²·K⁴)
    
    Returns:
        Temperature derivative dT/dt (K/s)
    """
    power_out = stefan_boltzmann_power(temperature, radius, emissivity, sigma)
    dT_dt = -power_out / (mass * specific_heat)
    return dT_dt


def update_temperature_euler(
    temperature: float,
    mass: float,
    radius: float,
    emissivity: float,
    specific_heat: float,
    dt: float,
    sigma: float = 5.67e-8,
) -> float:
    """
    Update temperature using Euler integration.
    
    T(t+dt) = T(t) + dT/dt × dt
    
    Args:
        temperature: Current temperature (K)
        mass: Packet mass (kg)
        radius: Packet radius (m)
        emissivity: Emissivity (dimensionless)
        specific_heat: Specific heat capacity (J/kg·K)
        dt: Time step (s)
        sigma: Stefan-Boltzmann constant (W/m²·K⁴)
    
    Returns:
        Updated temperature (K)
    """
    dT_dt = temperature_derivative(temperature, mass, radius, emissivity, specific_heat, sigma)
    new_temperature = temperature + dT_dt * dt
    
    # Prevent temperature from going below absolute zero
    new_temperature = max(new_temperature, 0.0)
    
    return new_temperature


def check_thermal_limits(
    temperature: float,
    limits: ThermalLimits = None,
) -> tuple[bool, str]:
    """
    Check if temperature is within thermal safety limits.
    
    Args:
        temperature: Temperature to check (K)
        limits: ThermalLimits object, default if None
    
    Returns:
        (within_limits, violation_type) tuple
    """
    if limits is None:
        limits = ThermalLimits()
    
    if temperature > limits.max_packet_temp:
        return False, "packet_temp_exceeded"
    
    if temperature > limits.max_node_temp:
        return False, "node_temp_exceeded"
    
    return True, "none"


def thermal_time_constant(
    mass: float,
    radius: float,
    emissivity: float,
    specific_heat: float,
    temperature: float = 300.0,
    sigma: float = 5.67e-8,
) -> float:
    """
    Compute thermal time constant for radiation cooling.
    
    τ = m · c_p / (dP/dT) where dP/dT = 4εσAT³
    
    Args:
        mass: Packet mass (kg)
        radius: Packet radius (m)
        emissivity: Emissivity (dimensionless)
        specific_heat: Specific heat capacity (J/kg·K)
        temperature: Operating temperature (K)
        sigma: Stefan-Boltzmann constant (W/m²·K⁴)
    
    Returns:
        Thermal time constant (s)
    """
    surface_area = 4 * np.pi * radius**2
    dP_dT = 4 * emissivity * sigma * surface_area * temperature**3
    tau = (mass * specific_heat) / dP_dT
    return tau

"""
Thermionic emission model for cathode electron emission.

Implements Richardson-Dushman equation with Schottky enhancement
for electrodynamic tether cathode emission.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Physical constants
ELECTRON_CHARGE = 1.602176634e-19  # C
BOLTZMANN = 1.380649e-23  # J/K
EPSILON_0 = 8.854187817e-12  # F/m (vacuum permittivity)


@dataclass
class CathodeSpec:
    """Cathode material specifications."""
    work_function: float = 4.5  # eV (barium oxide)
    richardson_constant: float = 120.0  # A/m²K²
    max_temperature: float = 2200.0  # K
    degradation_rate: float = 0.001  # % per 1000 hours


class ThermionicEmitter:
    """
    Thermionic emission model using Richardson-Dushman equation.

    Implements Richardson-Dushman with Schottky enhancement for
    high electric fields and space charge limits.
    """

    def __init__(
        self,
        work_function: float = 4.5,  # eV (barium oxide)
        richardson_constant: float = 120.0,  # A/m²K²
        cathode_area: float = 1e-4,  # m² (1 cm²)
        max_temperature: float = 2200.0,  # K
        degradation_rate: float = 0.001,  # % per 1000 hours
    ):
        """
        Initialize thermionic emitter.

        Args:
            work_function: Work function (eV)
            richardson_constant: Richardson constant (A/m²K²)
            cathode_area: Cathode area (m²)
            max_temperature: Maximum operating temperature (K)
            degradation_rate: Degradation rate (% per 1000 hours)
        """
        self.work_function = work_function  # eV
        self.richardson_constant = richardson_constant  # A/m²K²
        self.cathode_area = cathode_area  # m²
        self.max_temperature = max_temperature  # K
        self.degradation_rate = degradation_rate  # % per 1000 hours

        # Convert work function to Joules
        self.work_function_J = work_function * ELECTRON_CHARGE

        logger.info(
            f"Thermionic emitter initialized: W={work_function} eV, "
            f"A_G={richardson_constant} A/m²K², area={cathode_area} m²"
        )

    def emission_current_density(
        self,
        temperature: float,
        electric_field: float = 0.0,
    ) -> float:
        """
        Richardson-Dushman with Schottky enhancement.

        J = A_G T² exp(-W/kT) * exp(√(e³E/(4πε₀kT)))

        Args:
            temperature: Cathode temperature (K)
            electric_field: Electric field (V/m)

        Returns:
            Emission current density (A/m²)
        """
        # Temperature limit
        if temperature > self.max_temperature:
            logger.warning(
                f"Temperature {temperature} K exceeds max {self.max_temperature} K"
            )
            temperature = self.max_temperature

        # Richardson-Dushman equation
        # J_RD = A_G T² exp(-W/kT)
        exponent = -self.work_function_J / (BOLTZMANN * temperature)
        j_rd = self.richardson_constant * temperature ** 2 * np.exp(exponent)

        # Schottky enhancement
        # J_Schottky = J_RD * exp(√(e³E/(4πε₀kT)))
        if electric_field > 0:
            schottky_factor = np.sqrt(
                (ELECTRON_CHARGE ** 3 * electric_field)
                / (4 * np.pi * EPSILON_0 * BOLTZMANN * temperature)
            )
            j_schottky = j_rd * np.exp(schottky_factor)
            return j_schottky

        return j_rd

    def emission_current(
        self,
        temperature: float,
        electric_field: float = 0.0,
    ) -> float:
        """
        Total emission current: I = J × A

        Args:
            temperature: Cathode temperature (K)
            electric_field: Electric field (V/m)

        Returns:
            Emission current (A)
        """
        j = self.emission_current_density(temperature, electric_field)
        return j * self.cathode_area

    def space_charge_limit(
        self,
        anode_distance: float,
        voltage: float,
    ) -> float:
        """
        Child-Langmuir space charge limit.

        J_CL = (4ε₀/9) √(2e/m) V^(3/2) / d²

        Args:
            anode_distance: Anode-cathode distance (m)
            voltage: Applied voltage (V)

        Returns:
            Space charge limited current density (A/m²)
        """
        if voltage <= 0:
            return 0.0

        # Child-Langmuir current density
        j_cl = (
            (4 * EPSILON_0 / 9)
            * np.sqrt(2 * ELECTRON_CHARGE / 9.10938356e-31)
            * voltage ** 1.5
            / anode_distance ** 2
        )

        return j_cl

    def evaporation_rate(self, temperature: float) -> float:
        """
        Cathode material evaporation rate (simplified model).

        Uses Arrhenius-type equation: rate = A exp(-E_a/kT)

        Args:
            temperature: Cathode temperature (K)

        Returns:
            Evaporation rate (kg/s)
        """
        # Simplified model for barium oxide
        # Activation energy ~4 eV
        activation_energy = 4.0 * ELECTRON_CHARGE  # J
        pre_exponential = 1e-4  # kg/s (typical)

        rate = pre_exponential * np.exp(-activation_energy / (BOLTZMANN * temperature))

        return rate

    def effective_work_function(
        self,
        temperature: float,
        degradation_hours: float = 0.0,
    ) -> float:
        """
        Temperature-dependent work function with degradation.

        W(T) = W₀ + αT + degradation_factor

        Args:
            temperature: Cathode temperature (K)
            degradation_hours: Operating hours (for degradation)

        Returns:
            Effective work function (eV)
        """
        # Temperature coefficient (typical: 1e-4 eV/K)
        alpha = 1e-4

        # Temperature-dependent work function
        w_temp = self.work_function + alpha * temperature

        # Degradation factor (0.1-1% per 1000 hours)
        if degradation_hours > 0:
            degradation = self.degradation_rate * (degradation_hours / 1000.0)
            w_temp *= (1.0 + degradation / 100.0)

        return w_temp


def analytical_richardson_dushman(
    temperature: float,
    work_function: float,
    richardson_constant: float = 120.0,
) -> float:
    """
    Analytical Richardson-Dushman equation: J = A_G T² exp(-W/kT)

    Args:
        temperature: Temperature (K)
        work_function: Work function (eV)
        richardson_constant: Richardson constant (A/m²K²)

    Returns:
        Current density (A/m²)
    """
    work_function_J = work_function * ELECTRON_CHARGE
    exponent = -work_function_J / (BOLTZMANN * temperature)
    j = richardson_constant * temperature ** 2 * np.exp(exponent)
    return j


def analytical_schottky_enhancement(
    base_current_density: float,
    electric_field: float,
    temperature: float,
) -> float:
    """
    Analytical Schottky enhancement factor.

    Enhancement = exp(√(e³E/(4πε₀kT)))

    Args:
        base_current_density: Base Richardson-Dushman current (A/m²)
        electric_field: Electric field (V/m)
        temperature: Temperature (K)

    Returns:
        Enhanced current density (A/m²)
    """
    schottky_factor = np.sqrt(
        (ELECTRON_CHARGE ** 3 * electric_field)
        / (4 * np.pi * EPSILON_0 * BOLTZMANN * temperature)
    )
    return base_current_density * np.exp(schottky_factor)

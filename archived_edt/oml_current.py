"""
Orbital Motion Limited (OML) current collection model for bare tethers.

Implements OML theory for electron collection from ionospheric plasma,
including ion ram collection, photoemission, and space charge limits.

Phase 4A: OML current collection with literature validation
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Physical constants
ELECTRON_CHARGE = 1.602176634e-19  # C
ELECTRON_MASS = 9.10938356e-31  # kg
BOLTZMANN = 1.380649e-23  # J/K
STEFAN_BOLTZMANN = 5.670374419e-8  # W/m²/K⁴


@dataclass
class PlasmaParameters:
    """Ionospheric plasma parameters."""
    electron_density: float = 1e11  # m⁻³ (typical LEO)
    ion_density: float = 1e11  # m⁻³ (quasi-neutral)
    electron_temp: float = 0.2  # eV (typical LEO)
    ion_temp: float = 0.1  # eV


class OMLCurrentModel:
    """
    Orbital Motion Limited (OML) current collection model.

    Implements OML theory for electron collection from bare tethers
    in ionospheric plasma, including space charge limits.
    """

    def __init__(
        self,
        tether_perimeter: float = 0.00628,  # m (2mm wire)
        plasma_params: Optional[PlasmaParameters] = None,
        work_function: float = 4.5,  # eV (aluminum)
    ):
        """
        Initialize OML current model.

        Args:
            tether_perimeter: Tether perimeter (m)
            plasma_params: Plasma parameters (uses defaults if None)
            work_function: Work function (eV)
        """
        self.tether_perimeter = tether_perimeter
        self.plasma_params = plasma_params or PlasmaParameters()
        self.work_function = work_function

        # Convert electron temperature from eV to K
        self.electron_temp_K = self.plasma_params.electron_temp * 11604.5  # 1 eV ≈ 11604.5 K

        logger.info(
            f"OML model initialized: perimeter={tether_perimeter}m, "
            f"n_e={self.plasma_params.electron_density:.2e} m⁻³"
        )

    def electron_current(self, bias_voltage: float) -> float:
        """
        OML electron collection: I_oml = 2πr e n_e √(2e|φ|/m_e)

        For bare tether, effective radius = perimeter / (2π)

        Args:
            bias_voltage: Bias voltage relative to plasma (V)

        Returns:
            Electron current (A)
        """
        if bias_voltage >= 0:
            # No electron collection for positive bias (anode)
            return 0.0

        # Effective radius from perimeter
        effective_radius = self.tether_perimeter / (2 * np.pi)

        # OML current density: J_OML = e n_e √(2e|φ|/m_e)
        # Current: I = J × A = J × 2πr × L (per unit length)
        # Per unit length: I/L = 2πr e n_e √(2e|φ|/m_e)

        voltage_magnitude = abs(bias_voltage)

        # OML current per unit length
        current_per_length = (
            2 * np.pi * effective_radius
            * ELECTRON_CHARGE
            * self.plasma_params.electron_density
            * np.sqrt(2 * ELECTRON_CHARGE * voltage_magnitude / ELECTRON_MASS)
        )

        # Total current (assuming 1m length for current per length)
        # In practice, multiply by tether length
        return current_per_length

    def ion_current(self, velocity: float, collection_area: Optional[float] = None) -> float:
        """
        Ion ram collection: I_ion = e n_i v_rel A_collection

        Args:
            velocity: Relative velocity (m/s)
            collection_area: Collection area (m²), defaults to tether cross-section

        Returns:
            Ion current (A)
        """
        if collection_area is None:
            # Assume circular cross-section from perimeter
            radius = self.tether_perimeter / (2 * np.pi)
            collection_area = np.pi * radius ** 2

        # Ion ram current: I = e n_i v A
        ion_current = (
            ELECTRON_CHARGE
            * self.plasma_params.ion_density
            * velocity
            * collection_area
        )

        return ion_current

    def photoemission_current(self, solar_flux: float = 1361.0) -> float:
        """
        Solar UV photoemission: I_photo = q_Φ Y_ph

        Args:
            solar_flux: Solar flux (W/m²), default is solar constant

        Returns:
            Photoemission current (A/m²)
        """
        # Photoemission yield (typical for aluminum: ~0.01-0.1)
        # Simplified: assume 1% of solar flux produces photoelectrons
        photo_yield = 0.01

        # Photoemission current density
        # J_photo = (solar_flux / photon_energy) * yield * e
        # Simplified: use empirical value ~1-10 µA/m²
        photo_current_density = 1e-5  # A/m² (typical)

        # Total current (per unit length)
        radius = self.tether_perimeter / (2 * np.pi)
        circumference = 2 * np.pi * radius
        photo_current = photo_current_density * circumference

        return photo_current

    def space_charge_limit(self, bias_voltage: float, anode_distance: float) -> float:
        """
        Child-Langmuir space charge limit for high bias.

        J_CL = (4ε₀/9) √(2e/m) V^(3/2) / d²

        Args:
            bias_voltage: Bias voltage (V)
            anode_distance: Anode-cathode distance (m)

        Returns:
            Space charge limited current density (A/m²)
        """
        epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)

        if bias_voltage <= 0:
            return 0.0

        # Child-Langmuir current density
        j_cl = (
            (4 * epsilon_0 / 9)
            * np.sqrt(2 * ELECTRON_CHARGE / ELECTRON_MASS)
            * bias_voltage ** 1.5
            / anode_distance ** 2
        )

        return j_cl

    def total_current(
        self,
        bias_voltage: float,
        velocity: float,
        apply_space_charge_limit: bool = True,
        anode_distance: float = 0.1,
    ) -> float:
        """
        Total current with space charge limit.

        Args:
            bias_voltage: Bias voltage (V)
            velocity: Relative velocity (m/s)
            apply_space_charge_limit: Whether to apply space charge limit
            anode_distance: Anode-cathode distance for space charge limit (m)

        Returns:
            Total current (A)
        """
        # Electron current (OML collection)
        i_electron = self.electron_current(bias_voltage)

        # Ion current (ram collection)
        i_ion = self.ion_current(velocity)

        # Photoemission current
        i_photo = self.photoemission_current()

        # Total current
        if bias_voltage < 0:
            # Cathode: collect electrons, emit ions
            i_total = i_electron - i_ion + i_photo
        else:
            # Anode: collect ions, emit electrons
            i_total = i_ion - i_electron + i_photo

        # Apply space charge limit
        if apply_space_charge_limit and abs(i_total) > 0:
            j_sc = self.space_charge_limit(abs(bias_voltage), anode_distance)
            radius = self.tether_perimeter / (2 * np.pi)
            collection_area = 2 * np.pi * radius  # Per unit length
            i_sc_limit = j_sc * collection_area

            if abs(i_total) > i_sc_limit:
                i_total = np.sign(i_total) * i_sc_limit

        return i_total


def analytical_oml_current(
    tether_radius: float,
    electron_density: float,
    bias_voltage: float,
) -> float:
    """
    Analytical OML current for validation: I = 2πr e n_e √(2e|φ|/m_e)

    Args:
        tether_radius: Tether radius (m)
        electron_density: Electron density (m⁻³)
        bias_voltage: Bias voltage (V)

    Returns:
        OML current (A/m)
    """
    if bias_voltage >= 0:
        return 0.0

    voltage_magnitude = abs(bias_voltage)

    current = (
        2 * np.pi * tether_radius
        * ELECTRON_CHARGE
        * electron_density
        * np.sqrt(2 * ELECTRON_CHARGE * voltage_magnitude / ELECTRON_MASS)
    )

    return current


def analytical_ion_current(
    ion_density: float,
    velocity: float,
    collection_area: float,
) -> float:
    """
    Analytical ion current for validation: I = e n_i v A

    Args:
        ion_density: Ion density (m⁻³)
        velocity: Velocity (m/s)
        collection_area: Collection area (m²)

    Returns:
        Ion current (A)
    """
    return ELECTRON_CHARGE * ion_density * velocity * collection_area

"""
EDT Packet subclass for electrodynamic tether integration.

Extends the base Packet class with EDT-specific state (current, voltage, resistance)
and thermal-electrical coupling capabilities.
"""

import logging
import numpy as np

from .multi_body import Packet

logger = logging.getLogger(__name__)


class EDTPacket(Packet):
    """
    EDT packet extending base Packet class for electrodynamic tether integration.

    Adds EDT-specific state and thermal-electrical coupling capabilities.
    Compatible with hybrid mode: EDTPacket alongside regular Packets in same MultiBodyStream.

    Attributes:
        current: Tether current (A)
        voltage: Tether voltage (V)
        tether_segment_id: Tether segment identifier (for multi-segment tethers)
        resistance: Segment resistance (Ω)
        temperature: Packet temperature (K)

    Future Features:
        libration_angle: Tether libration angle (rad) - Not yet implemented
        libration_rate: Tether libration rate (rad/s) - Not yet implemented
    """

    def __init__(
        self,
        *args,
        current: float = 0.0,
        voltage: float = 0.0,
        tether_segment_id: int = 0,
        resistance: float = 0.01,
        temperature: float = 300.0,
        **kwargs
    ):
        """
        Initialize EDT packet.

        Args:
            *args: Arguments passed to parent Packet class
            current: Tether current (A)
            voltage: Tether voltage (V)
            tether_segment_id: Tether segment identifier
            resistance: Segment resistance (Ω)
            temperature: Initial temperature (K)
            **kwargs: Keyword arguments passed to parent Packet class
        """
        super().__init__(*args, **kwargs)
        self.current = current
        self.voltage = voltage
        self.tether_segment_id = tether_segment_id
        self.resistance = resistance
        self.temperature = temperature  # Initialize explicitly to avoid relying on parent

        logger.info(
            f"EDTPacket initialized: segment_id={tether_segment_id}, "
            f"resistance={resistance} Ω, current={current} A, temperature={temperature} K"
        )

    def joule_heating(self) -> float:
        """
        Compute Joule heating power: P = I²R

        Returns:
            Joule heating power (W)
        """
        return self.current ** 2 * self.resistance

    def update_thermal_state(
        self,
        thermal_model,
        dt: float = 0.01,
    ) -> None:
        """
        Update temperature using extended JAX thermal model with EDT heat sources.

        Args:
            thermal_model: JAXThermalModel instance with predict_with_edt_heat method
            dt: Time step (s)
        """
        Q_joule = self.joule_heating()

        # Use EDT-specific thermal prediction method
        T_new = thermal_model.predict_with_edt_heat(
            T_initial=np.array([self.temperature]),
            Q_edt=np.array([Q_joule]),
            dt=dt,
        )
        
        # Handle both numpy array and scalar returns
        if isinstance(T_new, np.ndarray):
            self.temperature = float(T_new[0])
        else:
            self.temperature = float(T_new)

        logger.debug(
            f"EDTPacket thermal update: segment_id={self.tether_segment_id}, "
            f"Q_joule={Q_joule:.4f} W, T={self.temperature:.2f} K"
        )

    def get_edt_state(self) -> dict:
        """
        Get EDT-specific state for dashboard/debugging.

        Returns:
            Dictionary with EDT state
        """
        return {
            "current": self.current,
            "voltage": self.voltage,
            "tether_segment_id": self.tether_segment_id,
            "resistance": self.resistance,
            "joule_heating": self.joule_heating(),
            "temperature": self.temperature,
        }

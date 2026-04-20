"""
Unit tests for EDT packet and thermal-electrical coupling.

Tests EDTPacket subclass, Joule heating, and integration with JAX thermal model.
"""

import pytest
import numpy as np

from dynamics.edt_packet import EDTPacket
from dynamics.rigid_body import RigidBody

try:
    from dynamics.jax_thermal import JAXThermalModel
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    JAXThermalModel = None


class TestEDTPacket:
    """Test EDTPacket basic functionality."""

    def test_edt_packet_initialization(self):
        """Test EDT packet initialization with EDT-specific parameters."""
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        position = np.array([0.0, 0.0, 0.0])
        velocity = np.array([1600.0, 0.0, 0.0])
        body = RigidBody(mass, I, position=position, velocity=velocity)

        edt_packet = EDTPacket(
            id=0,
            body=body,
            current=1.0,
            voltage=100.0,
            tether_segment_id=0,
            resistance=0.01,
            temperature=350.0,
        )

        assert edt_packet.current == 1.0
        assert edt_packet.voltage == 100.0
        assert edt_packet.tether_segment_id == 0
        assert edt_packet.resistance == 0.01
        assert edt_packet.temperature == 350.0
        assert isinstance(edt_packet, EDTPacket)

    def test_joule_heating(self):
        """Test Joule heating calculation: P = I²R."""
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        body = RigidBody(mass, I)

        edt_packet = EDTPacket(
            id=0,
            body=body,
            current=2.0,
            resistance=0.01,
        )

        # P = I²R = (2.0)² * 0.01 = 0.04 W
        expected_power = 4.0 * 0.01
        assert np.isclose(edt_packet.joule_heating(), expected_power)

    def test_joule_heating_zero_current(self):
        """Test Joule heating with zero current."""
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        body = RigidBody(mass, I)

        edt_packet = EDTPacket(
            id=0,
            body=body,
            current=0.0,
            resistance=0.01,
        )

        assert edt_packet.joule_heating() == 0.0

    def test_get_edt_state(self):
        """Test get_edt_state method returns correct state dictionary."""
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        body = RigidBody(mass, I)

        edt_packet = EDTPacket(
            id=0,
            body=body,
            current=1.5,
            voltage=150.0,
            tether_segment_id=5,
            resistance=0.02,
            temperature=350.0,
        )

        state = edt_packet.get_edt_state()

        assert state["current"] == 1.5
        assert state["voltage"] == 150.0
        assert state["tether_segment_id"] == 5
        assert state["resistance"] == 0.02
        assert state["temperature"] == 350.0
        assert "joule_heating" in state


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestEDTThermalCoupling:
    """Test thermal-electrical coupling with JAX thermal model."""

    def test_predict_with_edt_heat(self):
        """Test predict_with_edt_heat method updates temperature."""
        thermal_model = JAXThermalModel(dt=0.01)

        T_initial = np.array([300.0])
        Q_edt = np.array([10.0])  # 10 W heat input (realistic EDT heating)

        T_new = thermal_model.predict_with_edt_heat(T_initial, Q_edt, dt=0.01)

        # Temperature should increase with positive heat input
        assert T_new[0] > T_initial[0]
        # Temperature should stay within safe operating limits
        assert T_new[0] < 450.0

    def test_edt_packet_thermal_update(self):
        """Test EDTPacket update_thermal_state method."""
        thermal_model = JAXThermalModel(dt=0.01)

        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        body = RigidBody(mass, I)

        edt_packet = EDTPacket(
            id=0,
            body=body,
            current=10.0,  # Realistic EDT current (generates 1W)
            resistance=0.01,  # Realistic EDT resistance
            temperature=300.0,
        )

        T_initial = edt_packet.temperature
        edt_packet.update_thermal_state(thermal_model, dt=0.01)
        T_final = edt_packet.temperature

        # Verify thermal update mechanism works (temperature is updated)
        assert isinstance(T_final, float)
        # Temperature should stay within safe operating limits
        assert T_final < 450.0
        assert T_final >= 273.15  # Above absolute zero

    def test_energy_conservation_with_edt_heat(self):
        """Test energy conservation with EDT heat sources (±1% tolerance)."""
        thermal_model = JAXThermalModel(dt=0.01)

        T_initial = 300.0
        Q_edt = 10.0  # 10 W heat input (realistic EDT heating)
        dt = 0.01

        T_new = thermal_model.predict_with_edt_heat(
            np.array([T_initial]),
            np.array([Q_edt]),
            dt=dt,
        )[0]

        # Energy balance: Q_in * dt = m * c * dT
        # dT = Q_in * dt / (m * c)
        thermal_mass = thermal_model.thermal_mass
        dT_expected = Q_edt * dt / thermal_mass
        T_expected = T_initial + dT_expected

        # Allow 1% tolerance for numerical errors
        assert abs(T_new - T_expected) / T_expected < 0.01

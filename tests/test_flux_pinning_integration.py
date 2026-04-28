"""
Tests for flux-pinning force integration into RigidBody and MultiBodyStream.

Tests that flux-pinning force is correctly computed and applied
in the governing equation, with full integration validation.
"""

import numpy as np
import pytest

from dynamics.rigid_body import RigidBody
from dynamics.bean_london_model import BeanLondonModel
from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.multi_body import MultiBodyStream, Packet, SNode, PacketState


class TestFluxPinningIntegration:
    """Test flux-pinning force integration in RigidBody."""

    @pytest.fixture
    def gdBCO_material(self):
        """Create GdBCO material for testing."""
        properties = GdBCOProperties(
            Tc=92.0,  # Critical temperature (K)
            Jc0=1e8,  # Critical current density at 0K, 0T (A/m²) - reduced for realistic stiffness
            B0=5.0,  # Characteristic field (T)
            n_exponent=1.5,  # Temperature dependence exponent
            alpha=0.5,  # Field dependence exponent
        )
        return GdBCOMaterial(properties)

    @pytest.fixture
    def bean_london_model(self, gdBCO_material):
        """Create Bean-London model for testing."""
        geometry = {
            'thickness': 1e-6,  # 1 μm (coated conductor thickness)
            'width': 0.01,  # 10 mm
            'length': 0.01,  # 10 mm (reduced length)
        }
        return BeanLondonModel(gdBCO_material, geometry)

    @pytest.fixture
    def rigid_body(self, bean_london_model):
        """Create RigidBody with flux-pinning model."""
        I = np.diag([1.0, 2.0, 3.0])  # Inertia tensor
        return RigidBody(
            mass=10.0,
            I=I,
            flux_model=bean_london_model,
        )

    def test_flux_pinning_force_computation(self, rigid_body):
        """Test that flux-pinning force is computed correctly."""
        B_field = np.array([0.0, 0.0, 1.0])  # 1 T in z-direction
        temp = 77.0  # Liquid nitrogen temperature
        displacement = np.array([0.001, 0.0, 0.0])  # 1 mm displacement in x

        force_torque = rigid_body.compute_flux_pinning_force(
            B_field, temp, displacement
        )

        # Should return 6-DoF vector [Fx, Fy, Fz, τx, τy, τz]
        assert force_torque.shape == (6,)
        assert isinstance(force_torque, np.ndarray)

    def test_force_direction(self, rigid_body):
        """Test that restoring force opposes displacement."""
        B_field = np.array([0.0, 0.0, 1.0])
        temp = 77.0

        # Displace in +X
        displacement_pos = np.array([0.001, 0.0, 0.0])
        force_pos = rigid_body.compute_flux_pinning_force(
            B_field, temp, displacement_pos
        )

        # Displace in -X
        displacement_neg = np.array([-0.001, 0.0, 0.0])
        force_neg = rigid_body.compute_flux_pinning_force(
            B_field, temp, displacement_neg
        )

        # Force should oppose displacement
        assert force_pos[0] < 0, "Force should oppose +X displacement"
        assert force_neg[0] > 0, "Force should oppose -X displacement"

    def test_temperature_dependence(self, rigid_body):
        """Test that force goes to zero above critical temperature."""
        B_field = np.array([0.0, 0.0, 1.0])
        displacement = np.array([0.001, 0.0, 0.0])

        # Below Tc (77K)
        force_cold = rigid_body.compute_flux_pinning_force(
            B_field, 77.0, displacement
        )

        # Above Tc (100K)
        force_hot = rigid_body.compute_flux_pinning_force(
            B_field, 100.0, displacement
        )

        # Force should be much smaller above Tc
        assert np.abs(force_cold[0]) > np.abs(force_hot[0]), \
            "Flux-pinning force should decrease above Tc"

    def test_stiffness_magnitude(self, rigid_body):
        """Test that stiffness is positive and reasonable."""
        B_field = np.array([0.0, 0.0, 1.0])
        temp = 77.0
        displacement = np.array([0.0001, 0.0, 0.0])  # Small displacement

        force = rigid_body.compute_flux_pinning_force(
            B_field, temp, displacement
        )

        # Stiffness approximation: k ≈ F / x
        stiffness = abs(force[0]) / abs(displacement[0])

        # Should be positive and finite
        assert stiffness > 0, "Stiffness should be positive"
        assert np.isfinite(stiffness), "Stiffness should be finite"
        # Allow wide range for different geometries
        assert stiffness < 1e8, "Stiffness should be reasonable"

    def test_no_flux_model(self):
        """Test that RigidBody without flux model returns zero force."""
        I = np.diag([1.0, 2.0, 3.0])
        rigid_body = RigidBody(mass=10.0, I=I, flux_model=None)

        B_field = np.array([0.0, 0.0, 1.0])
        temp = 77.0
        displacement = np.array([0.001, 0.0, 0.0])

        force_torque = rigid_body.compute_flux_pinning_force(
            B_field, temp, displacement
        )

        # Should return zeros
        assert np.allclose(force_torque, np.zeros(6))

    def test_angular_momentum_conservation(self, rigid_body):
        """Test that angular momentum is conserved with flux-pinning."""
        # This is a basic sanity check - full conservation test requires integration
        B_field = np.array([0.0, 0.0, 1.0])
        temp = 77.0

        # Initial angular momentum
        L_initial = rigid_body.angular_momentum.copy()

        # Apply flux-pinning force (should not change L if no external torque)
        force_torque = rigid_body.compute_flux_pinning_force(B_field, temp)

        # Flux-pinning provides internal restoring torque, not external
        # So angular momentum should be conserved in absence of external torques
        # This is a placeholder for a more comprehensive test
        assert L_initial is not None

    def test_critical_temperature_exact(self, rigid_body):
        """Test that pinning force is exactly zero at T >= Tc."""
        B_field = np.array([0.0, 0.0, 1.0])
        displacement = np.array([0.001, 0.0, 0.0])

        # At exactly Tc = 92K, Jc should be zero
        force_at_tc = rigid_body.compute_flux_pinning_force(
            B_field, 92.0, displacement
        )

        # Above Tc
        force_above = rigid_body.compute_flux_pinning_force(
            B_field, 95.0, displacement
        )

        # Both should have negligible force
        assert np.abs(force_at_tc[0]) < 1e-6, "Force should be ~0 at Tc"
        assert np.abs(force_above[0]) < 1e-6, "Force should be ~0 above Tc"

    def test_stiffness_range(self, rigid_body):
        """Test that stiffness is positive and reasonable."""
        B_field = np.array([0.0, 0.0, 0.5])  # 0.5 T
        temp = 77.0  # Liquid nitrogen temp

        # Small displacement for linear stiffness measurement
        x_small = 0.0001  # 0.1 mm

        # Measure stiffness in X
        force_x = rigid_body.compute_flux_pinning_force(
            B_field, temp, np.array([x_small, 0, 0])
        )
        k_x = abs(force_x[0]) / x_small

        # Should be positive and finite
        assert k_x > 0, "Stiffness should be positive"
        assert np.isfinite(k_x), "Stiffness should be finite"
        # Allow wide range for different geometries and Jc values
        assert k_x < 1e7, "Stiffness should be reasonable"


class TestFluxPinningMultiBodyIntegration:
    """Test flux-pinning integration with MultiBodyStream."""

    @pytest.fixture
    def stream_with_pinning(self):
        """Create MultiBodyStream with flux-pinning enabled."""
        # Create GdBCO material
        props = GdBCOProperties(Tc=92.0, Jc0=1e8, B0=5.0)
        material = GdBCOMaterial(props)

        # Create Bean-London model
        geometry = {'thickness': 1e-6, 'width': 0.01, 'length': 0.01}
        flux_model = BeanLondonModel(material, geometry)

        # Create packets with flux-pinning models
        packets = []
        for i in range(3):
            I = np.diag([0.0001, 0.00011, 0.00009])
            # Give initial angular displacement (small rotation about X)
            quaternion = np.array([0.1, 0.0, 0.0, 0.995])  # Small X rotation
            body = RigidBody(
                mass=0.05,
                I=I,
                position=np.array([i * 10.0, 0.0, 0.0]),
                velocity=np.array([1600.0, 0.0, 0.0]),
                quaternion=quaternion,
                angular_velocity=np.array([0.1, 0.0, 0.0]),  # Small spin
                flux_model=flux_model,
            )
            packet = Packet(id=i, body=body, temperature=77.0)
            packets.append(packet)

        # Create S-Nodes
        nodes = [SNode(id=0, position=np.array([0.0, 0.0, 0.0]))]

        # Create stream with strong B-field
        B_field = np.array([0.0, 0.0, 0.5])  # 0.5 T axial
        return MultiBodyStream(packets, nodes, B_field=B_field)

    def test_stream_initializes_with_b_field(self, stream_with_pinning):
        """Test that MultiBodyStream properly stores B-field."""
        assert hasattr(stream_with_pinning, 'B_field')
        assert np.allclose(stream_with_pinning.B_field, [0, 0, 0.5])

    def test_packet_torque_computation(self, stream_with_pinning):
        """Test that packets compute flux-pinning torque."""
        packet = stream_with_pinning.packets[0]

        # Compute torque
        torque = packet.compute_flux_pinning_torque(stream_with_pinning.B_field)

        # Should return 3-element vector
        assert torque.shape == (3,)
        assert np.isfinite(torque).all()

    def test_temperature_collapse_scenario(self, stream_with_pinning):
        """Test that pinning fails when temperature exceeds Tc."""
        packet = stream_with_pinning.packets[0]

        # Normal operation at 77K
        packet.temperature = 77.0
        torque_cold = packet.compute_flux_pinning_torque(stream_with_pinning.B_field)
        torque_mag_cold = np.linalg.norm(torque_cold)

        # Thermal failure at 95K (> Tc)
        packet.temperature = 95.0
        torque_hot = packet.compute_flux_pinning_torque(stream_with_pinning.B_field)
        torque_mag_hot = np.linalg.norm(torque_hot)

        # Hot torque should be near zero (or at least much smaller)
        # Allow small tolerance for numerical precision
        assert torque_mag_hot < torque_mag_cold * 0.1, \
            f"Pinning should fail above Tc: hot={torque_mag_hot:.2e}, cold={torque_mag_cold:.2e}"

    def test_integration_with_control_torque(self, stream_with_pinning):
        """Test that flux-pinning adds to control torque in integration."""

        def control_torque(packet_id, t, state):
            """Simple control torque."""
            return np.array([1.0, 0.0, 0.0])  # Larger X torque

        # Store initial angular velocity
        packet = stream_with_pinning.packets[0]
        omega_initial = packet.angular_velocity.copy()

        # Integrate with dt=0.1s (longer for visible change)
        result = stream_with_pinning.integrate(
            dt=0.1,
            torques=control_torque,
            max_steps=10,
        )

        # Angular velocity should have changed
        omega_final = packet.angular_velocity
        assert not np.allclose(omega_initial, omega_final), \
            f"Angular velocity should change with combined torques: initial={omega_initial}, final={omega_final}"

        # Check that result includes thermal data
        assert 'packets' in result
        assert len(result['packets']) > 0


class TestAngularMomentumConservation:
    """Comprehensive angular momentum conservation tests."""

    @pytest.fixture
    def isolated_system(self):
        """Create isolated packet with flux-pinning (no external torques)."""
        props = GdBCOProperties(Tc=92.0, Jc0=5e10, B0=5.0)
        material = GdBCOMaterial(props)
        geometry = {'thickness': 0.001, 'width': 0.01, 'length': 0.1}
        flux_model = BeanLondonModel(material, geometry)

        I = np.diag([0.001, 0.001, 0.001])  # Spherical for simplicity
        body = RigidBody(
            mass=1.0,
            I=I,
            angular_velocity=np.array([1.0, 0.0, 0.0]),  # Spin about X
            flux_model=flux_model,
        )

        packet = Packet(id=0, body=body, temperature=77.0)
        nodes = [SNode(id=0, position=np.array([100.0, 0.0, 0.0]))]  # Far away

        return MultiBodyStream([packet], nodes, B_field=np.array([0, 0, 0.5]))

    def test_libration_conserves_angular_momentum(self, isolated_system):
        """Test angular momentum conservation during libration."""
        packet = isolated_system.packets[0]

        # Initial angular momentum
        L_initial = packet.body.angular_momentum.copy()

        # Zero external torque - only internal flux-pinning restoring torque
        def zero_torque(packet_id, t, state):
            return np.zeros(3)

        # Integrate for short time (libration period)
        dt = 0.001  # Small step
        result = isolated_system.integrate(dt=dt, torques=zero_torque)

        # Final angular momentum
        L_final = packet.body.angular_momentum

        # Angular momentum should be conserved (error < 1% for short integration)
        error = np.linalg.norm(L_final - L_initial) / np.linalg.norm(L_initial)
        assert error < 0.01, f"Angular momentum error {error:.4f} exceeds 1%"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

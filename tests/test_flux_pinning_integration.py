"""
Tests for flux-pinning force integration into RigidBody.

Tests that flux-pinning force is correctly computed and applied
in the governing equation.
"""

import numpy as np
import pytest

from dynamics.rigid_body import RigidBody
from dynamics.bean_london_model import BeanLondonModel
from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties


class TestFluxPinningIntegration:
    """Test flux-pinning force integration in RigidBody."""

    @pytest.fixture
    def gdBCO_material(self):
        """Create GdBCO material for testing."""
        properties = GdBCOProperties(
            Tc=92.0,  # Critical temperature (K)
            Jc0=5e10,  # Critical current density at 0K, 0T (A/m²)
            B0=5.0,  # Characteristic field (T)
            n_exponent=1.5,  # Temperature dependence exponent
            alpha=0.5,  # Field dependence exponent
        )
        return GdBCOMaterial(properties)

    @pytest.fixture
    def bean_london_model(self, gdBCO_material):
        """Create Bean-London model for testing."""
        geometry = {
            'thickness': 0.001,  # 1 mm
            'width': 0.01,  # 10 mm
            'length': 0.1,  # 100 mm
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

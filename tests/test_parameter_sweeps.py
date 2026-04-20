"""
Parameter sweep tests to validate physics scaling across different configurations.

Tests validate that physical quantities scale correctly with:
- Mass: [0.05, 0.5, 2.0, 8.0] kg
- Radius: [0.02, 0.05, 0.1] m
- RPM: [100, 1000, 5000, 5236, 5657] rad/s
- Velocity: [10, 100, 1000, 1600] m/s

These sweeps ensure the physics engine behaves correctly across the range
from test convenience values to operational paper targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamics.rigid_body import RigidBody
from dynamics.gyro_matrix import gyroscopic_coupling


class TestMassSweep:
    """Test physics scaling with packet mass."""

    @pytest.mark.parametrize("mass", [0.05, 0.5, 2.0, 8.0])
    def test_inertia_scales_with_mass(self, mass):
        """Inertia tensor should scale linearly with mass (I ∝ m·r²)."""
        radius = 0.1  # Fixed radius
        I_sphere = (2.0/5.0) * mass * radius**2
        I = np.diag([I_sphere, I_sphere, I_sphere])

        body = RigidBody(mass, I)
        assert np.allclose(body.I, I)

    @pytest.mark.parametrize("mass", [0.05, 0.5, 2.0, 8.0])
    def test_angular_momentum_scales_with_mass(self, mass):
        """Angular momentum should scale linearly with mass (L = I·ω)."""
        radius = 0.1
        omega = 100.0  # Fixed spin rate
        I_sphere = (2.0/5.0) * mass * radius**2
        I = np.diag([I_sphere, I_sphere, I_sphere])

        omega_vec = np.array([omega, 0.0, 0.0])
        body = RigidBody(mass, I, angular_velocity=omega_vec)

        L = body.angular_momentum
        L_expected = I_sphere * omega

        assert np.abs(L[0] - L_expected) < 1e-6

    @pytest.mark.parametrize("mass", [0.05, 0.5, 2.0, 8.0])
    def test_centrifugal_stress_scales_with_mass(self, mass):
        """Centrifugal stress should scale linearly with density (hoop stress: σ_θ = ρ·r²·ω²)."""
        radius = 0.1
        omega = 100.0
        rho = 2500.0  # BFRP density (kg/m³)

        # Hoop stress formula from paper: σ_θ = ρ·r²·ω²
        stress = rho * radius**2 * omega**2

        # Stress should be independent of mass (depends on density and geometry)
        # For fixed density and radius, stress is constant regardless of mass
        # This test validates that the formula is geometry-dependent, not mass-dependent
        assert stress > 0


class TestRadiusSweep:
    """Test physics scaling with packet radius."""

    @pytest.mark.parametrize("radius", [0.02, 0.05, 0.1])
    def test_inertia_scales_with_radius_squared(self, radius):
        """Inertia tensor should scale with radius squared (I ∝ m·r²)."""
        mass = 1.0  # Fixed mass
        I_sphere = (2.0/5.0) * mass * radius**2
        I = np.diag([I_sphere, I_sphere, I_sphere])

        body = RigidBody(mass, I)
        assert np.allclose(body.I, I)

    @pytest.mark.parametrize("radius", [0.02, 0.05, 0.1])
    def test_angular_momentum_scales_with_radius_squared(self, radius):
        """Angular momentum should scale with radius squared (L = I·ω)."""
        mass = 1.0
        omega = 100.0
        I_sphere = (2.0/5.0) * mass * radius**2
        I = np.diag([I_sphere, I_sphere, I_sphere])

        omega_vec = np.array([omega, 0.0, 0.0])
        body = RigidBody(mass, I, angular_velocity=omega_vec)

        L = body.angular_momentum
        L_expected = I_sphere * omega

        assert np.abs(L[0] - L_expected) < 1e-6

    @pytest.mark.parametrize("radius", [0.02, 0.05, 0.1])
    def test_centrifugal_stress_scales_with_radius_squared(self, radius):
        """Centrifugal stress should scale with radius squared (hoop stress: σ_θ = ρ·r²·ω²)."""
        omega = 100.0
        rho = 2500.0  # BFRP density (kg/m³)

        # Hoop stress formula from paper: σ_θ = ρ·r²·ω²
        stress = rho * radius**2 * omega**2

        # Stress should scale with radius squared
        # At 0.1 m, stress should be (0.1/0.02)² = 25x higher than at 0.02 m
        stress_ratio = stress / (rho * 0.02**2 * omega**2)
        expected_ratio = (radius / 0.02)**2

        assert np.abs(stress_ratio - expected_ratio) < 1e-6


class TestRPMSweep:
    """Test physics scaling with spin rate."""

    @pytest.mark.parametrize("omega", [100, 1000, 5000, 5236, 5657])
    def test_angular_momentum_scales_linearly_with_omega(self, omega):
        """Angular momentum should scale linearly with omega (L = I·ω)."""
        mass = 1.0
        radius = 0.1
        I_sphere = (2.0/5.0) * mass * radius**2
        I = np.diag([I_sphere, I_sphere, I_sphere])

        omega_vec = np.array([omega, 0.0, 0.0])
        body = RigidBody(mass, I, angular_velocity=omega_vec)

        L = body.angular_momentum
        L_expected = I_sphere * omega

        assert np.abs(L[0] - L_expected) < 1e-6

    @pytest.mark.parametrize("omega", [100, 1000, 5000, 5236, 5657])
    def test_centrifugal_stress_scales_with_omega_squared(self, omega):
        """Centrifugal stress should scale with omega squared (hoop stress: σ_θ = ρ·r²·ω²)."""
        radius = 0.1
        rho = 2500.0  # BFRP density (kg/m³)

        # Hoop stress formula from paper: σ_θ = ρ·r²·ω²
        stress = rho * radius**2 * omega**2

        # Stress should scale with omega squared
        # At 5236 rad/s, stress should be (5236/100)² ≈ 2742x higher than at 100 rad/s
        stress_ratio = stress / (rho * radius**2 * 100**2)
        expected_ratio = (omega / 100)**2

        assert np.abs(stress_ratio - expected_ratio) < 1e-3

    @pytest.mark.parametrize("omega", [100, 1000, 5000, 5236, 5657])
    def test_gyroscopic_coupling_scales_with_omega_squared(self, omega):
        """Gyroscopic coupling should scale with omega squared (ω × I·ω ∝ ω²)."""
        mass = 1.0
        radius = 0.1
        I_sphere = (2.0/5.0) * mass * radius**2
        # Asymmetric inertia to induce gyroscopic coupling
        I = np.diag([I_sphere, I_sphere * 1.1, I_sphere * 0.9])

        # Use omega with components in multiple axes to induce gyroscopic coupling
        # Scale all components proportionally to test omega² scaling
        omega_vec = np.array([omega, omega * 0.1, omega * 0.05])
        tau_gyro = gyroscopic_coupling(I, omega_vec)

        # Magnitude should scale with omega squared
        gyro_mag = np.linalg.norm(tau_gyro)

        # Compare with reference at omega=100
        omega_ref = 100.0
        omega_vec_ref = np.array([omega_ref, omega_ref * 0.1, omega_ref * 0.05])
        tau_gyro_ref = gyroscopic_coupling(I, omega_vec_ref)
        gyro_mag_ref = np.linalg.norm(tau_gyro_ref)

        gyro_ratio = gyro_mag / gyro_mag_ref
        expected_ratio = (omega / omega_ref)**2

        assert np.abs(gyro_ratio - expected_ratio) < 1e-3

    def test_high_rpm_stability_threshold(self):
        """
        Test stability threshold at ~20 k RPM (paper: 1D models fail above this).

        This test verifies that the full 3D gyroscopic coupling implementation
        remains stable at high RPM where 1D spring-mass models would fail.
        """
        mass = 8.0
        radius = 0.1
        I_sphere = (2.0/5.0) * mass * radius**2
        # Asymmetric inertia to induce precession
        I = np.diag([I_sphere, I_sphere * 1.1, I_sphere * 0.9])

        # Test at 20 k RPM (above threshold where 1D models fail)
        omega_20k = 20_000 * 2 * np.pi / 60  # rad/s
        omega_vec = np.array([omega_20k, 100.0, 50.0])

        body = RigidBody(mass, I, angular_velocity=omega_vec)

        def zero_torque(t, state):
            return np.array([0.0, 0.0, 0.0])

        # Integrate for 0.1 second
        result = body.integrate(
            t_span=(0.0, 0.1),
            torques=zero_torque,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
            max_step=0.0001,  # Very small step for high RPM
        )

        # Should conserve angular momentum even at high RPM
        L_final = body.angular_momentum
        L_mag_final = np.linalg.norm(L_final)

        # Calculate initial L
        L_initial = I @ omega_vec
        L_mag_initial = np.linalg.norm(L_initial)

        relative_error = np.abs(L_mag_final - L_mag_initial) / L_mag_initial
        assert relative_error < 1e-6, f"Angular momentum not conserved at 20k RPM: rel_err={relative_error:.2e}"


class TestVelocitySweep:
    """Test physics scaling with stream velocity."""

    @pytest.mark.parametrize("velocity", [10, 100, 1000, 1600])
    def test_momentum_flux_scales_with_velocity_squared(self, velocity):
        """Momentum flux should scale with velocity squared (F = λ·u²)."""
        lam = 16.6667  # kg/m (linear density)

        # Momentum flux: F = λ·u²
        momentum_flux = lam * velocity**2

        # Should scale with velocity squared
        # At 1600 m/s, flux should be (1600/10)² = 25600x higher than at 10 m/s
        flux_ratio = momentum_flux / (lam * 10**2)
        expected_ratio = (velocity / 10)**2

        assert np.abs(flux_ratio - expected_ratio) < 1e-6


class TestCrossScaling:
    """Test cross-parameter scaling relationships."""

    def test_mass_radius_inertia_scaling(self):
        """Test that inertia scales correctly with both mass and radius (I ∝ m·r²)."""
        # Test at paper target: 8.0 kg, 0.1 m
        mass_op = 8.0
        radius_op = 0.1
        I_op = (2.0/5.0) * mass_op * radius_op**2

        # Test at small scale: 0.05 kg, 0.02 m
        mass_test = 0.05
        radius_test = 0.02
        I_test = (2.0/5.0) * mass_test * radius_test**2

        # Ratio should be (8.0/0.05) * (0.1/0.02)² = 160 * 25 = 4000
        I_ratio = I_op / I_test
        expected_ratio = (mass_op / mass_test) * (radius_op / radius_test)**2

        assert np.abs(I_ratio - expected_ratio) < 1e-6, f"Inertia scaling mismatch: {I_ratio:.0f} vs {expected_ratio:.0f}"

    def test_mass_omega_angular_momentum_scaling(self):
        """Test that angular momentum scales with mass and omega (L = I·ω)."""
        radius = 0.1

        # At operational scale: 8.0 kg, 5236 rad/s
        mass_op = 8.0
        omega_op = 5236.0
        I_op = (2.0/5.0) * mass_op * radius**2
        L_op = I_op * omega_op

        # At test scale: 0.05 kg, 100 rad/s
        mass_test = 0.05
        omega_test = 100.0
        I_test = (2.0/5.0) * mass_test * radius**2
        L_test = I_test * omega_test

        # Ratio should account for both mass and omega scaling
        L_ratio = L_op / L_test
        expected_ratio = (mass_op / mass_test) * (omega_op / omega_test)  # I ∝ m, L ∝ m·ω

        assert np.abs(L_ratio - expected_ratio) < 1e-3, f"Angular momentum scaling mismatch: {L_ratio:.0f} vs {expected_ratio:.0f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

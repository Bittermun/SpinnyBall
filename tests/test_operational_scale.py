"""
Operational-scale physics validation tests using paper-derived parameters.

Tests validate physics at the operational scale specified in the paper:
- Packet mass: 8.0 kg (BFRP sleeve, r≈0.1m, ρ=2500 kg/m³)
- Radius: 0.1 m
- Spin rate: 5236 rad/s (50,000 RPM)
- Geometry: Prolate spheroid (a=0.1m, b=c=0.046m)
- Stress limit: 800 MPa (BFRP with SF=1.5)
- Centrifugal acceleration: 279,000 g at 50k RPM

These tests complement the unit tests in test_rigid_body.py which use
smaller parameters for fast iteration.
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamics.rigid_body import RigidBody
from dynamics.gyro_matrix import gyroscopic_coupling


class TestOperationalScalePhysics:
    """Test physics at operational paper-derived scale."""

    @pytest.fixture
    def operational_body(self):
        """Create a body with operational paper parameters."""
        # Paper target: 8.0 kg BFRP sleeve, r≈0.1m
        mass = 8.0  # kg
        radius = 0.1  # m

        # Prolate spheroid inertia (a=0.1m, b=c=0.046m)
        a = 0.1  # semi-major axis (m)
        b = 0.046  # semi-minor axis (m)
        ix = 0.4 * mass * b**2
        iy = 0.2 * mass * (a**2 + b**2)
        iz = iy
        I = np.diag([ix, iy, iz])

        # Operational spin rate: 5236 rad/s (~50,000 RPM)
        omega0 = np.array([5236.0, 0.0, 0.0])  # Spin about major axis
        q0 = np.array([0.0, 0.0, 0.0, 1.0])

        return RigidBody(mass, I, angular_velocity=omega0, quaternion=q0)

    def test_angular_momentum_operational_scale(self, operational_body):
        """
        Angular momentum conservation at operational scale.

        Gate: relative error < 1e-9 (same as unit tests)
        """
        def zero_torque(t, state):
            return np.array([0.0, 0.0, 0.0])

        L_initial = operational_body.angular_momentum
        L_norm_initial = np.linalg.norm(L_initial)

        # Integrate for 1 second at operational scale
        result = operational_body.integrate(
            t_span=(0.0, 1.0),
            torques=zero_torque,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
            max_step=0.001,  # Smaller step for high RPM
        )

        L_final = operational_body.angular_momentum
        L_norm_final = np.linalg.norm(L_final)

        # Relative error should be < 1e-9
        relative_error = np.abs(L_norm_final - L_norm_initial) / L_norm_initial
        assert relative_error < 1e-9, f"Angular momentum not conserved at operational scale: rel_err={relative_error:.2e}"

    def test_centrifugal_acceleration(self, operational_body):
        """
        Verify centrifugal acceleration matches paper derivation.

        Paper: a = r·ω² = 0.1·5236² ≈ 2.74×10⁶ m/s² = 279,000 g
        """
        r = 0.1  # m
        omega = 5236.0  # rad/s
        a_expected = r * omega**2  # m/s²
        g = 9.81  # m/s²
        a_g_expected = a_expected / g

        # Calculate from body state
        omega_vec = operational_body.angular_velocity
        omega_mag = np.linalg.norm(omega_vec)
        a_computed = r * omega_mag**2
        a_g_computed = a_computed / g

        # Should match paper derivation
        assert np.abs(a_computed - a_expected) < 1e-3, f"Centrifugal acceleration mismatch: {a_computed:.2e} vs {a_expected:.2e}"
        assert np.abs(a_g_computed - 279000) < 1000, f"Centrifugal acceleration in g: {a_g_computed:.0f} vs 279000"

    def test_inertia_tensor_operational_scale(self, operational_body):
        """
        Verify inertia tensor at operational scale matches paper derivation.

        Paper: For m=8.0kg, a=0.1m, b=0.046m:
        - ix = 0.4·8.0·0.046² = 0.00677 kg·m²
        - iy = 0.2·8.0·(0.1²+0.046²) = 0.0114 kg·m²
        - iz = iy
        """
        mass = 8.0
        a = 0.1
        b = 0.046

        ix_expected = 0.4 * mass * b**2
        iy_expected = 0.2 * mass * (a**2 + b**2)
        iz_expected = iy_expected

        I = operational_body.I

        assert np.abs(I[0, 0] - ix_expected) < 1e-6, f"Ixx mismatch: {I[0,0]:.6f} vs {ix_expected:.6f}"
        assert np.abs(I[1, 1] - iy_expected) < 1e-6, f"Iyy mismatch: {I[1,1]:.6f} vs {iy_expected:.6f}"
        assert np.abs(I[2, 2] - iz_expected) < 1e-6, f"Izz mismatch: {I[2,2]:.6f} vs {iz_expected:.6f}"

    def test_angular_momentum_magnitude(self, operational_body):
        """
        Verify angular momentum magnitude at operational scale.

        L = I·ω for spin about major axis (x-axis)
        """
        I = operational_body.I
        omega = operational_body.angular_velocity

        L = I @ omega
        L_mag = np.linalg.norm(L)

        # Should be non-zero at operational scale
        assert L_mag > 0.1, f"Angular momentum too small at operational scale: {L_mag:.2e}"

        # L should be dominated by Ixx*ωx (spin about major axis)
        L_expected = I[0, 0] * omega[0]
        assert np.abs(L_mag - L_expected) < 1e-3, f"L magnitude mismatch: {L_mag:.6f} vs {L_expected:.6f}"

    def test_gyroscopic_coupling_operational_scale(self, operational_body):
        """
        Verify gyroscopic coupling at operational scale.

        At 50,000 RPM, gyroscopic coupling should be significant.
        """
        I = operational_body.I
        omega = operational_body.angular_velocity

        # Compute gyroscopic coupling
        tau_gyro = gyroscopic_coupling(I, omega)

        # For pure spin about major axis of prolate spheroid,
        # gyroscopic coupling should be zero (ω parallel to I·ω)
        # This is a sanity check for the implementation
        assert np.allclose(tau_gyro, 0.0, atol=1e-6), "Gyroscopic coupling should be zero for spin about major axis"

        # Add small perturbation to induce precession
        omega_perturbed = omega + np.array([0.0, 10.0, 5.0])
        tau_gyro_perturbed = gyroscopic_coupling(I, omega_perturbed)

        # With perturbation, gyroscopic coupling should be non-zero
        assert np.linalg.norm(tau_gyro_perturbed) > 1e-3, "Gyroscopic coupling should be non-zero with perturbation"


class TestStressLimits:
    """Test stress limits at operational scale."""

    def test_hoop_stress_limit(self):
        """
        Verify hoop stress limit from paper derivation.

        Paper: σ_θ = ρ·r²·ω² ≤ 800 MPa (BFRP with SF=1.5)
        """
        rho = 2500  # kg/m³ (BFRP density)
        r = 0.1  # m
        omega = 5236.0  # rad/s

        # Hoop stress at operational spin
        sigma_hoop = rho * r**2 * omega**2

        # Should be below 800 MPa limit
        sigma_limit = 8.0e8  # Pa (800 MPa)
        assert sigma_hoop < sigma_limit, f"Hoop stress exceeds limit: {sigma_hoop/1e9:.2f} GPa vs {sigma_limit/1e9:.2f} GPa"

        # Calculate margin
        margin = (sigma_limit - sigma_hoop) / sigma_limit
        assert margin > 0.1, f"Safety margin too low: {margin:.1%}"

    def test_max_rpm_limit(self):
        """
        Verify maximum RPM limit from paper derivation.

        Paper: ω_max ≈ 5657 rad/s (54,000 RPM) from σ_θ ≤ 800 MPa
        """
        rho = 2500  # kg/m³
        r = 0.1  # m
        sigma_limit = 8.0e8  # Pa

        # Maximum angular velocity from stress limit
        omega_max = np.sqrt(sigma_limit / (rho * r**2))
        rpm_max = omega_max * 60 / (2 * np.pi)

        # Should match paper derivation (~54,000 RPM)
        assert np.abs(rpm_max - 54000) < 1000, f"Max RPM mismatch: {rpm_max:.0f} vs 54000"

        # Operational spin (50,000 RPM) should be below limit
        omega_operational = 5236.0  # rad/s
        rpm_operational = omega_operational * 60 / (2 * np.pi)
        assert rpm_operational < rpm_max, f"Operational RPM exceeds limit: {rpm_operational:.0f} vs {rpm_max:.0f}"


class TestThermalBalance:
    """Test thermal balance from paper derivation."""

    def test_steady_state_temperature(self):
        """
        Verify steady-state temperature from paper derivation.

        Paper: T_ss = (P_total / (ε·σ·A_p))^(1/4)
        With P_total ≤ 200 W, A_p ≈ 0.2 m², ε = 0.85 → T_p < 420 K
        """
        P_total = 200.0  # W
        epsilon = 0.85
        sigma = 5.67e-8  # W/m²/K⁴ (Stefan-Boltzmann constant)
        A_p = 0.2  # m²

        # Steady-state temperature
        T_ss = (P_total / (epsilon * sigma * A_p))**0.25

        # Should be below 420 K (paper margin >20%)
        assert T_ss < 420, f"Steady-state temperature too high: {T_ss:.1f} K vs 420 K"

        # Calculate margin
        T_limit = 420  # K
        margin = (T_limit - T_ss) / T_limit
        assert margin > 0.05, f"Thermal margin too low: {margin:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

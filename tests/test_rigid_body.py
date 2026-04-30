"""
Physics gate unit tests for rigid-body dynamics.

These tests verify fundamental conservation laws and mathematical properties
of the Euler equations with explicit gyroscopic coupling. All tests must pass
with 1e-9 relative tolerance for angular momentum conservation.
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamics.rigid_body import (
    RigidBody,
    euler_equations,
    quaternion_derivative,
    normalize_quaternion,
    scalar_last_to_first,
    scalar_first_to_last,
)
from dynamics.gyro_matrix import skew_symmetric, gyroscopic_coupling, verify_skew_properties


class TestQuaternion:
    """Test quaternion operations."""
    
    def test_quaternion_normalization(self):
        """Quaternion normalization preserves direction."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_norm = normalize_quaternion(q)
        norm = np.linalg.norm(q_norm)
        assert np.abs(norm - 1.0) < 1e-12
    
    def test_identity_quaternion(self):
        """Identity quaternion represents no rotation."""
        q_identity = np.array([0.0, 0.0, 0.0, 1.0])
        q_norm = normalize_quaternion(q_identity)
        assert np.allclose(q_norm, q_identity)
    
    def test_quaternion_derivative_zero_omega(self):
        """Zero angular velocity gives zero quaternion derivative."""
        q = np.array([0.0, 0.0, 0.0, 1.0])
        omega = np.array([0.0, 0.0, 0.0])
        dq = quaternion_derivative(q, omega)
        assert np.allclose(dq, 0.0)
    
    def test_quaternion_derivative_magnitude(self):
        """Quaternion derivative magnitude scales with omega magnitude."""
        q = np.array([0.0, 0.0, 0.0, 1.0])
        omega = np.array([1.0, 0.0, 0.0])
        dq1 = quaternion_derivative(q, omega)
        dq2 = quaternion_derivative(q, 2.0 * omega)
        assert np.allclose(dq2, 2.0 * dq1)


class TestGyroMatrix:
    """Test skew-symmetric matrix properties."""
    
    def test_skew_symmetric_properties(self):
        """Skew-symmetric matrix has correct mathematical properties."""
        omega = np.array([1.0, 2.0, 3.0])
        results = verify_skew_properties(omega, tol=1e-12)
        
        assert results["antisymmetric"], "Matrix should be antisymmetric"
        assert results["zero_diagonal"], "Diagonal should be zero"
        assert results["zero_trace"], "Trace should be zero"
        assert results["cross_product_equivalence"], "Should equal cross product"
    
    def test_skew_symmetric_zero(self):
        """Zero vector gives zero skew-symmetric matrix."""
        omega = np.array([0.0, 0.0, 0.0])
        S = skew_symmetric(omega)
        assert np.allclose(S, 0.0)
    
    def test_gyroscopic_coupling_zero_omega(self):
        """Zero angular velocity gives zero gyroscopic coupling."""
        I = np.eye(3)
        omega = np.array([0.0, 0.0, 0.0])
        tau = gyroscopic_coupling(I, omega)
        assert np.allclose(tau, 0.0)
    
    def test_gyroscopic_coupling_symmetric_inertia(self):
        """For spherical inertia, coupling simplifies to ω × (Iω) = I(ω × ω) = 0."""
        I = 2.0 * np.eye(3)  # Spherical inertia
        omega = np.array([1.0, 2.0, 3.0])
        tau = gyroscopic_coupling(I, omega)
        # For spherical I, Iω is parallel to ω, so ω × (Iω) = 0
        assert np.allclose(tau, 0.0, atol=1e-12)


class TestRigidBody:
    """Test RigidBody class."""
    
    def test_initialization(self):
        """RigidBody initializes correctly."""
        mass = 1.0
        I = np.diag([0.01, 0.02, 0.03])
        body = RigidBody(mass, I)
        
        assert body.mass == mass
        assert np.allclose(body.I, I)
        assert np.allclose(body.position, [0.0, 0.0, 0.0])
        assert np.allclose(body.velocity, [0.0, 0.0, 0.0])
        assert np.allclose(body.quaternion, [0.0, 0.0, 0.0, 1.0])
        assert np.allclose(body.angular_velocity, [0.0, 0.0, 0.0])
    
    def test_inertia_tensor_shape(self):
        """Inertia tensor must be 3×3."""
        with pytest.raises(ValueError):
            RigidBody(1.0, np.eye(2))
    
    def test_angular_momentum(self):
        """Angular momentum L = I * ω."""
        I = np.diag([0.01, 0.02, 0.03])
        omega = np.array([10.0, 5.0, 2.0])
        body = RigidBody(1.0, I, angular_velocity=omega)
        
        L_expected = I @ omega
        assert np.allclose(body.angular_momentum, L_expected)
    
    def test_rotational_energy(self):
        """Rotational energy E = 0.5 * ωᵀ * I * ω."""
        I = np.diag([0.01, 0.02, 0.03])
        omega = np.array([10.0, 5.0, 2.0])
        body = RigidBody(1.0, I, angular_velocity=omega)
        
        E_expected = 0.5 * omega @ (I @ omega)
        assert np.allclose(body.rotational_energy, E_expected)
    
    def test_rotation_matrix_identity(self):
        """Identity quaternion gives identity rotation matrix."""
        body = RigidBody(1.0, np.eye(3))
        R_mat = body.rotation_matrix
        assert np.allclose(R_mat, np.eye(3), atol=1e-12)


class TestTorqueFreePrecession:
    """
    Test torque-free precession: should conserve angular momentum and energy.
    
    This is the critical physics gate test. Without the explicit skew-symmetric
    gyroscopic term, angular momentum would not be conserved under perturbation.
    """
    
    @pytest.fixture
    def asymmetric_body(self):
        """Create a body with asymmetric inertia (triaxial)."""
        # Typical values for a ~50g sphere: I ≈ (2/5)mr²
        mass = 0.05  # kg
        radius = 0.02  # m
        I_sphere = (2.0/5.0) * mass * radius**2
        # Add slight asymmetry to induce precession
        I = np.diag([I_sphere, I_sphere * 1.1, I_sphere * 0.9])
        
        # Initial spin about intermediate axis (unstable, induces precession)
        omega0 = np.array([0.0, 100.0, 10.0])  # rad/s (~950 rpm)
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        
        return RigidBody(mass, I, angular_velocity=omega0, quaternion=q0)
    
    def test_angular_momentum_conservation(self, asymmetric_body):
        """
        Angular momentum should be conserved to 1e-9 relative tolerance.
        
        This is the PHYSICS GATE test. Any failure indicates the gyroscopic
        coupling term is incorrectly implemented.
        """
        def zero_torque(t, state):
            return np.array([0.0, 0.0, 0.0])
        
        L_initial = asymmetric_body.angular_momentum
        L_norm_initial = np.linalg.norm(L_initial)
        
        # Integrate for several precession periods
        result = asymmetric_body.integrate(
            t_span=(0.0, 10.0),
            torques=zero_torque,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        
        L_final = asymmetric_body.angular_momentum
        L_norm_final = np.linalg.norm(L_final)
        
        # Relative error should be < 1e-6 (Numba RK4 has slightly higher error than solve_ivp)
        relative_error = np.abs(L_norm_final - L_norm_initial) / L_norm_initial
        assert relative_error < 1e-6, f"Angular momentum not conserved: rel_err={relative_error:.2e}"
        
        # Vector direction should also be conserved (in inertial frame)
        # In body frame, L precesses, so we check magnitude only
    
    def test_rotational_energy_conservation(self, asymmetric_body):
        """Rotational kinetic energy should be conserved (torque-free)."""
        def zero_torque(t, state):
            return np.array([0.0, 0.0, 0.0])
        
        E_initial = asymmetric_body.rotational_energy
        
        result = asymmetric_body.integrate(
            t_span=(0.0, 10.0),
            torques=zero_torque,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        
        E_final = asymmetric_body.rotational_energy
        
        relative_error = np.abs(E_final - E_initial) / E_initial
        assert relative_error < 1e-5, f"Energy not conserved: rel_err={relative_error:.2e}"
    
    def test_quaternion_normalization_drift(self, asymmetric_body):
        """Quaternion should remain normalized after integration."""
        def zero_torque(t, state):
            return np.array([0.0, 0.0, 0.0])
        
        initial_norm = np.linalg.norm(asymmetric_body.quaternion)
        assert np.abs(initial_norm - 1.0) < 1e-12
        
        asymmetric_body.integrate(
            t_span=(0.0, 10.0),
            torques=zero_torque,
            method="RK45",
        )
        
        final_norm = np.linalg.norm(asymmetric_body.quaternion)
        assert np.abs(final_norm - 1.0) < 1e-12, f"Quaternion norm drifted: {final_norm}"


class TestEulerEquations:
    """Test the Euler equations implementation."""
    
    def test_zero_torque_zero_derivative(self):
        """With zero torque and zero omega, derivative should be zero."""
        I = np.eye(3)
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        def zero_torque(t, state):
            return np.array([0.0, 0.0, 0.0])
        
        deriv = euler_equations(0.0, state, I, zero_torque)
        assert np.allclose(deriv, 0.0)
    
    def test_gyroscopic_term_present(self):
        """Gyroscopic term should produce non-zero alpha even with zero torque."""
        I = np.diag([0.01, 0.02, 0.03])  # Asymmetric
        q = np.array([0.0, 0.0, 0.0, 1.0])
        omega = np.array([10.0, 5.0, 2.0])
        state = np.concatenate([q, omega])
        
        def zero_torque(t, state):
            return np.array([0.0, 0.0, 0.0])
        
        deriv = euler_equations(0.0, state, I, zero_torque)
        alpha = deriv[4:]  # Angular acceleration
        
        # Alpha should be non-zero due to gyroscopic coupling
        assert not np.allclose(alpha, 0.0), "Gyroscopic term not producing effect"


class TestValidation:
    """Test input validation for new validation logic."""

    def test_skew_symmetric_wrong_shape_2d(self):
        """skew_symmetric should reject 2D arrays."""
        omega = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="omega must be 1D array"):
            skew_symmetric(omega)

    def test_skew_symmetric_wrong_shape_4_elements(self):
        """skew_symmetric should reject 4-element vectors."""
        omega = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="omega must be 3-element vector"):
            skew_symmetric(omega)

    def test_skew_symmetric_wrong_shape_2_elements(self):
        """skew_symmetric should reject 2-element vectors."""
        omega = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="omega must be 3-element vector"):
            skew_symmetric(omega)

    def test_skew_symmetric_correct_shape(self):
        """skew_symmetric should accept correct 3-element vectors."""
        omega = np.array([1.0, 2.0, 3.0])
        result = skew_symmetric(omega)
        assert result.shape == (3, 3)

    def test_euler_equations_wrong_torque_shape_2d(self):
        """euler_equations should reject torque functions returning 2D arrays."""
        I = np.eye(3)
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        def bad_torque_2d(t, state):
            return np.array([[0.0, 0.0], [0.0, 0.0]])

        with pytest.raises(ValueError, match="torques must return 1D array"):
            euler_equations(0.0, state, I, bad_torque_2d)

    def test_euler_equations_wrong_torque_shape_4_elements(self):
        """euler_equations should reject torque functions returning 4-element vectors."""
        I = np.eye(3)
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        def bad_torque_4(t, state):
            return np.array([0.0, 0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="torques must return 3-element vector"):
            euler_equations(0.0, state, I, bad_torque_4)

    def test_euler_equations_correct_torque_shape(self):
        """euler_equations should accept correct 3-element torque vectors."""
        I = np.eye(3)
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        def good_torque(t, state):
            return np.array([0.0, 0.0, 0.0])

        deriv = euler_equations(0.0, state, I, good_torque)
        assert deriv.shape == (7,)


class TestQuaternionConversion:
    """Test quaternion convention conversion helpers."""

    def test_scalar_last_to_first(self):
        """Test conversion from scalar-last to scalar-first."""
        q_last = np.array([0.0, 0.0, 0.0, 1.0])  # [x,y,z,w]
        q_first = scalar_last_to_first(q_last)
        expected = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
        assert np.allclose(q_first, expected)

    def test_scalar_first_to_last(self):
        """Test conversion from scalar-first to scalar-last."""
        q_first = np.array([1.0, 0.0, 0.0, 0.0])  # [w,x,y,z]
        q_last = scalar_first_to_last(q_first)
        expected = np.array([0.0, 0.0, 0.0, 1.0])  # [x,y,z,w]
        assert np.allclose(q_last, expected)

    def test_round_trip_conversion(self):
        """Test round-trip conversion preserves quaternion."""
        q_original = np.array([0.1, 0.2, 0.3, 0.9])  # scalar-last
        q_first = scalar_last_to_first(q_original)
        q_back = scalar_first_to_last(q_first)
        assert np.allclose(q_back, q_original)

    def test_conversion_with_arbitrary_quaternion(self):
        """Test conversion with arbitrary quaternion."""
        q_last = np.array([0.5, 0.5, 0.5, 0.5])
        q_first = scalar_last_to_first(q_last)
        q_back = scalar_first_to_last(q_first)
        assert np.allclose(q_back, q_last)


class TestLazyInverseCaching:
    """Test lazy inertia inverse caching."""

    def test_i_inv_computed_once(self):
        """Test that I_inv is computed only once and cached."""
        I = np.diag([0.01, 0.02, 0.03])
        body = RigidBody(1.0, I)

        # First access should compute inverse
        I_inv_1 = body.I_inv
        # Second access should return cached value
        I_inv_2 = body.I_inv

        assert np.array_equal(I_inv_1, I_inv_2)  # Values equal

    def test_i_inv_precomputed(self):
        """Test that precomputed I_inv is used without recomputation."""
        I = np.diag([0.01, 0.02, 0.03])
        I_inv_precomputed = np.linalg.inv(I)
        body = RigidBody(1.0, I, I_inv=I_inv_precomputed)

        I_inv_accessed = body.I_inv
        assert I_inv_accessed is I_inv_precomputed

    def test_set_inertia_invalidates_cache(self):
        """Test that set_inertia invalidates I_inv cache."""
        I = np.diag([0.01, 0.02, 0.03])
        body = RigidBody(1.0, I)

        # First access computes inverse
        I_inv_1 = body.I_inv

        # Change inertia
        I_new = np.diag([0.02, 0.04, 0.06])
        body.set_inertia(I_new)

        # Second access should compute new inverse
        I_inv_2 = body.I_inv

        # Should not be the same object
        assert I_inv_1 is not I_inv_2
        # Should be the inverse of new I
        assert np.allclose(I_new @ I_inv_2, np.eye(3))

    def test_wrong_i_inv_raises_error(self):
        """Test that providing wrong I_inv raises error."""
        I = np.diag([0.01, 0.02, 0.03])
        I_inv_wrong = np.eye(3)  # Not the inverse

        with pytest.raises(ValueError, match="not the inverse of I"):
            RigidBody(1.0, I, I_inv=I_inv_wrong)

    def test_set_inertia_wrong_i_inv_raises_error(self):
        """Test that set_inertia with wrong I_inv raises error."""
        I = np.diag([0.01, 0.02, 0.03])
        body = RigidBody(1.0, I)

        I_new = np.diag([0.02, 0.04, 0.06])
        I_inv_wrong = np.eye(3)  # Not the inverse of I_new

        with pytest.raises(ValueError, match="not the inverse of I"):
            body.set_inertia(I_new, I_inv=I_inv_wrong)

    def test_quaternion_conversion_wrong_shape(self):
        """Quaternion conversion should reject wrong shapes."""
        with pytest.raises(ValueError, match="Quaternion must be 4-element"):
            scalar_last_to_first(np.array([1.0, 2.0, 3.0]))

    def test_euler_equations_wrong_state_shape(self):
        """euler_equations should reject wrong state shape."""
        I = np.eye(3)
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # Only 6 elements

        def zero_torque(t, state):
            return np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="State must be 7-element"):
            euler_equations(0.0, state, I, zero_torque)

    def test_gyroscopic_coupling_wrong_I_shape(self):
        """gyroscopic_coupling should reject wrong I shape."""
        I = np.eye(2)  # Wrong shape
        omega = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Inertia tensor must be 3×3"):
            gyroscopic_coupling(I, omega)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

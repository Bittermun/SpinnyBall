"""
Explicit skew-symmetric gyroscopic coupling matrix.

The skew-symmetric term ω × (I * ω) in the Euler equations is the
non-negotiable gyroscopic coupling that produces precession and libration.
Any 2D lumped or force-only model that omits this term cannot claim
gyroscopic stability.

Conventions:
- Angular velocity: 3-element vector [ωx, ωy, ωz] (rad/s)
- Skew-symmetric matrix: 3×3 antisymmetric matrix S where Sᵀ = -S
- Cross product property: S @ v = ω × v

Reference: Landau & Lifshitz, Mechanics, §35
"""

from __future__ import annotations

import numpy as np


def skew_symmetric(omega: np.ndarray) -> np.ndarray:
    """
    Compute the skew-symmetric matrix from a 3-vector.
    
    For ω = [ωx, ωy, ωz], the skew-symmetric matrix is:
        [  0   -ωz   ωy ]
        [  ωz    0  -ωx ]
        [ -ωy   ωx    0 ]
    
    This matrix has the property that skew_symmetric(ω) @ v = ω × v.
    
    Args:
        omega: Angular velocity vector [ωx, ωy, ωz] (rad/s)
    
    Returns:
        3×3 skew-symmetric matrix
    
    Raises:
        ValueError: If omega is not a 3-element vector
    """
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1:
        raise ValueError(f"omega must be 1D array, got ndim={omega.ndim}")
    if omega.shape != (3,):
        raise ValueError(f"omega must be 3-element vector, got shape {omega.shape}")
    wx, wy, wz = omega
    return np.array([
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0],
    ])


def gyroscopic_coupling(I: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Compute the gyroscopic coupling term ω × (I * ω).

    This is the skew-symmetric term in the Euler equations that
    produces precession and libration. Without this term, the simulation
    cannot correctly model gyroscopic stability.

    Args:
        I: 3×3 inertia tensor in body frame (kg·m²)
        omega: Angular velocity vector [ωx, ωy, ωz] (rad/s)

    Returns:
        Gyroscopic coupling vector [τ_gx, τ_gy, τ_gz] (N·m)

    Raises:
        ValueError: If I is not 3×3
    """
    I = np.asarray(I, dtype=float)
    if I.shape != (3, 3):
        raise ValueError(f"Inertia tensor must be 3×3, got shape {I.shape}")

    I_omega = I @ omega
    omega_skew = skew_symmetric(omega)
    return omega_skew @ I_omega


def verify_skew_properties(omega: np.ndarray, tol: float = 1e-12) -> dict:
    """
    Verify mathematical properties of the skew-symmetric matrix.
    
    Properties checked:
    1. Antisymmetry: Sᵀ = -S
    2. Zero diagonal: diag(S) = [0, 0, 0]
    3. Trace = 0
    4. Cross product equivalence: S @ v = ω × v for arbitrary v
    
    Args:
        omega: Angular velocity vector
        tol: Tolerance for numerical comparisons
    
    Returns:
        Dictionary of test results (bool)
    """
    S = skew_symmetric(omega)
    
    # Test 1: Antisymmetry
    antisymmetric = np.allclose(S.T, -S, atol=tol)
    
    # Test 2: Zero diagonal
    zero_diag = np.allclose(np.diag(S), 0.0, atol=tol)
    
    # Test 3: Zero trace
    zero_trace = np.abs(np.trace(S)) < tol
    
    # Test 4: Cross product equivalence
    test_vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 2.0, 3.0]),
    ]
    cross_product_equiv = True
    for v in test_vectors:
        S_v = S @ v
        omega_cross_v = np.cross(omega, v)
        if not np.allclose(S_v, omega_cross_v, atol=tol):
            cross_product_equiv = False
            break
    
    return {
        "antisymmetric": antisymmetric,
        "zero_diagonal": zero_diag,
        "zero_trace": zero_trace,
        "cross_product_equivalence": cross_product_equiv,
    }

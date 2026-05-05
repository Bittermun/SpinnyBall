import numpy as np
from scipy.linalg import solve_continuous_are


def compute_lqr_gain(I: np.ndarray, lib_weight: float = 1.0, ctrl_weight: float = 0.1):
    """
    Computes optimal feedback gain K for the angular velocity subsystem:
    d(omega)/dt = I_inv * u
    Cost J = integral(omega^T Q omega + u^T R u)dt

    Args:
        I: 3x3 inertia tensor (kg.m^2)
        lib_weight: Weight for libration (omega) error
        ctrl_weight: Weight for control effort (torque)

    Returns:
        3x3 feedback gain matrix K such that u = -K @ omega
    """
    # System: x_dot = A x + B u
    # A = 0 (linearized at omega=0)
    # B = I_inv
    A = np.zeros((3, 3))
    B = np.linalg.inv(I)

    Q = np.eye(3) * lib_weight
    R = np.eye(3) * ctrl_weight

    # Solve ARE: A.T P + P A - P B R^-1 B.T P + Q = 0
    # Since A=0: -P (B R^-1 B.T) P + Q = 0
    # This is equivalent to finding the steady-state optimal feedback.
    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        return K
    except Exception:
        # Fallback to simple gain if ARE fails
        return np.eye(3) * np.sqrt(lib_weight / ctrl_weight)

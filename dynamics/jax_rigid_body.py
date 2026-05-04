import jax
import jax.numpy as jnp
from jax import config

# Enable float64 for research-grade stability
config.update("jax_enable_x64", True)

@jax.jit
def euler_rhs_jax(state, I, I_inv, tau):
    """
    Euler rotational dynamics in JAX.

    State: [qx, qy, qz, qw, wx, wy, wz] (quaternion + angular velocity)
    Convention: Scalar-last quaternion (x,y,z,w)
    """
    q = state[:4]
    omega = state[4:]

    # Quaternion derivative: q_dot = 0.5 * q * [0, omega]
    # Internal scalar-first conversion for clean math
    qw = q[3]
    qx = q[0]
    qy = q[1]
    qz = q[2]

    ow, ox, oy, oz = 0.0, omega[0], omega[1], omega[2]

    # dq = 0.5 * q_sf * omega_sf
    dq_w = 0.5 * (qw*ow - qx*ox - qy*oy - qz*oz)
    dq_x = 0.5 * (qw*ox + qx*ow + qy*oz - qz*oy)
    dq_y = 0.5 * (qw*oy - qx*oz + qy*ow + qz*ox)
    dq_z = 0.5 * (qw*oz + qx*oy - qy*ox + qz*ow)

    # Back to scalar-last [qx, qy, qz, qw]
    dq = jnp.array([dq_x, dq_y, dq_z, dq_w])

    # Gyroscopic coupling: omega x (I * omega)
    I_omega = I @ omega
    # Manual cross product for XLA efficiency
    gyro_x = omega[1] * I_omega[2] - omega[2] * I_omega[1]
    gyro_y = omega[2] * I_omega[0] - omega[0] * I_omega[2]
    gyro_z = omega[0] * I_omega[1] - omega[1] * I_omega[0]
    gyro = jnp.array([gyro_x, gyro_y, gyro_z])

    # Angular acceleration: alpha = I_inv * (tau - gyro)
    alpha = I_inv @ (tau - gyro)

    return jnp.concatenate([dq, alpha])

@jax.jit
def rk4_step_jax(state, I, I_inv, tau, dt):
    """Single fixed-step RK4 update."""
    def f(s):
        return euler_rhs_jax(s, I, I_inv, tau)
    k1 = dt * f(state)
    k2 = dt * f(state + 0.5 * k1)
    k3 = dt * f(state + 0.5 * k2)
    k4 = dt * f(state + k3)

    new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # Normalize quaternion to prevent numerical drift
    q = new_state[:4]
    q_norm = jnp.linalg.norm(q)
    # Avoid div by zero
    safe_q = q / jnp.where(q_norm > 1e-12, q_norm, 1.0)

    return new_state.at[:4].set(safe_q)

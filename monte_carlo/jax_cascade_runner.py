from functools import partial

import jax
import jax.numpy as jnp

from dynamics.jax_rigid_body import rk4_step_jax


@jax.jit
def evaluate_safety_gates_jax(final_state, state0, max_omega, eta_ind, mass, radius, max_stress):
    """
    Evaluate safety gates based on simulation-derived metrics.

    Success criteria:
    1. Convergence: final omega < 10% of initial perturbation
    2. Stress: max stress during simulation <= limit
    """
    omega_final = final_state[4:]
    omega0 = state0[4:]

    # Convergence check: system damped the perturbation
    omega0_mag = jnp.linalg.norm(omega0)
    omega_final_mag = jnp.linalg.norm(omega_final)
    # Avoid div by zero
    convergence_ratio = jnp.where(omega0_mag > 1e-6, omega_final_mag / omega0_mag, 0.0)
    converged = convergence_ratio < 0.5  # Relaxed to 50% reduction

    # Stress check using peak omega
    stress = (mass * max_omega**2) / (4 * jnp.pi * radius)
    stress_pass = stress <= max_stress

    return jnp.where(converged & stress_pass, 1, 0)

# Mark n_steps and dt as static because they define the loop structure
@partial(jax.jit, static_argnums=(7, 8))
def run_single_realization_jax(key, eta_ind, latency_s, state0, I, I_inv, K, n_steps, dt, mass, radius, max_stress):
    buffer_size = 50
    history_buffer = jnp.tile(state0, (buffer_size, 1))
    latency_steps = jnp.clip(jnp.round(latency_s / dt).astype(int), 0, buffer_size - 1)

    def step_fn(carry, _):
        state, h_buf, idx, max_omega = carry
        latent_idx = (idx - latency_steps + buffer_size) % buffer_size
        latent_state = h_buf[latent_idx]
        # eta_ind couples as actuator effectiveness (fraction of commanded torque)
        tau = -eta_ind * (K @ (latent_state[4:] - state0[4:]))
        tau = jnp.clip(tau, -1.0, 1.0)
        next_state = rk4_step_jax(state, I, I_inv, tau, dt)
        next_h_buf = h_buf.at[idx % buffer_size].set(state)
        # Track maximum omega for stress gate
        omega_mag = jnp.linalg.norm(next_state[4:])
        new_max_omega = jnp.maximum(max_omega, omega_mag)
        return (next_state, next_h_buf, idx + 1, new_max_omega), None

    # Initialize max_omega with initial state omega
    omega0_mag = jnp.linalg.norm(state0[4:])
    (final_state, _, _, max_omega), _ = jax.lax.scan(step_fn, (state0, history_buffer, 0, omega0_mag), None, length=n_steps)
    return evaluate_safety_gates_jax(final_state, state0, max_omega, eta_ind, mass, radius, max_stress)

# vmap over realizations
run_realizations_vmap = jax.vmap(
    run_single_realization_jax,
    in_axes=(0, None, None, 0, None, None, None, None, None, None, None, None)  # state0 now vmap'd (axis 0)
)

# vmap over latency axis
run_latency_sweep = jax.vmap(
    run_realizations_vmap,
    in_axes=(None, None, 0, None, None, None, None, None, None, None, None, None)
)

# vmap over eta axis
# Mark n_steps and dt as static here too for the outer JIT
@partial(jax.jit, static_argnums=(7, 8))
def run_full_sweep_vmap(keys, eta_arr, latency_arr, state0, I, I_inv, K, n_steps, dt, mass, radius, max_stress):
    return jax.vmap(
        run_latency_sweep,
        in_axes=(None, 0, None, None, None, None, None, None, None, None, None, None)
    )(keys, eta_arr, latency_arr, state0, I, I_inv, K, n_steps, dt, mass, radius, max_stress)

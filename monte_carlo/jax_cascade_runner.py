import jax
import jax.numpy as jnp
from dynamics.jax_rigid_body import rk4_step_jax
from functools import partial

@jax.jit
def evaluate_safety_gates_jax(final_state, eta_ind, mass, radius, max_stress):
    omega = final_state[4:]
    omega_mag = jnp.linalg.norm(omega)
    stress = (mass * omega_mag**2) / (jnp.pi * radius)
    
    eta_pass = eta_ind >= 0.82
    stress_pass = stress <= max_stress
    
    return jnp.where(eta_pass & stress_pass, 1, 0)

# Mark n_steps and dt as static because they define the loop structure
@partial(jax.jit, static_argnums=(7, 8))
def run_single_realization_jax(key, eta_ind, latency_s, state0, I, I_inv, K, n_steps, dt, mass, radius, max_stress):
    buffer_size = 50
    history_buffer = jnp.tile(state0, (buffer_size, 1))
    latency_steps = jnp.clip(jnp.round(latency_s / dt).astype(int), 0, buffer_size - 1)

    def step_fn(carry, _):
        state, h_buf, idx = carry
        latent_idx = (idx - latency_steps + buffer_size) % buffer_size
        latent_state = h_buf[latent_idx]
        tau = -K @ (latent_state[4:] - state0[4:]) 
        tau = jnp.clip(tau, -1.0, 1.0)
        next_state = rk4_step_jax(state, I, I_inv, tau, dt)
        next_h_buf = h_buf.at[idx % buffer_size].set(state)
        return (next_state, next_h_buf, idx + 1), None

    (final_state, _, _), _ = jax.lax.scan(step_fn, (state0, history_buffer, 0), None, length=n_steps)
    return evaluate_safety_gates_jax(final_state, eta_ind, mass, radius, max_stress)

# vmap over realizations
run_realizations_vmap = jax.vmap(
    run_single_realization_jax,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None)
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

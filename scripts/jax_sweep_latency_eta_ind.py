import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from monte_carlo.jax_cascade_runner import run_full_sweep_vmap
from monte_carlo.lqr_gain import compute_lqr_gain


def run_jax_sweep():
    print(f"Starting JAX Sweep Refactor - {datetime.now().isoformat()}")
    print(f"Using JAX Devices: {jax.devices()}")

    # 1. Parameters (Mirroring sweep_latency_eta_ind.py)
    MASS = 0.05
    RADIUS = 0.02
    MAX_STRESS = 1.2e9
    DT = 0.01
    T_HORIZON = 2.0
    N_STEPS = int(T_HORIZON / DT)
    N_REALIZATIONS = 1000

    LATENCY_RANGE = np.linspace(0.005, 0.100, 16)  # 5ms to 100ms
    ETA_RANGE = np.linspace(0.75, 0.98, 16)       # 75% to 98%

    # 2. Setup Physics
    I = np.diag([0.0001, 0.00011, 0.00009])
    I_inv = np.linalg.inv(I)
    K = compute_lqr_gain(I)

    # JAX Arrays
    I_jax = jnp.array(I)
    I_inv_jax = jnp.array(I_inv)
    K_jax = jnp.array(K)

    # 3. Prepare Batch Data
    # For every grid point, we need N_REALIZATIONS keys and perturbed states
    key = jax.random.PRNGKey(42)

    # Split keys for perturbations and realizations
    perturb_key, sweep_key = jax.random.split(key)

    # Generate random perturbations for each realization
    # Perturbation range: 10-500 rad/s (~95-4775 RPM) - realistic disturbance
    perturb_keys = jax.random.split(perturb_key, N_REALIZATIONS)
    # Use vmap to generate perturbations in parallel
    def generate_perturbation(k):
        return jax.random.uniform(k, (3,), minval=10.0, maxval=500.0)
    omega_perturbations = jax.vmap(generate_perturbation)(perturb_keys)

    # Create batch of perturbed initial states
    # Base state: identity quaternion, zero omega
    base_state = jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    state0_batch = jnp.tile(base_state, (N_REALIZATIONS, 1))
    # Apply perturbations to omega components (indices 4-6)
    state0_batch = state0_batch.at[:, 4:7].set(omega_perturbations)

    # Split keys for realizations
    keys = jax.random.split(sweep_key, N_REALIZATIONS)

    # Latency sampling logic: we'll use fixed latency per grid point for now
    # (plus optional jitter inside the realization if needed)
    # Actually, let's vmap over latency_ms directly.

    start_time = time.time()

    # Success grid: (n_eta, n_lat, n_realizations)
    # We pass the latency values as a 1D array to the inner vmap
    latency_arr = jnp.array(LATENCY_RANGE)
    eta_arr = jnp.array(ETA_RANGE)

    # We need to broadcast the latency array into (n_lat, n_realizations)
    # if we want jitter, or just (n_lat) and vmap.
    # Let's do fixed latency per grid point first.

    # success_batch: shape (n_eta, n_lat, n_realizations)
    success_batch = run_full_sweep_vmap(
        keys,
        eta_arr,
        latency_arr,
        state0_batch,  # Now passing batch of perturbed states
        I_jax,
        I_inv_jax,
        K_jax,
        N_STEPS,
        DT,
        MASS,
        RADIUS,
        MAX_STRESS
    )

    # Average across realizations
    success_grid = jnp.mean(success_batch.astype(float), axis=2)

    # Trigger JIT and measure
    success_grid.block_until_ready()
    elapsed = time.time() - start_time

    print(f"Sweep complete in {elapsed:.2f} seconds.")
    print(f"Speedup vs 3600s baseline: {3600/elapsed:.1f}x")

    # 4. Save Results
    results = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": elapsed,
        "n_realizations": N_REALIZATIONS,
        "latency_range": LATENCY_RANGE.tolist(),
        "eta_range": ETA_RANGE.tolist(),
        "success_rate_grid": success_grid.tolist()
    }

    output_path = "sweep_results/jax_t1_highres.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_jax_sweep()

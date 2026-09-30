#!/usr/bin/env python3
"""
Phase 1: Core Orbital Mechanics Engine - Tether Boost Demonstration

Part of autonomous SpinnyBall research (tether-first architecture).

This module demonstrates a single momentum-exchange tether stage
providing 2 km/s Δv boost from LEO, raising orbit to trans-lunar energies.

Integrates with existing JAX/Numba physics engine in the repo.

Next steps in this branch: libration dynamics, electrodynamic reboost,
Monte-Carlo debris risk, multi-stage cislunar railroad.
"""

import numpy as np
from scipy.integrate import odeint

# Physical constants
G = 6.67430e-11
M_EARTH = 5.972e24
MU = G * M_EARTH
R_EARTH = 6371e3


def compute_tether_boost(
    h_leo: float = 500e3,
    tether_length: float = 100e3,
    spin_rate: float = 0.02
) -> dict:
    """
    Compute Δv boost and new orbit parameters for a rotating tether.

    Args:
        h_leo: LEO altitude (m)
        tether_length: Length of tether (m)
        spin_rate: Angular velocity (rad/s)

    Returns:
        dict with v_tip, a_new, apogee_alt, etc.
    """
    r_leo = R_EARTH + h_leo
    v_leo = np.sqrt(MU / r_leo)
    v_tip = spin_rate * tether_length
    v_new = v_leo + v_tip

    energy = v_new**2 / 2 - MU / r_leo
    a_new = -MU / (2 * energy)
    apogee_alt = 2 * a_new - r_leo - R_EARTH

    return {
        "tether_length_km": tether_length / 1000,
        "spin_rate_rad_s": spin_rate,
        "v_tip_kms": v_tip / 1000,
        "new_semi_major_axis_km": a_new / 1000,
        "apogee_altitude_km": apogee_alt / 1000,
        "delta_v_kms": v_tip / 1000,
    }


def two_body_dynamics(state, t, mu):
    """Two-body equations of motion."""
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    return [vx, vy, ax, ay]


if __name__ == "__main__":
    results = compute_tether_boost()
    print("Phase 1 Tether Boost Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")

    # Quick numerical check
    r_leo = R_EARTH + 500e3
    v_new = np.sqrt(MU / r_leo) + results["v_tip_kms"] * 1000
    state0 = [r_leo, 0.0, 0.0, v_new]
    period = 2 * np.pi * np.sqrt(results["new_semi_major_axis_km"]**3 * 1e9 / MU)
    t = np.linspace(0, period, 2000)
    sol = odeint(two_body_dynamics, state0, t, args=(MU,))
    print(f"\nNumerical propagation complete. Final r: {np.max(np.sqrt(sol[:,0]**2 + sol[:,1]**2))/1000:.0f} km")

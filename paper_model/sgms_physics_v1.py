"""
Hardened: Core Analytical Physics Baseline (v1.0.0-IEEE-2026)

This module contains the frozen analytical models and physics functions used for 
the project papers. It is decoupled from the engineering simulation 
scripts to ensure research results are reproducible against the baseline.

Core Restoring Force Equation:
    F = lambda * u^2 * theta

LQR Stability Matrix A:
    A = [[0, 1], [-k_eff/m_s, -c_damp/m_s]]
"""

import math
import numpy as np

# Paper Baseline Parameters (Moderate-U Canonical)
PAPER_BASELINE_PARAMS = {
    "u": 10.0,             # Stream velocity (m/s)
    "lam": 0.5,            # Stream linear density (kg/m)
    "mp": 0.05,            # Packet mass (kg)
    "theta_bias": 0.087,   # Nominal deflection (rad)
    "g_gain": 0.05,        # Control gain (rad/m)
    "ms": 1000.0,          # Station mass (kg)
    "c_damp": 4.0,         # Damping coefficient (N-s/m)
    "k_fp": 0.0,           # Pinning stiffness (N/m)
}

def analytical_metrics(params: dict) -> dict:
    """
    Calculates key analytical performance metrics for the anchor system.
    Returns:
        A dictionary containing stiffness, frequency, period, and damping ratio.
    """
    lam = params.get("lam", 0.5)
    u = params.get("u", 10.0)
    theta_bias = params.get("theta_bias", 0.087)
    g_gain = params.get("g_gain", 0.05)
    ms = params.get("ms", 1000.0)
    c_damp = params.get("c_damp", 4.0)
    k_fp = params.get("k_fp", 0.0)

    f_stream = lam * u**2 * theta_bias
    k_control = lam * u**2 * g_gain
    k_total = k_control + k_fp
    
    omega_n = math.sqrt(k_total / ms) if k_total > 0.0 else 0.0
    period = (2.0 * math.pi / omega_n) if omega_n > 0.0 else math.inf
    zeta = c_damp / (2.0 * math.sqrt(k_total * ms)) if k_total > 0.0 else math.inf
    
    packet_rate = lam * u / params.get("mp", 0.01) if params.get("mp", 0.01) > 0.0 else math.inf
    packet_period = 1.0 / packet_rate if packet_rate > 0.0 else math.inf
    
    return {
        "force_per_stream_n": f_stream,
        "k_control": k_control,
        "k_fp": k_fp,
        "k_total": k_total,
        "k_eff": k_total,  # Paper compatibility alias
        "omega_n_rad_s": omega_n,
        "period_s": period,
        "zeta": zeta,
        "packet_period_s": packet_period,
    }

def get_stream_forces(x: float, vx: float, dist_theta: float, params: dict) -> tuple[float, float, float, float]:
    """
    Calculates constituent forces for the anchor state.
    """
    lam_u2 = params["lam"] * params["u"] ** 2
    theta_cmd = params["g_gain"] * x

    # Command split logic (Paper Baseline v1)
    theta_plus = params["theta_bias"] - 0.5 * theta_cmd + dist_theta
    theta_minus = params["theta_bias"] + 0.5 * theta_cmd - dist_theta

    f_plus = lam_u2 * theta_plus
    f_minus = -lam_u2 * theta_minus
    f_pin = -params.get("k_fp", 0.0) * x
    f_damp = -params["c_damp"] * vx
    
    return f_plus, f_minus, f_pin, f_damp

def packet_modulation(t: float, period: float, sigma: float, phase: float = 0.0) -> float:
    """
    Models the temporal modulation of momentum flux from discrete mass packets.
    """
    if not math.isfinite(period) or period <= 0.0:
        return 1.0
    if sigma <= 0.0:
        frac = (t - phase) / period
        return period * 1e12 if abs(frac - round(frac)) < 1e-12 else 0.0

    center = int(round((t - phase) / period))
    radius = max(3, int(math.ceil(6.0 * sigma / period)))
    total = 0.0
    norm = sigma * math.sqrt(2.0 * math.pi)
    for k in range(center - radius, center + radius + 1):
        dt = t - (phase + k * period)
        total += math.exp(-(dt * dt) / (2.0 * sigma * sigma)) / norm
    return period * total

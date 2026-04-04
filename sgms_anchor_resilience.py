"""
Stream Resilience and Fracture Recovery for Aethelgard mass-packet architecture.
Formalizes the 'Catcher-Recovery' and 'VPD Compression' protocols.
Targets < 0.5mm displacement during 'Stream Fracture' events.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sgms_anchor_v1 import analytical_metrics, DEFAULT_PARAMS, _stream_forces

def simulate_resilience_event(params: dict, fracture_idx: int = 50, n_compensation: int = 10):
    """
    Simulates a single packet loss and the subsequent VPD (Variable Packet Density) compression.
    """
    m = analytical_metrics(params)
    period = m["packet_period_s"]
    t_max = params["t_max"]
    dt = 0.0001
    t = np.arange(0, t_max, dt)
    
    # Pre-calculate packet arrival times
    # Normal arrivals every period
    t_arrivals = np.arange(period, t_max + period, period)
    
    # 1. TRIGGER FRACTURE (Delete one packet)
    t_arrivals = np.delete(t_arrivals, fracture_idx)
    
    # 2. APPLY VPD COMPRESSION (Shift next N packets to fill momentum void)
    # The 'Gap' is one full period. To fill it over 'n' packets,
    # we reduce each of the next n packet delays by (period / n).
    shift_step = period / n_compensation
    for i in range(fracture_idx, fracture_idx + n_compensation):
        if i < len(t_arrivals):
            # Each packet moves up by a cumulative shift to squeeze the group
            t_arrivals[i:] -= shift_step

    # State vectors
    x = np.zeros_like(t)
    vx = np.zeros_like(t)
    force = np.zeros_like(t)
    
    # Initial displacement (trim)
    x[0] = params["x0"]
    vx[0] = params["v0"]

    # Simple RK4 integration
    ms = params["ms"]
    sigma = params["packet_sigma_s"]

    def get_discrete_mod(time_pt, arrivals):
        # Gaussian pulses for each packet arrival
        # For performance, only look at nearby arrivals
        idx = np.searchsorted(arrivals, time_pt)
        window = arrivals[max(0, idx-2):min(len(arrivals), idx+2)]
        total = 0.0
        norm = sigma * np.sqrt(2.0 * np.pi)
        for t_a in window:
            dt_p = time_pt - t_a
            total += np.exp(-(dt_p**2) / (2.0 * sigma**2)) / norm
        return total * period

    # Pre-calculate the packet pulses (for speed)
    print("Pre-calculating pulse stream...")
    plus_stream = np.array([get_discrete_mod(ti, t_arrivals) for ti in t])
    # Assume minus stream is perfectly coherent (no loss) for worst-case imbalance comparison
    t_minus = np.arange(period * 0.5, t_max + period, period)
    minus_stream = np.array([get_discrete_mod(ti, t_minus) for ti in t])

    print("Running integration...")
    for i in range(len(t) - 1):
        # Active control + Pinning
        fp, fm, fpin, fd = _stream_forces(x[i], vx[i], 0.0, params)
        
        # Modulate by discretized stream (with fracture)
        f_net = fp * plus_stream[i] + fm * minus_stream[i] + fpin + fd
        force[i] = f_net
        
        # RK4
        k1v = f_net / ms
        k1x = vx[i]
        
        # (Simplified Euler for this resilience high-res pass to save cycles)
        vx[i+1] = vx[i] + k1v * dt
        x[i+1] = x[i] + k1x * dt

    return t, x, force, t_arrivals[fracture_idx]

def plot_resilience(t, x, force, fracture_time, filename="artifacts/resilience_recovery.png"):
    Path("artifacts").mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Highlight the fracture window
    window = (t > fracture_time - 0.2) & (t < fracture_time + 1.0)
    
    axes[0].plot(t[window], x[window] * 1000, color="#79c0ff", linewidth=2)
    axes[0].axvline(fracture_time, color="#ff7b72", linestyle="--", alpha=0.8, label="Fracture Event")
    axes[0].set_ylabel("Displacement (mm)")
    axes[0].set_title("Aethelgard Stream Resilience: VPD Compensation Protocol")
    axes[0].grid(True, alpha=0.3)
    
    # 0.5mm target line
    axes[0].axhline(0.5, color="#f2cc60", linestyle=":", alpha=0.6, label="Compliance Limit (0.5mm)")
    axes[0].axhline(-0.5, color="#f2cc60", linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].plot(t[window], force[window], color="#7ee787", linewidth=1)
    axes[1].set_ylabel("Force (N)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Resilience plot saved to {filename}")

if __name__ == "__main__":
    # Using hardcoded parameters for resilience standalone run
    res_params = {
        "u": 10.0,
        "lam": 16.6667,
        "mp": 2.0,
        "g_gain": 0.05,
        "ms": 1000.0,
        "eps": 0.0001,
        "c_damp": 20.0, # High damping for resilience test
        "theta_bias": 0.087,
        "k_fp": 4500.0,
        "packet_sigma_s": 0.002, # Sharp pulses
        "t_max": 2.0,
        "x0": 0.0,
        "v0": 0.0
    }
    
    t, x, f, f_time = simulate_resilience_event(res_params)
    peak_err = np.max(np.abs(x)) * 1000
    print(f"Peak Fracture Deviation: {peak_err:.4f} mm")
    if peak_err < 0.5:
        print("RESULT: SUCCESS (< 0.5mm limit maintained)")
    else:
        print("RESULT: FAILURE (> 0.5mm limit breached)")
        
    plot_resilience(t, x, f, f_time)

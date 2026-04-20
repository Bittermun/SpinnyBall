"""
Reduced-order dynamic-anchor model for moderate-u counter-stream operation.

This is intentionally not a clone of ``sgms_v1.py``. The original file is a
single-pass pulsed-coil steering simulation; this module is a lower-order anchor
model built around the momentum-flux restoring-force law discussed in the memo:

    F = lambda * u^2 * theta

The counter-stream controller is written to preserve the memo baseline:

    k_eff = lambda * u^2 * g_gain

by splitting the feedback command evenly across the two streams.
"""

from __future__ import annotations

import csv
import math
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.bean_london_model import BeanLondonModel


# Default Bean-London model configuration (for backward compatibility)
DEFAULT_GDBCO_PROPS = GdBCOProperties(
    Tc=92.0,
    Jc0=3e10,
    n_exponent=1.5,
    B0=0.1,
    alpha=0.5,
    thickness=1e-6,
    width=0.012,
    length=1.0,  # 1 meter of tape
)
DEFAULT_FLUX_PINNING_GEOMETRY = {
    "thickness": DEFAULT_GDBCO_PROPS.thickness,
    "width": DEFAULT_GDBCO_PROPS.width,
    "length": 1.0,
}


DEFAULT_PARAMS = {
    "u": 10.0,
    "lam": 0.5,
    "mp": 0.05,
    "theta_bias": 0.087,
    "g_gain": 0.05,
    "ms": 1000.0,
    "c_damp": 4.0,
    "eps": 0.0,
    "disturbance_theta_std": 0.0,
    "disturbance_hold_s": 1.0,
    "packet_sigma_s": 0.01,
    "packet_phase_s": 0.0,
    "k_fp": 0.0,
    "x0": 0.1,
    "v0": 0.0,
    "t_max": 400.0,
    "rtol": 1e-8,
    "atol": 1e-10,
    "max_step": 0.25,
}


def analytical_metrics(params: dict, flux_model: BeanLondonModel | None = None) -> dict:
    """Calculate metrics with backward compatibility.

    If params contains "k_fp", use linear stiffness (old behavior).
    Otherwise, use dynamic Bean-London stiffness (new behavior).
    
    Args:
        params: System parameters
        flux_model: Optional BeanLondonModel instance. If None, creates default.
    
    Returns:
        Dictionary with calculated metrics
    """
    lam = params["lam"]
    u = params["u"]
    mp = params["mp"]
    theta_bias = params["theta_bias"]
    g_gain = params["g_gain"]
    ms = params["ms"]
    eps = params["eps"]
    c_damp = params["c_damp"]

    f_stream = lam * u**2 * theta_bias
    k_control = lam * u**2 * g_gain
    
    if "k_fp" in params:
        # Legacy linear stiffness
        k_fp = params["k_fp"]
    else:
        # New dynamic Bean-London stiffness
        if flux_model is None:
            # Create default model for backward compatibility
            material = GdBCOMaterial(DEFAULT_GDBCO_PROPS)
            flux_model = BeanLondonModel(material, DEFAULT_FLUX_PINNING_GEOMETRY)
        
        temperature = params.get("temperature", 77.0)
        B_field = params.get("B_field", 1.0)
        displacement = params.get("x0", 0.0)
        k_fp = flux_model.get_stiffness(displacement, B_field, temperature)

    k_total = k_control + k_fp
    
    omega_n = math.sqrt(k_total / ms) if k_total > 0.0 else 0.0
    period = (2.0 * math.pi / omega_n) if omega_n > 0.0 else math.inf
    zeta = c_damp / (2.0 * math.sqrt(k_total * ms)) if k_total > 0.0 else math.inf
    bias_force = 2.0 * eps * f_stream
    static_offset = bias_force / k_total if k_total > 0.0 else math.inf
    packet_rate = lam * u / mp if mp > 0.0 else math.inf
    packet_period = 1.0 / packet_rate if packet_rate > 0.0 else math.inf
    return {
        "force_per_stream_n": f_stream,
        "k_control": k_control,
        "k_fp": k_fp,
        "k_total": k_total,
        "k_eff": k_total,  # Alias for control analysis suite compatibility
        "omega_n_rad_s": omega_n,
        "period_s": period,
        "zeta": zeta,
        "bias_force_n": bias_force,
        "static_offset_m": static_offset,
        "packet_rate_hz": packet_rate,
        "packet_period_s": packet_period,
    }


def simulate_anchor_with_flux_pinning(
    params: dict,
    t_eval: np.ndarray,
    temperature_profile: np.ndarray | None = None,
    B_field_profile: np.ndarray | None = None,
    flux_model: BeanLondonModel | None = None,
) -> dict:
    """Simulate anchor with dynamic Bean-London flux-pinning stiffness.

    Args:
        params: System parameters
        t_eval: Time evaluation points
        temperature_profile: Optional temperature profile over time (K)
        B_field_profile: Optional magnetic field profile over time (T)
        flux_model: Optional BeanLondonModel instance. If None, creates default.

    Returns:
        Dictionary with time series including dynamic k_fp
    """
    # Initialize flux model if not provided
    if flux_model is None:
        material = GdBCOMaterial(DEFAULT_GDBCO_PROPS)
        flux_model = BeanLondonModel(material, DEFAULT_FLUX_PINNING_GEOMETRY)
    
    # Initialize temperature and field profiles
    if temperature_profile is None:
        temperature_profile = np.full_like(t_eval, 77.0)  # Constant 77K
    if B_field_profile is None:
        B_field_profile = np.full_like(t_eval, 1.0)  # Constant 1T

    # Initialize state
    x = params["x0"]
    v = params["v0"]
    dt = t_eval[1] - t_eval[0]

    # Storage for results
    results = {
        "t": t_eval,
        "x": [],
        "v": [],
        "k_fp": [],
        "k_eff": [],
        "temperature": [],
        "B_field": [],
    }

    # Simulation loop
    for i, t in enumerate(t_eval):
        # Get current temperature and field
        T = temperature_profile[i]
        B = B_field_profile[i]

        # Calculate dynamic flux-pinning stiffness
        k_fp = flux_model.get_stiffness(x, B, T)

        # Effective stiffness
        k_eff = k_fp + params["k_structural"] if "k_structural" in params else k_fp

        # Update dynamics (simplified Euler)
        # m_s * x_ddot + c_damp * x_dot + k_eff * x = 0
        a = -(params["c_damp"] * v + k_eff * x) / params["ms"]
        v += a * dt
        x += v * dt

        # Store results
        results["x"].append(x)
        results["v"].append(v)
        results["k_fp"].append(k_fp)
        results["k_eff"].append(k_eff)
        results["temperature"].append(T)
        results["B_field"].append(B)

    return results


def make_disturbance_series(params: dict, dt: float, steps: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = params["disturbance_theta_std"]
    if sigma <= 0.0:
        return np.zeros(steps)
    return rng.normal(0.0, sigma, size=steps)


def _stream_forces(x: float, vx: float, disturbance_theta: float, params: dict) -> tuple[float, float, float, float]:
    lam_u2 = params["lam"] * params["u"] ** 2
    theta_cmd = params["g_gain"] * x

    # Split the command evenly between the streams
    theta_plus = params["theta_bias"] - 0.5 * theta_cmd + disturbance_theta
    theta_minus = params["theta_bias"] + 0.5 * theta_cmd - disturbance_theta

    f_plus = (1.0 + params["eps"]) * lam_u2 * theta_plus
    f_minus = -(1.0 - params["eps"]) * lam_u2 * theta_minus
    f_pin = -params.get("k_fp", 0.0) * x
    f_damp = -params["c_damp"] * vx
    return f_plus, f_minus, f_pin, f_damp


def packet_modulation(t: float, period: float, sigma: float, phase: float = 0.0, radius: int = 3) -> float:
    if not math.isfinite(period) or period <= 0.0:
        return 1.0
    if sigma <= 0.0:
        frac = (t - phase) / period
        nearest = abs(frac - round(frac))
        return period * 1e12 if nearest < 1e-12 else 0.0

    center = int(round((t - phase) / period))
    radius = max(radius, int(math.ceil(6.0 * sigma / period)))
    total = 0.0
    norm = sigma * math.sqrt(2.0 * math.pi)
    for k in range(center - radius, center + radius + 1):
        dt = t - (phase + k * period)
        total += math.exp(-(dt * dt) / (2.0 * sigma * sigma)) / norm
    return period * total


def net_anchor_force(x: float, vx: float, t: float, params: dict, disturbance_theta: float = 0.0) -> float:
    del t  # Force law is time-invariant aside from the supplied disturbance.
    f_p, f_m, f_pin, f_d = _stream_forces(x, vx, disturbance_theta, params)
    return f_p + f_m + f_pin + f_d


def discrete_packet_force(x: float, vx: float, t: float, params: dict, disturbance_theta: float = 0.0) -> float:
    metrics = analytical_metrics(params)
    period = metrics["packet_period_s"]
    sigma = params["packet_sigma_s"]
    phase = params["packet_phase_s"]
    plus_mod = packet_modulation(t, period, sigma, phase=phase)
    minus_mod = packet_modulation(t, period, sigma, phase=phase + 0.5 * period)

    f_plus, f_minus, f_pin, f_damp = _stream_forces(x, vx, disturbance_theta, params)
    return f_plus * plus_mod + f_minus * minus_mod + f_pin + f_damp


def simulate_anchor(params: dict | None = None, t_eval: np.ndarray | None = None, seed: int = 0) -> dict:
    params = DEFAULT_PARAMS.copy() if params is None else params.copy()
    if t_eval is None:
        points = max(4000, int(params["t_max"] / params["max_step"]) + 1)
        t_eval = np.linspace(0.0, params["t_max"], points)
    else:
        t_eval = np.asarray(t_eval, dtype=float)

    dt_noise = params["disturbance_hold_s"]
    noise_steps = max(2, int(math.ceil(t_eval[-1] / dt_noise)) + 2)
    noise_t = np.linspace(0.0, dt_noise * (noise_steps - 1), noise_steps)
    disturbance_theta = make_disturbance_series(params, dt=dt_noise, steps=noise_steps, seed=seed)

    def noise_at(t: float) -> float:
        return float(np.interp(t, noise_t, disturbance_theta))

    def rhs(t: float, y: np.ndarray) -> list[float]:
        x, vx = y
        force = net_anchor_force(x, vx, t, params, disturbance_theta=noise_at(t))
        return [vx, force / params["ms"]]

    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        [params["x0"], params["v0"]],
        method="RK45",
        t_eval=t_eval,
        rtol=params["rtol"],
        atol=params["atol"],
        max_step=params["max_step"],
    )

    force = np.empty_like(sol.t)
    f_plus = np.empty_like(sol.t)
    f_minus = np.empty_like(sol.t)
    f_damp = np.empty_like(sol.t)
    disturbance = np.empty_like(sol.t)
    for i, t in enumerate(sol.t):
        disturbance[i] = noise_at(float(t))
        fp, fm, fpin, fd = _stream_forces(sol.y[0][i], sol.y[1][i], disturbance[i], params)
        f_plus[i], f_minus[i], f_damp[i] = fp, fm, fd
        force[i] = fp + fm + fpin + fd

    metrics = analytical_metrics(params)
    metrics.update(
        {
            "x_final_m": float(sol.y[0][-1]),
            "vx_final_m_s": float(sol.y[1][-1]),
            "x_peak_m": float(np.max(np.abs(sol.y[0]))),
            "force_peak_n": float(np.max(np.abs(force))),
        }
    )

    return {
        "t": sol.t,
        "x": sol.y[0],
        "vx": sol.y[1],
        "force": force,
        "f_plus": f_plus,
        "f_minus": f_minus,
        "f_damp": f_damp,
        "disturbance_theta": disturbance,
        "metrics": metrics,
        "params": params,
    }


def simulate_discrete_anchor(params: dict | None = None, t_eval: np.ndarray | None = None, seed: int = 0) -> dict:
    params = DEFAULT_PARAMS.copy() if params is None else params.copy()
    if t_eval is None:
        points = max(4000, int(params["t_max"] / params["max_step"]) + 1)
        t_eval = np.linspace(0.0, params["t_max"], points)
    else:
        t_eval = np.asarray(t_eval, dtype=float)

    dt_noise = params["disturbance_hold_s"]
    noise_steps = max(2, int(math.ceil(t_eval[-1] / dt_noise)) + 2)
    noise_t = np.linspace(0.0, dt_noise * (noise_steps - 1), noise_steps)
    disturbance_theta = make_disturbance_series(params, dt=dt_noise, steps=noise_steps, seed=seed)

    def noise_at(t: float) -> float:
        return float(np.interp(t, noise_t, disturbance_theta))

    x = np.empty_like(t_eval)
    vx = np.empty_like(t_eval)
    force = np.empty_like(t_eval)
    f_plus = np.empty_like(t_eval)
    f_minus = np.empty_like(t_eval)
    f_damp = np.empty_like(t_eval)
    disturbance = np.empty_like(t_eval)

    x[0] = params["x0"]
    vx[0] = params["v0"]

    def rhs(t: float, state_x: float, state_vx: float) -> tuple[float, float]:
        local_disturbance = noise_at(t)
        local_force = discrete_packet_force(state_x, state_vx, t, params, disturbance_theta=local_disturbance)
        return state_vx, local_force / params["ms"]

    for i in range(len(t_eval) - 1):
        t = float(t_eval[i])
        dt = float(t_eval[i + 1] - t_eval[i])

        k1x, k1v = rhs(t, x[i], vx[i])
        k2x, k2v = rhs(t + 0.5 * dt, x[i] + 0.5 * dt * k1x, vx[i] + 0.5 * dt * k1v)
        k3x, k3v = rhs(t + 0.5 * dt, x[i] + 0.5 * dt * k2x, vx[i] + 0.5 * dt * k2v)
        k4x, k4v = rhs(t + dt, x[i] + dt * k3x, vx[i] + dt * k3v)

        x[i + 1] = x[i] + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        vx[i + 1] = vx[i] + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    for i, t in enumerate(t_eval):
        disturbance[i] = noise_at(float(t))
        metrics = analytical_metrics(params)
        period = metrics["packet_period_s"]
        plus_mod = packet_modulation(float(t), period, params["packet_sigma_s"], phase=params["packet_phase_s"])
        minus_mod = packet_modulation(
            float(t),
            period,
            params["packet_sigma_s"],
            phase=params["packet_phase_s"] + 0.5 * period,
        )
        base_plus, base_minus, f_pin, f_damp[i] = _stream_forces(x[i], vx[i], disturbance[i], params)
        f_plus[i] = base_plus * plus_mod
        f_minus[i] = base_minus * minus_mod
        force[i] = f_plus[i] + f_minus[i] + f_pin + f_damp[i]

    metrics = analytical_metrics(params)
    metrics.update(
        {
            "x_final_m": float(x[-1]),
            "vx_final_m_s": float(vx[-1]),
            "x_peak_m": float(np.max(np.abs(x))),
            "force_peak_n": float(np.max(np.abs(force))),
        }
    )

    return {
        "t": t_eval,
        "x": x,
        "vx": vx,
        "force": force,
        "f_plus": f_plus,
        "f_minus": f_minus,
        "f_damp": f_damp,
        "disturbance_theta": disturbance,
        "metrics": metrics,
        "params": params,
    }


def estimate_period(t: np.ndarray, x: np.ndarray) -> float | None:
    peaks = []
    for i in range(1, len(x) - 1):
        if x[i - 1] < x[i] and x[i] >= x[i + 1] and x[i] > 0.0:
            peaks.append(float(t[i]))
    if len(peaks) < 2:
        return None
    return float(np.mean(np.diff(peaks[:4])))


def sweep_velocity(params: dict | None = None, u_values: np.ndarray | None = None) -> dict:
    params = DEFAULT_PARAMS.copy() if params is None else params.copy()
    if u_values is None:
        u_values = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 520.0])
    u_values = np.asarray(u_values, dtype=float)

    force_per_stream = []
    k_total = []
    periods = []
    static_offset = []
    for u in u_values:
        local = params.copy()
        local["u"] = float(u)
        metrics = analytical_metrics(local)
        force_per_stream.append(metrics["force_per_stream_n"])
        k_total.append(metrics["k_total"])
        periods.append(metrics["period_s"])
        static_offset.append(metrics["static_offset_m"])

    return {
        "u": u_values,
        "force_per_stream_n": np.array(force_per_stream),
        "k_total": np.array(k_total),
        "period_s": np.array(periods),
        "static_offset_m": np.array(static_offset),
    }


def sweep_anchor_grid(
    params: dict | None = None,
    u_values: list[float] | np.ndarray | None = None,
    g_values: list[float] | np.ndarray | None = None,
    eps_values: list[float] | np.ndarray | None = None,
) -> list[dict]:
    params = DEFAULT_PARAMS.copy() if params is None else params.copy()
    u_values = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 520.0] if u_values is None else list(u_values)
    g_values = [0.02, 0.05, 0.1, 0.2] if g_values is None else list(g_values)
    eps_values = [0.0, 1e-4, 1e-3] if eps_values is None else list(eps_values)

    rows = []
    for u in u_values:
        for g_gain in g_values:
            for eps in eps_values:
                local = params.copy()
                local["u"] = float(u)
                local["g_gain"] = float(g_gain)
                local["eps"] = float(eps)
                metrics = analytical_metrics(local)
                row = {
                    "u": float(u),
                    "g_gain": float(g_gain),
                    "eps": float(eps),
                }
                row.update(metrics)
                rows.append(row)
    return rows


def export_sweep_csv(rows: list[dict], filename: str | Path) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = list(rows[0].keys())
    with filename.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_anchor_response(result: dict, filename: str = "sgms_anchor_v1_displacement.png") -> None:
    metrics = result["metrics"]
    plt.figure(figsize=(10, 6))
    plt.plot(result["t"], result["x"], label="Node displacement", linewidth=2)
    plt.axhline(metrics["static_offset_m"], color="#f2cc60", linestyle=":", linewidth=1.5, label="Static offset")
    plt.axhline(0.0, color="#ff7b72", linestyle="--", alpha=0.7, label="Geometric center")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement x (m)")
    plt.title(
        "Moderate-u Dynamic Anchor Response\n"
        f"u = {result['params']['u']:.1f} m/s | "
        f"k_total = {metrics['k_total']:.2f} N/m | "
        f"period = {metrics['period_s']:.1f} s"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_velocity_sweep(sweep: dict, filename: str = "sgms_anchor_v1_sweep.png") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    axes[0].plot(sweep["u"], sweep["force_per_stream_n"], "o-", color="#79c0ff")
    axes[0].set_title("Force Per Stream")
    axes[0].set_xlabel("u (m/s)")
    axes[0].set_ylabel("F_stream (N)")

    axes[1].plot(sweep["u"], sweep["k_total"], "o-", color="#7ee787")
    axes[1].set_title("Total Stiffness (Active + Pinning)")
    axes[1].set_xlabel("u (m/s)")
    axes[1].set_ylabel("k_total (N/m)")

    axes[2].plot(sweep["u"], sweep["period_s"], "o-", color="#f2cc60")
    axes[2].set_title("Oscillation Period")
    axes[2].set_xlabel("u (m/s)")
    axes[2].set_ylabel("Period (s)")

    axes[3].plot(sweep["u"], sweep["static_offset_m"] * 1e3, "o-", color="#ff7b72")
    axes[3].set_title("Static Offset Under Imbalance")
    axes[3].set_xlabel("u (m/s)")
    axes[3].set_ylabel("Offset (mm)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_continuum_vs_packet(
    continuum: dict,
    discrete: dict,
    filename: str = "sgms_anchor_v1_packet_compare.png",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(continuum["t"], continuum["x"], label="Continuum", linewidth=2)
    axes[0].plot(discrete["t"], discrete["x"], label="Discrete packets", linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel("x (m)")
    axes[0].set_title("Continuum vs Discrete Packet Anchor Response")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(continuum["t"], continuum["force"], label="Continuum force", linewidth=2)
    axes[1].plot(discrete["t"], discrete["force"], label="Discrete packet force", linewidth=1.2, alpha=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Force (N)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_sweep_heatmaps(
    rows: list[dict],
    eps: float = 1e-3,
    filename: str = "sgms_anchor_v1_heatmaps.png",
) -> None:
    filtered = [row for row in rows if abs(row["eps"] - eps) < 1e-15]
    if not filtered:
        raise ValueError("No rows found for requested eps slice")

    u_vals = sorted({row["u"] for row in filtered})
    g_vals = sorted({row["g_gain"] for row in filtered})
    k_grid = np.zeros((len(g_vals), len(u_vals)))
    offset_grid = np.zeros((len(g_vals), len(u_vals)))
    for row in filtered:
        i = g_vals.index(row["g_gain"])
        j = u_vals.index(row["u"])
        k_grid[i, j] = row["k_total"]
        offset_grid[i, j] = row["static_offset_m"] * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    im1 = axes[0].imshow(k_grid, aspect="auto", origin="lower")
    axes[0].set_title(f"k_total (N/m) at eps={eps:g}")
    axes[0].set_xticks(range(len(u_vals)), [f"{u:g}" for u in u_vals])
    axes[0].set_yticks(range(len(g_vals)), [f"{g:g}" for g in g_vals])
    axes[0].set_xlabel("u (m/s)")
    axes[0].set_ylabel("g_gain (rad/m)")
    fig.colorbar(im1, ax=axes[0], shrink=0.9)

    im2 = axes[1].imshow(offset_grid, aspect="auto", origin="lower")
    axes[1].set_title(f"Static offset (mm) at eps={eps:g}")
    axes[1].set_xticks(range(len(u_vals)), [f"{u:g}" for u in u_vals])
    axes[1].set_yticks(range(len(g_vals)), [f"{g:g}" for g in g_vals])
    axes[1].set_xlabel("u (m/s)")
    axes[1].set_ylabel("g_gain (rad/m)")
    fig.colorbar(im2, ax=axes[1], shrink=0.9)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def print_summary(metrics: dict, estimated_period: float | None = None) -> None:
    print("=== MODERATE-U DYNAMIC ANCHOR METRICS ===")
    print(f"Force per stream:   {metrics['force_per_stream_n']:.3f} N")
    print(f"Control stiffness:  {metrics['k_control']:.3f} N/m")
    print(f"Pinning stiffness:  {metrics['k_fp']:.1f} N/m")
    print(f"Total stiffness:    {metrics['k_total']:.3f} N/m")
    print(f"Natural frequency:  {metrics['omega_n_rad_s']:.5f} rad/s")
    print(f"Analytic period:    {metrics['period_s']:.3f} s")
    print(f"Damping ratio:      {metrics['zeta']:.4f}")
    print(f"Bias force:         {metrics['bias_force_n']:.6f} N")
    print(f"Static offset:      {metrics['static_offset_m'] * 1e3:.3f} mm")
    print(f"Packet rate:        {metrics['packet_rate_hz']:.3f} Hz")
    print(f"Packet period:      {metrics['packet_period_s']:.6f} s")
    if estimated_period is not None:
        print(f"Measured period:    {estimated_period:.3f} s")
    print(f"Final displacement: {metrics['x_final_m']:.6f} m")
    print(f"Peak displacement:  {metrics['x_peak_m']:.6f} m")
    print(f"Peak force:         {metrics['force_peak_n']:.6f} N")


def main() -> None:
    parser = argparse.ArgumentParser(description="Moderate-U Dynamic Anchor Simulation")
    parser.add_argument("--u", type=float, default=DEFAULT_PARAMS["u"], help="Stream velocity (m/s)")
    parser.add_argument("--lam", type=float, default=DEFAULT_PARAMS["lam"], help="Stream density (kg/m)")
    parser.add_argument("--g_gain", type=float, default=DEFAULT_PARAMS["g_gain"], help="Control gain (rad/m)")
    parser.add_argument("--k_fp", type=float, default=DEFAULT_PARAMS["k_fp"], help="Pinning stiffness (N/m)")
    parser.add_argument("--ms", type=float, default=DEFAULT_PARAMS["ms"], help="Anchor mass (kg)")
    parser.add_argument("--audit", action="store_true", help="Run full suite audit and sweep")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    params.update({
        "u": args.u,
        "lam": args.lam,
        "g_gain": args.g_gain,
        "k_fp": args.k_fp,
        "ms": args.ms,
    })

    if args.audit:
        # Full Audit / Sweep Mode
        t_eval = np.linspace(0.0, params["t_max"], 6000)
        result = simulate_anchor(params, t_eval=t_eval, seed=7)
        packet_result = simulate_discrete_anchor(params, t_eval=t_eval, seed=7)
        estimated_period = estimate_period(result["t"], result["x"])
        packet_period = estimate_period(packet_result["t"], packet_result["x"])
        
        plot_anchor_response(result)
        plot_continuum_vs_packet(result, packet_result)
        
        sweep_params = params.copy()
        sweep = sweep_velocity(sweep_params)
        plot_velocity_sweep(sweep)
        
        rows = sweep_anchor_grid(sweep_params)
        export_sweep_csv(rows, "sgms_anchor_v1_grid.csv")
        plot_sweep_heatmaps(rows, eps=1e-3)
        
        print_summary(result["metrics"], estimated_period=estimated_period)
        if packet_period is not None:
            print(f"Discrete period:    {packet_period:.3f} s")
    else:
        # Single Simulation Mode
        t_eval = np.linspace(0.0, params["t_max"], 4000)
        result = simulate_anchor(params, t_eval=t_eval, seed=7)
        print_summary(result["metrics"])
        print(f"Simulation complete. Final x: {result['x'][-1]:.6f} m")

if __name__ == "__main__":
    main()

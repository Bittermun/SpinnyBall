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
import warnings
from pathlib import Path
from typing import Optional, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.bean_london_model import BeanLondonModel

try:
    from control_layer.stream_balance import StreamBalanceController, StreamBalanceConfig
    STREAM_BALANCE_AVAILABLE = True
except ImportError:
    STREAM_BALANCE_AVAILABLE = False
    StreamBalanceController = None
    StreamBalanceConfig = None

# Optional MPC controller
try:
    from control_layer.mpc_controller import MPCController, ConfigurationMode
    MPC_AVAILABLE = True
except ImportError:
    MPC_AVAILABLE = False
    MPCController = None
    ConfigurationMode = None

# Optional orbital perturbations
try:
    from dynamics.orbital_perturbations import get_orbital_perturbation_force, create_orbital_state_from_params
    ORBITAL_PERTURBATIONS_AVAILABLE = True
except ImportError:
    ORBITAL_PERTURBATIONS_AVAILABLE = False
    get_orbital_perturbation_force = None
    create_orbital_state_from_params = None


# Default Bean-London model configuration (for backward compatibility)
DEFAULT_GDBCO_PROPS = GdBCOProperties(
    Tc=92.0,
    Jc0=3e10,
    n_exponent=1.5,
    B0=0.1,
    alpha=0.5,
    thickness=1e-6,
    width=0.012,
)
DEFAULT_FLUX_PINNING_GEOMETRY = {
    "thickness": DEFAULT_GDBCO_PROPS.thickness,
    "width": DEFAULT_GDBCO_PROPS.width,
    "length": 1.0,  # 1 meter of tape (geometry parameter, not material property)
}


def material_profile_to_properties(material_profile: dict | None) -> GdBCOProperties:
    """Convert material profile dict to GdBCOProperties.
    
    Args:
        material_profile: Material profile dict from catalog, or None for defaults
        
    Returns:
        GdBCOProperties instance
        
    Note:
        This function uses a simplified scaling model where Jc0 is scaled linearly
        based on the midpoint of the k_fp_range. This is a first-order approximation
        and may not accurately reflect the true relationship between flux-pinning
        stiffness and critical current density for different GdBCO material variants.
        For more accurate material modeling, consider adding explicit jc0 values
        to material profiles.
    """
    if material_profile is None:
        return DEFAULT_GDBCO_PROPS
    
    # Extract k_fp_range from material profile
    k_fp_range = material_profile.get("k_fp_range", [80000, 120000])
    # Use midpoint of range as baseline, scale Jc0 proportionally
    # NOTE: This is a simplification - assumes linear relationship between
    # flux-pinning stiffness and critical current density
    k_fp_baseline = (k_fp_range[0] + k_fp_range[1]) / 2
    k_fp_default = (80000 + 120000) / 2  # Default baseline
    jc0_scaled = DEFAULT_GDBCO_PROPS.Jc0 * (k_fp_baseline / k_fp_default)
    
    return GdBCOProperties(
        Tc=DEFAULT_GDBCO_PROPS.Tc,
        Jc0=jc0_scaled,
        n_exponent=DEFAULT_GDBCO_PROPS.n_exponent,
        B0=DEFAULT_GDBCO_PROPS.B0,
        alpha=DEFAULT_GDBCO_PROPS.alpha,
        thickness=DEFAULT_GDBCO_PROPS.thickness,
        width=DEFAULT_GDBCO_PROPS.width,
    )


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
    "k_structural": 0.0,  # Structural stiffness (N/m)
    "k_drag": 0.01,  # Eddy-current drag coefficient (N·s/m)
    "x0": 0.1,
    "v0": 0.0,
    "t_max": 400.0,
    "rtol": 1e-8,
    "atol": 1e-10,
    "max_step": 0.25,
    "use_dynamic_epsilon": False,  # Enable dynamic stream balance control
    "epsilon_target": 1e-4,  # Target ε < 10⁻⁴
    "cryocooler_power": 5.0,  # Cryocooler cooling power (W)
    "temperature": 77.0,  # Operating temperature (K)
    "B_field": 1.0,  # Magnetic field (T)
    "altitude_km": 400.0,  # Orbital altitude (km)
    "inclination_deg": 51.6,  # Orbital inclination (degrees)
    "include_j2": False,  # Include J2 perturbation
    "include_srp": False,  # Include solar radiation pressure
    "include_drag": False,  # Include atmospheric drag
    "enable_thermal_dynamics": False,  # Enable temperature evolution
    "enable_eclipse": False,  # Enable eclipse detection in thermal model
    "control_mode": "pid",  # Control mode: "pid" or "mpc"
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
        if "material_profile" in params:
            warnings.warn(
                "Both 'k_fp' (legacy) and 'material_profile' (new) are present. "
                "Using legacy 'k_fp' value and ignoring material_profile. "
                "Remove 'k_fp' to use material-based Bean-London stiffness.",
                UserWarning,
                stacklevel=2
            )
        k_fp = params["k_fp"]
    else:
        # New dynamic Bean-London stiffness
        if flux_model is None:
            # Check for material_profile in params
            material_profile = params.get("material_profile")
            gdBCO_props = material_profile_to_properties(material_profile)
            material = GdBCOMaterial(gdBCO_props)
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


def net_anchor_force(x: float, vx: float, t: float, params: dict, disturbance_theta: float = 0.0, orbital_state=None) -> float:
    f_p, f_m, f_pin, f_d = _stream_forces(x, vx, disturbance_theta, params)
    f_total = f_p + f_m + f_pin + f_d
    if orbital_state is not None and ORBITAL_PERTURBATIONS_AVAILABLE:
        if params.get("include_j2", False) or params.get("include_srp", False) or params.get("include_drag", False):
            try:
                # Compute perturbation force directly from current orbital state
                # (state already propagated correctly in rhs closure)
                f_orbital = get_orbital_perturbation_force(params, orbital_state, t, packet_mass=params["ms"])
                # Scale factor 0.01 to couple orbital perturbations appropriately to anchor dynamics
                f_total += f_orbital[0] * 0.01
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Orbital perturbation force calculation failed at t={t:.3f}: {e}")
    return f_total


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

    # Initialize stream balance controller if enabled
    balance_controller = None
    dynamic_epsilon = params.get("eps", 0.0)  # Track separately to avoid mutating params
    eps_log: list[tuple[float, float]] = []  # (t, epsilon) recorded during integration
    if params.get("use_dynamic_epsilon", False) and STREAM_BALANCE_AVAILABLE:
        config = StreamBalanceConfig(target_epsilon=params.get("epsilon_target", 1e-4))
        balance_controller = StreamBalanceController(config)

    # Initialize MPC controller if requested
    mpc_controller = None
    control_mode = params.get("control_mode", "pid")
    if control_mode == "mpc" and MPC_AVAILABLE:
        try:
            mpc_controller = MPCController(horizon=10, dt=0.01, configuration_mode=ConfigurationMode.OPERATIONAL)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"MPC initialization failed, falling back to PID: {e}")
            control_mode = "pid"

    # Initialize orbital state if perturbations enabled or thermal dynamics with eclipse
    orbital_state = None
    orbital_propagator = None  # Create once to avoid recreation on every RK45 step
    _orbital_initial_state = None  # Cache initial state for pure-function propagation
    enable_orbital = (
        params.get("include_j2", False) or 
        params.get("include_srp", False) or 
        params.get("include_drag", False) or
        (params.get("enable_thermal_dynamics", False) and params.get("enable_eclipse", False))
    )
    if ORBITAL_PERTURBATIONS_AVAILABLE and enable_orbital:
        try:
            orbital_state = create_orbital_state_from_params(params)
            _orbital_initial_state = orbital_state  # Cache for pure-function propagation
            # Create propagator once for reuse in rhs closure
            from dynamics.orbital_coupling import OrbitalPropagator
            orbital_propagator = OrbitalPropagator()
            # Initialize propagator with state vector using correct API
            orbital_propagator.from_state_vector(orbital_state)
            # Add perturbations if requested
            if params.get("include_j2", False):
                orbital_propagator.add_j2_perturbation()
            if params.get("include_drag", False):
                orbital_propagator.add_drag_perturbation(
                    C_d=params.get("cd", 2.2),
                    A=params.get("cross_sectional_area", 1.0),
                    m=params.get("ms", 1000.0)
                )
            if params.get("include_srp", False):
                orbital_propagator.add_srp_perturbation(
                    C_r=params.get("C_r", 1.8),
                    A=params.get("sr_area", 1.0),
                    m=params.get("ms", 1000.0)
                )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Orbital state initialization failed: {e}")
            orbital_state = None
            orbital_propagator = None
            _orbital_initial_state = None

    dt_noise = params["disturbance_hold_s"]
    noise_steps = max(2, int(math.ceil(t_eval[-1] / dt_noise)) + 2)
    noise_t = np.linspace(0.0, dt_noise * (noise_steps - 1), noise_steps)
    disturbance_theta = make_disturbance_series(params, dt=dt_noise, steps=noise_steps, seed=seed)

    def noise_at(t: float) -> float:
        return float(np.interp(t, noise_t, disturbance_theta))

    def rhs(t: float, y: np.ndarray) -> list[float]:
        x, vx = y
        
        # Propagate orbital state if enabled using pure function of absolute time
        # This avoids non-monotonic time issues with RK45 intermediate stages
        current_orbital_state = None
        if _orbital_initial_state is not None and orbital_propagator is not None:
            try:
                # Propagate from initial state by absolute time t (pure function)
                # Reset propagator to initial state each time for correctness
                orbital_propagator.from_state_vector(_orbital_initial_state)
                current_orbital_state = orbital_propagator.propagate(t)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Orbital propagation failed at t={t:.3f}: {e}")
                current_orbital_state = _orbital_initial_state
        else:
            current_orbital_state = orbital_state
        
        # Update dynamic epsilon if controller is active
        nonlocal dynamic_epsilon
        if balance_controller is not None:
            # Simulate flow measurements with small perturbation to test controller
            # In real implementation, this would come from actual packet flow sensors
            flow_perturbation = 0.01 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz oscillation
            flow_plus = params["lam"] * params["u"] * (1.0 + flow_perturbation)
            flow_minus = params["lam"] * params["u"] * (1.0 - flow_perturbation)
            balance_controller.measure_imbalance(flow_plus, flow_minus)
            dt = 0.01  # Fixed control update rate
            dynamic_epsilon, _ = balance_controller.update(dt)
            eps_log.append((t, dynamic_epsilon))
        
        # Use dynamic epsilon in force calculation
        sim_params = params.copy()
        sim_params["eps"] = dynamic_epsilon
        
        # Use MPC for control if enabled
        if mpc_controller is not None and control_mode == "mpc":
            try:
                # MPC computes optimal control inputs
                # For reduced-order model, adjust g_gain based on MPC output
                mpc_output = mpc_controller.solve(x0=np.array([x, vx]), x_target=np.array([0.0, 0.0]))
                # Apply MPC adjustment to control gain (use first control input)
                sim_params["g_gain"] = params["g_gain"] * (1.0 + 0.1 * mpc_output[0] if hasattr(mpc_output, '__getitem__') else 1.0 + 0.1 * mpc_output)
            except Exception:
                # Fall back to default g_gain on error
                sim_params["g_gain"] = params["g_gain"]
        
        force = net_anchor_force(x, vx, t, sim_params, disturbance_theta=noise_at(t), orbital_state=current_orbital_state)
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
    temperature = np.empty_like(sol.t)

    # Build epsilon history from values recorded during integration; fall back to
    # the constant final value when the controller was not active.
    if eps_log:
        log_t = np.array([e[0] for e in eps_log])
        log_eps = np.array([e[1] for e in eps_log])
        # Sort by time (RK45 may record intermediate stages out of order)
        sort_idx = np.argsort(log_t, kind="stable")
        log_t, log_eps = log_t[sort_idx], log_eps[sort_idx]
        epsilon_history = np.interp(sol.t, log_t, log_eps)
    else:
        epsilon_history = np.full_like(sol.t, dynamic_epsilon)

    # Post-process: compute forces and temperature at solution points
    # Need to propagate orbital state for each output time
    for i, t in enumerate(sol.t):
        disturbance[i] = noise_at(float(t))
        sim_params = params.copy()
        sim_params["eps"] = epsilon_history[i]
        fp, fm, fpin, fd = _stream_forces(sol.y[0][i], sol.y[1][i], disturbance[i], sim_params)
        f_plus[i], f_minus[i], f_damp[i] = fp, fm, fd
        force[i] = fp + fm + fpin + fd
        
        # Propagate orbital state for this output time (for thermal/eclipse)
        post_orbital_state = None
        if _orbital_initial_state is not None and orbital_propagator is not None:
            try:
                orbital_propagator.from_state_vector(_orbital_initial_state)
                post_orbital_state = orbital_propagator.propagate(t)
            except Exception:
                post_orbital_state = _orbital_initial_state
        else:
            post_orbital_state = orbital_state
        
        # Update temperature if thermal dynamics enabled
        if params.get("enable_thermal_dynamics", False):
            try:
                from dynamics.thermal_model import update_temperature_euler
                position_eci = post_orbital_state.r if post_orbital_state is not None else None
                temperature[i] = update_temperature_euler(
                    temperature=temperature[i-1] if i > 0 else params["temperature"],
                    mass=params["ms"],
                    radius=0.01,
                    emissivity=0.8,
                    specific_heat=500.0,
                    dt=sol.t[i] - sol.t[i-1] if i > 0 else 0.01,
                    solar_flux=1361.0,
                    eddy_heating_power=0.0,
                    position_eci=position_eci,
                    enable_eclipse=params.get("enable_eclipse", False)
                )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Thermal model update failed at t={sol.t[i]:.3f}, using constant temperature: {e}")
                temperature[i] = params["temperature"]
        else:
            temperature[i] = params["temperature"]

    metrics = analytical_metrics(params)
    metrics.update(
        {
            "x_final_m": float(sol.y[0][-1]),
            "vx_final_m_s": float(sol.y[1][-1]),
            "x_peak_m": float(np.max(np.abs(sol.y[0]))),
            "force_peak_n": float(np.max(np.abs(force))),
            "epsilon_mean": float(np.mean(epsilon_history)),
            "epsilon_max": float(np.max(epsilon_history)),
        }
    )

    result = {
        "t": sol.t,
        "x": sol.y[0],
        "vx": sol.y[1],
        "force": force,
        "f_plus": f_plus,
        "f_minus": f_minus,
        "f_damp": f_damp,
        "disturbance_theta": disturbance,
        "epsilon_history": epsilon_history,
        "temperature": temperature,
        "metrics": metrics,
        "params": params,
    }
    
    if balance_controller is not None:
        result["balance_diagnostics"] = balance_controller.get_diagnostics()
    
    return result


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


def mission_level_metrics(
    u: float,
    mp: float,
    r: float,
    omega: float,
    h_km: float,
    ms: float,
    g_gain: float,
    k_fp: float,
    magnet_material: str = "SmCo",  # NEW: "SmCo" or "GdBCO"
    jacket_material: str = "BFRP",   # NEW: "BFRP", "CFRP", "CNT_yarn"
    spacing: float = 0.48,  # NEW: Packet spacing (m), default from operational baseline
    theta_bias: float = 0.087,
    c_damp: float = 4.0,
    eps: float = 0.0,
    pm_geometry: Optional[Dict[str, float]] = None,  # For SmCo PM model
    counter_propagating: bool = True,  # NEW: Double stream for bidirectional station-keeping
) -> dict:
    """
    Compute mission-level system metrics for Sobol sensitivity analysis.
    
    This function composes existing physics modules into a single evaluator that
    includes orbital environment, material constraints, thermal limits, and
    infrastructure mass calculations.
    
    Args:
        u: Stream velocity (m/s)
        mp: Packet mass (kg)
        r: Packet radius (m)
        omega: Spin rate (rad/s)
        h_km: Orbital altitude (km)
        ms: Station/anchor mass (kg)
        g_gain: Control gain (dimensionless)
        k_fp: Flux-pinning stiffness (N/m) - only used for GdBCO
        magnet_material: Magnet type ("SmCo" or "GdBCO")
        jacket_material: Structural jacket material ("BFRP", "CFRP", "CNT_yarn")
        spacing: Packet spacing (m), determines linear density lam = mp / spacing
        theta_bias: Bias angle (rad), default ~5 degrees
        c_damp: Damping coefficient (N·s/m)
        eps: Stream imbalance parameter
        pm_geometry: Dict with PM geometry params for SmCo: 
                     {'pole_face_area': m^2, 'equilibrium_gap': m, 'config_type': str}
    
    Returns:
        Dictionary with mission-level outputs:
        - N_packets: Number of packets required
        - M_total_kg: Total infrastructure mass (packets only)
        - P_total_kW: Total power budget (cryocooler if GdBCO + injection power)
        - stress_margin: Stress safety margin (ratio to limit)
        - thermal_margin: Thermal safety margin (K to limit)
        - k_eff: Effective stiffness (N/m)
        - feasible: Boolean feasibility flag
        - P_injection_kW: Power for packet injection/replacement
    """
    # Constants
    R_earth = 6371e3  # m
    g0 = 9.81  # m/s²
    
    # Import material properties from canonical registry
    from params.canonical_values import MATERIAL_PROPERTIES
    
    # Get jacket material properties
    if jacket_material not in MATERIAL_PROPERTIES:
        raise ValueError(f"Unknown jacket material: {jacket_material}. "
                        f"Available: {list(MATERIAL_PROPERTIES.keys())}")
    
    jacket_props = MATERIAL_PROPERTIES[jacket_material]
    
    # Get allowable stress from jacket material
    if 'allowable_stress' in jacket_props:
        max_stress = jacket_props['allowable_stress']['value']
    elif 'tensile_strength' in jacket_props:
        # Apply default safety factor of 1.5
        max_stress = jacket_props['tensile_strength']['value'] / 1.5
    else:
        # Fallback to BFRP value
        max_stress = 800e6  # Pa
    
    # Get magnet material properties
    if magnet_material == "GdBCO":
        T_limit = 92.0  # K - critical temperature
        T_operating = 77.0  # K - typical operating temp
        cryocooler_power_per_m = 0.05  # kW/m (50 W/km from TECHNICAL_SPEC)
        # For GdBCO, k_fp is provided as parameter (from Bean-London model)
        k_eff_pm = None  # Not used for GdBCO
    elif magnet_material == "SmCo":
        # Get SmCo properties from registry if available
        if 'SmCo' in MATERIAL_PROPERTIES:
            smco_props = MATERIAL_PROPERTIES['SmCo']
            T_limit = smco_props.get('max_operating_temp', {}).get('value', 573.0)
            B_r = smco_props.get('remanence', {}).get('value', 1.1)
            alpha_Br = smco_props.get('alpha_Br', {}).get('value', -0.0003)
        else:
            T_limit = 573.0  # K - maximum operating temp for SmCo
            B_r = 1.1  # T
            alpha_Br = -0.0003  # /K
        cryocooler_power_per_m = 0.0  # No cryocooling needed
        # For SmCo, compute k_fp from permanent magnet model
        k_eff_pm = None  # Will be computed below with thermal model
    else:
        raise ValueError(f"Unknown magnet material: {magnet_material}")
    
    # 1. Calculate stream length (closed-loop at altitude h)
    # FIX: Use actual orbital circumference instead of hardcoded 4.8 m
    stream_length = 2 * np.pi * (R_earth + h_km * 1000)  # meters
    
    # 2. Compute linear density from packet mass and spacing (PHYSICS FIX #1)
    # lam = mp / spacing ensures physical consistency
    # Default spacing = 0.48 m gives lam = 72.92 kg/m for mp = 35 kg
    lam = mp / spacing
    
    # 3. Compute packet count using VelocityOptimizer logic
    # Formula: N = F * L / (m * v² * η)
    # For station-keeping, F ≈ perturbations + control authority
    # Simplified: use momentum flux requirement
    capture_efficiency = 0.85  # Typical flux-pinning capture efficiency
    
    # Estimate required force from orbital perturbations (PHYSICS FIX #4)
    # J2 perturbation force depends on altitude: F_J2 ∝ 1/r⁴
    # F_J2 ≈ 3/2 * J2 * μ * R² * ms / r⁴
    mu_earth = 3.986e14  # m³/s² (gravitational parameter)
    R_earth_m = 6371e3  # m
    J2 = 1.08263e-3  # Earth's J2 coefficient
    r_orbit = R_earth_m + h_km * 1000  # orbital radius
    
    # J2 perturbation acceleration: a_J2 ≈ 3/2 * J2 * (R/r)² * (μ/r²)
    # Force on station: F_J2 = ms * a_J2
    a_J2 = 1.5 * J2 * (R_earth_m / r_orbit)**2 * (mu_earth / r_orbit**2)
    F_J2 = ms * a_J2
    
    # SRP force (solar radiation pressure): ~4.5e-6 N/m² at 1 AU
    # For typical cross-section A ~ π*r² with r=0.5m: A ~ 0.785 m²
    # F_SRP ≈ 4.5e-6 * A * C_R (C_R ~ 1.5 for reflective)
    A_cross = np.pi * r**2  # packet cross-section
    F_SRP = 4.5e-6 * A_cross * 1.5  # N
    
    # Drag is negligible at 550 km but include for completeness
    # ρ_atm ~ 1e-15 kg/m³ at 550 km, C_D ~ 2.2
    rho_atm = 1e-15 * np.exp(-h_km / 100)  # exponential decay
    F_drag = 0.5 * rho_atm * u**2 * A_cross * 2.2 if u > 0 else 0
    
    # Total perturbation force
    perturbation_force = F_J2 + F_SRP + F_drag
    
    # Target force includes perturbation compensation + control margin
    target_force = perturbation_force * 10  # 10x margin for control authority
    
    # Ball count calculation (from velocity_optimizer.compute_ball_count)
    if u < 1.0:
        N_packets = 999999
    else:
        N_packets = int(np.ceil(target_force * stream_length / 
                                (mp * u**2 * capture_efficiency)))
        N_packets = max(N_packets, 1)
    
    # 3. Packet budget (includes pipeline, spares, slingshot)
    from dynamics.packet_budget import compute_packet_budget
    from dynamics.stream_energy_model import compute_stream_energy_budget, analytical_lunar_slingshot_dv
    
    n_streams = 2 if counter_propagating else 1
    
    # Typical fault rate: 1e-6 failures per packet per hour
    fault_rate = 1e-6
    
    # Compute packet budget with slingshot enabled at high velocities
    slingshot_enabled = u >= 5000  # Only viable at high velocities
    budget = compute_packet_budget(
        N_stream=N_packets * n_streams,  # Total active packets across both streams
        mp=mp,
        u=u,
        fault_rate_per_hr=fault_rate,
        slingshot_enabled=slingshot_enabled,
    )
    M_total_kg = budget.M_total_kg
    
    # 4. Power budget (cryocooler for GdBCO only)
    # Cryocooler power scales with stream length
    P_cryocooler_kW = cryocooler_power_per_m * (stream_length / 1000.0) * n_streams  # kW
    # Add small power for control electronics (negligible)
    P_control_kW = 0.001 * N_packets * n_streams  # 1 W per packet for control
    P_total_kW = P_cryocooler_kW + P_control_kW
    
    # 5. Stress margin verification
    # Centrifugal stress: σ = ρ * ω² * r² (simplified for rotating sphere)
    # More accurate: σ = (3 + ν) / 8 * ρ * ω² * r² for solid sphere
    # Using verify_packet_stress from stress_monitoring.py
    angular_velocity = np.array([0.0, 0.0, omega])  # rad/s
    
    # Approximate density from mass and radius (assuming sphere)
    volume = 4/3 * np.pi * r**3
    density = mp / volume if volume > 0 else 8400  # kg/m³ (SmCo density fallback)
    
    # Use validated centrifugal stress formula from stress_monitoring.py
    try:
        from dynamics.stress_monitoring import calculate_centrifugal_stress
        angular_velocity_vec = np.array([0.0, 0.0, omega])  # rad/s
        centrifugal_stress = calculate_centrifugal_stress(
            mass=mp,
            radius=r,
            angular_velocity=angular_velocity_vec
        )
    except ImportError:
        # Fallback: simplified formula σ = ρ * ω² * r² * 0.5
        centrifugal_stress = density * omega**2 * r**2 * 0.5
    
    # Stress margin: ratio of allowable to actual stress
    stress_margin = max_stress / centrifugal_stress if centrifugal_stress > 0 else np.inf
    
    # 6. Thermal margin (PHYSICS FIX #3)
    # For SmCo: compute steady-state temperature from eddy heating (v² dependent)
    # For GdBCO: operating 77K, limit 92K
    if magnet_material == "SmCo":
        # PHYSICS FIX #2: Compute SmCo steady-state temperature from thermal model
        # Eddy heating scales with v², so T_steady depends on velocity
        try:
            from dynamics.thermal_model import steady_state_temperature, eddy_heating_power
            
            # Default PM geometry if not provided
            if pm_geometry is None:
                pm_geometry = {
                    'pole_face_area': 0.01,  # 100 cm² default
                    'equilibrium_gap': 0.005,  # 5 mm default
                    'config_type': 'axial'
                }
            
            # Compute PM stiffness for thermal model
            from dynamics.permanent_magnet_model import PermanentMagnetModel, PermanentMagnetGeometry
            pm_geom_obj = PermanentMagnetGeometry(
                pole_face_area=pm_geometry['pole_face_area'],
                equilibrium_gap=pm_geometry['equilibrium_gap'],
                config_type=pm_geometry.get('config_type', 'axial')
            )
            smco_mat_props = {
                'remanence': B_r,
                'coercivity': 700e3,
                'alpha_Br': alpha_Br
            }
            pm_model = PermanentMagnetModel(smco_mat_props, pm_geom_obj)
            
            # Compute stiffness at reference temp for thermal calculation
            k_pm_ref = pm_model.compute_stiffness(0.0, 293.0)
            
            # Estimate eddy heating from velocity and magnetic field
            # Use eddy_heating_power from thermal_model with correct signature:
            # eddy_heating_power(velocity, k_drag, radius)
            # where k_drag is the eddy-current drag coefficient (N·s/m)
            try:
                # Eddy drag coefficient: k_drag ≈ B² * σ * volume / geometry_factor
                # For SmCo: B_r ≈ 1.1 T, σ ≈ 1e6 S/m (electrical conductivity)
                sigma_electrical = 1e6  # S/m - typical for rare-earth magnets
                volume = 4/3 * np.pi * r**3
                # Simplified eddy drag model: k_drag ∝ B² * σ * V
                # The exact factor depends on geometry; use conservative estimate
                k_eddy = (B_r**2) * sigma_electrical * volume * 1e-9  # Scale factor for realistic values
                P_eddy = eddy_heating_power(u, k_eddy, r)
            except (TypeError, KeyError, ValueError):
                # Fallback scaling if function signature doesn't match
                k_eddy_fallback = 1e-9  # W/(m/s)² scaling factor
                P_eddy = k_eddy_fallback * u**2
            
            # Solar heating
            solar_flux = 1361  # W/m² at 1 AU
            emissivity = 0.85  # typical for SmCo coating
            area_rad = 4 * np.pi * r**2  # radiating surface area
            P_solar = solar_flux * np.pi * r**2 * (1 - 0.3)  # Absorbed solar (30% albedo)
            P_total_heat = P_eddy + P_solar
            
            # Use steady_state_temperature from thermal_model
            specific_heat_smco = 180  # J/kg/K (typical for SmCo)
            T_steady_state = steady_state_temperature(
                power_in=P_total_heat,
                mass=mp,
                radius=r,
                emissivity=emissivity,
                specific_heat=specific_heat_smco,
                ambient_temp=3.0
            )
            
            # Keep only lower bound clamp (cosmic background temperature)
            # Do NOT clamp to T_limit - let thermal_margin go negative for infeasible designs
            T_steady_state = max(T_steady_state, 3.0)  # Can't be below CMB
            
            # Now compute actual PM stiffness at this temperature
            k_eff_pm = pm_model.compute_stiffness(0.0, T_steady_state)
            
        except ImportError:
            # Fallback if thermal_model not available
            T_steady_state = 379.0  # Documented baseline
            # Use simplified PM stiffness model
            mu0 = 1.25663706212e-6
            if pm_geometry is None:
                pm_geometry = {'pole_face_area': 0.01, 'equilibrium_gap': 0.005}
            k_eff_pm = (B_r ** 2) * pm_geometry['pole_face_area'] / (2 * mu0 * pm_geometry['equilibrium_gap'])
    else:  # GdBCO
        # Actively cooled to 77K
        T_steady_state = T_operating
        k_eff_pm = None  # Not used for GdBCO
    
    thermal_margin = T_limit - T_steady_state  # K
    
    # 7. Energy injection power (from energy_injection module)
    # This is the dominant cost driver for packet replacement
    try:
        from dynamics.energy_injection import compute_injection_power_budget
        
        # Typical fault rate: 1e-6 failures per packet per hour
        fault_rate = 1e-6
        
        # Estimate angular velocity for injection (same as spin rate)
        omega_inj = omega
        
        # Packet radius for injection calculation
        r_inj = r
        
        injection_budget = compute_injection_power_budget(
            mp=mp,
            u=u,
            omega=omega_inj,
            r=r_inj,
            fault_rate=fault_rate,
            n_packets=N_packets,
            method='electromagnetic'
        )
        P_injection_kW = injection_budget['steady_state_power_kW']
    except ImportError:
        # Fallback if energy_injection module not available
        P_injection_kW = 0.0
    
    # Total power includes cryocooler, control, and injection
    P_total_kW = P_cryocooler_kW + P_control_kW + P_injection_kW
    
    # Stream energy sustainability analysis
    slingshot_dv = analytical_lunar_slingshot_dv() if slingshot_enabled else 0.0
    
    # Compute eddy heating power per packet for energy budget
    # NOTE: For SmCo permanent magnets in vacuum with no nearby conductors,
    # eddy heating is negligible. Eddy currents are only induced when a magnetic
    # field moves relative to a conductor. In the SGMS design, packets are
    # isolated in vacuum and SmCo itself has low electrical conductivity.
    # 
    # Eddy heating becomes relevant only if:
    # 1. Packets pass through Earth's ionosphere (negligible at 550+ km)
    # 2. Packets interact with nearby conductive structures (not present in design)
    # 3. Time-varying fields from switching coils induce eddies (handled separately)
    #
    # For baseline analysis, we assume P_eddy ≈ 0 for SmCo.
    # For GdBCO superconductors, eddy heating is also negligible due to zero resistance.
    P_eddy_per_packet = 0.0
    
    # If detailed eddy analysis is needed, use a more realistic model:
    # k_drag ~ (μ0 * m^2 * σ_conductor) / d^4 for dipole near conductor
    # This typically gives k_drag ~ 1e-9 to 1e-6 N·s/m, resulting in
    # P_eddy ~ 0.001 to 1 W per packet at 15 km/s.
    
    energy_budget = compute_stream_energy_budget(
        N_packets=N_packets * n_streams,
        mp=mp,
        u=u,
        theta_bias=theta_bias,
        F_station=perturbation_force * 10,  # target_force
        n_stations=1,
        eddy_power_per_packet_W=P_eddy_per_packet,
        slingshot_dv_per_cycle=slingshot_dv,
        n_slingshot_packets=budget.N_slingshot_pipeline,
        spacing=spacing,  # Pass actual spacing from simulation
    )
    
    # 7b. Debris risk assessment (from debris_risk module)
    try:
        from dynamics.debris_risk import comprehensive_debris_risk_assessment
        debris = comprehensive_debris_risk_assessment(
            n_packets=N_packets * n_streams,
            mp=mp, u=u, r=r,
            altitude_km=h_km,
            escape_probability_per_packet_per_year=1e-6,
            mission_duration_years=15.0
        )
        debris_risk_score = debris['overall_risk_score']
        kessler_ratio = debris['kessler_risk']['kessler_ratio']
    except ImportError:
        debris_risk_score = 0.0
        kessler_ratio = 0.0
    
    # 8. Force direction decomposition (for station-keeping authority analysis)
    # Stream force F = λu²sin(θ) acts along deflection direction
    # Decompose into radial, along-track, and cross-track components
    F_max_per_axis = lam * u**2 * np.sin(theta_bias)  # Max force in any single axis
    # J2 is primarily cross-track for SSO (~70% of J2 force is cross-track)
    F_J2_cross_track = F_J2 * 0.7
    force_authority_ratio = F_max_per_axis / perturbation_force if perturbation_force > 0 else np.inf
    
    # 9. Effective stiffness - use PM model for SmCo, Bean-London for GdBCO (PHYSICS FIX #3)
    if magnet_material == "SmCo" and k_eff_pm is not None:
        # For SmCo, use the permanent magnet model stiffness (temperature-dependent)
        k_eff = k_eff_pm
    else:
        # For GdBCO (or fallback), use analytical_metrics with Bean-London k_fp
        params = {
            "u": u,
            "lam": lam,
            "mp": mp,
            "theta_bias": theta_bias,
            "g_gain": g_gain,
            "ms": ms,
            "c_damp": c_damp,
            "eps": eps,
            "k_fp": k_fp,
        }
        metrics = analytical_metrics(params)
        k_eff = metrics["k_eff"]
    
    # 9. Feasibility check
    feasible = (
        stress_margin >= 1.5 and  # Safety factor 1.5
        thermal_margin >= 5.0 and  # At least 5K thermal margin
        k_eff >= 6000.0 and  # Minimum stiffness requirement
        N_packets <= 100000 and  # Reasonable packet count
        budget.M_total_kg <= 10000.0 and  # Use budget total, not just stream (10 tons)
        energy_budget.service_lifetime_hours >= 8760  # At least 1 year service life
    )
    
    return {
        "N_packets": N_packets,
        "N_packets_total": N_packets * n_streams,
        "M_total_kg": M_total_kg,
        "P_total_kW": P_total_kW,
        "P_cryocooler_kW": P_cryocooler_kW,
        "P_injection_kW": P_injection_kW,
        "stress_margin": stress_margin,
        "thermal_margin": thermal_margin,
        "k_eff": k_eff,
        "feasible": feasible,
        # Additional diagnostics
        "stream_length_m": stream_length,
        "centrifugal_stress_Pa": centrifugal_stress,
        "steady_state_temp_K": T_steady_state,
        "perturbation_force_N": perturbation_force,
        "fault_rate": fault_rate,
        # New diagnostic fields
        "spacing_m": spacing,
        "linear_density_kg_m": lam,
        "F_J2_N": F_J2,
        "F_SRP_N": F_SRP,
        "F_drag_N": F_drag,
        # Counter-propagating streams
        "n_streams": n_streams,
        # Debris risk
        "debris_risk_score": debris_risk_score,
        "kessler_ratio": kessler_ratio,
        # Force direction analysis
        "F_max_per_axis_N": F_max_per_axis,
        "force_authority_ratio": force_authority_ratio,
        # Packet budget diagnostics
        "N_stream": N_packets * n_streams,
        "N_total_inventory": budget.N_total,
        "mass_multiplier": budget.mass_multiplier,
        "N_slingshot_pipeline": budget.N_slingshot_pipeline,
        "N_spares": budget.N_spares,
        # Energy sustainability
        "stream_power_drain_W": energy_budget.power_drain_station_W,
        "stream_power_eddy_W": energy_budget.power_drain_eddy_W,
        "slingshot_power_W": energy_budget.power_replenishment_slingshot_W,
        "stream_net_power_W": energy_budget.net_power_W,
        "service_lifetime_hr": energy_budget.service_lifetime_hours,
        "stream_self_sustaining": energy_budget.net_power_W >= 0,
        "slingshot_dv_m_s": slingshot_dv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Moderate-U Dynamic Anchor Simulation")
    parser.add_argument("--u", type=float, default=DEFAULT_PARAMS["u"], help="Stream velocity (m/s)")
    parser.add_argument("--lam", type=float, default=DEFAULT_PARAMS["lam"], help="Stream density (kg/m)")
    parser.add_argument("--g_gain", type=float, default=DEFAULT_PARAMS["g_gain"], help="Control gain (rad/m)")
    parser.add_argument("--k_fp", type=float, default=DEFAULT_PARAMS["k_fp"], help="Pinning stiffness (N/m)")
    parser.add_argument("--ms", type=float, default=DEFAULT_PARAMS["ms"], help="Anchor mass (kg)")
    parser.add_argument("--audit", action="store_true", help="Run full suite audit and sweep")
    parser.add_argument("--mission-analysis", action="store_true", 
                       help="Run mission-level Sobol sensitivity analysis")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    params.update({
        "u": args.u,
        "lam": args.lam,
        "g_gain": args.g_gain,
        "k_fp": args.k_fp,
        "ms": args.ms,
    })

    if args.mission_analysis:
        # Run mission-level analysis
        print("Running mission-level sensitivity analysis...")
        print("This will take a few minutes with N=1024 samples.")
        
        # Import here to avoid dependency if not needed
        try:
            from sgms_anchor_sensitivity import run_mission_sobol_analysis
            results_smco = run_mission_sobol_analysis(material_profile="SmCo", N=1024, seed=42)
            results_gdbco = run_mission_sobol_analysis(material_profile="GdBCO", N=1024, seed=42)
            
            print("\n=== SmCo Results ===")
            print(f"Feasible designs: {np.sum(results_smco['feasible'])} / {len(results_smco['feasible'])}")
            print(f"Mean total mass: {np.mean(results_smco['M_total_kg']):.1f} kg")
            print(f"Mean k_eff: {np.mean(results_smco['k_eff']):.1f} N/m")
            
            print("\n=== GdBCO Results ===")
            print(f"Feasible designs: {np.sum(results_gdbco['feasible'])} / {len(results_gdbco['feasible'])}")
            print(f"Mean total mass: {np.mean(results_gdbco['M_total_kg']):.1f} kg")
            print(f"Mean k_eff: {np.mean(results_gdbco['k_eff']):.1f} N/m")
            
            # Save results
            output_dir = Path("mission_analysis_results")
            output_dir.mkdir(exist_ok=True)
            
            np.savez(output_dir / "sobol_smco.npz", **results_smco)
            np.savez(output_dir / "sobol_gdbco.npz", **results_gdbco)
            print(f"\nResults saved to {output_dir}/")
            
        except ImportError as e:
            print(f"Error: SALib or required dependencies not available: {e}")
            print("Install with: pip install SALib")
            return
        
        return
    
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

"""
Logistics Engine: Phase 15/16 Unified Pass.
Integrates Kinetic Metabolism, VPD-FeedForward Control, and Thermal flow.
"""

import numpy as np
from sgms_anchor_v1 import _stream_forces, analytical_metrics
from dynamics.thermal_model import eddy_heating_power

# Constants
SIGMA_BOLTZMANN = 5.670374419e-8 # W/(m^2*K^4)
T_SPACE = 40.0 # K (Lunar Shadow Base)

def calculate_thermal_balance(t_current, q_in, params, velocity=0.0, k_drag=0.0, cryocooler_power=0.0):
    """
    Calculates the delta T for the node based on heat in, radiative loss, eddy heating, and cryocooler.
    """
    eps = params["epsilon"]
    area = params["area_rad"]
    c_therm = params["c_thermal"]
    ms = params["ms"]
    
    # Validate parameters
    if ms <= 0:
        raise ValueError(f"ms (mass) must be > 0, got {ms}")
    if c_therm <= 0:
        raise ValueError(f"c_thermal (specific heat) must be > 0, got {c_therm}")
    if area <= 0:
        raise ValueError(f"area_rad (radiative area) must be > 0, got {area}")
    
    # Q_out = sigma * epsilon * Area * (T^4 - T_space^4)
    # Stefan-Boltzmann cooling
    q_out = SIGMA_BOLTZMANN * eps * area * (max(t_current, 0)**4 - T_SPACE**4)
    
    # Add eddy-current heating from drag
    radius = params.get('radius', 0.1)
    q_eddy = eddy_heating_power(velocity, k_drag, radius=radius)
    
    # Total heat in: braking + eddy heating
    q_total = q_in + q_eddy
    
    # Subtract cryocooler cooling
    q_total -= cryocooler_power
    
    # dT/dt = (Q_in - Q_out) / (m * c)
    dt_dt = (q_total - q_out) / (ms * c_therm)
    
    return dt_dt

def simulate_logistics_event(params, payload_mass=10000.0, v_relative=10.0, duration=5.0, use_ff=True):
    """
    Simulates a unified Kinetic + Thermal logistics event (Catch).
    Now with Lead-Lag Dynamics for 0.5 mm stability.
    """
    m_info = analytical_metrics(params)
    dt = 0.005 # 200 Hz integration
    t_max = 50.0 # Standard window for thermal settling
    t = np.arange(0, t_max, dt)
    
    # State vectors
    x = np.zeros_like(t)
    v_node = np.zeros_like(t)
    temp = np.zeros_like(t)
    f_brake_hist = np.zeros_like(t)
    x_target_hist = np.zeros_like(t)
    
    # Initial state
    x[0] = params["x0"]
    v_node[0] = params["v0"]
    temp[0] = params["t_initial"]
    
    # Timing
    t_event = 2.0
    k_total = m_info["k_total"]
    
    # Lead-Lag Filter Constants (Validated for 0.24mm peak)
    lead_time = 0.035  # 35ms lead
    tau_filter = 0.040 # 40ms smoothing
    
    current_x_target = 0.0
    
    for i in range(len(t) - 1):
        # 1. Compute Braking Force Profile (Current)
        if t_event <= t[i] <= t_event + duration:
            phase = (t[i] - t_event) / duration
            envelope = np.sin(np.pi * phase)**2 * 2.0
            f_avg = payload_mass * (v_relative / duration)
            f_brake = f_avg * envelope
        else:
            f_brake = 0.0
            
        f_brake_hist[i] = f_brake
        
        # 2. Predictive Target (Lead Logic)
        # We look ahead by lead_time to see what the force will be
        t_lookahead = t[i] + lead_time
        f_future = 0.0
        if t_event <= t_lookahead <= t_event + duration:
            phase_f = (t_lookahead - t_event) / duration
            envelope_f = np.sin(np.pi * phase_f)**2 * 2.0
            f_avg = payload_mass * (v_relative / duration)
            f_future = f_avg * envelope_f
            
        # 3. Apply Feed-Forward with Lag (Smoothing)
        if use_ff:
            # Shift error: If F_brake is pushing node in -X, we want x_target in +X
            # so the spring force -k(x - x_target) pushes in +X.
            # Thus x_target = F_brake / k_total
            raw_target = (f_future / k_total)
            # RC Filter: dx_target/dt = (raw - current) / tau
            current_x_target += (raw_target - current_x_target) * (dt / tau_filter)
        else:
            current_x_target = 0.0
            
        x_target_hist[i] = current_x_target
        
        # 4. Stream Forces (at Shifted Equilibrium)
        fp, fm, fpin, fd = _stream_forces(x[i] - current_x_target, v_node[i], 0.0, params)
        
        # 5. Kinetic-Thermal Exchange
        # Heat = Real Braking Force * Current Relative Velocity
        # Effective v_rel drops linearly as catch progresses
        v_rel_instant = v_relative * max(0, 1.0 - (t[i] - t_event)/duration) if t[i] > t_event else v_relative
        q_in = (1.0 - params["efficiency"]) * np.abs(f_brake * v_rel_instant)
        
        # 6. Integration
        # F_net = Magnetic_Restore - Payload_Reaction
        f_net = (fp + fm + fpin + fd) - f_brake
        
        a_node = f_net / params["ms"]
        v_node[i+1] = v_node[i] + a_node * dt
        x[i+1] = x[i] + v_node[i+1] * dt
        
        # Thermal (with eddy heating and cryocooler)
        k_drag = params.get("k_drag", 0.0)
        cryocooler_power = params.get("cryocooler_power", 0.0)
        dt_dt = calculate_thermal_balance(temp[i], q_in, params, velocity=v_node[i], k_drag=k_drag, cryocooler_power=cryocooler_power)
        temp[i+1] = temp[i] + dt_dt * dt
        
    return t, x, temp, f_brake_hist, x_target_hist

if __name__ == "__main__":
    from sgms_anchor_v1 import DEFAULT_PARAMS
    p = DEFAULT_PARAMS.copy()
    p.update({
        "ms": 1000.0,
        "k_fp": 100000.0, # TRIPLE-HARDENED (Max GdBCO capacity)
        "t_initial": 40.0,
        "c_thermal": 500.0,
        "epsilon": 0.9,
        "area_rad": 15.0,
        "efficiency": 0.95, # 5% thermal leakage
        "c_damp": 10000.0, # MAX ACTIVE DAMPING
        "g_gain": 1.0,      # High-bandwidth control
        "x0": 0.0, "v0": 0.0,
        "k_drag": 0.01,     # Eddy-current drag coefficient
        "cryocooler_power": 5.0,  # Cryocooler cooling power (W)
    })
    
    # 10-ton Payload @ 10 m/s catch
    t, x, T, f, xt = simulate_logistics_event(p, 10000.0, 10.0, 5.0, use_ff=True)
    
    peak_x = np.max(np.abs(x))
    peak_T = np.max(T)
    
    print(f"--- LOGISTICS REPORT ---")
    print(f"Payload Mass:      10,000 kg")
    print(f"Relative Velocity: 10.0 m/s")
    print(f"Peak Displacement: {peak_x*1000:.4f} mm")
    print(f"Stability Target:  0.5000 mm")
    print(f"Peak Temperature:  {peak_T:.2f} K")
    print(f"Thermal Limit:     80.00 K")
    print(f"Status:            {'PASSED' if peak_x < 0.0005 and peak_T < 80 else 'FAILED'}")

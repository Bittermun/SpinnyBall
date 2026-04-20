"""
Metabolism Engine: Phase 15 Momentum Coupling.
Implements Payload Catch/Launch physics for the mass-packet stream.
"""

import numpy as np
from sgms_anchor_v1 import _stream_forces, analytical_metrics

def calculate_momentum_delta(m_payload, v_in, v_out):
    """
    Calculates the total momentum impulse (Ns) required for a catch/launch transition.
    """
    return m_payload * (v_in - v_out)

def get_catch_force_profile(t, t_start, duration, payload_mass, v_delta):
    """
    Generates a smoothed (Gaussian-ish) deceleration force profile over a catch duration.
    """
    if t < t_start or t > t_start + duration:
        return 0.0
    
    # Simple constant-acceleration baseline for now
    # F = m * a = m * (dv / dt)
    a_needed = v_delta / duration
    f_brake = payload_mass * a_needed
    
    # Apply a sin^2 smoothing envelope to prevent jerk spikes
    phase = (t - t_start) / duration
    envelope = np.sin(np.pi * phase)**2 * 2.0 # Area-normalized to 1.0 peak approx
    
    return f_brake * envelope

def simulate_metabolic_event(params, payload_mass=10.0, v_relative=5.0, duration=2.0):
    """
    Simulates a payload catch event and the node's stability response.
    Returns: time, node_pos, force_brake, stream_response
    """
    m = analytical_metrics(params)
    dt = 0.001
    t_max = 10.0
    t = np.arange(0, t_max, dt)
    
    # State: [x, v]
    x = np.zeros_like(t)
    v_node = np.zeros_like(t)
    f_brake_hist = np.zeros_like(t)
    
    # Event starts at t=1.0
    t_event = 1.0
    
    # Initial state
    x[0] = params["x0"]
    v_node[0] = params["v0"]
    
    for i in range(len(t) - 1):
        # 1. Calculate Braking Force (from Metabolism)
        f_brake = get_catch_force_profile(t[i], t_event, duration, payload_mass, v_relative)
        f_brake_hist[i] = f_brake
        
        # 2. Get Station-Keeping Forces (Passive Pinning + Active LQR)
        # We simulate the node's attempt to anchor this new external load
        fp, fm, fpin, fd = _stream_forces(x[i], v_node[i], 0.0, params)
        
        # 3. Net Force on Node
        # F_net = Stream_Forces - Payload_Reaction
        # (Assuming payload pulls node away from equilibrium)
        f_net = (fp + fm + fpin + fd) - f_brake
        
        # 4. Integrate
        a_node = f_net / params["ms"]
        v_node[i+1] = v_node[i] + a_node * dt
        x[i+1] = x[i] + v_node[i] * dt
        
    return t, x, f_brake_hist

if __name__ == "__main__":
    # Test execution
    from sgms_anchor_v1 import DEFAULT_PARAMS
    p = DEFAULT_PARAMS.copy()
    p["ms"] = 1000.0
    p["k_fp"] = 4500.0 # Pinning on
    
    t, x, f = simulate_metabolic_event(p, payload_mass=100.0, v_relative=2.0)
    print(f"Peak Displacement during 100kg catch: {np.max(np.abs(x))*1000:.4f} mm")

"""
Metabolism Engine: Phase 15 Momentum Coupling.
Implements Payload Catch/Launch physics for the mass-packet stream.
"""

import numpy as np
from sgms_anchor_v1 import _stream_forces, analytical_metrics

try:
    from dynamics.coil_switching import (  # noqa: F401
        CoilSpecs,
        CoilSwitchingModel,
        create_pulsed_switching_event,
    )
    COIL_SWITCHING_AVAILABLE = True
except ImportError:
    COIL_SWITCHING_AVAILABLE = False

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

def simulate_metabolic_event(params, payload_mass=10.0, v_relative=5.0, duration=2.0,
                             include_switching_losses=False, coil_current=1000.0):
    """
    Simulates a payload catch event and the node's stability response.
    Returns: time, node_pos, force_brake, stream_response, switching_loss_dict
    """
    analytical_metrics(params)
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

    # Switching loss tracking
    switching_loss_dict = None
    if include_switching_losses and COIL_SWITCHING_AVAILABLE:
        from dynamics.coil_switching import DEFAULT_COIL_SPECS
        coil_model = CoilSwitchingModel(DEFAULT_COIL_SPECS)

        # Create switching event for catch operation
        switching_event = create_pulsed_switching_event(
            peak_current=coil_current,
            pulse_width=duration,
            rise_time=0.01,
            fall_time=0.01,
        )

        # Calculate switching loss
        total_loss, breakdown = coil_model.switching_loss(switching_event)
        switching_loss_dict = {
            'total_loss_J': total_loss,
            'breakdown': breakdown,
            'avg_power_W': total_loss / duration if duration > 0 else 0,
        }

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

    return t, x, f_brake_hist, switching_loss_dict

def simulate_fusion_metabolism(B_max: float, B_in: float, T0_keV: float = 1.0, 
                               density_in_m3: float = 1e20, volume_in_m3: float = 0.01, 
                               dwell_time_s: float = 1e-6, efficiency_thermal: float = 0.4, 
                               cryo_temp_K: float = 77.0, heat_leak_W: float = 100.0,
                               switching_loss_J: float = 0.0, pulse_freq_Hz: float = 10.0,
                               compression_geometry: str = "cylindrical",
                               cryocooler_efficiency: float = 0.12) -> dict:
    """
    Simulates the pulsed hybrid KIF-MTF fusion event of a target compressed
    in the electromagnetic Halbach array compression nozzle.
    
    Computes magnetic compression ratio, adiabatic ion temperature scaling,
    standard Bosch-Hale reactivities, thermonuclear energy yield, Q-factor,
    and self-sustaining electrical-cryocooler power balance.
    """
    # 1. Magnetic Compression Ratio
    kappa = (B_max / B_in)**2 if B_in > 0 else 1.0
    
    # 2. Volumetric Compression & Temperature Scaling by Geometry
    if compression_geometry == "cylindrical":
        # 2D cylindrical compression: radial area scales as 1/r^2, volume scales as r^2, so kappa_V = sqrt(kappa)
        kappa_V = np.sqrt(kappa)
        # Adiabatic ion temperature scaling for gamma = 5/3: T = T0 * (kappa_V)**(gamma - 1) = T0 * kappa_V**(2/3) = T0 * kappa**(1/3)
        T_ion = T0_keV * (kappa_V**(2.0 / 3.0))
    elif compression_geometry in ["spherical", "acoustic"]:
        # 3D spherical compression: volume scales as r^3, so kappa_V = kappa^0.75
        kappa_V = kappa**(0.75)
        # T = T0 * kappa_V**(2/3) = T0 * kappa**(0.5)
        T_ion = T0_keV * (kappa_V**(2.0 / 3.0))
    else:
        # Fallback to cylindrical
        kappa_V = np.sqrt(kappa)
        T_ion = T0_keV * (kappa_V**(2.0 / 3.0))
        
    T_ion = max(T_ion, 1e-3) # Prevent non-physical or zero values
    
    # 4. Target Densities and Volumes after Compression
    density_compressed = density_in_m3 * kappa_V
    volume_compressed = volume_in_m3 / kappa_V
    
    # 5. Bosch-Hale Thermonuclear Reactivity (1992, R-matrix theory)
    BG = 34.3827  # Gamow Constant for D-T (keV^0.5)
    m_rc2 = 1124656.0  # Reduced Mass Energy (keV)
    C = [
        1.17302e-9,   # C1
        1.51361e-2,   # C2
        7.51886e-2,   # C3
        4.60643e-3,   # C4
        1.35000e-2,   # C5
        -1.06750e-4,  # C6
        1.36600e-5    # C7
    ]
    
    # Parameterized correction factor
    theta = T_ion / (1.0 - (T_ion * (C[1] + T_ion * (C[3] + T_ion * C[5]))) / 
                           (1.0 + T_ion * (C[2] + T_ion * (C[4] + T_ion * C[6]))))
    # Gamow parameter
    xi = (BG**2 / (4.0 * T_ion))**(1.0/3.0)
    
    # Reactivity in m^3 / s
    sigma_v = C[0] * theta * np.sqrt(xi / (m_rc2 * T_ion**3)) * np.exp(-3.0 * xi)
    
    # 6. Thermonuclear Power and Energy Yield
    # D-T Energy per event: Ef = 17.59 MeV = 2.818e-12 Joules
    Ef = 2.818e-12 
    power_density_W_m3 = 0.25 * (density_compressed**2) * sigma_v * Ef
    yield_energy_J = power_density_W_m3 * volume_compressed * dwell_time_s
    
    # 7. Energy Inputs (Thermal + Magnetic Energy)
    # 1 keV = 1.6022e-16 Joules
    E_thermal_input = 3.0 * density_compressed * (T_ion * 1.6022e-16) * volume_compressed
    mu_0 = 4.0 * np.pi * 1e-7
    E_magnetic_input = (B_max**2 / (2.0 * mu_0)) * volume_compressed
    total_input_J = E_thermal_input + E_magnetic_input
    
    # Q-factor
    Q_factor = yield_energy_J / total_input_J if total_input_J > 0 else 0.0
    
    # 8. Power Balance & Cryo Metabolism
    # Electrical power output generated
    P_elec_generated = yield_energy_J * efficiency_thermal * pulse_freq_Hz
    
    # Cryocooler power consumption using Carnot COP coefficient
    # Coefficient of Performance (COP) = cryocooler_efficiency * [T_cryo / (300 - T_cryo)]
    cryo_temp_K = max(1.0, cryo_temp_K)
    cop = cryocooler_efficiency * (cryo_temp_K / (300.0 - cryo_temp_K))
    P_cryo = heat_leak_W / cop if cop > 0 else 0.0
    
    # Coil-switching power losses
    P_switching = switching_loss_J * pulse_freq_Hz
    
    # Net metabolic power margin
    P_net = P_elec_generated - P_cryo - P_switching
    is_self_sustaining = P_net >= 0.0
    
    return {
        'magnetic_compression_ratio': kappa,
        'volumetric_compression_ratio': kappa_V,
        'ion_temperature_keV': T_ion,
        'reactivity_m3_s': sigma_v,
        'fusion_yield_energy_J': yield_energy_J,
        'q_factor': Q_factor,
        'power_electrical_gen_W': P_elec_generated,
        'power_cryocooler_W': P_cryo,
        'power_switching_loss_W': P_switching,
        'power_net_margin_W': P_net,
        'is_self_sustaining': is_self_sustaining,
        'compression_geometry': compression_geometry,
        'cryocooler_efficiency': cryocooler_efficiency
    }


def simulate_gravity_assist_replenishment(payload_mass_kg: float, v_relative_m_s: float, 
                                         n_slingshots: int, v_inf_m_s: float, 
                                         periapsis_alt_m: float) -> dict:
    """
    Calculates the net momentum and metabolic energy balance of the anchor node
    by pairing payload catch/launch events with lunar gravity assist replenishment.
    """
    # Momentum lost in catch/launch operation: delta_p = m * dv
    momentum_drain_Ns = payload_mass_kg * v_relative_m_s
    
    # Patched-conic gravity assist hyperbolic flyby mechanics
    mu_moon = 4.9048695e12 # Moon gravitational parameter
    R_moon = 1737e3 # Moon radius in meters
    v_moon = 1022.0 # Moon orbital speed around Earth in m/s
    r_p = R_moon + periapsis_alt_m
    
    # Hyperbolic eccentricity
    e = 1.0 + r_p * v_inf_m_s**2 / mu_moon
    
    # Turn angle
    turn_angle = 2.0 * np.arcsin(1.0 / e) if e > 1.0 else np.pi
    
    # Deflection velocity magnitude
    delta_v_mag = 2.0 * v_inf_m_s * np.sin(turn_angle / 2.0)
    
    # Specific geocentric energy gain (optimal retrograde approach harvesting orbital energy)
    energy_gain_specific_J_kg = v_moon * delta_v_mag
    
    # Replenished momentum: delta_p_gained = N * m_packet * delta_v_mag
    packet_mass_kg = 0.05 # 50-gram standard circulating packet
    momentum_gained_Ns = n_slingshots * packet_mass_kg * delta_v_mag
    
    # Net momentum balance
    net_momentum_Ns = momentum_gained_Ns - momentum_drain_Ns
    is_self_sustaining = net_momentum_Ns >= 0.0
    
    return {
        'eccentricity': e,
        'turn_angle_deg': np.degrees(turn_angle),
        'delta_v_slingshot_m_s': delta_v_mag,
        'energy_gain_specific_J_kg': energy_gain_specific_J_kg,
        'momentum_drain_Ns': momentum_drain_Ns,
        'momentum_gained_Ns': momentum_gained_Ns,
        'net_momentum_balance_Ns': net_momentum_Ns,
        'is_self_sustaining': is_self_sustaining
    }


if __name__ == "__main__":
    # Test execution
    from sgms_anchor_v1 import DEFAULT_PARAMS
    p = DEFAULT_PARAMS.copy()
    p["ms"] = 1000.0
    p["k_fp"] = 4500.0 # Pinning on

    t, x, f_brake_hist, switching_loss_dict = simulate_metabolic_event(p, payload_mass=100.0, v_relative=2.0)
    print(f"Peak Displacement during 100kg catch: {np.max(np.abs(x))*1000:.4f} mm")

    # Run quick fusion & slingshot metabolic tests
    fusion_res = simulate_fusion_metabolism(B_max=20.0, B_in=2.0, T0_keV=2.0, density_in_m3=2e20)
    print(f"Fusion Q-Factor: {fusion_res['q_factor']:.4f}, Net Power: {fusion_res['power_net_margin_W']/1e3:.4f} kW")

    slingshot_res = simulate_gravity_assist_replenishment(payload_mass_kg=1000.0, v_relative_m_s=50.0,
                                                          n_slingshots=2000, v_inf_m_s=1500.0, periapsis_alt_m=100e3)
    print(f"Slingshot Turn Angle: {slingshot_res['turn_angle_deg']:.2f} deg, Net Momentum: {slingshot_res['net_momentum_balance_Ns']:.2f} Ns")


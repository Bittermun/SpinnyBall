# dynamics/stream_energy_model.py

"""
Stream Energy Budget Model for SGMS Packet Stream.

This module computes the complete energy budget for the packet stream,
including:
- Total kinetic energy of all packets
- Power extracted by station deflections
- Power lost to eddy heating
- Power gained from lunar slingshots
- Power from new packet injection
- Service lifetime before replenishment needed
- Replacement rate to maintain stream velocity

The key question: does the stream gain more energy from slingshots
than it loses to station deflections and eddy heating?
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class StreamEnergyBudget:
    """Complete energy budget for the packet stream."""
    total_stream_KE_J: float          # Total kinetic energy of all packets
    power_drain_station_W: float      # Power extracted by station deflection
    power_drain_eddy_W: float         # Power lost to eddy heating (per packet × N)
    power_replenishment_slingshot_W: float  # Power gained from lunar slingshots
    power_replenishment_injection_W: float  # Power from new packet injection
    net_power_W: float                # Net power balance
    service_lifetime_hours: float     # Time until stream KE drops below threshold
    packets_replaced_per_year: float  # Replacement rate to maintain stream


def compute_stream_energy_budget(
    N_packets: int,
    mp: float,           # kg
    u: float,            # m/s
    theta_bias: float,   # rad - bias deflection angle (NOT used for power calc)
    F_station: float,    # N - force extracted by station
    n_stations: int = 1, # number of stations on the loop
    eddy_power_per_packet_W: float = 0.0,  # from thermal_model.eddy_heating_power
    slingshot_dv_per_cycle: float = 0.0,    # m/s gained per slingshot cycle
    slingshot_cycle_time_s: float = 30*86400, # time per slingshot cycle (~30 days)
    n_slingshot_packets: int = 0,   # packets dedicated to slingshot replenishment
    spacing: float = None,  # m - packet spacing (needed to compute lambda)
) -> StreamEnergyBudget:
    """
    Compute the complete energy budget for the stream.
    
    The key question: does the stream gain more energy from slingshots
    than it loses to station deflections and eddy heating?
    
    PHYSICS NOTE: The power drain from station-keeping is determined by the
    ACTUAL deflection angle needed to produce F_station, NOT the bias angle.
    
    For station-keeping: F = lambda * u^2 * sin(theta_eff)
    Therefore: theta_eff = F / (lambda * u^2)
    And power drain: P = F * u * sin(theta_eff) ≈ F * u * theta_eff (small angle)
    
    This means at high velocities with dense streams, the power drain is
    surprisingly small (~0.016 W for 4.2 N at 15 km/s with lambda=72.92 kg/m).
    
    Args:
        N_packets: Number of packets in the active stream
        mp: Packet mass (kg)
        u: Stream velocity (m/s)
        theta_bias: Bias deflection angle (rad) - geometric setup, NOT used for power
        F_station: Force extracted by station for station-keeping (N)
        n_stations: Number of stations on the loop
        eddy_power_per_packet_W: Eddy heating power loss per packet (W)
        slingshot_dv_per_cycle: Delta-v gained per slingshot cycle (m/s)
        slingshot_cycle_time_s: Duration of one slingshot cycle (s)
        n_slingshot_packets: Number of packets dedicated to slingshot replenishment
        spacing: Packet spacing (m). If None, estimated from typical stream length
    
    Returns:
        StreamEnergyBudget with complete energy breakdown
    """
    # Total stream KE
    total_KE = 0.5 * N_packets * mp * u**2 if N_packets > 0 and u > 0 else 0.0
    
    # Station deflection power drain
    # CRITICAL: Use the EFFECTIVE deflection angle, not the bias angle!
    # The bias angle (0.087 rad) is the geometric setup, but station-keeping
    # only requires a tiny additional deflection to extract 4.2 N.
    # 
    # From F = lambda * u^2 * sin(theta_eff):
    #   theta_eff = F / (lambda * u^2)
    # 
    # Power drain: P = F * u * sin(theta_eff) ≈ F * u * theta_eff (small angle)
    # 
    # If spacing not provided, estimate from typical stream (43,500 km / N_packets)
    if spacing is None:
        stream_length = 43.5e6  # m - typical Earth circumference for LEO
        spacing = stream_length / N_packets if N_packets > 0 else 1.0
    
    lambda_val = mp / spacing if spacing > 0 else mp  # kg/m
    
    if lambda_val > 0 and u > 0:
        # Small angle approximation: theta_eff = F / (lambda * u^2)
        theta_effective = F_station / (lambda_val * u**2)
        # Clamp to reasonable range (should be tiny for station-keeping)
        theta_effective = min(theta_effective, 0.1)  # Sanity clamp
        P_station = F_station * u * np.sin(theta_effective) * n_stations
    else:
        P_station = 0.0
    
    # Eddy heating drain (total across all packets)
    P_eddy_total = eddy_power_per_packet_W * N_packets if N_packets > 0 else 0.0
    
    # Slingshot replenishment
    # Each slingshot packet gains dv, adding KE = 0.5 * mp * ((u+dv)^2 - u^2) ≈ mp * u * dv
    if n_slingshot_packets > 0 and slingshot_dv_per_cycle > 0 and u > 0:
        KE_gain_per_cycle = n_slingshot_packets * mp * u * slingshot_dv_per_cycle
        P_slingshot = KE_gain_per_cycle / slingshot_cycle_time_s
    else:
        P_slingshot = 0.0
    
    # Injection replenishment (new packets at full velocity)
    # This is handled by energy_injection.py, just track the power
    P_injection = 0.0  # Computed externally
    
    # Net power balance
    P_net = P_slingshot + P_injection - P_station - P_eddy_total
    
    # Service lifetime (time until KE drops to 90% of initial)
    if P_net < 0 and total_KE > 0:
        # Stream is losing energy
        KE_threshold = 0.9 * total_KE
        energy_to_lose = total_KE - KE_threshold
        lifetime_s = energy_to_lose / abs(P_net)
        lifetime_hr = lifetime_s / 3600
    elif P_net >= 0:
        lifetime_hr = float('inf')  # Self-sustaining
    else:
        lifetime_hr = 0.0  # No energy to lose
    
    # Replacement rate to maintain stream velocity
    # If P_net < 0, need to inject new packets to compensate
    if P_net < 0 and u > 0:
        # Energy per replacement packet
        KE_per_packet = 0.5 * mp * u**2
        packets_per_second = abs(P_net) / KE_per_packet if KE_per_packet > 0 else 0.0
        packets_per_year = packets_per_second * 3600 * 24 * 365
    else:
        packets_per_year = 0.0
    
    return StreamEnergyBudget(
        total_stream_KE_J=total_KE,
        power_drain_station_W=P_station,
        power_drain_eddy_W=P_eddy_total,
        power_replenishment_slingshot_W=P_slingshot,
        power_replenishment_injection_W=P_injection,
        net_power_W=P_net,
        service_lifetime_hours=lifetime_hr,
        packets_replaced_per_year=packets_per_year,
    )


def analytical_lunar_slingshot_dv(v_inf: float = 1000.0, periapsis_alt: float = 100e3) -> float:
    """
    Quick analytical delta-v from a single lunar gravity assist.
    
    Uses the hyperbolic orbit formula from patched-conic approximation:
    - e = 1 + r_p * v_inf^2 / mu_moon
    - turn_angle = 2 * arcsin(1/e)
    - delta_v = 2 * v_inf * sin(turn_angle/2)
    
    Args:
        v_inf: Approach velocity relative to Moon (m/s)
        periapsis_alt: Closest approach altitude above Moon surface (m)
    
    Returns:
        Delta-v magnitude (m/s)
    """
    mu_moon = 4.904e12  # m³/s²
    R_moon = 1.737e6    # m
    r_p = R_moon + periapsis_alt
    
    e = 1.0 + r_p * v_inf**2 / mu_moon
    if e <= 1.0:
        return 0.0
    turn_angle = 2.0 * np.arcsin(1.0 / e)
    return 2.0 * v_inf * np.sin(turn_angle / 2.0)


def compute_multi_cycle_slingshot_dv(
    v_initial: float = 10900.0,
    n_cycles: int = 10,
    v_inf_base: float = 1000.0,
    periapsis_alt: float = 100e3,
) -> Dict[str, float]:
    """
    Compute cumulative delta-v from multiple slingshot cycles.
    
    Each cycle increases velocity, which changes the approach velocity
    for the next cycle. This provides a quick estimate without running
    the full earth_moon_pumping simulation.
    
    Args:
        v_initial: Initial stream velocity (m/s)
        n_cycles: Number of slingshot cycles
        v_inf_base: Base approach velocity relative to Moon (m/s)
        periapsis_alt: Periapsis altitude (m)
    
    Returns:
        Dict with:
            - v_final: Final velocity after all cycles (m/s)
            - total_dv: Total delta-v accumulated (m/s)
            - dv_per_cycle: Average delta-v per cycle (m/s)
    """
    v_current = v_initial
    total_dv = 0.0
    
    for i in range(n_cycles):
        # Approach velocity scales with stream velocity
        v_inf = v_inf_base * (v_current / v_initial)
        dv = analytical_lunar_slingshot_dv(v_inf, periapsis_alt)
        v_current += dv
        total_dv += dv
    
    return {
        'v_final': v_current,
        'total_dv': total_dv,
        'dv_per_cycle': total_dv / n_cycles if n_cycles > 0 else 0.0,
        'n_cycles': n_cycles,
    }

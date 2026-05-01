"""
Energy injection model for packet stream replenishment.

Computes the energy required to inject packets into the stream,
which is a dominant cost driver for mission-level analysis.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class InjectionEnergyResult:
    """Results from energy injection calculation."""
    # Translational kinetic energy (J)
    KE_translational_J: float
    
    # Rotational kinetic energy (J)
    KE_rotational_J: float
    
    # Total energy required including efficiency losses (J)
    total_energy_J: float
    
    # Total energy in kWh
    total_energy_kWh: float
    
    # Launch method used
    method: str
    
    # Wall-plug efficiency
    efficiency: float


def compute_injection_energy(
    mp: float,
    u: float,
    omega: float,
    r: float,
    method: str = 'electromagnetic'
) -> InjectionEnergyResult:
    """Total energy to inject one packet into the stream.
    
    Components:
    1. Translational KE: 0.5 * mp * u^2
    2. Rotational KE: 0.5 * I * omega^2 (where I = 2/5 * mp * r^2 for sphere)
    3. Launch overhead: efficiency factor (0.3 for EM launcher, 0.01 for chemical)
    
    Args:
        mp: Packet mass (kg)
        u: Stream velocity (m/s)
        omega: Angular velocity (rad/s)
        r: Packet radius (m)
        method: Launch method. One of:
            - 'electromagnetic': EM railgun/coilgun (30% efficiency)
            - 'chemical': Chemical rocket (~1% efficiency)
            - 'lunar_slingshot': Lunar gravity assist (only need transfer orbit)
    
    Returns:
        InjectionEnergyResult with energy breakdown
    """
    # Translational kinetic energy
    if method == 'lunar_slingshot':
        # Only need to reach transfer orbit (~10.9 km/s), rest from gravity assists
        u_effective = 10900.0  # Transfer orbit velocity (m/s)
    else:
        u_effective = u
    
    KE_trans = 0.5 * mp * u_effective ** 2
    
    # Rotational kinetic energy
    # For solid sphere: I = 2/5 * m * r^2
    I = 2.0 / 5.0 * mp * r ** 2
    KE_rot = 0.5 * I * omega ** 2
    
    # Efficiency based on launch method
    if method == 'electromagnetic':
        efficiency = 0.30  # 30% wall-plug efficiency for EM launcher
    elif method == 'chemical':
        efficiency = 0.01  # ~1% for chemical rocket
    elif method == 'lunar_slingshot':
        # Only need to reach transfer orbit (~10.9 km/s), rest from gravity assists
        efficiency = 0.30  # Same EM launcher efficiency
    else:
        raise ValueError(f"Unknown launch method: {method}. "
                        f"Available: 'electromagnetic', 'chemical', 'lunar_slingshot'")
    
    # Total energy required (including efficiency losses)
    total_energy = (KE_trans + KE_rot) / efficiency
    
    return InjectionEnergyResult(
        KE_translational_J=KE_trans,
        KE_rotational_J=KE_rot,
        total_energy_J=total_energy,
        total_energy_kWh=total_energy / 3.6e6,
        method=method,
        efficiency=efficiency,
    )


def compute_replacement_rate(
    fault_rate: float,
    n_packets: int,
    mission_duration_hr: float = 8760.0  # 1 year default
) -> float:
    """Packets lost per hour requiring replacement.
    
    Args:
        fault_rate: Fault rate (failures per packet per hour)
        n_packets: Total number of packets in stream
        mission_duration_hr: Mission duration (hours)
    
    Returns:
        Replacement rate (packets per hour)
    """
    return fault_rate * n_packets


def compute_steady_state_power(
    replacement_rate: float,
    energy_per_packet: float
) -> float:
    """Continuous power needed to replace lost packets.
    
    Args:
        replacement_rate: Packets lost per hour
        energy_per_packet: Energy per injection (J)
    
    Returns:
        Continuous power requirement (Watts)
    """
    # Convert packets/hour to packets/second and multiply by energy
    return replacement_rate * energy_per_packet / 3600.0  # Watts


def compute_injection_power_budget(
    mp: float,
    u: float,
    omega: float,
    r: float,
    fault_rate: float,
    n_packets: int,
    method: str = 'electromagnetic'
) -> Dict[str, Any]:
    """Complete power budget for packet injection and replacement.
    
    Args:
        mp: Packet mass (kg)
        u: Stream velocity (m/s)
        omega: Angular velocity (rad/s)
        r: Packet radius (m)
        fault_rate: Fault rate (failures per packet per hour)
        n_packets: Total number of packets
        method: Launch method
    
    Returns:
        Dictionary with complete power budget metrics
    """
    # Energy per injection
    energy_result = compute_injection_energy(mp, u, omega, r, method)
    
    # Replacement rate
    replacement_rate = compute_replacement_rate(fault_rate, n_packets)
    
    # Steady-state power for replacement
    steady_state_power = compute_steady_state_power(
        replacement_rate, 
        energy_result.total_energy_J
    )
    
    # Annual energy consumption
    annual_energy_J = steady_state_power * 3600 * 24 * 365  # J/year
    annual_energy_kWh = annual_energy_J / 3.6e6  # kWh/year
    
    return {
        'energy_per_packet_J': energy_result.total_energy_J,
        'energy_per_packet_kWh': energy_result.total_energy_kWh,
        'KE_translational_J': energy_result.KE_translational_J,
        'KE_rotational_J': energy_result.KE_rotational_J,
        'launch_efficiency': energy_result.efficiency,
        'launch_method': method,
        'replacement_rate_per_hour': replacement_rate,
        'replacement_rate_per_day': replacement_rate * 24,
        'replacement_rate_per_year': replacement_rate * 24 * 365,
        'steady_state_power_W': steady_state_power,
        'steady_state_power_kW': steady_state_power / 1000,
        'annual_energy_J': annual_energy_J,
        'annual_energy_kWh': annual_energy_kWh,
        'n_packets': n_packets,
        'fault_rate': fault_rate,
    }


def compare_launch_methods(
    mp: float,
    u: float,
    omega: float,
    r: float,
    fault_rate: float,
    n_packets: int
) -> Dict[str, Dict[str, Any]]:
    """Compare different launch methods for the same mission parameters.
    
    Args:
        mp: Packet mass (kg)
        u: Stream velocity (m/s)
        omega: Angular velocity (rad/s)
        r: Packet radius (m)
        fault_rate: Fault rate (failures per packet per hour)
        n_packets: Total number of packets
    
    Returns:
        Dictionary comparing all launch methods
    """
    methods = ['electromagnetic', 'chemical', 'lunar_slingshot']
    results = {}
    
    for method in methods:
        results[method] = compute_injection_power_budget(
            mp, u, omega, r, fault_rate, n_packets, method
        )
    
    # Add comparison metrics
    em_power = results['electromagnetic']['steady_state_power_W']
    for method in methods:
        power = results[method]['steady_state_power_W']
        results[method]['power_ratio_vs_EM'] = power / em_power if em_power > 0 else float('inf')
    
    return results

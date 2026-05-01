# dynamics/packet_budget.py

"""
Packet Budget Model for SGMS Stream Sustainability.

This module computes the total packet inventory required for a mission,
including:
- Active stream packets (N_stream)
- Slingshot pipeline packets (on lunar transfer orbits)
- Spare replacement inventory
- Injection queue (packets being prepared for launch)

The total inventory can be 1.1-2x the active stream count, which is
critical for accurate mass budgeting in mission-level analysis.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass 
class PacketBudget:
    """Complete packet inventory breakdown."""
    N_stream: int           # Packets in the active stream (current N_packets)
    N_slingshot_pipeline: int  # Packets on lunar transfer orbits
    N_spares: int           # Replacement inventory
    N_injection_queue: int  # Being prepared for launch
    N_total: int            # Grand total
    M_total_kg: float       # Total mass (all packets)
    mass_multiplier: float  # N_total / N_stream (the "overhead factor")


def compute_packet_budget(
    N_stream: int,
    mp: float,
    u: float,
    fault_rate_per_hr: float = 1e-6,
    mission_duration_years: float = 15.0,
    slingshot_enabled: bool = True,
    slingshot_cycle_days: float = 30.0,
    slingshot_fraction: float = 0.05,  # fraction of stream dedicated to slingshot
    spare_margin: float = 0.10,        # 10% spare inventory
    injection_lead_time_days: float = 7.0,  # time to prepare a replacement
) -> PacketBudget:
    """
    Compute total packet inventory including pipeline, spares, and slingshot.
    
    Args:
        N_stream: Packets needed in the active stream
        mp: Packet mass (kg)
        u: Stream velocity (m/s)
        fault_rate_per_hr: Failure rate per packet per hour
        mission_duration_years: Mission lifetime
        slingshot_enabled: Whether lunar slingshot replenishment is used
        slingshot_cycle_days: Duration of one slingshot cycle
        slingshot_fraction: Fraction of stream packets in slingshot pipeline at any time
        spare_margin: Fractional spare inventory
        injection_lead_time_days: Time to prepare a replacement packet
    
    Returns:
        PacketBudget with complete inventory breakdown
    """
    # Slingshot pipeline: packets currently on lunar transfer orbits
    if slingshot_enabled:
        N_slingshot = int(np.ceil(N_stream * slingshot_fraction))
    else:
        N_slingshot = 0
    
    # Replacement rate from faults
    replacements_per_year = fault_rate_per_hr * N_stream * 8760
    
    # Injection queue: packets being prepared (lead time × replacement rate)
    injection_rate_per_day = replacements_per_year / 365
    N_injection = int(np.ceil(injection_rate_per_day * injection_lead_time_days))
    N_injection = max(N_injection, 1)  # At least 1 in queue
    
    # Spares: buffer inventory
    N_spares = int(np.ceil(N_stream * spare_margin))
    
    # Total
    N_total = N_stream + N_slingshot + N_spares + N_injection
    M_total = N_total * mp
    multiplier = N_total / N_stream if N_stream > 0 else 1.0
    
    return PacketBudget(
        N_stream=N_stream,
        N_slingshot_pipeline=N_slingshot,
        N_spares=N_spares,
        N_injection_queue=N_injection,
        N_total=N_total,
        M_total_kg=M_total,
        mass_multiplier=multiplier,
    )


def compute_replacement_schedule(
    N_stream: int,
    fault_rate_per_hr: float,
    mission_duration_years: float,
    mp: float,
    u: float,
) -> Dict[str, float]:
    """
    Compute packet replacement schedule over mission lifetime.
    
    Args:
        N_stream: Number of packets in active stream
        fault_rate_per_hr: Failure rate per packet per hour
        mission_duration_years: Mission duration
        mp: Packet mass (kg)
        u: Stream velocity (m/s)
    
    Returns:
        Dict with:
            - total_replacements: Total packets replaced over mission
            - replacements_per_year: Average replacement rate
            - total_mass_kg: Total mass of replacement packets
            - energy_for_injection_J: Total KE needed for replacements
    """
    # Total replacements over mission
    total_replacements = fault_rate_per_hr * N_stream * 8760 * mission_duration_years
    
    # Energy needed to inject each replacement packet
    KE_per_packet = 0.5 * mp * u**2
    total_energy = total_replacements * KE_per_packet
    
    return {
        'total_replacements': total_replacements,
        'replacements_per_year': total_replacements / mission_duration_years,
        'total_mass_kg': total_replacements * mp,
        'energy_for_injection_J': total_energy,
    }


def estimate_slingshot_pipeline_capacity(
    N_stream: int,
    slingshot_cycle_days: float = 30.0,
    replenishment_fraction: float = 0.05,
) -> int:
    """
    Estimate number of packets in slingshot pipeline at any time.
    
    The pipeline holds packets that are currently on lunar transfer
    orbits being velocity-pumped. This is a fraction of the total
    stream, determined by the cycle time and replenishment strategy.
    
    Args:
        N_stream: Active stream packet count
        slingshot_cycle_days: Duration of one slingshot cycle
        replenishment_fraction: Fraction of stream in pipeline
    
    Returns:
        Number of packets in slingshot pipeline
    """
    return int(np.ceil(N_stream * replenishment_fraction))

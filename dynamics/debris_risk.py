"""
Debris risk quantification module for orbital packet stream systems.

Provides order-of-magnitude estimates for:
1. Collision probability with existing orbital debris
2. Risk from escaped packets becoming debris
3. Kessler syndrome threshold assessment

Based on NASA ORDEM (Orbital Debris Engineering Model) guidelines.
"""

import numpy as np
from typing import Dict, Any


# Constants
R_EARTH = 6371e3  # Earth radius (m)
MU_EARTH = 3.986004418e14  # Gravitational parameter (m³/s²)

# NASA debris mitigation thresholds
LETHAL_KE_THRESHOLD_J = 40.0  # J - KE threshold for lethal damage (1cm object)
CRITICAL_SIZE_CM = 1.0  # cm - critical debris size


def get_orbital_debris_density(altitude_km: float) -> float:
    """
    Get orbital debris density from simplified NASA ORDEM model.
    
    This is a simplified approximation based on ORDEM 3.0 data.
    Real applications should use the full ORDEM API or ESA MASTER model.
    
    Args:
        altitude_km: Orbital altitude (km)
        
    Returns:
        Debris density (objects/km³) for objects > 1cm
        
    Notes:
        Peak debris density is around 800-1000 km altitude.
        Below 600 km, atmospheric drag provides natural cleanup.
        Above 1000 km, debris persists for centuries.
    """
    # Simplified ORDEM-based debris density model
    # Peak at ~850 km, exponential decay above and below
    
    if altitude_km < 200:
        return 1e-6  # Very low - rapid atmospheric decay
    elif altitude_km < 400:
        return 1e-4  # Low - ISS region, monitored closely
    elif altitude_km < 600:
        return 1e-3  # Moderate - Sun-synchronous orbit region
    elif altitude_km < 800:
        return 5e-3  # High - debris accumulation zone
    elif altitude_km < 1000:
        return 1e-2  # Peak debris density
    elif altitude_km < 1200:
        return 5e-3  # Declining
    else:
        # Exponential decay at very high altitudes
        return 1e-2 * np.exp(-(altitude_km - 1000) / 500)


def compute_orbital_velocity(altitude_km: float) -> float:
    """
    Compute circular orbital velocity at given altitude.
    
    Args:
        altitude_km: Orbital altitude (km)
        
    Returns:
        Orbital velocity (m/s)
    """
    r = R_EARTH + altitude_km * 1000
    return np.sqrt(MU_EARTH / r)


def compute_collision_probability(
    n_packets: int,
    cross_section: float,
    altitude_km: float,
    orbital_debris_density: float = None,
    mission_duration_years: float = 1.0
) -> Dict[str, Any]:
    """
    Annual probability of collision between a packet and existing debris.
    
    Uses kinetic theory approach: P = n * sigma * v_rel * dt
    
    Args:
        n_packets: Number of packets in the stream
        cross_section: Cross-sectional area per packet (m²)
        altitude_km: Orbital altitude (km)
        orbital_debris_density: Debris density (objects/km³), or None to auto-calculate
        mission_duration_years: Mission duration (years)
        
    Returns:
        Dictionary with collision probability metrics:
        - annual_collision_probability: Probability per year
        - mission_collision_probability: Probability over mission duration
        - expected_collisions_per_year: Expected number of collisions/year
        - mean_time_between_collisions_years: MTBF in years
        
    Notes:
        Assumes isotropic debris distribution (conservative).
        Relative velocity assumed ~10 km/s for crossing orbits.
    """
    # Auto-calculate debris density if not provided
    if orbital_debris_density is None:
        orbital_debris_density = get_orbital_debris_density(altitude_km)
    
    # Convert debris density to objects/m³
    debris_density_m3 = orbital_debris_density / 1e9  # km³ to m³
    
    # Typical relative velocity for crossing orbits
    # Conservative estimate: sum of orbital velocities for head-on collision
    v_orbital = compute_orbital_velocity(altitude_km)
    v_rel = 10000.0  # 10 km/s typical for random crossings
    
    # Total cross-section (all packets)
    total_cross_section = n_packets * cross_section  # m²
    
    # Collision rate (collisions per second)
    # Rate = n_debris * sigma_total * v_rel
    collision_rate_per_sec = debris_density_m3 * total_cross_section * v_rel
    
    # Annual collision probability (for small probabilities, P ≈ rate * time)
    seconds_per_year = 365.25 * 24 * 3600
    annual_collision_probability = collision_rate_per_sec * seconds_per_year
    
    # Mission collision probability
    mission_collision_probability = 1.0 - np.exp(-collision_rate_per_sec * seconds_per_year * mission_duration_years)
    
    # Expected collisions per year
    expected_collisions_per_year = collision_rate_per_sec * seconds_per_year
    
    # Mean time between collisions
    if collision_rate_per_sec > 0:
        mtbf_years = 1.0 / (collision_rate_per_sec * seconds_per_year)
    else:
        mtbf_years = float('inf')
    
    return {
        'annual_collision_probability': annual_collision_probability,
        'mission_collision_probability': mission_collision_probability,
        'expected_collisions_per_year': expected_collisions_per_year,
        'mean_time_between_collisions_years': mtbf_years,
        'debris_density_objects_per_km3': orbital_debris_density,
        'relative_velocity_m_s': v_rel,
        'total_cross_section_m2': total_cross_section,
    }


def compute_escaped_packet_risk(
    mp: float,
    u: float,
    altitude_km: float,
    escape_probability_per_packet_per_year: float,
    n_packets: int
) -> Dict[str, Any]:
    """
    Risk metric for escaped packets becoming debris.
    
    Compares kinetic energy to NASA debris mitigation guidelines.
    
    Args:
        mp: Packet mass (kg)
        u: Stream velocity (m/s) - also escape velocity
        altitude_km: Orbital altitude (km)
        escape_probability_per_packet_per_year: Probability of escape per packet per year
        n_packets: Total number of packets
        
    Returns:
        Dictionary with risk metrics:
        - KE_per_packet_J: Kinetic energy per escaped packet
        - exceeds_lethal_threshold: Boolean - does KE exceed 40J?
        - expected_escapes_per_year: Expected number of escapes/year
        - risk_score: Normalized risk metric (0-1 scale)
        - recommendation: Mitigation recommendation string
        
    Notes:
        NASA guideline: Objects with KE > 40J are considered lethal.
        Escaped packets at 15 km/s carry enormous energy (~10 MJ for 100kg).
    """
    # Kinetic energy per escaped packet
    KE_per_packet_J = 0.5 * mp * u**2
    
    # Check against lethal threshold
    exceeds_lethal_threshold = KE_per_packet_J > LETHAL_KE_THRESHOLD_J
    
    # Expected escapes per year
    expected_escapes_per_year = n_packets * escape_probability_per_packet_per_year
    
    # Risk score (normalized 0-1)
    # Based on: (energy ratio) × (escape rate)
    energy_ratio = min(KE_per_packet_J / LETHAL_KE_THRESHOLD_J, 1e6)  # Cap at 1M×
    escape_rate_factor = min(expected_escapes_per_year, 100)  # Cap at 100/year
    risk_score = min(1.0, (np.log10(energy_ratio + 1) / 6) * (escape_rate_factor / 100))
    
    # Recommendation
    if risk_score > 0.8:
        recommendation = "CRITICAL: Implement active deorbit mechanism for all packets"
    elif risk_score > 0.5:
        recommendation = "HIGH RISK: Add passive drag augmentation for natural decay"
    elif risk_score > 0.2:
        recommendation = "MODERATE: Monitor and plan end-of-life disposal"
    else:
        recommendation = "LOW RISK: Standard debris mitigation practices sufficient"
    
    return {
        'KE_per_packet_J': KE_per_packet_J,
        'exceeds_lethal_threshold': exceeds_lethal_threshold,
        'expected_escapes_per_year': expected_escapes_per_year,
        'risk_score': risk_score,
        'recommendation': recommendation,
        'lethal_threshold_J': LETHAL_KE_THRESHOLD_J,
        'energy_ratio': energy_ratio,
    }


def compute_kessler_threshold(
    n_packets: int,
    altitude_km: float,
    cross_section: float,
    mission_duration_years: float = 25.0
) -> Dict[str, Any]:
    """
    Estimate whether the packet population exceeds the Kessler syndrome threshold.
    
    Kessler syndrome occurs when collision rate > removal rate, leading to
    cascading collisions that populate the orbit with debris.
    
    Args:
        n_packets: Number of packets
        altitude_km: Orbital altitude (km)
        cross_section: Cross-sectional area per packet (m²)
        mission_duration_years: Expected mission lifetime (years)
        
    Returns:
        Dictionary with Kessler assessment:
        - collision_rate_per_year: Expected collisions per year
        - removal_rate_per_year: Natural removal rate (atmospheric drag)
        - kessler_ratio: collision_rate / removal_rate
        - exceeds_threshold: Boolean - does system exceed Kessler threshold?
        - characteristic_time_years: Time to cascade if threshold exceeded
        - assessment: Text assessment
        
    Notes:
        At 550 km, atmospheric drag removes objects in ~25 years.
        Below 600 km, natural cleanup is relatively fast.
        Above 800 km, debris persists for centuries.
        Kessler threshold: ratio > 1.0 indicates unstable population.
    """
    # Get debris density
    debris_density = get_orbital_debris_density(altitude_km)
    
    # Calculate collision rate (from previous function)
    collision_metrics = compute_collision_probability(
        n_packets=n_packets,
        cross_section=cross_section,
        altitude_km=altitude_km,
        orbital_debris_density=debris_density,
        mission_duration_years=1.0
    )
    collision_rate_per_year = collision_metrics['expected_collisions_per_year']
    
    # Estimate natural removal rate (atmospheric drag)
    # Decay time depends strongly on altitude
    if altitude_km < 400:
        decay_time_years = 1.0  # Very rapid decay
    elif altitude_km < 500:
        decay_time_years = 5.0
    elif altitude_km < 600:
        decay_time_years = 25.0  # Typical Sun-sync orbit
    elif altitude_km < 700:
        decay_time_years = 100.0
    elif altitude_km < 800:
        decay_time_years = 500.0
    else:
        decay_time_years = 1000.0  # Essentially permanent
    
    # Removal rate (fraction of population removed per year)
    removal_rate_per_year = n_packets / decay_time_years
    
    # Kessler ratio
    # If collision_rate > removal_rate, population grows unstably
    if removal_rate_per_year > 0:
        kessler_ratio = collision_rate_per_year / removal_rate_per_year
    else:
        kessler_ratio = float('inf') if collision_rate_per_year > 0 else 0.0
    
    # Threshold check
    exceeds_threshold = kessler_ratio > 1.0
    
    # Characteristic time to cascade (if threshold exceeded)
    if exceeds_threshold and collision_rate_per_year > 0:
        # Time for collision rate to double (rough estimate)
        characteristic_time_years = np.log(2) / (collision_rate_per_year / n_packets) if n_packets > 0 else float('inf')
    else:
        characteristic_time_years = float('inf')
    
    # Assessment
    if exceeds_threshold:
        assessment = f"WARNING: Kessler threshold exceeded (ratio={kessler_ratio:.2f}). " \
                    f"Cascading collisions possible within {characteristic_time_years:.1f} years."
    elif kessler_ratio > 0.5:
        assessment = f"CAUTION: Approaching Kessler threshold (ratio={kessler_ratio:.2f}). " \
                    f"Monitor collision rate closely."
    else:
        assessment = f"SAFE: Well below Kessler threshold (ratio={kessler_ratio:.4f}). " \
                    f"Natural removal dominates collision production."
    
    return {
        'collision_rate_per_year': collision_rate_per_year,
        'removal_rate_per_year': removal_rate_per_year,
        'kessler_ratio': kessler_ratio,
        'exceeds_threshold': exceeds_threshold,
        'characteristic_time_years': characteristic_time_years,
        'assessment': assessment,
        'decay_time_years': decay_time_years,
        'altitude_km': altitude_km,
    }


def comprehensive_debris_risk_assessment(
    n_packets: int,
    mp: float,
    u: float,
    r: float,
    altitude_km: float,
    escape_probability_per_packet_per_year: float = 1e-6,
    mission_duration_years: float = 15.0
) -> Dict[str, Any]:
    """
    Comprehensive debris risk assessment combining all metrics.
    
    Args:
        n_packets: Number of packets
        mp: Packet mass (kg)
        u: Stream velocity (m/s)
        r: Packet radius (m)
        altitude_km: Orbital altitude (km)
        escape_probability_per_packet_per_year: Escape probability per packet per year
        mission_duration_years: Mission duration (years)
        
    Returns:
        Comprehensive risk assessment dictionary
    """
    # Cross-sectional area (assuming spherical packets)
    cross_section = np.pi * r**2
    
    # Run all three assessments
    collision_risk = compute_collision_probability(
        n_packets=n_packets,
        cross_section=cross_section,
        altitude_km=altitude_km,
        mission_duration_years=mission_duration_years
    )
    
    escape_risk = compute_escaped_packet_risk(
        mp=mp,
        u=u,
        altitude_km=altitude_km,
        escape_probability_per_packet_per_year=escape_probability_per_packet_per_year,
        n_packets=n_packets
    )
    
    kessler_risk = compute_kessler_threshold(
        n_packets=n_packets,
        altitude_km=altitude_km,
        cross_section=cross_section,
        mission_duration_years=mission_duration_years
    )
    
    # Overall risk score (weighted average)
    overall_risk_score = (
        0.4 * min(1.0, collision_risk['mission_collision_probability']) +
        0.4 * escape_risk['risk_score'] +
        0.2 * min(1.0, kessler_risk['kessler_ratio'])
    )
    
    # Overall recommendation
    if overall_risk_score > 0.7:
        overall_recommendation = "HIGH RISK: Significant debris concerns. Redesign required."
    elif overall_risk_score > 0.4:
        overall_recommendation = "MODERATE RISK: Implement mitigation measures."
    else:
        overall_recommendation = "LOW RISK: Acceptable debris profile with standard mitigations."
    
    return {
        'collision_risk': collision_risk,
        'escape_risk': escape_risk,
        'kessler_risk': kessler_risk,
        'overall_risk_score': overall_risk_score,
        'overall_recommendation': overall_recommendation,
        'mission_duration_years': mission_duration_years,
    }

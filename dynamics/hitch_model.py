"""
Scaled proxy model for inelastic hitch (payload capture).

This module implements a scaled proxy for hypervelocity hitch physics
at 100-500 m/s with configurable energy-loss factor. Full 15 km/s
hypervelocity physics is deferred to TRL 5 gate (mandatory vacuum test).

Acceptable thresholds (from FMECA v1.2):
- Energy dissipation < 0.1% of packet KE (requires e > 0.999)
- Post-capture misalignment < 10 cm
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class HitchResult:
    """Result of inelastic hitch calculation."""
    final_velocity: np.ndarray  # m/s
    energy_dissipated: float  # J
    energy_dissipation_fraction: float  # dimensionless
    coefficient_of_restitution: float  # dimensionless
    misalignment: float  # m
    meets_energy_threshold: bool  # < 0.1% dissipation
    meets_misalignment_threshold: bool  # < 10 cm
    extrapolation_warning: bool  # True if results may not scale to 15 km/s


@dataclass
class HitchConfig:
    """Configuration for hitch proxy model."""
    # Velocity range for proxy model (m/s)
    min_proxy_velocity: float = 100.0
    max_proxy_velocity: float = 500.0
    
    # Full hypervelocity target (m/s) - for extrapolation warning
    full_hypervelocity: float = 15000.0
    
    # Energy dissipation coefficient (e = coefficient of restitution)
    # e > 0.999 required for < 0.1% energy dissipation
    coefficient_of_restitution: float = 0.9995
    
    # Misalignment threshold (m) - from FMECA v1.2 kill criteria
    misalignment_threshold: float = 0.1  # 10 cm
    
    # Energy dissipation threshold (fraction) - from FMECA v1.2
    energy_dissipation_threshold: float = 0.001  # 0.1%


def calculate_inelastic_hitch(
    packet_mass: float,
    packet_velocity: np.ndarray,
    payload_mass: float,
    payload_velocity: np.ndarray,
    config: HitchConfig = None,
) -> HitchResult:
    """
    Calculate inelastic hitch (momentum exchange with energy dissipation).
    
    Uses scaled proxy model for 100-500 m/s range. Flags extrapolation
    warnings for velocities that may not scale to 15 km/s.
    
    Args:
        packet_mass: Mass of packet (kg)
        packet_velocity: Velocity of packet [vx, vy, vz] (m/s)
        payload_mass: Mass of payload (kg)
        payload_velocity: Velocity of payload [vx, vy, vz] (m/s)
        config: Hitch configuration (uses defaults if None)
    
    Returns:
        HitchResult with final state and threshold checks
    """
    if config is None:
        config = HitchConfig()
    
    # Calculate relative velocity magnitude
    v_rel = packet_velocity - payload_velocity
    v_rel_mag = np.linalg.norm(v_rel)
    
    # Check if velocity is within proxy range
    extrapolation_warning = False
    if v_rel_mag > config.max_proxy_velocity:
        extrapolation_warning = True
        logger.warning(
            f"Velocity {v_rel_mag:.1f} m/s exceeds proxy range "
            f"({config.max_proxy_velocity} m/s). Results may not scale to "
            f"{config.full_hypervelocity} m/s hypervelocity regime."
        )
    
    # Calculate initial kinetic energy
    KE_packet = 0.5 * packet_mass * np.linalg.norm(packet_velocity)**2
    KE_payload = 0.5 * payload_mass * np.linalg.norm(payload_velocity)**2
    KE_total_initial = KE_packet + KE_payload
    
    # Inelastic momentum exchange (perfectly inelastic collision)
    # v_final = (m1*v1 + m2*v2) / (m1 + m2)
    total_mass = packet_mass + payload_mass
    momentum_packet = packet_mass * packet_velocity
    momentum_payload = payload_mass * payload_velocity
    final_velocity = (momentum_packet + momentum_payload) / total_mass
    
    # Calculate final kinetic energy
    KE_final = 0.5 * total_mass * np.linalg.norm(final_velocity)**2
    
    # Energy dissipated (lost to plastic deformation, thermal, etc.)
    energy_dissipated = KE_total_initial - KE_final
    energy_dissipation_fraction = energy_dissipated / KE_total_initial if KE_total_initial > 0 else 0.0
    
    # Apply coefficient of restitution (energy recovery factor)
    # Higher e means less energy dissipation
    # Physics: In COM frame, only relative velocity is affected by COR
    if config.coefficient_of_restitution < 1.0:
        # Center-of-mass velocity (invariant in collision)
        v_cm = (packet_mass * packet_velocity + payload_mass * payload_velocity) / total_mass
        
        # Relative velocity in CoM frame
        v_rel_cm = packet_velocity - payload_velocity
        
        # Apply COR to relative velocity only (correct physics)
        v_rel_cm_final = -config.coefficient_of_restitution * v_rel_cm
        
        # Reconstruct lab-frame velocities from CoM frame
        packet_velocity_final = v_cm + (payload_mass / total_mass) * v_rel_cm_final
        payload_velocity_final = v_cm - (packet_mass / total_mass) * v_rel_cm_final
        
        # For hitch, we assume payload is captured, so use combined mass moving at v_cm
        # plus the corrected relative motion contribution
        final_velocity = v_cm  # Captured system moves at CoM velocity
        
        # Recalculate energy with adjusted velocity
        KE_final = 0.5 * total_mass * np.linalg.norm(final_velocity)**2
        energy_dissipated = KE_total_initial - KE_final
        energy_dissipation_fraction = energy_dissipated / KE_total_initial if KE_total_initial > 0 else 0.0
    
    # Calculate misalignment (simplified: perpendicular component of relative velocity)
    # In full model, this would depend on capture geometry and flux-pinning compliance
    v_rel_parallel = np.dot(v_rel, final_velocity / np.linalg.norm(final_velocity)) if np.linalg.norm(final_velocity) > 0 else 0
    v_rel_perp = np.sqrt(v_rel_mag**2 - v_rel_parallel**2) if v_rel_mag > v_rel_parallel else 0
    misalignment = v_rel_perp * 0.01  # Simplified scaling factor
    
    # Check thresholds
    meets_energy_threshold = energy_dissipation_fraction < config.energy_dissipation_threshold
    meets_misalignment_threshold = misalignment < config.misalignment_threshold
    
    return HitchResult(
        final_velocity=final_velocity,
        energy_dissipated=energy_dissipated,
        energy_dissipation_fraction=energy_dissipation_fraction,
        coefficient_of_restitution=config.coefficient_of_restitution,
        misalignment=misalignment,
        meets_energy_threshold=meets_energy_threshold,
        meets_misalignment_threshold=meets_misalignment_threshold,
        extrapolation_warning=extrapolation_warning,
    )


def calculate_hitch_energy_budget(
    packet_mass: float,
    packet_velocity: float,
    payload_mass: float,
    config: HitchConfig = None,
) -> dict:
    """
    Calculate energy budget for hitch at different velocities.
    
    Provides energy dissipation estimates across velocity range to
    identify safe operating envelope for proxy model.
    
    Args:
        packet_mass: Mass of packet (kg)
        packet_velocity: Velocity magnitude (m/s)
        payload_mass: Mass of payload (kg)
        config: Hitch configuration
    
    Returns:
        Dictionary with energy budget metrics
    """
    if config is None:
        config = HitchConfig()
    
    # Test at proxy range velocities
    velocities = np.linspace(config.min_proxy_velocity, config.max_proxy_velocity, 10)
    results = []
    
    for v in velocities:
        v_packet = np.array([v, 0, 0])
        v_payload = np.array([0, 0, 0])
        
        result = calculate_inelastic_hitch(
            packet_mass, v_packet, payload_mass, v_payload, config
        )
        
        results.append({
            "velocity": v,
            "energy_dissipated_j": result.energy_dissipated,
            "energy_dissipation_fraction": result.energy_dissipation_fraction,
            "meets_threshold": result.meets_energy_threshold,
        })
    
    # Calculate full hypervelocity energy (for reference)
    v_full = config.full_hypervelocity
    v_packet_full = np.array([v_full, 0, 0])
    result_full = calculate_inelastic_hitch(
        packet_mass, v_packet_full, payload_mass, np.array([0, 0, 0]), config
    )
    
    return {
        "proxy_range_results": results,
        "full_hypervelocity_reference": {
            "velocity_m_s": v_full,
            "energy_dissipated_mj": result_full.energy_dissipated / 1e6,
            "energy_dissipation_fraction": result_full.energy_dissipation_fraction,
            "extrapolation_warning": True,
        },
        "packet_mass_kg": packet_mass,
        "payload_mass_kg": payload_mass,
    }

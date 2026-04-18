"""
Centrifugal stress monitoring for gyroscopic mass-stream system.

Implements stress calculation and verification against the constraint:
σ ≤ 1.2 GPa with safety factor SF=1.5.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class StressMetrics:
    """Stress metrics for a packet."""
    centrifugal_stress: float  # Pa
    max_stress_limit: float  # Pa
    safety_factor: float
    within_limit: bool
    utilization: float  # fraction of limit


def calculate_centrifugal_stress(
    mass: float,
    radius: float,
    angular_velocity: np.ndarray,
) -> float:
    """
    Calculate centrifugal stress for a spinning sphere.
    
    For a solid sphere of uniform density:
    σ = (1/2) * ρ * ω² * r² = (3m/4πr³) * ω² * r² / 2 = (3mω²)/(8πr)
    
    Alternatively, for a thin spherical shell:
    σ = m * ω² * r / (4πr²) = mω²/(4πr)
    
    Args:
        mass: Mass (kg)
        radius: Radius (m)
        angular_velocity: Angular velocity vector [ωx, ωy, ωz] (rad/s)
    
    Returns:
        Centrifugal stress (Pa)
    """
    omega_mag = np.linalg.norm(angular_velocity)
    # Using thin shell approximation (more conservative)
    # σ = m * ω² * r / (4 * π * r²) = m * ω² / (4 * π * r)
    stress = mass * omega_mag**2 / (4 * np.pi * radius)
    return stress


def verify_stress_constraint(
    stress: float,
    max_stress: float = 1.2e9,  # 1.2 GPa
    safety_factor: float = 1.5,
) -> StressMetrics:
    """
    Verify that stress is within allowable limit.
    
    Args:
        stress: Calculated stress (Pa)
        max_stress: Maximum allowable stress (Pa)
        safety_factor: Safety factor to apply
    
    Returns:
        StressMetrics object with verification results
    """
    allowable_stress = max_stress / safety_factor
    within_limit = stress <= allowable_stress
    utilization = stress / allowable_stress
    
    return StressMetrics(
        centrifugal_stress=stress,
        max_stress_limit=allowable_stress,
        safety_factor=safety_factor,
        within_limit=within_limit,
        utilization=utilization,
    )


def verify_packet_stress(
    mass: float,
    radius: float,
    angular_velocity: np.ndarray,
    max_stress: float = 1.2e9,
    safety_factor: float = 1.5,
) -> StressMetrics:
    """
    Calculate and verify stress for a packet.
    
    Args:
        mass: Packet mass (kg)
        radius: Packet radius (m)
        angular_velocity: Angular velocity vector (rad/s)
        max_stress: Maximum allowable stress (Pa)
        safety_factor: Safety factor
    
    Returns:
        StressMetrics object
    """
    stress = calculate_centrifugal_stress(mass, radius, angular_velocity)
    return verify_stress_constraint(stress, max_stress, safety_factor)


def get_stress_alert_level(metrics: StressMetrics) -> str:
    """
    Get alert level based on stress utilization.
    
    Args:
        metrics: StressMetrics object
    
    Returns:
        Alert level: "safe", "caution", "warning", "critical"
    """
    if not metrics.within_limit:
        return "critical"
    elif metrics.utilization > 0.9:
        return "warning"
    elif metrics.utilization > 0.7:
        return "caution"
    else:
        return "safe"

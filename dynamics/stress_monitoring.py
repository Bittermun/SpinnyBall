"""
Centrifugal stress monitoring for gyroscopic mass-stream system.

Implements stress calculation and verification against the constraint:
σ ≤ 1.2 GPa with safety factor SF=1.5.

Includes configurable heritage scaling multipliers (6-10×) from FMECA v1.2
for conservative bounding runs at high-RPM regimes beyond heritage data.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HeritageScalingConfig:
    """Configuration for heritage scaling multipliers from FMECA v1.2."""
    # Scaling multiplier for centrifugal stress/creep (6-10×)
    stress_multiplier: float = 1.0  # Default: nominal (no scaling)
    
    # Scaling multiplier for hysteresis/eddy losses (6-10×)
    hysteresis_multiplier: float = 1.0
    
    # Scaling multiplier for stiffness (4-7×)
    stiffness_multiplier: float = 1.0
    
    # Label for documentation
    mode: str = "nominal"  # "nominal" or "conservative (FMECA 6-10× heritage)"
    
    def __post_init__(self):
        """Validate and update mode label based on multipliers."""
        if self.stress_multiplier > 1.0 or self.hysteresis_multiplier > 1.0:
            self.mode = f"conservative (stress×{self.stress_multiplier}, hysteresis×{self.hysteresis_multiplier})"
        elif self.stiffness_multiplier > 1.0:
            self.mode = f"conservative (stiffness×{self.stiffness_multiplier})"


@dataclass
class StressMetrics:
    """Stress metrics for a packet."""
    centrifugal_stress: float  # Pa
    max_stress_limit: float  # Pa
    safety_factor: float
    within_limit: bool
    utilization: float  # fraction of limit
    scaling_multiplier: float = 1.0  # Heritage scaling applied
    scaling_mode: str = "nominal"  # Documentation of scaling mode


def calculate_centrifugal_stress(
    mass: float,
    radius: float,
    angular_velocity: np.ndarray,
    scaling_config: Optional[HeritageScalingConfig] = None,
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
        scaling_config: Heritage scaling configuration (optional)
    
    Returns:
        Centrifugal stress (Pa) with heritage scaling applied if configured
    """
    if scaling_config is None:
        scaling_config = HeritageScalingConfig()
    
    omega_mag = np.linalg.norm(angular_velocity)
    # Using thin shell approximation (more conservative)
    # σ = m * ω² * r / (4 * π * r²) = m * ω² / (4 * π * r)
    stress = mass * omega_mag**2 / (4 * np.pi * radius)
    
    # Apply heritage scaling multiplier for stress/creep
    stress *= scaling_config.stress_multiplier
    
    return stress


def verify_stress_constraint(
    stress: float,
    max_stress: float = 1.2e9,  # 1.2 GPa
    safety_factor: float = 1.5,
    scaling_config: Optional[HeritageScalingConfig] = None,
) -> StressMetrics:
    """
    Verify that stress is within allowable limit.
    
    Args:
        stress: Calculated stress (Pa)
        max_stress: Maximum allowable stress (Pa)
        safety_factor: Safety factor to apply
        scaling_config: Heritage scaling configuration (optional)
    
    Returns:
        StressMetrics object with verification results and scaling info
    """
    if scaling_config is None:
        scaling_config = HeritageScalingConfig()
    
    allowable_stress = max_stress / safety_factor
    within_limit = stress <= allowable_stress
    utilization = stress / allowable_stress
    
    return StressMetrics(
        centrifugal_stress=stress,
        max_stress_limit=allowable_stress,
        safety_factor=safety_factor,
        within_limit=within_limit,
        utilization=utilization,
        scaling_multiplier=scaling_config.stress_multiplier,
        scaling_mode=scaling_config.mode,
    )


def verify_packet_stress(
    mass: float,
    radius: float,
    angular_velocity: np.ndarray,
    max_stress: float = 1.2e9,
    safety_factor: float = 1.5,
    scaling_config: Optional[HeritageScalingConfig] = None,
) -> StressMetrics:
    """
    Calculate and verify stress for a packet.
    
    Args:
        mass: Packet mass (kg)
        radius: Packet radius (m)
        angular_velocity: Angular velocity vector (rad/s)
        max_stress: Maximum allowable stress (Pa)
        safety_factor: Safety factor
        scaling_config: Heritage scaling configuration (optional)
    
    Returns:
        StressMetrics object with scaling info
    """
    stress = calculate_centrifugal_stress(mass, radius, angular_velocity, scaling_config)
    return verify_stress_constraint(stress, max_stress, safety_factor, scaling_config)


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

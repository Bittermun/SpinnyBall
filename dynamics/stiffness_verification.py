"""
Effective stiffness (k_eff) verification for gyroscopic mass-stream system.

Implements verification against the constraint: k_eff ≥ 6,000 N/m.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class StiffnessMetrics:
    """Stiffness metrics for the stream."""
    k_eff: float  # Effective stiffness (N/m)
    min_k_eff: float  # Minimum required stiffness (N/m)
    within_limit: bool
    utilization: float  # fraction of minimum


def calculate_effective_stiffness(
    lambda_density: float,
    velocity: float,
    g_gain: float,
    k_fp: float = 0.0,
) -> float:
    """
    Calculate effective stiffness of the mass-stream anchor system.
    
    From the anchor model:
    k_eff = lambda * u^2 * g_gain + k_fp
    
    Args:
        lambda_density: Stream density (kg/m)
        velocity: Stream velocity (m/s)
        g_gain: Control gain (rad/m)
        k_fp: Flux-pinning stiffness (N/m)
    
    Returns:
        Effective stiffness k_eff (N/m)
    """
    k_control = lambda_density * velocity**2 * g_gain
    k_eff = k_control + k_fp
    return k_eff


def verify_stiffness_constraint(
    k_eff: float,
    min_k_eff: float = 6000.0,  # N/m
) -> StiffnessMetrics:
    """
    Verify that effective stiffness meets minimum requirement.
    
    Args:
        k_eff: Calculated effective stiffness (N/m)
        min_k_eff: Minimum required stiffness (N/m)
    
    Returns:
        StiffnessMetrics object with verification results
    """
    within_limit = k_eff >= min_k_eff
    utilization = k_eff / min_k_eff if min_k_eff > 0 else float('inf')
    
    return StiffnessMetrics(
        k_eff=k_eff,
        min_k_eff=min_k_eff,
        within_limit=within_limit,
        utilization=utilization,
    )


def verify_anchor_stiffness(
    lambda_density: float,
    velocity: float,
    g_gain: float,
    k_fp: float = 0.0,
    min_k_eff: float = 6000.0,
) -> StiffnessMetrics:
    """
    Calculate and verify stiffness for an anchor.
    
    Args:
        lambda_density: Stream density (kg/m)
        velocity: Stream velocity (m/s)
        g_gain: Control gain (rad/m)
        k_fp: Flux-pinning stiffness (N/m)
        min_k_eff: Minimum required stiffness (N/m)
    
    Returns:
        StiffnessMetrics object
    """
    k_eff = calculate_effective_stiffness(lambda_density, velocity, g_gain, k_fp)
    return verify_stiffness_constraint(k_eff, min_k_eff)


def get_stiffness_alert_level(metrics: StiffnessMetrics) -> str:
    """
    Get alert level based on stiffness utilization.
    
    Args:
        metrics: StiffnessMetrics object
    
    Returns:
        Alert level: "safe", "caution", "warning", "critical"
    """
    if not metrics.within_limit:
        return "critical"
    elif metrics.utilization < 1.2:
        return "safe"
    elif metrics.utilization < 2.0:
        return "caution"
    else:
        return "warning"


def sweep_stiffness_velocity(
    lambda_density: float,
    g_gain: float,
    k_fp: float = 0.0,
    velocities: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    Sweep velocity to analyze stiffness behavior.
    
    Args:
        lambda_density: Stream density (kg/m)
        g_gain: Control gain (rad/m)
        k_fp: Flux-pinning stiffness (N/m)
        velocities: Velocity array to sweep (m/s)
    
    Returns:
        Dictionary with sweep results
    """
    if velocities is None:
        velocities = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 520.0])
    
    k_eff_values = []
    for u in velocities:
        k_eff = calculate_effective_stiffness(lambda_density, u, g_gain, k_fp)
        k_eff_values.append(k_eff)
    
    return {
        'velocity': velocities,
        'k_eff': np.array(k_eff_values),
    }

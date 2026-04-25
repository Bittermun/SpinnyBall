"""
Effective stiffness (k_eff) verification for gyroscopic mass-stream system.

Implements verification against the constraint: k_eff ≥ 6,000 N/m.

Includes configurable heritage scaling multipliers (4-7×) from FMECA v1.2
for conservative bounding runs at high-speed regimes beyond heritage data.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from dynamics.gdBCO_material import GdBCOMaterial
from dynamics.bean_london_model import BeanLondonModel


@dataclass
class HeritageScalingConfig:
    """Configuration for heritage scaling multipliers from FMECA v1.2."""
    # Scaling multiplier for stiffness (4-7×)
    stiffness_multiplier: float = 1.0  # Default: nominal (no scaling)
    
    # Label for documentation
    mode: str = "nominal"  # "nominal" or "conservative (FMECA 4-7× heritage)"
    
    def __post_init__(self):
        """Update mode label based on multiplier."""
        if self.stiffness_multiplier > 1.0:
            self.mode = f"conservative (stiffness×{self.stiffness_multiplier})"


@dataclass
class StiffnessMetrics:
    """Stiffness metrics for the stream."""
    k_eff: float  # Effective stiffness (N/m)
    min_k_eff: float  # Minimum required stiffness (N/m)
    within_limit: bool
    utilization: float  # fraction of minimum
    scaling_multiplier: float = 1.0  # Heritage scaling applied
    scaling_mode: str = "nominal"  # Documentation of scaling mode


def calculate_effective_stiffness(
    lambda_density: float,
    velocity: float,
    g_gain: float,
    k_fp: float = 0.0,
    scaling_config: Optional[HeritageScalingConfig] = None,
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
        scaling_config: Heritage scaling configuration (optional)
    
    Returns:
        Effective stiffness k_eff (N/m) with heritage scaling applied if configured
    """
    if scaling_config is None:
        scaling_config = HeritageScalingConfig()
    
    k_control = lambda_density * velocity**2 * g_gain
    k_eff = k_control + k_fp
    
    # Apply heritage scaling multiplier for stiffness
    k_eff *= scaling_config.stiffness_multiplier
    
    return k_eff


def verify_stiffness_constraint(
    k_eff: float,
    min_k_eff: float = 6000.0,  # N/m
    scaling_config: Optional[HeritageScalingConfig] = None,
) -> StiffnessMetrics:
    """
    Verify that effective stiffness meets minimum requirement.
    
    Args:
        k_eff: Calculated effective stiffness (N/m)
        min_k_eff: Minimum required stiffness (N/m)
        scaling_config: Heritage scaling configuration (optional)
    
    Returns:
        StiffnessMetrics object with verification results and scaling info
    """
    if scaling_config is None:
        scaling_config = HeritageScalingConfig()
    
    within_limit = k_eff >= min_k_eff
    utilization = k_eff / min_k_eff if min_k_eff > 0 else float('inf')
    
    return StiffnessMetrics(
        k_eff=k_eff,
        min_k_eff=min_k_eff,
        within_limit=within_limit,
        utilization=utilization,
        scaling_multiplier=scaling_config.stiffness_multiplier,
        scaling_mode=scaling_config.mode,
    )


def verify_anchor_stiffness(
    lambda_density: float,
    velocity: float,
    g_gain: float,
    k_fp: float = 0.0,
    min_k_eff: float = 6000.0,
    scaling_config: Optional[HeritageScalingConfig] = None,
) -> StiffnessMetrics:
    """
    Calculate and verify stiffness for an anchor.
    
    Args:
        lambda_density: Stream density (kg/m)
        velocity: Stream velocity (m/s)
        g_gain: Control gain (rad/m)
        k_fp: Flux-pinning stiffness (N/m)
        min_k_eff: Minimum required stiffness (N/m)
        scaling_config: Heritage scaling configuration (optional)
    
    Returns:
        StiffnessMetrics object with scaling info
    """
    k_eff = calculate_effective_stiffness(lambda_density, velocity, g_gain, k_fp, scaling_config)
    return verify_stiffness_constraint(k_eff, min_k_eff, scaling_config)


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


def calculate_flux_pinning_stiffness(
    displacement: float,
    B_field: float,
    temperature: float,
    material: GdBCOMaterial,
    geometry: dict,
) -> float:
    """Calculate flux-pinning stiffness using Bean-London model.

    Args:
        displacement: Relative displacement (m)
        B_field: Magnetic flux density (T)
        temperature: Temperature (K)
        material: GdBCO material properties
        geometry: Geometry parameters

    Returns:
        Effective stiffness (N/m)
    """
    model = BeanLondonModel(material, geometry)
    return model.get_stiffness(displacement, B_field, temperature)

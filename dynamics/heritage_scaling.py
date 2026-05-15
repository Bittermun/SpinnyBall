"""Heritage scaling configuration shared across stress and stiffness modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HeritageScalingConfig:
    """Configuration for heritage scaling multipliers from FMECA v1.2.

    Used by both stress_monitoring and stiffness_verification modules.
    """
    stress_multiplier: float = 1.0
    hysteresis_multiplier: float = 1.0
    stiffness_multiplier: float = 1.0
    mode: str = "nominal"

    def __post_init__(self):
        if self.stress_multiplier > 1.0 or self.hysteresis_multiplier > 1.0:
            self.mode = f"conservative (stress×{self.stress_multiplier}, hysteresis×{self.hysteresis_multiplier})"
        elif self.stiffness_multiplier > 1.0:
            self.mode = f"conservative (stiffness×{self.stiffness_multiplier})"

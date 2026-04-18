"""
Monte-Carlo stability analysis framework.

This module implements Monte-Carlo risk assessment and pass/fail gates
for the closed-loop mass-stream system.
"""

from .cascade_runner import (
    CascadeRunner,
    MonteCarloConfig,
    Perturbation,
    PerturbationType,
    RealizationResult,
    create_default_perturbations,
)
from .pass_fail_gates import (
    PassFailGate,
    GateStatus,
    GateResult,
    GateSet,
    InductionEfficiencyGate,
    StressGate,
    StiffnessGate,
    CascadeProbabilityGate,
    create_default_gate_set,
    evaluate_monte_carlo_gates,
)

__all__ = [
    "CascadeRunner",
    "MonteCarloConfig",
    "Perturbation",
    "PerturbationType",
    "RealizationResult",
    "create_default_perturbations",
    "PassFailGate",
    "GateStatus",
    "GateResult",
    "GateSet",
    "InductionEfficiencyGate",
    "StressGate",
    "StiffnessGate",
    "CascadeProbabilityGate",
    "create_default_gate_set",
    "evaluate_monte_carlo_gates",
]

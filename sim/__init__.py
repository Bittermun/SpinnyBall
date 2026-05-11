"""
Simulation engine for SpinnyBall multi-physics system.

Provides:
- Uncertainty quantification (UncertainQuantity)
- Domain-specific physics adapters with regime validity
- Macro-step scheduler with operator splitting
- Structure-preserving integrators
"""

from sim.uncertainty import UncertainQuantity, UncertainArray
from sim.domain_base import DomainAdapter, DomainOutput, AdvanceResult
from sim.scheduler import MacroScheduler, TimeScale, CouplingStrength

__all__ = [
    'UncertainQuantity',
    'UncertainArray',
    'DomainAdapter',
    'DomainOutput',
    'AdvanceResult',
    'MacroScheduler',
    'TimeScale',
    'CouplingStrength',
]

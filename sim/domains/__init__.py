"""
Physics domain adapters for the simulation engine.

Each domain wraps existing physics code with:
- Consistent advance() interface
- Structure-preserving integrators
- Uncertainty quantification
- Regime validity checking
"""

from sim.domains.mechanics_stream import MechanicsStreamDomain
from sim.domains.attitude_fluxgyro import AttitudeFluxGyroDomain
from sim.domains.thermal_anchor import ThermalAnchorDomain
from sim.domains.orbit_env import OrbitalEnvironmentDomain

__all__ = [
    'MechanicsStreamDomain',
    'AttitudeFluxGyroDomain',
    'ThermalAnchorDomain',
    'OrbitalEnvironmentDomain',
]

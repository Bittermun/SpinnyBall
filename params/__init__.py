"""
Centralized Parameter Registry for SGMS Anchor Simulation

This module provides a single source of truth for all physical constants
and simulation parameters, eliminating discrepancies across modules.

Usage:
    from params import PHYSICAL_CONSTANTS, SIMULATION_PARAMS
    
    # Access canonical values
    Jc0 = PHYSICAL_CONSTANTS['GdBCO']['Jc0']
    thickness = SIMULATION_PARAMS['flux_pinning']['thickness']
    
Provenance:
    All values include source citations and uncertainty estimates where available.
"""

from .canonical_values import (
    PHYSICAL_CONSTANTS,
    SIMULATION_PARAMS,
    MATERIAL_PROPERTIES,
    GEOMETRY_PARAMS,
    get_parameter,
    validate_parameters,
)

__all__ = [
    'PHYSICAL_CONSTANTS',
    'SIMULATION_PARAMS', 
    'MATERIAL_PROPERTIES',
    'GEOMETRY_PARAMS',
    'get_parameter',
    'validate_parameters',
]

"""
Mission scenario scripts for orbital operations.

Provides predefined orbital scenarios for launch, transfer, station-keeping, and deorbit.
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass

from dynamics.orbital_coupling import (
    OrbitalState, OrbitalPropagator, OrbitalElements, 
    create_circular_orbit, eci_to_lvlh, lvlh_to_eci
)


@dataclass
class MissionScenario:
    """Mission scenario definition."""
    name: str
    description: str
    initial_state: OrbitalState
    duration: float  # seconds
    target_state: Optional[OrbitalState] = None


def create_leo_scenario(altitude: float = 400.0, inclination: float = 51.6) -> MissionScenario:
    """Create LEO station-keeping scenario.
    
    Args:
        altitude: Orbital altitude (km)
        inclination: Orbital inclination (deg)
        
    Returns:
        MissionScenario for LEO operations
    """
    initial_state = create_circular_orbit(altitude, inclination)
    
    return MissionScenario(
        name="LEO Station-Keeping",
        description=f"Low Earth Orbit at {altitude} km altitude, {inclination}° inclination",
        initial_state=initial_state,
        duration=90 * 60,  # 90 minutes (one orbit)
    )


def create_transfer_scenario(
    r1: float = 400.0,
    r2: float = 1000.0,
    inclination: float = 0.0
) -> MissionScenario:
    """Create Hohmann transfer scenario.
    
    Args:
        r1: Initial orbit altitude (km)
        r2: Target orbit altitude (km)
        inclination: Orbital inclination (deg)
        
    Returns:
        MissionScenario for Hohmann transfer
    """
    # Initial circular orbit
    initial_state = create_circular_orbit(r1, inclination)
    
    # Target circular orbit
    target_state = create_circular_orbit(r2, inclination)
    
    # Hohmann transfer time (half period of transfer ellipse)
    mu = 398600.4418  # Earth gravitational parameter (km^3/s^2)
    r1_km = r1 + 6371.0
    r2_km = r2 + 6371.0
    a_transfer = (r1_km + r2_km) / 2.0
    transfer_time = np.pi * np.sqrt(a_transfer**3 / mu)
    
    return MissionScenario(
        name="Hohmann Transfer",
        description=f"Transfer from {r1} km to {r2} km altitude",
        initial_state=initial_state,
        duration=transfer_time,
        target_state=target_state,
    )


def create_deorbit_scenario(
    initial_altitude: float = 400.0,
    inclination: float = 51.6
) -> MissionScenario:
    """Create deorbit scenario.
    
    Args:
        initial_altitude: Initial orbital altitude (km)
        inclination: Orbital inclination (deg)
        
    Returns:
        MissionScenario for deorbit operations
    """
    initial_state = create_circular_orbit(initial_altitude, inclination)
    
    # Target: suborbital trajectory
    # Simplified: reduce velocity to start decay
    target_state = OrbitalState(
        r=initial_state.r.copy(),
        v=initial_state.v * 0.8,  # 80% of orbital velocity
    )
    
    return MissionScenario(
        name="Deorbit",
        description=f"Deorbit from {initial_altitude} km altitude",
        initial_state=initial_state,
        duration=30 * 60,  # 30 minutes for re-entry
        target_state=target_state,
    )


def create_polar_scenario(altitude: float = 600.0) -> MissionScenario:
    """Create polar orbit scenario.
    
    Args:
        altitude: Orbital altitude (km)
        
    Returns:
        MissionScenario for polar orbit
    """
    initial_state = create_circular_orbit(altitude, inclination=98.0)
    
    return MissionScenario(
        name="Polar Orbit",
        description=f"Polar orbit at {altitude} km altitude for Earth observation",
        initial_state=initial_state,
        duration=90 * 60,
    )


def create_geo_scenario() -> MissionScenario:
    """Create GEO scenario.
    
    Returns:
        MissionScenario for geostationary orbit
    """
    # GEO altitude: ~35,786 km
    initial_state = create_circular_orbit(35786.0, inclination=0.0)
    
    return MissionScenario(
        name="Geostationary Orbit",
        description="Geostationary orbit at 35,786 km altitude",
        initial_state=initial_state,
        duration=24 * 3600,  # 24 hours
    )


def get_all_scenarios() -> List[MissionScenario]:
    """Get all predefined mission scenarios.
    
    Returns:
        List of MissionScenario objects
    """
    return [
        create_leo_scenario(),
        create_transfer_scenario(),
        create_deorbit_scenario(),
        create_polar_scenario(),
        create_geo_scenario(),
    ]


# Predefined scenario library
SCENARIO_LIBRARY = {
    "leo": create_leo_scenario,
    "transfer": create_transfer_scenario,
    "deorbit": create_deorbit_scenario,
    "polar": create_polar_scenario,
    "geo": create_geo_scenario,
}


def get_scenario(name: str, **kwargs) -> Optional[MissionScenario]:
    """Get scenario by name.
    
    Args:
        name: Scenario name (key from SCENARIO_LIBRARY)
        **kwargs: Additional parameters for scenario creation
        
    Returns:
        MissionScenario or None if not found
    """
    if name in SCENARIO_LIBRARY:
        return SCENARIO_LIBRARY[name](**kwargs)
    return None

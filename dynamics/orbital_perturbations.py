"""
Orbital perturbation utilities for SGMS anchor simulation.

This module provides helper functions to compute orbital perturbation forces
and create orbital states from simulation parameters, bridging the gap between
the orbital_coupling module and the sgms_anchor_v1 simulation.
"""

import numpy as np
from typing import Optional, Tuple

# Import from orbital_coupling
try:
    from dynamics.orbital_coupling import (
        OrbitalPropagator, 
        OrbitalState, 
        POLIASTRO_AVAILABLE,
        R_earth,
        M_earth,
    )
    ORBITAL_COUPLING_AVAILABLE = True
except ImportError:
    ORBITAL_COUPLING_AVAILABLE = False
    OrbitalPropagator = None
    OrbitalState = None
    POLIASTRO_AVAILABLE = False
    R_earth = 6371.0
    M_earth = 5.972e24


def create_orbital_state_from_params(params: dict) -> Optional['OrbitalState']:
    """Create an initial orbital state from simulation parameters.
    
    Args:
        params: Dictionary containing orbital parameters:
            - altitude_km: Orbital altitude in km (default: 400.0)
            - inclination_deg: Orbital inclination in degrees (default: 51.6)
            - mu: Gravitational parameter in km^3/s^2 (default: Earth's)
            
    Returns:
        OrbitalState instance for circular orbit at specified altitude,
        or None if orbital coupling is not available
    """
    if not ORBITAL_COUPLING_AVAILABLE:
        return None
    
    altitude_km = params.get("altitude_km", 400.0)
    inclination_deg = params.get("inclination_deg", 51.6)
    mu = params.get("mu", 398600.4418)  # Earth's gravitational parameter
    
    # Compute circular orbit velocity
    r_km = R_earth + altitude_km
    v_circular = np.sqrt(mu / r_km)  # km/s
    
    # Create state vector for circular orbit in ECI frame
    # Position: along x-axis
    # Velocity: in y-direction (perpendicular to position for circular orbit)
    position_eci = np.array([r_km, 0.0, 0.0])
    
    # Rotate velocity vector by inclination
    inc_rad = np.radians(inclination_deg)
    velocity_eci = np.array([0.0, v_circular * np.cos(inc_rad), v_circular * np.sin(inc_rad)])
    
    return OrbitalState(r=position_eci, v=velocity_eci, epoch=0.0)


def get_orbital_perturbation_force(
    params: dict,
    orbital_state: 'OrbitalState',
    t: float,
    packet_mass: float = 1000.0
) -> np.ndarray:
    """Compute orbital perturbation force on the packet.
    
    Args:
        params: Dictionary containing perturbation flags and parameters:
            - include_j2: Include J2 perturbation (default: False)
            - include_srp: Include solar radiation pressure (default: False)
            - include_drag: Include atmospheric drag (default: False)
            - sr_area: Solar radiation area in m^2 (default: 1.0)
            - cd: Drag coefficient (default: 2.2)
            - sr_pressure: Solar radiation pressure at 1 AU in Pa (default: 4.56e-6)
        orbital_state: Current orbital state
        t: Current time in seconds
        packet_mass: Mass of the packet in kg
        
    Returns:
        3-element numpy array with perturbation force components in N
        (in ECI frame)
    """
    if not ORBITAL_COUPLING_AVAILABLE:
        return np.zeros(3)
    
    force = np.zeros(3)
    
    # J2 perturbation (Earth oblateness)
    if params.get("include_j2", False):
        force += _j2_perturbation(orbital_state, packet_mass)
    
    # Solar Radiation Pressure
    if params.get("include_srp", False):
        force += _srp_perturbation(orbital_state, t, params, packet_mass)
    
    # Atmospheric Drag
    if params.get("include_drag", False):
        force += _drag_perturbation(orbital_state, params, packet_mass)
    
    return force


def _j2_perturbation(orbital_state: 'OrbitalState', mass: float) -> np.ndarray:
    """Compute J2 perturbation force due to Earth's oblateness.
    
    Args:
        orbital_state: Current orbital state
        mass: Spacecraft mass in kg
        
    Returns:
        J2 perturbation force vector in N (ECI frame)
    """
    mu = 398600.4418  # km^3/s^2
    J2 = 1.08263e-3
    R_e = R_earth  # km
    
    r = orbital_state.r
    r_mag = np.linalg.norm(r)
    
    if r_mag < R_e:
        return np.zeros(3)
    
    # J2 acceleration formula (in km/s^2)
    # a_J2 = -1.5 * J2 * mu * R_e^2 / r^5 * [
    #     x * (1 - 5*z^2/r^2),
    #     y * (1 - 5*z^2/r^2),
    #     z * (3 - 5*z^2/r^2)
    # ]
    factor = -1.5 * J2 * mu * R_e**2 / r_mag**5
    z_ratio = r[2]**2 / r_mag**2
    
    ax = factor * r[0] * (1 - 5 * z_ratio)
    ay = factor * r[1] * (1 - 5 * z_ratio)
    az = factor * r[2] * (3 - 5 * z_ratio)
    
    # Convert from km/s^2 to m/s^2 and multiply by mass to get force in N
    accel = np.array([ax, ay, az]) * 1000.0  # km/s^2 -> m/s^2
    return accel * mass


def _srp_perturbation(
    orbital_state: 'OrbitalState',
    t: float,
    params: dict,
    mass: float
) -> np.ndarray:
    """Compute Solar Radiation Pressure perturbation force.
    
    Args:
        orbital_state: Current orbital state
        t: Current time in seconds
        params: Parameters including sr_area and sr_pressure
        mass: Spacecraft mass in kg
        
    Returns:
        SRP force vector in N (ECI frame)
    """
    # Simplified SRP model: constant pressure from Sun direction
    # In reality, Sun position varies with time
    sr_area = params.get("sr_area", 1.0)  # m^2
    sr_pressure = params.get("sr_pressure", 4.56e-6)  # Pa at 1 AU
    C_r = params.get("C_r", 1.8)  # Reflectivity coefficient
    
    # Simplified: assume Sun is in +x direction (not time-varying)
    # A more accurate model would compute Sun position based on t
    sun_direction = np.array([1.0, 0.0, 0.0])
    
    # Check if spacecraft is in eclipse (simplified)
    # If position.x < 0 and |position.y|, |position.z| are small, it's in Earth's shadow
    r = orbital_state.r
    if r[0] < 0 and np.abs(r[1]) < R_earth and np.abs(r[2]) < R_earth:
        # In eclipse, no SRP
        return np.zeros(3)
    
    # Force magnitude: F = P * A * C_r
    force_mag = sr_pressure * sr_area * C_r
    
    return force_mag * sun_direction


def _drag_perturbation(
    orbital_state: 'OrbitalState',
    params: dict,
    mass: float
) -> np.ndarray:
    """Compute atmospheric drag perturbation force.
    
    Args:
        orbital_state: Current orbital state
        params: Parameters including cd and cross_sectional_area
        mass: Spacecraft mass in kg
        
    Returns:
        Drag force vector in N (ECI frame)
    """
    cd = params.get("cd", 2.2)  # Drag coefficient
    area = params.get("cross_sectional_area", 1.0)  # m^2
    
    # Exponential atmosphere model
    altitude_km = np.linalg.norm(orbital_state.r) - R_earth
    altitude_m = altitude_km * 1000.0
    
    # Scale height and sea-level density
    H = 8500.0  # scale height in m
    rho_0 = 1.225  # kg/m^3 at sea level
    
    # Density at altitude
    if altitude_m < 0:
        rho = rho_0
    else:
        rho = rho_0 * np.exp(-altitude_m / H)
    
    # Only significant below ~1000 km
    if altitude_km > 1000:
        return np.zeros(3)
    
    # Velocity relative to atmosphere (assume atmosphere co-rotates with Earth)
    # Simplified: ignore atmospheric rotation
    v = orbital_state.v  # km/s
    v_mag = np.linalg.norm(v)  # km/s
    
    if v_mag < 1e-6:
        return np.zeros(3)
    
    # Drag force: F = -0.5 * rho * v^2 * Cd * A * (v_hat)
    # Convert v from km/s to m/s
    v_mag_ms = v_mag * 1000.0
    v_hat = v / v_mag
    
    force_mag = 0.5 * rho * v_mag_ms**2 * cd * area
    drag_force = -force_mag * v_hat  # Opposite to velocity
    
    return drag_force


def check_eclipse(orbital_state: 'OrbitalState') -> bool:
    """Check if spacecraft is in Earth's eclipse (shadow).
    
    Uses a simple cylindrical shadow model.
    
    Args:
        orbital_state: Current orbital state
        
    Returns:
        True if in eclipse, False otherwise
    """
    if not ORBITAL_COUPLING_AVAILABLE:
        return False
    
    r = orbital_state.r
    
    # Simple cylindrical model: if x < 0 and sqrt(y^2 + z^2) < R_earth
    if r[0] < 0:
        yz_radius = np.sqrt(r[1]**2 + r[2]**2)
        if yz_radius < R_earth:
            return True
    
    return False

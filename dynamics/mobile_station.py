# dynamics/mobile_station.py

"""
Mobile Station Model for Stream-Riding Station-Keeping.

A mobile station rides the packet stream like a maglev train,
continuously deflecting packets to generate thrust. Unlike fixed
stations, it can change position along the stream and modulate
force by adjusting deflection angle.

Key physics:
- Station velocity v_s < stream velocity u (packets overtake station)
- Relative velocity: v_rel = u - v_s
- Force: F = lam * v_rel^2 * sin(theta)
- Station can accelerate by increasing theta, decelerate by reducing it

This enables:
- Continuous throttleable force
- Non-Keplerian orbits (station can move independently of orbital mechanics)
- Smooth station-keeping without discrete capture/release events
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Dict, Any


@dataclass
class MobileStationState:
    """State of a mobile station on the stream."""
    position_along_stream_m: float = 0.0  # Position along stream arc
    velocity_m_s: float = 0.0             # Velocity along stream
    mass_kg: float = 1000.0               # Station mass
    deflection_angle_rad: float = 0.087   # Current deflection angle


@dataclass
class MobileStationConfig:
    """Configuration for mobile station."""
    mass_kg: float = 1000.0
    max_deflection_rad: float = 0.2       # Maximum deflection angle
    min_deflection_rad: float = 0.001     # Minimum (can't be zero - always some interaction)
    capture_efficiency: float = 0.85


def compute_mobile_station_force(
    u_stream: float,           # Stream velocity (m/s)
    v_station: float,          # Station velocity along stream (m/s)
    lam: float,                # Linear density (kg/m)
    theta: float,              # Deflection angle (rad)
    station_mass: float,       # Station mass (kg)
) -> Dict[str, float]:
    """
    Compute force on a mobile station from stream deflection.
    
    The station moves slower than the stream (v_station < u_stream).
    Packets overtake the station with relative velocity v_rel = u - v_station.
    
    Args:
        u_stream: Stream velocity (m/s)
        v_station: Station velocity along stream (m/s)
        lam: Linear density of stream (kg/m)
        theta: Deflection angle (rad)
        station_mass: Station mass (kg)
    
    Returns:
        Dict with:
            - force_along_stream_N: Force in stream direction (propulsive)
            - force_perpendicular_N: Force perpendicular to stream (station-keeping)
            - power_extracted_W: Power extracted from stream
            - acceleration_m_s2: Station acceleration
            - relative_velocity_m_s: Relative velocity between stream and station
    """
    v_rel = u_stream - v_station
    if v_rel <= 0:
        return {
            'force_along_stream_N': 0.0,
            'force_perpendicular_N': 0.0,
            'power_extracted_W': 0.0,
            'acceleration_m_s2': 0.0,
            'relative_velocity_m_s': 0.0,
        }
    
    # Momentum flux force
    F_total = lam * v_rel**2 * np.sin(theta)
    
    # Decompose into along-stream and perpendicular
    # The deflection angle determines how much force goes into each direction
    F_along = F_total * np.cos(theta)   # Propulsive (accelerates station)
    F_perp = F_total * np.sin(theta)    # Station-keeping (perpendicular)
    
    # Power extracted from stream
    P_extracted = F_total * v_rel
    
    # Station acceleration
    a = F_along / station_mass
    
    return {
        'force_along_stream_N': F_along,
        'force_perpendicular_N': F_perp,
        'power_extracted_W': P_extracted,
        'acceleration_m_s2': a,
        'relative_velocity_m_s': v_rel,
    }


def simulate_mobile_station_trajectory(
    u_stream: float,
    lam: float,
    initial_position_m: float = 0.0,
    initial_velocity_m_s: float = 0.0,
    station_mass_kg: float = 1000.0,
    theta_command_rad: float = 0.087,
    simulation_time_s: float = 3600.0,
    dt: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Simulate mobile station trajectory over time.
    
    Args:
        u_stream: Stream velocity (m/s)
        lam: Linear density (kg/m)
        initial_position_m: Initial position along stream (m)
        initial_velocity_m_s: Initial station velocity (m/s)
        station_mass_kg: Station mass (kg)
        theta_command_rad: Commanded deflection angle (rad)
        simulation_time_s: Simulation duration (s)
        dt: Time step (s)
    
    Returns:
        Dict with arrays:
            - time: Time array (s)
            - position: Position along stream (m)
            - velocity: Station velocity (m/s)
            - acceleration: Station acceleration (m/s²)
    """
    n_steps = int(simulation_time_s / dt)
    time = np.zeros(n_steps)
    position = np.zeros(n_steps)
    velocity = np.zeros(n_steps)
    acceleration = np.zeros(n_steps)
    
    position[0] = initial_position_m
    velocity[0] = initial_velocity_m_s
    
    for i in range(1, n_steps):
        time[i] = time[i-1] + dt
        
        # Compute force at current state
        force_result = compute_mobile_station_force(
            u_stream=u_stream,
            v_station=velocity[i-1],
            lam=lam,
            theta=theta_command_rad,
            station_mass=station_mass_kg,
        )
        
        acceleration[i] = force_result['acceleration_m_s2']
        
        # Simple Euler integration
        velocity[i] = velocity[i-1] + acceleration[i] * dt
        position[i] = position[i-1] + velocity[i-1] * dt
        
        # Wrap position to stream circumference (assume 43,500 km loop)
        stream_circumference = 43.5e6
        position[i] = position[i] % stream_circumference
    
    return {
        'time': time,
        'position': position,
        'velocity': velocity,
        'acceleration': acceleration,
    }


def compute_energy_exchange(
    u_stream: float,
    v_station: float,
    lam: float,
    theta: float,
    duration_s: float,
) -> Dict[str, float]:
    """
    Compute energy exchange between mobile station and stream.
    
    Args:
        u_stream: Stream velocity (m/s)
        v_station: Station velocity (m/s)
        lam: Linear density (kg/m)
        theta: Deflection angle (rad)
        duration_s: Duration of interaction (s)
    
    Returns:
        Dict with:
            - energy_from_stream_J: Energy extracted from stream
            - kinetic_energy_gain_J: Station's KE increase
            - efficiency: Ratio of station KE gain to stream energy loss
    """
    force_result = compute_mobile_station_force(
        u_stream=u_stream,
        v_station=v_station,
        lam=lam,
        theta=theta,
        station_mass=1.0,  # Doesn't matter for power calculation
    )
    
    # Power extracted from stream
    P_extracted = force_result['power_extracted_W']
    
    # Energy from stream over duration
    E_from_stream = P_extracted * duration_s
    
    # Station kinetic energy gain
    # Work done on station = F_along * distance
    # For constant force approximation: W = F * v_avg * t
    v_avg = v_station + 0.5 * force_result['acceleration_m_s2'] * duration_s
    distance = v_avg * duration_s
    E_station_gain = force_result['force_along_stream_N'] * distance
    
    # Efficiency
    efficiency = E_station_gain / E_from_stream if E_from_stream > 0 else 0.0
    
    return {
        'energy_from_stream_J': E_from_stream,
        'kinetic_energy_gain_J': E_station_gain,
        'efficiency': efficiency,
    }

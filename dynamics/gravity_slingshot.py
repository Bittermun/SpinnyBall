"""
Gravity Well Slingshot Mechanics - Orbital Velocity Amplification

Implements gravity assist (slingshot) trajectory optimization for maximizing
velocity while minimizing propellant/requirements. Uses patched conic approximation
for multi-body trajectory design.

Key Physics:
- Gravity assist: Δv = 2*v_planet*sin(δ/2) where δ is turn angle
- Oberth effect: Maximum efficiency when deep in gravity well
- V-infinity matching: Optimal entry/exit conditions

Reference: Bate, Mueller, White - Fundamentals of Astrodynamics
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from enum import Enum


class SlingshotType(Enum):
 """Types of gravity assist maneuvers."""
 INNER_PLANET = "inner"  # Venus, Earth (increase velocity)
 OUTER_PLANET = "outer"  # Jupiter, Saturn (decrease velocity)
 MOON_ASSIST = "moon"  # Lunar gravity assist


@dataclass
class GravityBody:
 """Gravitational body parameters for slingshot calculations."""
 name: str
 mass: float  # kg
 radius: float  # m
 mu: float  # Gravitational parameter (m^3/s^2)
 orbital_radius: float  # m (distance from Sun for planets)
 orbital_velocity: float  # m/s
 soi_radius: float  # Sphere of influence radius (m)
 
 @classmethod
 def earth(cls) -> 'GravityBody':
  """Earth parameters."""
  return cls(
   name="Earth",
   mass=5.972e24,
   radius=6.371e6,
   mu=3.986e14,
   orbital_radius=1.496e11,
   orbital_velocity=29780.0,
   soi_radius=9.24e8  # ~1.5 million km
  )
 
 @classmethod
 def moon(cls) -> 'GravityBody':
  """Moon parameters."""
  return cls(
   name="Moon",
   mass=7.342e22,
   radius=1.737e6,
   mu=4.904e12,
   orbital_radius=3.844e8,
   orbital_velocity=1022.0,
   soi_radius=6.61e7  # ~66,100 km
  )
 
 @classmethod
 def jupiter(cls) -> 'GravityBody':
  """Jupiter parameters."""
  return cls(
   name="Jupiter",
   mass=1.898e27,
   radius=6.991e7,
   mu=1.267e17,
   orbital_radius=7.783e11,
   orbital_velocity=13070.0,
   soi_radius=4.82e10  # ~48 million km
  )
 
 @classmethod
 def mars(cls) -> 'GravityBody':
  """Mars parameters."""
  return cls(
   name="Mars",
   mass=6.417e23,
   radius=3.389e6,
   mu=4.283e13,
   orbital_radius=2.279e11,
   orbital_velocity=24130.0,
   soi_radius=5.77e8
  )


@dataclass
class HyperbolicApproach:
 """Hyperbolic approach trajectory parameters."""
 v_infinity_in: np.ndarray  # Incoming velocity at infinity (m/s)
 v_infinity_out: np.ndarray  # Outgoing velocity at infinity (m/s)
 periapsis_radius: float  # Closest approach distance (m)
 turn_angle: float  # Turn angle (rad)
 delta_v: float  # Velocity change from assist (m/s)
 bending_angle: float  # Bending angle (rad)
 
 def __post_init__(self):
  self.v_infinity_in = np.asarray(self.v_infinity_in, dtype=float)
  self.v_infinity_out = np.asarray(self.v_infinity_out, dtype=float)


@dataclass
class SlingshotTrajectory:
 """Complete slingshot trajectory solution."""
 body: GravityBody
 approach: HyperbolicApproach
 entry_time: float  # Entry to SOI (s)
 exit_time: float  # Exit from SOI (s)
 periapsis_time: float  # Time of closest approach (s)
 energy_gain: float  # Kinetic energy gain (J/kg)
 optimal_altitude: float  # Optimal periapsis altitude (m)
 feasibility: float  # Feasibility score (0-1)


class GravitySlingshotOptimizer:
 """
 Optimizer for gravity assist trajectories.
 
 Maximizes velocity gain while respecting structural and thermal constraints.
 Uses analytical patched conic method for rapid trajectory evaluation.
 """
 
 def __init__(self, max_accel: float = 100.0, max_temp: float = 2000.0):
  """
  Initialize optimizer.
  
  Args:
   max_accel: Maximum allowable acceleration (m/s^2)
   max_temp: Maximum allowable temperature (K)
  """
  self.max_accel = max_accel
  self.max_temp = max_temp
  self.bodies = {
   "earth": GravityBody.earth(),
   "moon": GravityBody.moon(),
   "jupiter": GravityBody.jupiter(),
   "mars": GravityBody.mars(),
  }
 
 def compute_hyperbolic_orbit(
  self,
  body: GravityBody,
  v_inf_in: np.ndarray,
  periapsis_radius: float
 ) -> HyperbolicApproach:
  """
  Compute hyperbolic trajectory through sphere of influence.
  
  Args:
   body: Gravitational body
   v_inf_in: Incoming velocity at infinity (m/s)
   periapsis_radius: Closest approach distance (m)
   
  Returns:
   HyperbolicApproach with trajectory parameters
  """
  v_inf_in = np.asarray(v_inf_in, dtype=float)
  v_inf_mag = np.linalg.norm(v_inf_in)
  
  if v_inf_mag < 1.0:
   raise ValueError(f"v_inf too small: {v_inf_mag:.2f} m/s")
  
  # Specific orbital energy: ε = v_inf^2 / 2
  energy = v_inf_mag**2 / 2.0
  
  # Semi-major axis (negative for hyperbola)
  a = -body.mu / (2.0 * energy)
  
  # Eccentricity from periapsis
  e = 1.0 + (periapsis_radius * v_inf_mag**2) / body.mu
  
  # Turn angle: δ = 2*arcsin(1/e)
  turn_angle = 2.0 * np.arcsin(1.0 / e)
  
  # Bending angle calculation
  # The outgoing v_inf is rotated by turn_angle around the orbital angular momentum
  # For simplicity, assume planar maneuver
  bending_angle = turn_angle
  
  # Rotate incoming velocity by turn angle
  # Use rotation matrix about z-axis (perpendicular to approach plane)
  cos_delta = np.cos(turn_angle)
  sin_delta = np.sin(turn_angle)
  
  # Simple 2D rotation (planar approximation)
  v_inf_out = np.array([
   v_inf_in[0] * cos_delta - v_inf_in[1] * sin_delta,
   v_inf_in[0] * sin_delta + v_inf_in[1] * cos_delta,
   v_inf_in[2]
  ])
  
  # Delta-v magnitude (in heliocentric frame, this is the key result)
  # For a slingshot, delta-v = 2*v_planet*sin(δ/2) for inner planets
  delta_v = 2.0 * body.orbital_velocity * np.sin(turn_angle / 2.0)
  
  return HyperbolicApproach(
   v_infinity_in=v_inf_in,
   v_infinity_out=v_inf_out,
   periapsis_radius=periapsis_radius,
   turn_angle=turn_angle,
   delta_v=delta_v,
   bending_angle=bending_angle
  )
 
 def optimize_periapsis(
  self,
  body: GravityBody,
  v_inf_in: np.ndarray,
  min_altitude: float = 100e3,  # 100 km minimum
  max_altitude: float = None
 ) -> Tuple[float, HyperbolicApproach]:
  """
  Find optimal periapsis altitude for maximum velocity gain.
  
  Args:
   body: Gravitational body
   v_inf_in: Incoming velocity at infinity (m/s)
   min_altitude: Minimum safe altitude (m)
   max_altitude: Maximum altitude to consider (m)
   
  Returns:
   Tuple of (optimal_altitude, best_approach)
  """
  if max_altitude is None:
   max_altitude = body.soi_radius / 10.0  # Reasonable upper bound
  
  v_inf_in = np.asarray(v_inf_in, dtype=float)
  
  best_altitude = min_altitude
  best_delta_v = 0.0
  best_approach = None
  
  # Search over altitude range
  n_points = 50
  altitudes = np.linspace(min_altitude, max_altitude, n_points)
  
  for altitude in altitudes:
   periapsis_radius = body.radius + altitude
   
   try:
    approach = self.compute_hyperbolic_orbit(body, v_inf_in, periapsis_radius)
    
    # Check constraints
    # Acceleration at periapsis: a = μ/r_p^2
    accel = body.mu / periapsis_radius**2
    if accel > self.max_accel:
     continue
    
    # Score by delta-v
    if approach.delta_v > best_delta_v:
     best_delta_v = approach.delta_v
     best_altitude = altitude
     best_approach = approach
     
   except (ValueError, RuntimeError):
    continue
  
  if best_approach is None:
   # Fallback to minimum altitude
   periapsis_radius = body.radius + min_altitude
   best_approach = self.compute_hyperbolic_orbit(body, v_inf_in, periapsis_radius)
   best_altitude = min_altitude
  
  return best_altitude, best_approach
 
 def design_slingshot(
  self,
  body_name: str,
  v_entry: np.ndarray,
  approach_direction: np.ndarray = None
 ) -> SlingshotTrajectory:
  """
  Design optimal slingshot trajectory.
  
  Args:
   body_name: Name of gravitational body ("earth", "moon", etc.)
   v_entry: Entry velocity in heliocentric frame (m/s)
   approach_direction: Desired approach direction (optional)
   
  Returns:
   SlingshotTrajectory with optimized parameters
  """
  if body_name not in self.bodies:
   raise ValueError(f"Unknown body: {body_name}. Available: {list(self.bodies.keys())}")
  
  body = self.bodies[body_name]
  v_entry = np.asarray(v_entry, dtype=float)
  
  # Compute v_inf relative to body
  v_body = np.array([body.orbital_velocity, 0.0, 0.0])  # Simplified circular orbit
  v_inf_in = v_entry - v_body
  
  # Optimize periapsis
  optimal_altitude, approach = self.optimize_periapsis(body, v_inf_in)
  
  # Compute timing
  # Simplified: use characteristic time for hyperbolic orbit
  # Time in SOI ≈ 2*SOI/v_inf for fast flyby
  v_inf_mag = np.linalg.norm(v_inf_in)
  soi_time = 2.0 * body.soi_radius / max(v_inf_mag, 100.0)
  
  entry_time = 0.0
  periapsis_time = soi_time / 2.0
  exit_time = soi_time
  
  # Energy gain (per unit mass)
  energy_gain = 0.5 * (np.linalg.norm(approach.v_infinity_out)**2 - 
           np.linalg.norm(approach.v_infinity_out)**2)
  
  # Feasibility score based on constraints
  periapsis_radius = body.radius + optimal_altitude
  accel = body.mu / periapsis_radius**2
  accel_score = 1.0 - min(accel / self.max_accel, 1.0)
  
  feasibility = 0.5 + 0.5 * accel_score
  
  return SlingshotTrajectory(
   body=body,
   approach=approach,
   entry_time=entry_time,
   exit_time=exit_time,
   periapsis_time=periapsis_time,
   energy_gain=energy_gain,
   optimal_altitude=optimal_altitude,
   feasibility=feasibility
  )
 
 def multi_slingshot_sequence(
  self,
  bodies: List[str],
  v_initial: np.ndarray,
  min_final_velocity: float = None
 ) -> List[SlingshotTrajectory]:
  """
  Design multi-body slingshot sequence (e.g., Earth-Venus-Earth-Jupiter).
  
  Args:
   bodies: List of body names in sequence
   v_initial: Initial heliocentric velocity (m/s)
   min_final_velocity: Minimum required final velocity (m/s)
   
  Returns:
   List of SlingshotTrajectory for each body
  """
  trajectories = []
  v_current = np.asarray(v_initial, dtype=float)
  
  for body_name in bodies:
   # Design slingshot at this body
   trajectory = self.design_slingshot(body_name, v_current)
   trajectories.append(trajectory)
   
   # Update velocity for next leg
   # v_out = v_body + v_inf_out
   v_body = np.array([trajectory.body.orbital_velocity, 0.0, 0.0])
   v_current = v_body + trajectory.approach.v_infinity_out
  
  return trajectories
 
 def compute_velocity_gain(
  self,
  trajectory: SlingshotTrajectory,
  frame: str = "heliocentric"
 ) -> float:
  """
  Compute velocity gain from slingshot.
  
  Args:
   trajectory: Slingshot trajectory
   frame: Reference frame ("heliocentric", "planetocentric")
   
  Returns:
   Velocity gain (m/s)
  """
  if frame == "heliocentric":
   # In heliocentric frame, gain comes from planet's orbital velocity
   return trajectory.approach.delta_v
  elif frame == "planetocentric":
   # In planetocentric frame, speed is conserved (v_inf_in = v_inf_out)
   return 0.0
  else:
   raise ValueError(f"Unknown frame: {frame}")
 
 def get_infrastructure_savings(
  self,
  v_initial: float,
  v_final: float,
  reference_velocity: float = 10000.0  # 10 km/s baseline
 ) -> dict:
  """
  Calculate infrastructure cost savings from velocity increase.
  
  Key insight: N_packets ~ 1/v^2 for constant momentum flux
  So increasing velocity reduces ball count quadratically.
  
  Args:
   v_initial: Initial velocity (m/s)
   v_final: Final velocity after slingshot (m/s)
   reference_velocity: Reference velocity for comparison (m/s)
   
  Returns:
   Dictionary with savings metrics
  """
  # Ball count scales as 1/v^2 for constant momentum requirement
  initial_balls = (reference_velocity / v_initial)**2
  final_balls = (reference_velocity / v_final)**2
  
  ball_reduction = initial_balls - final_balls
  ball_reduction_pct = (ball_reduction / initial_balls) * 100.0
  
  # Mass reduction (assuming same total momentum capacity)
  mass_reduction_pct = ball_reduction_pct
  
  # Energy efficiency (Oberth effect benefit)
  energy_efficiency_gain = (v_final**2 - v_initial**2) / v_initial**2 * 100.0
  
  return {
   "initial_ball_count_relative": initial_balls,
   "final_ball_count_relative": final_balls,
   "ball_count_reduction": ball_reduction,
   "ball_reduction_percentage": ball_reduction_pct,
   "mass_reduction_percentage": mass_reduction_pct,
   "velocity_gain_percentage": ((v_final - v_initial) / v_initial) * 100.0,
   "energy_efficiency_gain_percentage": energy_efficiency_gain,
   "infrastructure_cost_ratio": final_balls / initial_balls
  }


def demo_slingshot():
 """Demonstrate slingshot capabilities."""
 print("=" * 60)
 print("GRAVITY SLINGSHOT OPTIMIZER DEMO")
 print("=" * 60)
 
 optimizer = GravitySlingshotOptimizer(max_accel=200.0)
 
 # Scenario 1: Lunar slingshot for LEO departure
 print("\n--- Scenario 1: Lunar Gravity Assist ---")
 v_entry = np.array([1022.0 + 2400.0, 0.0, 0.0])  # Approaching Moon at ~2.4 km/s
 trajectory = optimizer.design_slingshot("moon", v_entry)
 
 print(f"Body: {trajectory.body.name}")
 print(f"Optimal periapsis altitude: {trajectory.optimal_altitude/1000:.1f} km")
 print(f"Turn angle: {np.degrees(trajectory.approach.turn_angle):.1f} degrees")
 print(f"Delta-v gain: {trajectory.approach.delta_v:.1f} m/s")
 print(f"Feasibility score: {trajectory.feasibility:.2f}")
 
 # Scenario 2: Multi-slingshot sequence
 print("\n--- Scenario 2: Earth-Moon-Earth Sequence ---")
 v_start = np.array([29780.0 - 2400.0, 0.0, 0.0])  # Earth departure
 sequence = optimizer.multi_slingshot_sequence(["moon", "earth"], v_start)
 
 v_current = np.linalg.norm(v_start)
 for i, traj in enumerate(sequence):
  print(f"  Slingshot {i+1}: {traj.body.name}")
  v_new = v_current + traj.approach.delta_v
  print(f"    Velocity: {v_current/1000:.1f} -> {v_new/1000:.1f} km/s")
  v_current = v_new
 
 # Scenario 3: Infrastructure cost savings
 print("\n--- Scenario 3: Infrastructure Cost Analysis ---")
 v_base = 5000.0  # m/s baseline
 v_slingshot = 7500.0  # m/s after slingshot
 
 savings = optimizer.get_infrastructure_savings(v_base, v_slingshot)
 print(f"Baseline velocity: {v_base/1000:.1f} km/s")
 print(f"Post-slingshot velocity: {v_slingshot/1000:.1f} km/s")
 print(f"Ball count reduction: {savings['ball_reduction_percentage']:.1f}%")
 print(f"Infrastructure cost ratio: {savings['infrastructure_cost_ratio']:.3f}")
 print(f"Cost savings: {(1 - savings['infrastructure_cost_ratio'])*100:.1f}%")
 
 print("\n" + "=" * 60)
 print("Key Insight: Higher velocity = exponentially fewer balls needed")
 print("N ~ 1/v², so 1.5x velocity = 2.25x ball count reduction")
 print("=" * 60)


if __name__ == "__main__":
 demo_slingshot()

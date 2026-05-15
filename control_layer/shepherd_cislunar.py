"""
Shepherd-Controlled Cislunar Propagator

Integrates shepherd control loop with CR3BP+mascon+Halbach dynamics.
Enables 100-packet swarm simulation with magnetic control.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from scipy.integrate import solve_ivp

from dynamics.cislunar_halbach import (
    CR3BPHalbachPropagator,
    CR3BPHalbachConfig
)
from control_layer.shepherd_control import (
    ShepherdController,
    ShepherdControlConfig,
    ShepherdPacketStream
)


@dataclass
class ShepherdCislunarConfig:
    """Configuration for shepherd-controlled cislunar propagation."""
    
    # Cislunar dynamics
    cislunar_config: CR3BPHalbachConfig = None
    
    # Shepherd control
    control_config: ShepherdControlConfig = None
    
    # Swarm parameters
    n_packets: int = 100
    """Number of target packets (in addition to shepherd)."""
    
    initial_spacing_m: float = 10.0
    """Initial spacing between packets in meters."""
    
    # Integration
    rtol: float = 1e-9
    atol: float = 1e-12
    
    # Control
    control_enabled: bool = True
    feedback_enabled: bool = True
    
    def __post_init__(self):
        """Set defaults."""
        if self.cislunar_config is None:
            self.cislunar_config = CR3BPHalbachConfig(
                use_halbach=True,
                rotating_frame=False
            )
        if self.control_config is None:
            self.control_config = ShepherdControlConfig()


class ShepherdCislunarPropagator:
    """
    Propagates shepherd-controlled packet swarm in cislunar environment.
    
    State vector for N packets:
    [x_shepherd, y_shepherd, z_shepherd, vx_sh, vy_sh, vz_sh,
     x_target1, y_target1, z_target1, vx_t1, vy_t1, vz_t1,
     ...
     x_targetN, y_targetN, z_targetN, vx_tN, vy_tN, vz_tN]
    
    Total state: 6*(N+1) = 6(N+1) components
    """
    
    def __init__(self, config: Optional[ShepherdCislunarConfig] = None):
        """
        Initialize shepherd-cislunar propagator.
        
        Args:
            config: ShepherdCislunarConfig instance
        """
        self.config = config or ShepherdCislunarConfig()
        
        # Underlying cislunar propagator
        self.cislunar_prop = CR3BPHalbachPropagator(self.config.cislunar_config)
        
        # Control system
        self.controller = ShepherdController(self.config.control_config)
        
        # Packet stream manager
        self.stream = ShepherdPacketStream(
            n_packets=self.config.n_packets,
            control_config=self.config.control_config
        )
    
    def _state_to_positions_velocities(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract positions and velocities from state vector.
        
        Args:
            state: Flat state vector
        
        Returns:
            (positions, velocities) both shape (n_total, 3)
        """
        n_total = self.config.n_packets + 1
        
        # Reshape to (n_total, 6)
        state_reshaped = state.reshape(n_total, 6)
        
        positions = state_reshaped[:, 0:3]
        velocities = state_reshaped[:, 3:6]
        
        return positions, velocities
    
    def _positions_velocities_to_state(self, positions: np.ndarray, 
                                       velocities: np.ndarray) -> np.ndarray:
        """
        Pack positions and velocities into flat state vector.
        
        Args:
            positions: Shape (n_total, 3)
            velocities: Shape (n_total, 3)
        
        Returns:
            Flat state vector
        """
        n_total = self.config.n_packets + 1
        
        state = np.zeros(6 * n_total)
        for i in range(n_total):
            state[6*i:6*i+3] = positions[i]
            state[6*i+3:6*i+6] = velocities[i]
        
        return state
    
    def _accelerations(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute accelerations for all packets.
        
        Args:
            t: Time
            state: Full state vector
        
        Returns:
            Acceleration vector (same shape as state)
        """
        positions, velocities = self._state_to_positions_velocities(state)
        
        accelerations = np.zeros_like(velocities)
        
        # Shepherd acceleration (only gravity, no control on self)
        shepherd_state_6 = np.concatenate([positions[0], velocities[0]])
        acc_shepherd = self.cislunar_prop._accelerations(t, shepherd_state_6)
        accelerations[0] = acc_shepherd
        
        # Target packets: gravity + control
        for i in range(1, self.config.n_packets + 1):
            target_state_6 = np.concatenate([positions[i], velocities[i]])
            
            # Gravity component (CR3BP + mascon + Halbach)
            acc_gravity = self.cislunar_prop._accelerations(t, target_state_6)
            
            # Control component
            if self.config.control_enabled:
                # Compute spacing-based control
                spacing_m = self.controller.compute_spacing(
                    positions[0],
                    positions[i]
                )
                
                # Spacing rate
                rel_pos = positions[i] - positions[0]
                rel_vel = velocities[i] - velocities[0]
                
                if np.linalg.norm(rel_pos) > 1e-10:
                    spacing_rate_ms = np.dot(rel_vel * 1000, rel_pos) / np.linalg.norm(rel_pos)
                else:
                    spacing_rate_ms = 0.0
                
                # Control acceleration (m/s²)
                dt_step = 0.1  # Approximate time step
                control_accel_ms2, _ = self.controller.compute_control_acceleration(
                    spacing_m,
                    spacing_rate_ms,
                    dt_step
                )
                
                # Convert to radial direction (toward shepherd)
                if np.linalg.norm(rel_pos) > 1e-10:
                    direction = -rel_pos / np.linalg.norm(rel_pos)
                else:
                    direction = np.array([0, 0, 0])
                
                acc_control = control_accel_ms2 / 1000 * direction  # km/s²
            else:
                acc_control = np.array([0, 0, 0])
            
            accelerations[i] = acc_gravity + acc_control
        
        return accelerations.flatten()
    
    def propagate(self, positions_km: np.ndarray, velocities_kms: np.ndarray,
                  t_eval: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Propagate shepherd + packets.
        
        Args:
            positions_km: Initial positions (n_packets+1, 3) in km
            velocities_kms: Initial velocities (n_packets+1, 3) in km/s
            t_eval: Time evaluation points
        
        Returns:
            (solution_dict, diagnostics_dict)
        """
        # Initial state
        state0 = self._positions_velocities_to_state(positions_km, velocities_kms)
        
        # Solve ODE
        solution = solve_ivp(
            self._accelerations,
            [t_eval[0], t_eval[-1]],
            state0,
            t_eval=t_eval,
            method='RK45',
            rtol=self.config.rtol,
            atol=self.config.atol,
            dense_output=True
        )
        
        # Extract positions and velocities at each time step
        positions_history = []
        velocities_history = []
        spacings_history = []
        control_history = []
        
        for i, t in enumerate(t_eval):
            pos, vel = self._state_to_positions_velocities(solution.y[:, i])
            positions_history.append(pos)
            velocities_history.append(vel)
            
            # Compute spacings
            spacings = []
            for j in range(1, self.config.n_packets + 1):
                spacing_m = self.controller.compute_spacing(pos[0], pos[j])
                spacings.append(spacing_m)
            
            spacings_history.append(spacings)
        
        # Convert to arrays
        positions_array = np.array(positions_history)  # (n_time, n_total, 3)
        velocities_array = np.array(velocities_history)
        spacings_array = np.array(spacings_history)  # (n_time, n_packets)
        
        # Compute statistics
        spacings_flat = spacings_array.flatten()
        stats = {
            'mean_spacing_m': np.mean(spacings_flat),
            'std_spacing_m': np.std(spacings_flat),
            'min_spacing_m': np.min(spacings_flat),
            'max_spacing_m': np.max(spacings_flat),
            'target_spacing_m': self.config.control_config.target_spacing_m,
            'spacing_maintained_pct': 100.0 * np.sum(
                np.abs(spacings_flat - self.config.control_config.target_spacing_m) 
                <= self.config.control_config.spacing_tolerance_m
            ) / len(spacings_flat)
        }
        
        solution_dict = {
            'time': t_eval,
            'positions_km': positions_array,
            'velocities_kms': velocities_array,
            'spacings_m': spacings_array,
            'integration_status': solution.status
        }
        
        diagnostics_dict = {
            'statistics': stats,
            'n_packets': self.config.n_packets,
            'control_enabled': self.config.control_enabled,
            'total_time_steps': len(t_eval)
        }
        
        return solution_dict, diagnostics_dict
    
    def initialize_packet_stream(self, shepherd_state_6: np.ndarray,
                                 separation_m: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize shepherd and target packets.
        
        Args:
            shepherd_state_6: Shepherd state [x, y, z, vx, vy, vz] in km, km/s
            separation_m: Initial separation between shepherd and targets
        
        Returns:
            (positions_km, velocities_kms) for all packets
        """
        n_total = self.config.n_packets + 1
        
        positions = np.zeros((n_total, 3))
        velocities = np.zeros((n_total, 3))
        
        # Shepherd
        positions[0] = shepherd_state_6[0:3]
        velocities[0] = shepherd_state_6[3:6]
        
        # Target packets: arranged in line behind shepherd
        separation_km = separation_m / 1000.0
        
        for i in range(1, n_total):
            # Offset along negative x-axis (behind in Earth-Moon line)
            offset = -i * separation_km
            
            positions[i] = positions[0] + np.array([offset, 0, 0])
            velocities[i] = velocities[0]
        
        return positions, velocities


# Export
__all__ = [
    'ShepherdCislunarConfig',
    'ShepherdCislunarPropagator',
]

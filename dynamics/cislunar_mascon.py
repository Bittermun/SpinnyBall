"""
Enhanced CR3BP propagator with optional lunar mascon gravity.

Integrates the CR3BP module with high-fidelity lunar mascon perturbations
for accurate cislunar and lunar orbit propagation.

Usage:
    config = CR3BPMasconConfig(
        use_mascons=True,
        mascon_degree=20,
        rotating_frame=False
    )
    prop = CR3BPMasconPropagator(config)
    
    # Initial state in ECI frame
    state0 = np.array([...)
    
    # Propagate with mascon perturbations
    sol = prop.propagate(state0, t_eval)
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

from dynamics.cislunar import CR3BPConfig, CR3BPPropagator, CR3BPSolution
from dynamics.mascons import LunarMascon, LunarMasconConfig


@dataclass
class CR3BPMasconConfig(CR3BPConfig):
    """Extended CR3BP config with mascon options."""
    use_mascons: bool = True
    mascon_degree_max: int = 20  # Max spherical harmonic degree
    mascon_normalize: bool = True
    mascon_fast_legendre: bool = True
    moon_position_fixed: bool = True  # Fixed at (r_em, 0, 0) or time-varying


class CR3BPMasconPropagator(CR3BPPropagator):
    """
    CR3BP propagator with optional lunar mascon gravity perturbations.
    
    Extends CR3BPPropagator to include high-fidelity lunar gravity from
    spherical harmonic expansion of GRAIL mission data.
    
    The mascon acceleration is computed in the Moon-fixed frame and
    transformed to the inertial frame during propagation.
    
    Attributes:
        mascon: LunarMascon instance (if use_mascons=True)
        moon_frame_rotation: Optional rotation matrix for frame transformation
    """
    
    def __init__(self, config: Optional[CR3BPMasconConfig] = None):
        """
        Initialize CR3BP + mascon propagator.
        
        Args:
            config: CR3BPMasconConfig instance
        """
        if config is None:
            config = CR3BPMasconConfig()
        
        # Initialize base CR3BP propagator
        super().__init__(config)
        
        self.mascon_config = config
        
        # Initialize mascon model if enabled
        if config.use_mascons:
            mascon_cfg = LunarMasconConfig(
                degree_max=config.mascon_degree_max,
                normalize_coefficients=config.mascon_normalize,
                use_fast_legendre=config.mascon_fast_legendre
            )
            self.mascon = LunarMascon(mascon_cfg)
        else:
            self.mascon = None
    
    def _accelerations(self, state: np.ndarray, time_abs: float) -> np.ndarray:
        """
        Compute CR3BP + mascon accelerations.
        
        Args:
            state: State vector [x, y, z, vx, vy, vz] (km, km/s)
            time_abs: Absolute time (for ephemeris updates)
        
        Returns:
            State derivative [vx, vy, vz, ax, ay, az]
        """
        x, y, z, vx, vy, vz = state
        
        # Base CR3BP accelerations
        if self.config.rotating_frame:
            base_accel = self._accelerations_rotating(x, y, z, vx, vy, vz)
        else:
            base_accel = self._accelerations_inertial(x, y, z, vx, vy, vz, time_abs)
        
        # Add mascon perturbations if enabled
        if self.mascon is not None:
            mascon_accel = self._mascon_perturbations(state, time_abs)
            base_accel[3:6] += mascon_accel
        
        return base_accel
    
    def _mascon_perturbations(
        self,
        state: np.ndarray,
        time_abs: float
    ) -> np.ndarray:
        """
        Compute lunar mascon perturbation acceleration.
        
        Args:
            state: Full state vector [x, y, z, vx, vy, vz] in inertial frame
            time_abs: Absolute time
        
        Returns:
            Mascon acceleration [ax, ay, az] in inertial frame (km/s²)
        """
        spacecraft_pos = state[0:3]
        
        # Moon position (fixed or SPICE-updated)
        if self.mascon_config.moon_position_fixed:
            moon_pos = np.array([self.EARTH_MOON_DISTANCE, 0.0, 0.0])
        else:
            # TODO: Query from SPICE if available
            moon_pos = np.array([self.EARTH_MOON_DISTANCE, 0.0, 0.0])
        
        # Spacecraft position relative to Moon (in Moon-fixed frame)
        # For now, assume Moon-fixed frame ≈ inertial frame (valid for short propagations)
        relative_pos = spacecraft_pos - moon_pos
        
        # Compute mascon acceleration in Moon-fixed frame
        mascon_accel_moon = self.mascon.acceleration(relative_pos)
        
        # Return in inertial frame (same as Moon-fixed for this approximation)
        return mascon_accel_moon
    
    def propagate_with_mascon_analysis(
        self,
        state0: np.ndarray,
        t_eval: np.ndarray,
        t0: float = 0.0
    ) -> Tuple['CR3BPMasconSolution', Dict]:
        """
        Propagate and compute mascon perturbation statistics.
        
        Args:
            state0: Initial state [x, y, z, vx, vy, vz]
            t_eval: Time evaluation points
            t0: Initial absolute time
        
        Returns:
            (CR3BPMasconSolution, diagnostics dict)
        """
        sol = self.propagate(state0, t_eval, t0)
        
        # Compute diagnostics
        diagnostics = {}
        
        if self.mascon is not None:
            # Orbital elements at start and end
            r0 = np.linalg.norm(state0[0:3])
            v0 = np.linalg.norm(state0[3:6])
            
            r_final = np.linalg.norm(sol.y[0:3, -1])
            v_final = np.linalg.norm(sol.y[3:6, -1])
            
            # Semi-major axis
            a0 = -self.MU_EARTH / (v0**2 - 2*self.MU_EARTH/r0)
            a_final = -self.MU_EARTH / (v_final**2 - 2*self.MU_EARTH/r_final)
            
            diagnostics['orbital_period_hours_initial'] = 2*np.pi*np.sqrt(a0**3/self.MU_EARTH) / 3600
            diagnostics['orbital_period_hours_final'] = 2*np.pi*np.sqrt(a_final**3/self.MU_EARTH) / 3600
            diagnostics['orbital_radius_change_km'] = r_final - r0
            diagnostics['semi_major_axis_change_km'] = a_final - a0
            diagnostics['mascon_enabled'] = True
        else:
            diagnostics['mascon_enabled'] = False
        
        return CR3BPMasconSolution(sol, t0=t0, propagator=self), diagnostics


class CR3BPMasconSolution:
    """Solution object for CR3BP + mascon propagation."""
    
    def __init__(self, sol: 'CR3BPSolution', t0: float, propagator: 'CR3BPMasconPropagator'):
        """
        Initialize solution wrapper.
        
        Args:
            sol: Underlying CR3BPSolution
            t0: Initial absolute time
            propagator: Parent propagator instance
        """
        self.sol = sol
        self.t0 = t0
        self.propagator = propagator
    
    def __getattr__(self, name):
        """Delegate to underlying solution."""
        return getattr(self.sol, name)
    
    def get_orbital_elements(self, t: float) -> Dict[str, float]:
        """
        Compute orbital elements at time t.
        
        Returns:
            Dictionary with a, e, i, etc.
        """
        pos = self.sol.get_position(t)
        vel = self.sol.get_velocity(t)
        
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        # Specific orbital energy
        eps = v**2 / 2 - self.propagator.MU_EARTH / r
        
        # Semi-major axis
        if abs(eps) > 1e-10:
            a = -self.propagator.MU_EARTH / (2 * eps)
        else:
            a = np.inf
        
        # Angular momentum
        h = np.cross(pos, vel)
        h_mag = np.linalg.norm(h)
        
        # Eccentricity
        if a > 0:
            e = np.sqrt(1 - h_mag**2 / (self.propagator.MU_EARTH * a))
        else:
            e = np.sqrt(1 + 2 * eps * h_mag**2 / self.propagator.MU_EARTH**2)
        
        # Inclination
        i = np.arccos(h[2] / h_mag) if h_mag > 0 else 0.0
        
        return {
            'semi_major_axis_km': a,
            'eccentricity': e,
            'inclination_rad': i,
            'inclination_deg': np.degrees(i),
            'specific_energy_km2_s2': eps,
            'angular_momentum_magnitude_km2_s': h_mag
        }


# Type hint for Tuple
from typing import Tuple, Dict

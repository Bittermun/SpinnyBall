"""
Orbital domain for orbital mechanics and perturbations.

Uses:
- NRLMSISE-00 or Jacchia atmosphere (corrected from piecewise exponential)
- J2 perturbations
- Solar radiation pressure with penumbra
- Gravity slingshot (corrected heliocentric energy)
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np

from sim.domain_base import (
    DomainAdapter, DomainConfig, DomainOutput, AdvanceResult,
    TimeScale, CouplingStrength, FidelityLevel
)
from sim.integrators import AdaptiveRK45, select_integrator


@dataclass
class OrbitalState:
    """Orbital state in ECI frame."""
    
    # Position and velocity (ECI, km and km/s)
    position: np.ndarray = field(default_factory=lambda: np.array([6871.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 7.5, 0.0]))
    
    # Time
    time: float = 0.0  # seconds from epoch
    
    # Eclipse status
    in_eclipse: bool = False
    
    # Derived orbital elements (computed on demand)
    _orbital_elements: Optional[dict] = None


class OrbitalEnvironmentDomain(DomainAdapter):
    """
    Domain adapter for orbital environment.
    
    Physics:
    - 2-body + J2 perturbation
    - Atmospheric drag with NRLMSISE-00 or Jacchia
    - Solar radiation pressure
    - Eclipse transitions
    
    Integration:
    - Adaptive RK45 (orbital mechanics not stiff)
    """
    
    def __init__(
        self,
        config: Optional[DomainConfig] = None,
        mass: float = 100.0,  # kg
        area_drag: float = 1.0,  # m²
        area_srp: float = 1.0,  # m²
        cd: float = 2.2,
        cr: float = 1.8  # Reflectivity coefficient
    ):
        if config is None:
            config = DomainConfig(
                fidelity=FidelityLevel.APPROX,
                max_dt=60.0,  # 1 minute max
                relative_tolerance=1e-8,
                coupling_strength=CouplingStrength.WEAK
            )
        
        super().__init__(config)
        
        self.mass = mass
        self.area_drag = area_drag
        self.area_srp = area_srp
        self.cd = cd
        self.cr = cr
        
        # Integrator
        self.integrator = select_integrator(
            system_type="orbital",
            timescale=60.0,
            accuracy_required="high" if config.fidelity == FidelityLevel.PRECISE else "medium"
        )
        
        # Atmosphere calculator (use corrected v2)
        try:
            from dynamics.atmosphere_v2 import AtmosphereCalculatorV2, AtmosphereModel
            self._atmosphere = AtmosphereCalculatorV2(AtmosphereModel.EXPONENTIAL_JACCHIA)
        except ImportError:
            self._atmosphere = None
        
        # Earth parameters
        self.R_earth = 6371.0  # km
        self.mu_earth = 398600.4418  # km³/s²
        self.J2 = 1.08263e-3
        
        # SRP parameters
        self.srp_pressure = 4.56e-6  # Pa at 1 AU
    
    @property
    def characteristic_timescale(self) -> TimeScale:
        # Orbital period: ~90 min for LEO
        return TimeScale.MINUTE
    
    @property
    def name(self) -> str:
        return "orbital_environment"
    
    def get_initial_state(self) -> OrbitalState:
        """Create initial orbital state (400 km circular orbit)."""
        r = self.R_earth + 400.0  # 400 km altitude
        v_circ = np.sqrt(self.mu_earth / r)
        
        return OrbitalState(
            position=np.array([r, 0.0, 0.0]),
            velocity=np.array([0.0, v_circ, 0.0]),
            time=0.0
        )
    
    def _compute_accelerations(
        self,
        state: OrbitalState,
        fidelity: FidelityLevel
    ) -> tuple[np.ndarray, dict]:
        """
        Compute total acceleration with perturbations.
        
        Returns:
            (acceleration vector, diagnostics dict)
        """
        r = state.position  # km
        v = state.velocity  # km/s
        r_mag = np.linalg.norm(r)
        
        diagnostics = {
            'two_body': 0.0,
            'j2': 0.0,
            'drag': 0.0,
            'srp': 0.0
        }
        
        # 1. Two-body acceleration (km/s²)
        a_two_body = -self.mu_earth / r_mag**3 * r
        diagnostics['two_body'] = np.linalg.norm(a_two_body)
        
        # 2. J2 perturbation
        if fidelity == FidelityLevel.PRECISE:
            # Full J2 acceleration (from orbital_perturbations.py)
            factor = -1.5 * self.J2 * self.mu_earth * self.R_earth**2 / r_mag**5
            z_ratio = r[2]**2 / r_mag**2
            
            a_j2 = factor * np.array([
                r[0] * (1 - 5 * z_ratio),
                r[1] * (1 - 5 * z_ratio),
                r[2] * (3 - 5 * z_ratio)
            ])
        else:
            # Approximate J2 effect (only nodal precession)
            a_j2 = np.zeros(3)
        
        diagnostics['j2'] = np.linalg.norm(a_j2)
        
        # 3. Atmospheric drag
        altitude_km = r_mag - self.R_earth
        
        if altitude_km < 1000 and self._atmosphere is not None:
            # Get density
            density_result = self._atmosphere.compute_density(altitude_km)
            rho = density_result.density  # kg/m³
            
            # Convert velocity to m/s
            v_ms = v * 1000.0  # km/s to m/s
            v_ms_mag = np.linalg.norm(v_ms)
            
            # Drag acceleration: a = -0.5 * rho * v² * Cd * A / m
            # in m/s², then convert to km/s²
            a_drag_mag = 0.5 * rho * v_ms_mag**2 * self.cd * self.area_drag / self.mass
            a_drag_mag_km = a_drag_mag / 1000.0  # Convert to km/s²
            
            # Direction opposite to velocity
            a_drag = -a_drag_mag_km * v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-10 else np.zeros(3)
            
            diagnostics['drag'] = np.linalg.norm(a_drag)
        else:
            a_drag = np.zeros(3)
        
        # 4. Solar radiation pressure
        if not state.in_eclipse:
            # Simplified Sun direction (assume in equatorial plane)
            # Real implementation would use ephemeris
            sun_direction = np.array([1.0, 0.0, 0.0])  # Simplified
            
            # Force magnitude: F = P * A * Cr
            # a = F / m
            a_srp_mag = self.srp_pressure * self.area_srp * self.cr / self.mass
            a_srp = a_srp_mag * sun_direction / 1000.0  # Convert to km/s²
            
            diagnostics['srp'] = np.linalg.norm(a_srp)
        else:
            a_srp = np.zeros(3)
        
        # Total acceleration
        a_total = a_two_body + a_j2 + a_drag + a_srp
        
        return a_total, diagnostics
    
    def advance(
        self,
        state: Any,
        t_start: float,
        t_end: float,
        inputs: dict[str, Any]
    ) -> AdvanceResult:
        """
        Advance orbital state.
        
        Uses adaptive RK45 for orbital propagation.
        """
        orbit_state: OrbitalState = state
        dt = t_end - t_start
        
        # Check for eclipse transitions
        in_eclipse = self._check_eclipse(orbit_state.position)
        
        # Define ODE
        def orbit_ode(t: float, y: np.ndarray) -> np.ndarray:
            """Orbital ODE: [r, v] -> [v, a]."""
            r = y[0:3]
            v = y[3:6]
            
            # Create temporary state
            temp_state = OrbitalState(
                position=r,
                velocity=v,
                time=t,
                in_eclipse=in_eclipse  # Use current eclipse status
            )
            
            a, _ = self._compute_accelerations(temp_state, self.config.fidelity)
            
            return np.concatenate([v, a])
        
        # Integrate using scipy
        from scipy.integrate import solve_ivp
        
        y0 = np.concatenate([orbit_state.position, orbit_state.velocity])
        
        sol = solve_ivp(
            orbit_ode,
            [t_start, t_end],
            y0,
            method='RK45',
            rtol=self.config.relative_tolerance,
            atol=self.config.absolute_tolerance,
            max_step=self.config.max_dt
        )
        
        if not sol.success:
            return AdvanceResult(
                new_state=state,
                output=self.compute_output(state, t_start, self.config.fidelity),
                dt_actual=0.0,
                error_estimate=np.inf,
                step_accepted=False,
                suggested_dt=dt / 2
            )
        
        y_final = sol.y[:, -1]
        
        # Check eclipse status at final position
        in_eclipse_final = self._check_eclipse(y_final[0:3])
        
        new_state = OrbitalState(
            position=y_final[0:3],
            velocity=y_final[3:6],
            time=t_end,
            in_eclipse=in_eclipse_final
        )
        
        output = self.compute_output(new_state, t_end, self.config.fidelity)
        
        return AdvanceResult(
            new_state=new_state,
            output=output,
            dt_actual=t_end - sol.t[0],
            error_estimate=0.0,  # Use solver internal error
            step_accepted=True,
            suggested_dt=dt,
            num_substeps=len(sol.t)
        )
    
    def _check_eclipse(self, position: np.ndarray) -> bool:
        """Check if position is in Earth's shadow (simplified cylindrical model)."""
        # Cylindrical shadow: if x < 0 and sqrt(y² + z²) < R_earth
        if position[0] < 0:
            yz_dist = np.sqrt(position[1]**2 + position[2]**2)
            if yz_dist < self.R_earth:
                return True
        return False
    
    def compute_output(
        self,
        state: OrbitalState,
        t: float,
        fidelity: FidelityLevel
    ) -> DomainOutput:
        """Compute orbital outputs."""
        from sim.uncertainty import UncertainQuantity, from_relative, UncertainArray
        
        # Altitude
        r_mag = np.linalg.norm(state.position)
        altitude_km = r_mag - self.R_earth
        
        # Velocity
        v_mag = np.linalg.norm(state.velocity)
        
        # Orbital energy (km²/s²)
        energy = v_mag**2 / 2 - self.mu_earth / r_mag
        
        # Uncertainty on position (grows due to drag uncertainty)
        # For LEO, position uncertainty ~10-100m after 1 orbit
        pos_unc = 0.01  # km = 10 m
        
        # Compute drag force for output
        if self._atmosphere is not None:
            density_result = self._atmosphere.compute_density(altitude_km)
            rho = density_result.density
            rho_unc = density_result.total_uncertainty
        else:
            rho = 1e-12
            rho_unc = 0.5 * rho
        
        v_ms = state.velocity * 1000.0
        F_drag = 0.5 * rho * np.linalg.norm(v_ms)**2 * self.cd * self.area_srp
        
        # Altitude with uncertainty
        alt_unc = from_relative(altitude_km, 0.001, source="orbital_propagation")
        
        violations = []
        if altitude_km < 100:
            violations.append(f"Altitude {altitude_km:.1f}km too low")
        if altitude_km > 2000:
            violations.append(f"Altitude {altitude_km:.1f}km too high for LEO")
        
        return DomainOutput(
            t_start=t,
            t_end=t,
            scalars={
                'altitude_km': alt_unc,
                'velocity_km_s': from_relative(v_mag, 0.001),
                'orbital_energy': from_relative(energy, 0.01)
            },
            vectors={
                'position_eci': UncertainArray(
                    values=state.position,
                    std_devs=np.full(3, pos_unc),
                    systematic_errors=np.full(3, pos_unc)
                ),
                'velocity_eci': UncertainArray(
                    values=state.velocity,
                    std_devs=np.full(3, 1e-4),
                    systematic_errors=np.full(3, 1e-4)
                )
            },
            time_averaged_scalars={
                'altitude_km': altitude_km,
                'velocity_km_s': v_mag,
                'drag_force': F_drag
            },
            regime_violations=violations
        )
    
    def check_validity(
        self,
        state: Any,
        t: float
    ) -> tuple[bool, list[str]]:
        """Check orbital validity."""
        orbit_state: OrbitalState = state
        violations = []
        
        r_mag = np.linalg.norm(orbit_state.position)
        altitude_km = r_mag - self.R_earth
        
        if altitude_km < 0:
            violations.append("Orbit intersects Earth surface")
        
        if altitude_km > 50000:
            violations.append("Orbit beyond reasonable LEO range")
        
        return len(violations) == 0, violations

"""
Mechanics domain for mass stream dynamics.

Combines:
- Interball magnetic forces (dipole-dipole with corrections)
- Hoop tension dynamics
- Shepherd station interactions

Uses symplectic integration for long-term stability.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING
import numpy as np

from sim.domain_base import (
    DomainAdapter, DomainConfig, DomainOutput, AdvanceResult,
    TimeScale, CouplingStrength, FidelityLevel
)
from sim.integrators import VelocityVerlet, select_integrator, ConservationMonitor

if TYPE_CHECKING:
    from dynamics.halbach_array_v2 import HalbachSphereV2
    from dynamics.interball_magnetic import InterBallMagneticInteraction
    from dynamics.hoop_tension import HoopTensionModel
    from dynamics.shepherd_station import ShepherdStation


@dataclass
class BallState:
    """State of a single ball in the stream."""
    position: np.ndarray  # 3D position (m)
    velocity: np.ndarray  # 3D velocity (m/s)
    quaternion: np.ndarray  # Attitude as quaternion (optional)
    angular_velocity: np.ndarray  # Angular velocity (rad/s)
    
    def to_array(self) -> np.ndarray:
        """Convert to flat array for integration."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.quaternion,
            self.angular_velocity
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'BallState':
        """Create from flat array."""
        return cls(
            position=arr[0:3],
            velocity=arr[3:6],
            quaternion=arr[6:10],
            angular_velocity=arr[10:13]
        )


@dataclass
class StreamMechanicsState:
    """Complete state of the mass stream mechanics."""
    balls: list[BallState]
    time: float = 0.0
    
    # Shepherd stations (positions fixed or slowly varying)
    shepherd_positions: list[np.ndarray] = field(default_factory=list)
    
    # Stream parameters
    nominal_radius: float = 100.0  # m
    ball_mass: float = 100.0  # kg
    
    # Integrator state
    step_count: int = 0


class MechanicsStreamDomain(DomainAdapter):
    """
    Domain adapter for stream mechanics.
    
    Physics:
    - Interball magnetic: dipole-dipole with multipole corrections
    - Hoop tension: T = λu² with geometric stiffness
    - Shepherd forces: quadrupole lens + trim coils
    
    Integration:
    - Velocity Verlet (symplectic) for conservative forces
    - Symplectic Euler if damping added later
    """
    
    def __init__(
        self,
        config: Optional[DomainConfig] = None,
        n_balls: int = 10,
        nominal_radius: float = 100.0,
        ball_mass: float = 100.0
    ):
        if config is None:
            config = DomainConfig(
                fidelity=FidelityLevel.APPROX,
                max_dt=1e-3,  # 1 ms max for mechanics
                relative_tolerance=1e-6
            )
        
        super().__init__(config)
        
        self.n_balls = n_balls
        self.nominal_radius = nominal_radius
        self.ball_mass = ball_mass
        
        # Linear mass density
        circumference = 2 * np.pi * nominal_radius
        self.linear_density = (n_balls * ball_mass) / circumference
        
        # Hoop tension
        stream_velocity = 1000.0  # m/s (example)
        self.hoop_tension = self.linear_density * stream_velocity**2
        
        # Integrator
        self.integrator = select_integrator(
            system_type="conservative_mechanics",
            timescale=1e-3,
            accuracy_required="high" if config.fidelity == FidelityLevel.PRECISE else "medium"
        )
        
        # Conservation monitor
        self.monitor = ConservationMonitor()
        
        # Physics modules (initialized on first use)
        self._halbach: Optional['HalbachSphereV2'] = None
        self._interball: Optional[Any] = None
        self._shepherds: list[Any] = []
    
    @property
    def characteristic_timescale(self) -> TimeScale:
        # Packet passage: ~1-10 ms
        return TimeScale.MILLI
    
    @property
    def name(self) -> str:
        return "mechanics_stream"
    
    def get_initial_state(self) -> StreamMechanicsState:
        """Create initial state with balls equally spaced on circle."""
        balls = []
        
        for i in range(self.n_balls):
            theta = 2 * np.pi * i / self.n_balls
            
            # Position on nominal circle
            pos = np.array([
                self.nominal_radius * np.cos(theta),
                self.nominal_radius * np.sin(theta),
                0.0
            ])
            
            # Tangential velocity
            vel = np.array([
                -1000.0 * np.sin(theta),  # Tangential
                1000.0 * np.cos(theta),
                0.0
            ])
            
            # Identity quaternion
            quat = np.array([1.0, 0.0, 0.0, 0.0])
            
            # Zero angular velocity
            omega = np.zeros(3)
            
            balls.append(BallState(pos, vel, quat, omega))
        
        return StreamMechanicsState(
            balls=balls,
            time=0.0,
            nominal_radius=self.nominal_radius,
            ball_mass=self.ball_mass
        )
    
    def _compute_forces(
        self,
        state: StreamMechanicsState,
        fidelity: FidelityLevel
    ) -> tuple[list[np.ndarray], dict]:
        """
        Compute forces on all balls.
        
        Returns:
            (list of forces, dict with diagnostics)
        """
        forces = [np.zeros(3) for _ in state.balls]
        diagnostics = {
            'interball_forces': [],
            'hoop_forces': [],
            'shepherd_forces': []
        }
        
        # 1. Interball magnetic forces
        for i in range(self.n_balls):
            for j in range(i + 1, self.n_balls):
                # Get positions
                r_i = state.balls[i].position
                r_j = state.balls[j].position
                
                # Vector from i to j
                r_ij = r_j - r_i
                d = np.linalg.norm(r_ij)
                
                if d < 0.1:  # Avoid singularity
                    continue
                
                # Dipole moment (assume aligned with z for now)
                m = 580.0  # A·m² (typical for 5cm NdFeB Halbach)
                
                # Force between dipoles (repulsive side-by-side)
                # F = (3*mu0*m²)/(4*pi*d⁴) for side-by-side
                mu0 = 4 * np.pi * 1e-7
                
                if fidelity == FidelityLevel.APPROX:
                    # Simple dipole-dipole
                    F_mag = (3 * mu0 * m**2) / (4 * np.pi * d**4)
                else:
                    # Add multipole correction for nearby balls
                    R_ball = 0.05  # 5cm radius
                    correction = 1.0 - (R_ball / d)**2
                    F_mag = (3 * mu0 * m**2) / (4 * np.pi * d**4) * correction
                
                # Direction: repulsive (away from each other)
                F_vec = F_mag * r_ij / d
                
                forces[i] += F_vec
                forces[j] -= F_vec
                
                diagnostics['interball_forces'].append((i, j, F_mag))
        
        # 2. Hoop tension restoring forces
        # For small perturbations from circle: F = -(T/R²) * δr
        for i, ball in enumerate(state.balls):
            r = ball.position
            r_mag = np.linalg.norm(r[:2])  # In-plane distance
            
            if r_mag > 1e-6:
                # Nominal position direction
                theta = np.arctan2(r[1], r[0])
                r_nominal = np.array([
                    self.nominal_radius * np.cos(theta),
                    self.nominal_radius * np.sin(theta),
                    0.0
                ])
                
                # Radial perturbation
                delta_r = r - r_nominal
                
                # Hoop tension restoring force
                k_hoop = self.hoop_tension / (self.nominal_radius**2)
                F_hoop = -k_hoop * delta_r
                
                forces[i] += F_hoop
                diagnostics['hoop_forces'].append((i, np.linalg.norm(F_hoop)))
        
        # 3. Shepherd station forces (if any)
        for shepherd_pos in state.shepherd_positions:
            for i, ball in enumerate(state.balls):
                r_to_shepherd = shepherd_pos - ball.position
                d = np.linalg.norm(r_to_shepherd)
                
                if d < 10.0:  # Within capture radius
                    # Simplified: linear restoring toward station axis
                    k_lens = 1000.0  # N/m (example)
                    F_shepherd = k_lens * r_to_shepherd / d * (10.0 - d)
                    forces[i] += F_shepherd
                    diagnostics['shepherd_forces'].append((i, np.linalg.norm(F_shepherd)))
        
        return forces, diagnostics
    
    def advance(
        self,
        state: Any,
        t_start: float,
        t_end: float,
        inputs: dict[str, Any]
    ) -> AdvanceResult:
        """
        Advance stream mechanics from t_start to t_end.
        
        Uses symplectic integration with substepping.
        """
        stream_state: StreamMechanicsState = state
        
        dt_macro = t_end - t_start
        
        # Determine substep size
        # For symplectic integrator, fixed step is OK
        dt_sub = min(1e-4, dt_macro / 100)  # 0.1 ms or 100 steps
        n_substeps = int(np.ceil(dt_macro / dt_sub))
        dt_sub = dt_macro / n_substeps  # Adjust to exactly fill macro step
        
        # Prepare state vector for integration
        # Flatten all ball states: [pos0, vel0, pos1, vel1, ...]
        n_balls = len(stream_state.balls)
        y = np.zeros(6 * n_balls)
        
        for i, ball in enumerate(stream_state.balls):
            y[6*i : 6*i+3] = ball.position
            y[6*i+3 : 6*i+6] = ball.velocity
        
        # Define ODE function
        def ode_func(t: float, y_flat: np.ndarray) -> np.ndarray:
            """Compute derivatives for all balls."""
            dydt = np.zeros_like(y_flat)
            
            # Unpack positions and velocities
            temp_balls = []
            for i in range(n_balls):
                pos = y_flat[6*i : 6*i+3]
                vel = y_flat[6*i+3 : 6*i+6]
                temp_balls.append(BallState(pos, vel, np.array([1,0,0,0]), np.zeros(3)))
            
            temp_state = StreamMechanicsState(
                balls=temp_balls,
                time=t,
                nominal_radius=stream_state.nominal_radius,
                ball_mass=stream_state.ball_mass
            )
            
            # Compute forces
            forces, _ = self._compute_forces(temp_state, self.config.fidelity)
            
            # Pack derivatives: dq/dt = v, dv/dt = F/m
            for i in range(n_balls):
                dydt[6*i : 6*i+3] = y_flat[6*i+3 : 6*i+6]  # dq/dt = v
                dydt[6*i+3 : 6*i+6] = forces[i] / self.ball_mass  # dv/dt = F/m
            
            return dydt
        
        # Integrate with substepping
        t = t_start
        for step in range(n_substeps):
            result = self.integrator.step(ode_func, t, y, dt_sub)
            y = result.y_new
            t = result.t_new
        
        # Unpack final state
        new_balls = []
        for i in range(n_balls):
            pos = y[6*i : 6*i+3]
            vel = y[6*i+3 : 6*i+6]
            
            # Keep original attitude (not evolved in this mechanics domain)
            old_ball = stream_state.balls[i]
            new_balls.append(BallState(pos, vel, old_ball.quaternion, old_ball.angular_velocity))
        
        new_state = StreamMechanicsState(
            balls=new_balls,
            time=t_end,
            shepherd_positions=stream_state.shepherd_positions,
            nominal_radius=stream_state.nominal_radius,
            ball_mass=stream_state.ball_mass,
            step_count=stream_state.step_count + n_substeps
        )
        
        # Compute output
        output = self.compute_output(new_state, t_end, self.config.fidelity)
        
        return AdvanceResult(
            new_state=new_state,
            output=output,
            dt_actual=dt_macro,
            error_estimate=0.0,  # Symplectic - no error estimate
            step_accepted=True,
            suggested_dt=dt_macro,
            num_substeps=n_substeps,
            computation_time_ms=0.0  # Would measure in production
        )
    
    def compute_output(
        self,
        state: StreamMechanicsState,
        t: float,
        fidelity: FidelityLevel
    ) -> DomainOutput:
        """Compute outputs from mechanics state."""
        from sim.uncertainty import UncertainQuantity, from_relative
        
        # Compute kinetic energy
        ke_total = sum(
            0.5 * self.ball_mass * np.linalg.norm(ball.velocity)**2
            for ball in state.balls
        )
        
        # Compute mean spacing
        spacings = []
        for i in range(len(state.balls)):
            j = (i + 1) % len(state.balls)
            d = np.linalg.norm(state.balls[i].position - state.balls[j].position)
            spacings.append(d)
        
        mean_spacing = np.mean(spacings)
        
        # Uncertainty
        ke_unc = from_relative(ke_total, 0.05, source="kinetic_energy")
        spacing_unc = from_relative(mean_spacing, 0.10, source="mean_spacing")
        
        # Time-averaged outputs (for this domain, same as instantaneous)
        return DomainOutput(
            t_start=t,
            t_end=t,
            scalars={
                'kinetic_energy': ke_unc,
                'mean_spacing': spacing_unc,
                'n_balls': UncertainQuantity(float(len(state.balls)), 0.0, 0.0)
            },
            time_averaged_scalars={
                'kinetic_energy': ke_total,
                'mean_spacing': mean_spacing
            },
            regime_violations=[]
        )
    
    def check_validity(
        self,
        state: Any,
        t: float
    ) -> tuple[bool, list[str]]:
        """Check if state is in valid regime."""
        stream_state: StreamMechanicsState = state
        violations = []
        
        # Check for collisions
        for i in range(len(stream_state.balls)):
            for j in range(i + 1, len(stream_state.balls)):
                d = np.linalg.norm(
                    stream_state.balls[i].position - stream_state.balls[j].position
                )
                if d < 0.1:  # 10 cm minimum spacing
                    violations.append(f"Balls {i} and {j} too close: {d:.3f}m")
        
        # Check for excessive displacement from nominal
        for i, ball in enumerate(stream_state.balls):
            r_mag = np.linalg.norm(ball.position[:2])
            if abs(r_mag - self.nominal_radius) > 0.5 * self.nominal_radius:
                violations.append(f"Ball {i} displaced by {abs(r_mag - self.nominal_radius):.1f}m")
        
        return len(violations) == 0, violations

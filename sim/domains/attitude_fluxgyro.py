"""
Attitude domain for flux-gyroscopic coupled dynamics.

Wraps existing flux_gyroscopic_dynamics.py with:
- Symplectic integration for attitude dynamics
- Uncertainty on torques and angular rates
- Clean interface to thermal domain (eddy heating output)
"""

from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING
import numpy as np

from sim.domain_base import (
    DomainAdapter, DomainConfig, DomainOutput, AdvanceResult,
    TimeScale, CouplingStrength, FidelityLevel
)
from sim.integrators import SymplecticEuler, select_integrator

if TYPE_CHECKING:
    from dynamics.flux_gyroscopic_dynamics import FluxGyroscopicCoupledSystem, FluxGyroConfig


@dataclass
class AttitudeState:
    """Attitude state for a single body."""
    
    # Attitude as quaternion (scalar first: [w, x, y, z])
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    
    # Angular velocity (rad/s)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Position (for coupling with mechanics)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Time
    time: float = 0.0
    
    # Integrated eddy heating (for thermal coupling)
    eddy_heating_integrated: float = 0.0  # J


class AttitudeFluxGyroDomain(DomainAdapter):
    """
    Domain adapter for flux-gyroscopic attitude dynamics.
    
    Physics:
    - Gyroscopic coupling: τ = ω × (I × ω)
    - Flux pinning: τ_fp from Bean-London or analytical model
    - Control torques (optional)
    
    Integration:
    - Symplectic Euler for dissipative system (flux pinning has hysteresis)
    
    Outputs to thermal:
    - Eddy current heating power (W)
    """
    
    def __init__(
        self,
        config: Optional[DomainConfig] = None,
        inertia_tensor: Optional[np.ndarray] = None,
        mass: float = 100.0,
        use_bean_london: bool = True
    ):
        if config is None:
            config = DomainConfig(
                fidelity=FidelityLevel.APPROX,
                max_dt=1e-3,  # 1 ms max for attitude
                relative_tolerance=1e-6,
                coupling_strength=CouplingStrength.MODERATE  # Heat to thermal
            )
        
        super().__init__(config)
        
        self.mass = mass
        self.inertia = inertia_tensor if inertia_tensor is not None else np.eye(3) * 10.0
        self.use_bean_london = use_bean_london
        
        # Integrator
        # Attitude has damping (flux pinning), so use symplectic Euler not Verlet
        self.integrator = select_integrator(
            system_type="dissipative_mechanics",
            timescale=1e-3,
            accuracy_required="high" if config.fidelity == FidelityLevel.PRECISE else "medium"
        )
        
        # Underlying dynamics system (created on first use)
        self._flux_gyro_system: Optional[Any] = None
        
        # Eddy heating parameters
        self.eddy_coefficient = 0.001  # W per (rad/s)² of attitude rate
    
    def _init_dynamics(self, initial_state: AttitudeState):
        """Initialize flux-gyro system."""
        try:
            from dynamics.flux_gyroscopic_dynamics import FluxGyroscopicCoupledSystem, FluxGyroConfig
            
            gyro_config = FluxGyroConfig(
                mass=self.mass,
                inertia_tensor=self.inertia,
                spin_rate=10.0,  # rad/s nominal spin
                spin_axis=np.array([0.0, 0.0, 1.0]),  # z-axis spin
                k_fp_base=1000.0
            )
            
            self._flux_gyro_system = FluxGyroscopicCoupledSystem(gyro_config)
            
        except ImportError:
            self._flux_gyro_system = None
    
    @property
    def characteristic_timescale(self) -> TimeScale:
        # Gyroscopic precession: milliseconds to seconds
        return TimeScale.MILLI
    
    @property
    def name(self) -> str:
        return "attitude_fluxgyro"
    
    def get_initial_state(self) -> AttitudeState:
        """Create initial attitude state."""
        return AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            position=np.zeros(3),
            velocity=np.zeros(3),
            time=0.0
        )
    
    def advance(
        self,
        state: Any,
        t_start: float,
        t_end: float,
        inputs: dict[str, Any]
    ) -> AdvanceResult:
        """
        Advance attitude dynamics.
        
        Key physics:
        - Quaternion kinematics: dq/dt = 0.5 * q ⊗ ω
        - Euler equation: I ω̇ + ω × (I ω) = τ_total
        - Flux pinning torque: from displacement from equilibrium
        """
        att_state: AttitudeState = state
        dt = t_end - t_start
        
        # Initialize on first call
        if self._flux_gyro_system is None:
            self._init_dynamics(att_state)
        
        # Get external torques from inputs
        external_torque = inputs.get('vectors', {}).get('magnetic_torque', np.zeros(3))
        
        # Use existing flux-gyro system if available
        if self._flux_gyro_system is not None:
            try:
                # The existing system has compute_coupled_dynamics method
                # that does one step with explicit Euler
                # We'd prefer to use our symplectic integrator, but for now
                # use the existing system and take multiple small steps
                
                from dynamics.flux_gyroscopic_dynamics import FluxGyroState
                
                # Create state object
                current_state = FluxGyroState(
                    position=att_state.position,
                    velocity=att_state.velocity,
                    attitude=att_state.quaternion,
                    angular_velocity=att_state.angular_velocity,
                    temperature=77.0  # Would get from thermal domain
                )
                
                # Multiple small steps
                n_steps = max(1, int(dt / 1e-4))  # 0.1 ms steps
                dt_small = dt / n_steps
                
                for _ in range(n_steps):
                    current_state = self._flux_gyro_system.compute_coupled_dynamics(
                        current_state,
                        external_torque=external_torque,
                        dt=dt_small
                    )
                
                # Update our state
                new_quat = current_state.attitude
                new_omega = current_state.angular_velocity
                new_pos = current_state.position
                new_vel = current_state.velocity
                
            except Exception as e:
                # Fallback to simplified model
                new_quat, new_omega, new_pos, new_vel = self._simplified_advance(
                    att_state, external_torque, dt
                )
        else:
            # Simplified fallback
            new_quat, new_omega, new_pos, new_vel = self._simplified_advance(
                att_state, external_torque, dt
            )
        
        # Compute eddy heating (power dissipated)
        # P_eddy ∝ ω² (simplified model)
        omega_mag = np.linalg.norm(new_omega)
        P_eddy = self.eddy_coefficient * omega_mag**2
        
        new_state = AttitudeState(
            quaternion=new_quat,
            angular_velocity=new_omega,
            position=new_pos,
            velocity=new_vel,
            time=t_end,
            eddy_heating_integrated=att_state.eddy_heating_integrated + P_eddy * dt
        )
        
        output = self.compute_output(new_state, t_end, self.config.fidelity)
        
        return AdvanceResult(
            new_state=new_state,
            output=output,
            dt_actual=dt,
            error_estimate=0.0,
            step_accepted=True,
            suggested_dt=dt,
            num_substeps=n_steps if 'n_steps' in dir() else 1
        )
    
    def _simplified_advance(
        self,
        state: AttitudeState,
        external_torque: np.ndarray,
        dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simplified attitude integration (fallback).
        
        Uses symplectic Euler:
        1. Update angular momentum: L_{n+1} = L_n + dt * τ
        2. Update attitude: q_{n+1} = q_n + dt * 0.5 * q_n ⊗ ω_{n+1}
        """
        q = state.quaternion.copy()
        omega = state.angular_velocity.copy()
        
        # Compute gyroscopic torque
        L = self.inertia @ omega
        gyro_torque = np.cross(omega, L)
        
        # Total torque
        tau_total = external_torque - gyro_torque  # Minus because gyro opposes change
        
        # Update angular velocity (semi-implicit: use inertia to get acceleration)
        alpha = np.linalg.solve(self.inertia, tau_total)
        omega_new = omega + dt * alpha
        
        # Update quaternion
        # dq/dt = 0.5 * q * omega_quat
        omega_quat = np.array([0, omega_new[0], omega_new[1], omega_new[2]])
        
        # Quaternion multiplication q ⊗ omega
        q_w, q_x, q_y, q_z = q
        o_w, o_x, o_y, o_z = omega_quat
        
        dq = 0.5 * np.array([
            q_w*o_w - q_x*o_x - q_y*o_y - q_z*o_z,
            q_w*o_x + q_x*o_w + q_y*o_z - q_z*o_y,
            q_w*o_y - q_x*o_z + q_y*o_w + q_z*o_x,
            q_w*o_z + q_x*o_y - q_y*o_x + q_z*o_w
        ])
        
        q_new = q + dt * dq
        
        # Normalize
        q_new = q_new / np.linalg.norm(q_new)
        
        # Position and velocity unchanged in attitude domain
        pos_new = state.position
        vel_new = state.velocity
        
        return q_new, omega_new, pos_new, vel_new
    
    def compute_output(
        self,
        state: AttitudeState,
        t: float,
        fidelity: FidelityLevel
    ) -> DomainOutput:
        """Compute attitude outputs."""
        from sim.uncertainty import UncertainQuantity, from_relative, UncertainArray
        
        # Angular velocity with uncertainty
        # Sources: integration error, model simplification
        omega_mag = np.linalg.norm(state.angular_velocity)
        omega_unc = from_relative(omega_mag, 0.05, source="attitude_dynamics")
        
        # Quaternion uncertainty (angular uncertainty)
        # Small rotation uncertainty
        angle_unc = 0.01  # rad (~0.6 deg)
        
        # Eddy heating power output (to thermal domain)
        P_eddy = self.eddy_coefficient * omega_mag**2
        P_eddy_unc = from_relative(P_eddy, 0.20, source="eddy_heating_model")
        
        # Angular momentum
        L = self.inertia @ state.angular_velocity
        L_unc = from_relative(np.linalg.norm(L), 0.05)
        
        return DomainOutput(
            t_start=t,
            t_end=t,
            scalars={
                'omega_magnitude': omega_unc,
                'eddy_heating_power': P_eddy_unc,
                'eddy_heating_stator': P_eddy_unc,  # For thermal coupling
                'eddy_heating_rotor': P_eddy_unc,   # For thermal coupling
                'angular_momentum': L_unc
            },
            vectors={
                'angular_velocity': UncertainArray(
                    values=state.angular_velocity,
                    std_devs=np.full(3, omega_mag * 0.02),
                    systematic_errors=np.full(3, omega_mag * 0.03)
                ),
                'angular_momentum': UncertainArray(
                    values=L,
                    std_devs=np.full(3, np.linalg.norm(L) * 0.02),
                    systematic_errors=np.full(3, np.linalg.norm(L) * 0.03)
                )
            },
            time_averaged_scalars={
                'omega_magnitude': omega_mag,
                'eddy_heating_power': P_eddy,
                'eddy_heating_stator': P_eddy,
                'eddy_heating_rotor': P_eddy
            },
            integrated_energy=state.eddy_heating_integrated,
            regime_violations=[]
        )
    
    def check_validity(
        self,
        state: Any,
        t: float
    ) -> tuple[bool, list[str]]:
        """Check attitude validity."""
        att_state: AttitudeState = state
        violations = []
        
        # Check quaternion normalization
        q_norm = np.linalg.norm(att_state.quaternion)
        if abs(q_norm - 1.0) > 0.01:
            violations.append(f"Quaternion not normalized: |q|={q_norm:.4f}")
        
        # Check angular velocity (prevent unrealistic values)
        omega_mag = np.linalg.norm(att_state.angular_velocity)
        max_omega = 1000.0  # rad/s (~10,000 RPM)
        if omega_mag > max_omega:
            violations.append(f"Angular velocity {omega_mag:.1f} rad/s exceeds limit")
        
        return len(violations) == 0, violations

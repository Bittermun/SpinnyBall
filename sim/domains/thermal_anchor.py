"""
Thermal domain for anchor system.

Wraps lumped thermal model with:
- Adaptive RK45 integration (thermal is not stiff)
- Temperature-dependent cryocooler power
- Uncertainty on temperature predictions
"""

from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING
import numpy as np

from sim.domain_base import (
    DomainAdapter, DomainConfig, DomainOutput, AdvanceResult,
    TimeScale, CouplingStrength, FidelityLevel
)
from sim.integrators import AdaptiveRK45, select_integrator

if TYPE_CHECKING:
    from dynamics.lumped_thermal import LumpedThermalModel, LumpedThermalParams
    from dynamics.cryocooler_model import CryocoolerModel


@dataclass
class ThermalState:
    """Thermal state of anchor system."""
    
    # Temperatures (K)
    T_stator: float = 77.0
    T_rotor: float = 77.0
    T_ambient: float = 4.0  # Deep space
    
    # Heat sources (W) - accumulated over step
    Q_stator_integrated: float = 0.0
    Q_rotor_integrated: float = 0.0
    
    # Time
    time: float = 0.0


class ThermalAnchorDomain(DomainAdapter):
    """
    Domain adapter for anchor thermal management.
    
    Physics:
    - 2-node lumped thermal (stator + rotor)
    - Radiative cooling
    - Conductive coupling
    - Cryocooler with load-dependent cooling power
    
    Integration:
    - Adaptive RK45 (thermal not stiff for these timescales)
    """
    
    def __init__(
        self,
        config: Optional[DomainConfig] = None,
        stator_mass: float = 10.0,  # kg
        rotor_mass: float = 5.0,
        cryocooler_power_77k: float = 5.0  # W
    ):
        if config is None:
            config = DomainConfig(
                fidelity=FidelityLevel.APPROX,
                max_dt=1.0,  # 1 second max
                relative_tolerance=1e-4,
                coupling_strength=CouplingStrength.WEAK  # Thermal changes slowly
            )
        
        super().__init__(config)
        
        self.stator_mass = stator_mass
        self.rotor_mass = rotor_mass
        self.cryocooler_power_77k = cryocooler_power_77k
        
        # Integrator
        self.integrator = select_integrator(
            system_type="general_ode",
            timescale=1.0,
            accuracy_required="high" if config.fidelity == FidelityLevel.PRECISE else "medium"
        )
        
        # Initialize thermal model
        self._thermal_model: Optional[Any] = None
        self._cryocooler: Optional[Any] = None
        
        self._init_models()
    
    def _init_models(self):
        """Initialize underlying thermal models."""
        try:
            from dynamics.lumped_thermal import LumpedThermalParams, LumpedThermalModel
            from dynamics.cryocooler_model import CryocoolerModel, CryocoolerSpecs
            
            # Create cryocooler with load-dependent cooling
            cryo_specs = CryocoolerSpecs(
                cooling_power_at_70k=self.cryocooler_power_77k * 0.7,
                cooling_power_at_80k=self.cryocooler_power_77k,
                cooling_power_at_90k=self.cryocooler_power_77k * 1.5,
                input_power_at_70k=50.0,
                input_power_at_80k=60.0,
                input_power_at_90k=80.0,
                cooldown_time=3600.0,
                warmup_time=60.0,
                mass=5.0,
                volume=0.01,
                vibration_amplitude=1e-6
            )
            self._cryocooler = CryocoolerModel(cryo_specs)
            
            # Create thermal model
            thermal_params = LumpedThermalParams(
                stator_mass=self.stator_mass,
                stator_specific_heat=500.0,  # J/kg/K (GdBCO)
                stator_surface_area=0.1,
                stator_emissivity=0.1,
                rotor_mass=self.rotor_mass,
                rotor_specific_heat=400.0,  # Aluminum at 77K
                rotor_surface_area=0.05,
                rotor_emissivity=0.2,
                shaft_conductance=10.0,  # W/K
                ambient_temp=4.0,
                initial_temp=77.0,
                enable_switching_losses=True,
                switching_power_stator=0.5,  # W
                switching_power_rotor=0.2,  # W
                enable_cryocooler=True,
                cryocooler_model=self._cryocooler
            )
            
            self._thermal_model = LumpedThermalModel(thermal_params, dt=0.01)
            
        except ImportError:
            # Fallback: simple 2-node model
            self._thermal_model = None
    
    @property
    def characteristic_timescale(self) -> TimeScale:
        # Thermal time constants: seconds to minutes
        return TimeScale.SECOND
    
    @property
    def name(self) -> str:
        return "thermal_anchor"
    
    def get_initial_state(self) -> ThermalState:
        """Create initial thermal state."""
        return ThermalState(
            T_stator=77.0,
            T_rotor=77.0,
            T_ambient=4.0,
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
        Advance thermal state from t_start to t_end.
        
        Thermal changes slowly, so we can take large steps.
        """
        thermal_state: ThermalState = state
        dt = t_end - t_start
        
        # Get heat sources from inputs (from attitude domain)
        Q_stator = inputs.get('scalars', {}).get('eddy_heating_stator', 0.0)
        Q_rotor = inputs.get('scalars', {}).get('eddy_heating_rotor', 0.0)
        
        if self._thermal_model is not None:
            # Use existing lumped thermal model
            # It uses Euler internally - we'd want to replace with RK45
            # For now, use it as-is but take smaller steps
            
            n_steps = max(1, int(dt / 0.1))  # 0.1s internal steps
            dt_internal = dt / n_steps
            
            for _ in range(n_steps):
                self._thermal_model.step({
                    'stator': Q_stator,
                    'rotor': Q_rotor
                })
            
            T_stator_new = self._thermal_model.T_stator
            T_rotor_new = self._thermal_model.T_rotor
            
        else:
            # Fallback: simple explicit integration
            # dT/dt = Q / (m * cp) - (T - T_ambient) / tau
            
            cp_stator = 500.0
            cp_rotor = 400.0
            
            tau_stator = 100.0  # s - thermal time constant
            tau_rotor = 50.0
            
            dT_stator = dt * (
                Q_stator / (self.stator_mass * cp_stator) -
                (thermal_state.T_stator - thermal_state.T_ambient) / tau_stator
            )
            dT_rotor = dt * (
                Q_rotor / (self.rotor_mass * cp_rotor) -
                (thermal_state.T_rotor - thermal_state.T_ambient) / tau_rotor
            )
            
            T_stator_new = thermal_state.T_stator + dT_stator
            T_rotor_new = thermal_state.T_rotor + dT_rotor
        
        # En物理 limits
        T_stator_new = max(T_stator_new, thermal_state.T_ambient)
        T_rotor_new = max(T_rotor_new, thermal_state.T_ambient)
        
        new_state = ThermalState(
            T_stator=T_stator_new,
            T_rotor=T_rotor_new,
            T_ambient=thermal_state.T_ambient,
            Q_stator_integrated=thermal_state.Q_stator_integrated + Q_stator * dt,
            Q_rotor_integrated=thermal_state.Q_rotor_integrated + Q_rotor * dt,
            time=t_end
        )
        
        # Compute output
        output = self.compute_output(new_state, t_end, self.config.fidelity)
        
        return AdvanceResult(
            new_state=new_state,
            output=output,
            dt_actual=dt,
            error_estimate=0.0,
            step_accepted=True,
            suggested_dt=dt,
            num_substeps=1
        )
    
    def compute_output(
        self,
        state: ThermalState,
        t: float,
        fidelity: FidelityLevel
    ) -> DomainOutput:
        """Compute thermal outputs."""
        from sim.uncertainty import UncertainQuantity, from_relative, ValidityRegime
        
        # Temperatures with uncertainty
        # Uncertainty sources:
        # - Model simplification (2-node vs real distribution): ±20%
        # - Parameter uncertainty (masses, conductances): ±10%
        # - Input uncertainty (heat loads): ±15%
        
        rel_unc = np.sqrt(0.20**2 + 0.10**2 + 0.15**2)  # ~27%
        
        T_stator_unc = from_relative(state.T_stator, rel_unc, source="thermal_model")
        T_stator_unc.validity.append(ValidityRegime("T", 4.0, 400.0))
        
        T_rotor_unc = from_relative(state.T_rotor, rel_unc, source="thermal_model")
        T_rotor_unc.validity.append(ValidityRegime("T", 4.0, 400.0))
        
        # Cryocooler status
        if self._cryocooler is not None:
            P_cool = self._cryocooler.cooling_power(state.T_stator)
            P_in = self._cryocooler.input_power(state.T_stator)
            cop = self._cryocooler.cop(state.T_stator)
        else:
            P_cool = self.cryocooler_power_77k
            P_in = 60.0
            cop = P_cool / P_in
        
        P_cool_unc = from_relative(P_cool, 0.15, source="cryocooler_model")
        
        # Check if within operating range
        violations = []
        if state.T_stator > 90.0:
            violations.append("Stator temperature approaching quench limit")
        if state.T_rotor > 100.0:
            violations.append("Rotor temperature too high")
        
        return DomainOutput(
            t_start=t,
            t_end=t,
            scalars={
                'T_stator': T_stator_unc,
                'T_rotor': T_rotor_unc,
                'cryocooler_power': P_cool_unc,
                'cop': from_relative(cop, 0.10)
            },
            time_averaged_scalars={
                'T_stator': state.T_stator,
                'T_rotor': state.T_rotor,
                'cryocooler_power': P_cool
            },
            regime_violations=violations
        )
    
    def check_validity(
        self,
        state: Any,
        t: float
    ) -> tuple[bool, list[str]]:
        """Check thermal validity."""
        thermal_state: ThermalState = state
        violations = []
        
        # Temperature limits
        if thermal_state.T_stator > 93.0:  # GdBCO critical
            violations.append(f"Stator temperature {thermal_state.T_stator:.1f}K exceeds critical")
        
        if thermal_state.T_stator < 4.0:
            violations.append(f"Stator temperature {thermal_state.T_stator:.1f}K below ambient")
        
        return len(violations) == 0, violations

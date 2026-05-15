"""
Base classes for domain-specific physics adapters.

Each physics domain (mechanics, thermal, orbital, attitude) implements
DomainAdapter with a consistent interface for the macro-step scheduler.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sim.uncertainty import UncertainQuantity, UncertainArray


class FidelityLevel(Enum):
    """Two fidelity levels - practical and sufficient."""
    APPROX = auto()    # Analytical, fast (~1ms), for sweeps
    PRECISE = auto()   # Numerical, slower (~10-100ms), for validation


class TimeScale(Enum):
    """Characteristic timescales for domain classification."""
    FAST = 1e-6       # microseconds (packet passage)
    MILLI = 1e-3      # milliseconds (attitude dynamics)
    SECOND = 1.0      # seconds (thermal)
    MINUTE = 60.0     # minutes
    HOUR = 3600.0     # hours (orbital)
    DAY = 86400.0     # days


class CouplingStrength(Enum):
    """How tightly this domain couples to others."""
    WEAK = auto()     # Can hold other domain inputs constant over macro-step
    MODERATE = auto() # Needs interpolation of inputs
    STRONG = auto()   # Needs substepping within macro-step


@dataclass
class DomainOutput:
    """
    Standardized output from a physics domain.
    
    Each domain emits outputs that may be consumed by other domains.
    All quantities include uncertainty bounds.
    """
    # Time this output is valid for
    t_start: float
    t_end: float
    
    # Primary outputs with uncertainty
    scalars: dict[str, 'UncertainQuantity'] = field(default_factory=dict)
    vectors: dict[str, 'UncertainArray'] = field(default_factory=dict)
    
    # Averaged quantities over [t_start, t_end]
    # (for passing to slower domains)
    time_averaged_scalars: dict[str, float] = field(default_factory=dict)
    time_averaged_vectors: dict[str, np.ndarray] = field(default_factory=dict)
    
    # Integrated impulses/energy transfers
    integrated_impulse: Optional[np.ndarray] = None  # N·s
    integrated_energy: Optional[float] = None  # J
    
    # Validity flags
    regime_violations: list[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if all outputs are within valid regime."""
        return len(self.regime_violations) == 0


@dataclass  
class AdvanceResult:
    """Result of advancing a domain forward in time."""
    # New state (domain-specific)
    new_state: Any
    
    # Output produced during this step
    output: DomainOutput
    
    # Actual step size taken (may differ from requested)
    dt_actual: float
    
    # Error estimate (for adaptive stepping)
    error_estimate: float = 0.0
    
    # Whether step was accepted
    step_accepted: bool = True
    
    # Suggested next step size
    suggested_dt: Optional[float] = None
    
    # Diagnostics
    num_substeps: int = 0
    computation_time_ms: float = 0.0


@dataclass
class DomainConfig:
    """Configuration for a physics domain."""
    fidelity: FidelityLevel = FidelityLevel.APPROX
    
    # Time step constraints
    min_dt: float = 1e-9
    max_dt: float = 1.0
    
    # Adaptive stepping tolerance
    relative_tolerance: float = 1e-4
    absolute_tolerance: float = 1e-8
    
    # Output control
    save_history: bool = False
    history_interval: float = 0.1
    
    # Coupling behavior
    coupling_strength: CouplingStrength = CouplingStrength.WEAK


class DomainAdapter(ABC):
    """
    Abstract base for physics domain adapters.
    
    Each domain (mechanics, thermal, orbital, attitude) implements this
    interface so the scheduler can coordinate them.
    
    Key principle: Domains are loosely coupled. The scheduler passes
    time-averaged or interpolated inputs, not instantaneous coupling.
    """
    
    def __init__(self, config: DomainConfig):
        self.config = config
        self.history: list[tuple[float, Any]] = []  # (t, state) pairs
        self._last_output: Optional[DomainOutput] = None
    
    @property
    @abstractmethod
    def characteristic_timescale(self) -> TimeScale:
        """Return the characteristic timescale of this domain."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable domain name."""
        pass
    
    @abstractmethod
    def get_initial_state(self) -> Any:
        """Return initial state for this domain."""
        pass
    
    @abstractmethod
    def advance(
        self,
        state: Any,
        t_start: float,
        t_end: float,
        inputs: dict[str, Any]
    ) -> AdvanceResult:
        """
        Advance domain from t_start to t_end.
        
        Args:
            state: Current domain state
            t_start: Start time
            t_end: End time (requested)
            inputs: Dictionary of inputs from other domains
                   (held constant or interpolated as appropriate)
        
        Returns:
            AdvanceResult with new state and outputs
        """
        pass
    
    @abstractmethod
    def compute_output(
        self,
        state: Any,
        t: float,
        fidelity: FidelityLevel
    ) -> DomainOutput:
        """
        Compute outputs from current state.
        
        Called at output intervals to produce DomainOutput for other domains.
        """
        pass
    
    def check_validity(self, state: Any, t: float) -> tuple[bool, list[str]]:
        """
        Check if current state is in valid regime.
        
        Override in subclass to add domain-specific validity checks.
        
        Returns:
            (is_valid, list of violation messages)
        """
        return True, []
    
    def get_time_averaged_inputs(
        self,
        other_output: DomainOutput,
        t_start: float,
        t_end: float
    ) -> dict[str, Any]:
        """
        Extract time-averaged inputs from another domain's output.
        
        Default implementation uses pre-computed averages.
        Subclasses can override for domain-specific treatment.
        """
        return {
            'scalars': other_output.time_averaged_scalars.copy(),
            'vectors': other_output.time_averaged_vectors.copy(),
            'impulse': other_output.integrated_impulse,
            'energy': other_output.integrated_energy,
        }
    
    def save_state(self, t: float, state: Any):
        """Save state to history if enabled."""
        if self.config.save_history:
            self.history.append((t, state))
    
    def get_history(self) -> list[tuple[float, Any]]:
        """Get saved history."""
        return self.history.copy()


# Input specification for cross-domain coupling

@dataclass
class CoupledInput:
    """
    Specification for how one domain consumes output from another.
    
    Example:
        thermal_input = CoupledInput(
            source_domain='attitude',
            source_quantity='eddy_heating_power',
            averaging='time_average',  # vs 'instantaneous'
            coupling_type='weak'       # can hold constant over macro-step
        )
    """
    source_domain: str
    source_quantity: str
    averaging: str = 'time_average'  # 'time_average', 'instantaneous', 'integrated'
    coupling_type: CouplingStrength = CouplingStrength.WEAK
    
    def extract(self, source_output: DomainOutput) -> Any:
        """Extract the specified quantity from source output."""
        if self.averaging == 'time_average':
            if self.source_quantity in source_output.time_averaged_scalars:
                return source_output.time_averaged_scalars[self.source_quantity]
            elif self.source_quantity in source_output.time_averaged_vectors:
                return source_output.time_averaged_vectors[self.source_quantity]
        
        elif self.averaging == 'instantaneous':
            if self.source_quantity in source_output.scalars:
                return source_output.scalars[self.source_quantity].value
            elif self.source_quantity in source_output.vectors:
                return source_output.vectors[self.source_quantity].values
        
        elif self.averaging == 'integrated':
            if self.source_quantity == 'impulse':
                return source_output.integrated_impulse
            elif self.source_quantity == 'energy':
                return source_output.integrated_energy
        
        raise KeyError(f"Quantity '{self.source_quantity}' not found in output")

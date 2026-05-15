"""
Macro-step scheduler for multi-timescale physics.

Implements operator splitting: each domain advances independently
over a macro-step, then exchanges averaged/integrated quantities.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from collections import defaultdict
import numpy as np

from sim.domain_base import (
    DomainAdapter, DomainOutput, AdvanceResult,
    TimeScale, CouplingStrength, CoupledInput
)


@dataclass
class SchedulerConfig:
    """Configuration for macro-step scheduler."""
    
    # Macro step size (coarse time grid)
    macro_dt: float = 1.0  # seconds
    
    # Adaptive macro-stepping
    adaptive_macro: bool = True
    macro_rel_tol: float = 1e-3
    macro_abs_tol: float = 1e-6
    
    # Event detection
    detect_events: bool = True
    event_tolerance: float = 1e-6
    
    # Output control
    save_interval: float = 1.0
    
    # Safety limits
    max_macro_dt: float = 60.0  # 1 minute max
    min_macro_dt: float = 1e-6  # 1 microsecond min


@dataclass
class CouplingGraph:
    """Defines which domains feed inputs to which other domains."""
    edges: dict[str, list[CoupledInput]] = field(default_factory=dict)
    
    def add_coupling(self, target: str, input_spec: CoupledInput):
        """Add a coupling edge."""
        if target not in self.edges:
            self.edges[target] = []
        self.edges[target].append(input_spec)
    
    def get_inputs_for(self, target: str) -> list[CoupledInput]:
        """Get all input specifications for a target domain."""
        return self.edges.get(target, [])
    
    def get_source_domains(self, target: str) -> set[str]:
        """Get set of domains that feed into target."""
        return set(inp.source_domain for inp in self.get_inputs_for(target))


@dataclass
class SystemSnapshot:
    """Complete system state at a time instant."""
    t: float
    states: dict[str, Any]  # domain_name -> state
    outputs: dict[str, DomainOutput]  # domain_name -> output
    
    def to_dict(self) -> dict:
        """Serialize snapshot."""
        return {
            't': self.t,
            'states': {k: str(v) for k, v in self.states.items()},  # simplified
            'outputs': {k: v.to_dict() for k, v in self.outputs.items()},
        }


class MacroScheduler:
    """
    Coordinates multiple physics domains with different timescales.
    
    Algorithm per macro-step:
    1. Determine current outputs from all domains
    2. For each domain, gather inputs from other domains (time-averaged)
    3. Advance each domain from t to t+dt (may substep internally)
    4. Update global state
    5. Check for events (eclipse, capture, threshold crossings)
    6. Adjust next macro-step if needed
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.domains: dict[str, DomainAdapter] = {}
        self.coupling = CouplingGraph()
        self.history: list[SystemSnapshot] = []
        self.event_handlers: list[callable] = []
        
        self._current_time: float = 0.0
        self._current_states: dict[str, Any] = {}
        self._current_outputs: dict[str, DomainOutput] = {}
    
    def register_domain(self, name: str, domain: DomainAdapter):
        """Register a physics domain."""
        self.domains[name] = domain
        self._current_states[name] = domain.get_initial_state()
    
    def register_coupling(self, target: str, input_spec: CoupledInput):
        """Register a coupling from source to target domain."""
        self.coupling.add_coupling(target, input_spec)
    
    def add_event_handler(self, handler: callable):
        """Add an event detection handler."""
        self.event_handlers.append(handler)
    
    def initialize(self):
        """Initialize all domains and compute initial outputs."""
        self._current_time = 0.0
        
        for name, domain in self.domains.items():
            state = self._current_states[name]
            output = domain.compute_output(state, 0.0, domain.config.fidelity)
            self._current_outputs[name] = output
        
        # Save initial snapshot
        self._save_snapshot()
    
    def step(self, dt: Optional[float] = None) -> bool:
        """
        Advance simulation by one macro-step.
        
        Args:
            dt: Override macro step size (uses config default if None)
        
        Returns:
            True if step was successful
        """
        if dt is None:
            dt = self.config.macro_dt
        
        dt = np.clip(dt, self.config.min_macro_dt, self.config.max_macro_dt)
        
        t_start = self._current_time
        t_end = t_start + dt
        
        # Phase 1: Gather inputs for each domain
        domain_inputs = self._gather_inputs(t_start, t_end)
        
        # Phase 2: Advance each domain
        advance_results = {}
        for name, domain in self.domains.items():
            inputs = domain_inputs.get(name, {})
            state = self._current_states[name]
            
            result = domain.advance(state, t_start, t_end, inputs)
            advance_results[name] = result
            
            if not result.step_accepted:
                # Domain rejected step - need to retry with smaller dt
                if result.suggested_dt is not None:
                    new_dt = result.suggested_dt
                    return self.step(new_dt)
                else:
                    # Halve the step and retry
                    return self.step(dt / 2)
        
        # Phase 3: Update states
        for name, result in advance_results.items():
            self._current_states[name] = result.new_state
            self._current_outputs[name] = result.output
        
        self._current_time = t_end
        
        # Phase 4: Check for events
        if self.config.detect_events:
            events = self._detect_events(t_start, t_end)
            for event in events:
                for handler in self.event_handlers:
                    handler(event)
        
        # Phase 5: Save snapshot if needed
        if len(self.history) == 0 or \
           (self._current_time - self.history[-1].t) >= self.config.save_interval:
            self._save_snapshot()
        
        return True
    
    def run(self, t_final: float) -> list[SystemSnapshot]:
        """
        Run simulation until t_final.
        
        Returns:
            List of saved snapshots
        """
        self.initialize()
        
        while self._current_time < t_final:
            remaining = t_final - self._current_time
            dt = min(self.config.macro_dt, remaining)
            
            success = self.step(dt)
            if not success:
                raise RuntimeError(f"Simulation failed at t={self._current_time}")
        
        return self.history
    
    def _gather_inputs(self, t_start: float, t_end: float) -> dict[str, dict]:
        """Gather time-averaged inputs for each domain."""
        inputs = defaultdict(dict)
        
        for target_name in self.domains:
            coupled_inputs = self.coupling.get_inputs_for(target_name)
            
            for inp in coupled_inputs:
                source_output = self._current_outputs[inp.source_domain]
                
                # Extract based on coupling strength
                if inp.coupling_type == CouplingStrength.WEAK:
                    # Can use simple time average
                    value = inp.extract(source_output)
                elif inp.coupling_type == CouplingStrength.MODERATE:
                    # May need interpolation (use average for now)
                    value = inp.extract(source_output)
                else:  # STRONG
                    # Would need fine-grained coupling (not implemented)
                    value = inp.extract(source_output)
                
                inputs[target_name][inp.source_quantity] = value
        
        return dict(inputs)
    
    def _detect_events(self, t_start: float, t_end: float) -> list[dict]:
        """Detect events in time interval."""
        events = []
        
        # Check for threshold crossings in outputs
        for name, output in self._current_outputs.items():
            if not output.is_valid():
                events.append({
                    'type': 'validity_violation',
                    'domain': name,
                    'violations': output.regime_violations,
                    't': t_end,
                })
        
        return events
    
    def _save_snapshot(self):
        """Save current state to history."""
        snapshot = SystemSnapshot(
            t=self._current_time,
            states=self._current_states.copy(),
            outputs={k: v for k, v in self._current_outputs.items()}
        )
        self.history.append(snapshot)
    
    def get_current_time(self) -> float:
        """Get current simulation time."""
        return self._current_time
    
    def get_current_state(self, domain: str) -> Any:
        """Get current state for a domain."""
        return self._current_states.get(domain)
    
    def get_current_output(self, domain: str) -> Optional[DomainOutput]:
        """Get current output for a domain."""
        return self._current_outputs.get(domain)


# Event types for common scenarios

class EclipseEvent:
    """Orbital eclipse entry/exit."""
    def __init__(self, t: float, entering: bool, body: str = "Earth"):
        self.t = t
        self.entering = entering  # True = entering eclipse, False = exiting
        self.body = body


class CaptureEvent:
    """Packet capture/decouple event."""
    def __init__(self, t: float, packet_id: str, capturing: bool, station_id: int):
        self.t = t
        self.packet_id = packet_id
        self.capturing = capturing
        self.station_id = station_id


class ThresholdEvent:
    """Generic threshold crossing."""
    def __init__(self, t: float, domain: str, quantity: str, 
                 threshold: float, crossing_up: bool, value: float):
        self.t = t
        self.domain = domain
        self.quantity = quantity
        self.threshold = threshold
        self.crossing_up = crossing_up
        self.value = value

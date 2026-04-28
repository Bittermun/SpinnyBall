"""
Coil switching loss model for pulsed magnetic systems.

Implements I²R resistive losses and eddy current losses during
coil switching transitions in cryogenic environments.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple


@dataclass
class CoilSpecs:
    """Coil physical and electrical specifications."""
    # Geometry
    length: float  # m - coil length
    radius: float  # m - coil radius
    turns: int  # number of turns
    
    # Electrical properties
    resistance: float  # Ohm - DC resistance at operating temperature
    inductance: float  # H - self-inductance
    
    # Material properties
    conductivity: float  # S/m - electrical conductivity
    permeability: float  # H/m - magnetic permeability
    
    # Operating conditions
    operating_temp: float  # K - operating temperature (cryogenic)
    skin_depth: float  # m - skin depth at switching frequency
    
    def __post_init__(self):
        """Validate coil specifications."""
        if self.resistance <= 0:
            raise ValueError(f"resistance must be > 0, got {self.resistance}")
        if self.inductance <= 0:
            raise ValueError(f"inductance must be > 0, got {self.inductance}")
        if self.turns <= 0:
            raise ValueError(f"turns must be > 0, got {self.turns}")
        if self.length <= 0:
            raise ValueError(f"length must be > 0, got {self.length}")
        if self.radius <= 0:
            raise ValueError(f"radius must be > 0, got {self.radius}")


@dataclass
class SwitchingEvent:
    """Single coil switching event."""
    current_start: float  # A - initial current
    current_end: float  # A - final current
    rise_time: float  # s - switching rise time
    fall_time: float  # s - switching fall time
    duty_cycle: float  # dimensionless - fraction of time active


class CoilSwitchingModel:
    """Model for coil switching losses including I²R and eddy currents."""
    
    def __init__(self, specs: CoilSpecs):
        """Initialize coil switching model.
        
        Args:
            specs: CoilSpecs with coil parameters
        """
        self.specs = specs
    
    def i2r_loss(self, current: float, duration: float) -> float:
        """Compute I²R resistive loss.
        
        P_loss = I² * R
        E_loss = P_loss * duration
        
        Args:
            current: Current (A)
            duration: Duration (s)
        
        Returns:
            Energy loss (J)
        """
        power_loss = current**2 * self.specs.resistance
        energy_loss = power_loss * duration
        return energy_loss
    
    def eddy_current_loss(
        self,
        current_change: float,
        switching_time: float,
        volume: Optional[float] = None
    ) -> float:
        """Compute eddy current loss during switching.
        
        P_eddy = σ * V * (dB/dt)²
        B ≈ μ * n * I (solenoid approximation)
        dB/dt ≈ μ * n * dI/dt
        
        Args:
            current_change: Change in current (A)
            switching_time: Switching time (s)
            volume: Volume of conductive material (m³). If None, estimated from coil geometry.
        
        Returns:
            Energy loss (J)
        """
        if volume is None:
            # Estimate volume from coil geometry
            wire_area = np.pi * (self.specs.skin_depth)**2
            wire_length = 2 * np.pi * self.specs.radius * self.specs.turns
            volume = wire_area * wire_length
        
        # Current rate of change
        dI_dt = current_change / switching_time if switching_time > 0 else 0
        
        # Magnetic field rate of change (solenoid approximation)
        # B = μ * n * I, where n = turns/length
        n = self.specs.turns / self.specs.length
        dB_dt = self.specs.permeability * n * dI_dt
        
        # Eddy current power density
        power_density = self.specs.conductivity * (dB_dt)**2
        
        # Total eddy current power
        power_eddy = power_density * volume
        
        # Energy loss (assume switching happens once)
        energy_loss = power_eddy * switching_time
        
        return energy_loss
    
    def switching_loss(
        self,
        event: SwitchingEvent,
        volume: Optional[float] = None
    ) -> Tuple[float, dict]:
        """Compute total switching loss for an event.
        
        Args:
            event: SwitchingEvent with switching parameters
            volume: Volume for eddy current calculation (optional)
        
        Returns:
            Tuple of (total_loss_J, breakdown_dict)
        """
        # I²R loss during rise phase
        avg_current_rise = (event.current_start + event.current_end) / 2
        loss_rise = self.i2r_loss(avg_current_rise, event.rise_time)
        
        # I²R loss during fall phase
        avg_current_fall = (event.current_end + 0) / 2
        loss_fall = self.i2r_loss(avg_current_fall, event.fall_time)
        
        # I²R loss during steady state (if any)
        steady_duration = 0  # Simplified: assume no steady state for pulsed operation
        loss_steady = self.i2r_loss(event.current_end, steady_duration)
        
        # Eddy current losses
        loss_eddy_rise = self.eddy_current_loss(
            event.current_end - event.current_start,
            event.rise_time,
            volume
        )
        loss_eddy_fall = self.eddy_current_loss(
            -event.current_end,
            event.fall_time,
            volume
        )
        
        total_i2r = loss_rise + loss_fall + loss_steady
        total_eddy = loss_eddy_rise + loss_eddy_fall
        total_loss = total_i2r + total_eddy
        
        breakdown = {
            'i2r_rise_J': loss_rise,
            'i2r_fall_J': loss_fall,
            'i2r_steady_J': loss_steady,
            'eddy_rise_J': loss_eddy_rise,
            'eddy_fall_J': loss_eddy_fall,
            'total_i2r_J': total_i2r,
            'total_eddy_J': total_eddy,
        }
        
        return total_loss, breakdown
    
    def average_power_loss(
        self,
        events: list[SwitchingEvent],
        period: float,
        volume: Optional[float] = None
    ) -> Tuple[float, dict]:
        """Compute average power loss over a period.
        
        Args:
            events: List of SwitchingEvent objects
            period: Time period (s)
            volume: Volume for eddy current calculation (optional)
        
        Returns:
            Tuple of (avg_power_W, breakdown_dict)
        """
        total_energy = 0.0
        total_i2r = 0.0
        total_eddy = 0.0
        
        for event in events:
            loss, breakdown = self.switching_loss(event, volume)
            total_energy += loss
            total_i2r += breakdown['total_i2r_J']
            total_eddy += breakdown['total_eddy_J']
        
        avg_power = total_energy / period if period > 0 else 0.0
        
        breakdown = {
            'total_energy_J': total_energy,
            'avg_power_W': avg_power,
            'avg_i2r_power_W': total_i2r / period if period > 0 else 0.0,
            'avg_eddy_power_W': total_eddy / period if period > 0 else 0.0,
            'num_events': len(events),
        }
        
        return avg_power, breakdown


# Default coil specifications (pulsed copper coil at 77K)
DEFAULT_COIL_SPECS = CoilSpecs(
    length=0.1,  # 10 cm coil length
    radius=0.05,  # 5 cm coil radius
    turns=100,  # 100 turns
    resistance=0.01,  # 10 mOhm at 77K
    inductance=1e-3,  # 1 mH
    conductivity=1e8,  # S/m (copper at 77K)
    permeability=4*np.pi*1e-7,  # H/m (vacuum permeability)
    operating_temp=77.0,  # K
    skin_depth=1e-4,  # 0.1 mm at typical switching frequencies
)


def create_pulsed_switching_event(
    peak_current: float,
    pulse_width: float,
    rise_time: float = 1e-5,
    fall_time: float = 1e-5,
) -> SwitchingEvent:
    """Create a typical pulsed switching event.
    
    Args:
        peak_current: Peak current (A)
        pulse_width: Pulse width (s)
        rise_time: Rise time (s)
        fall_time: Fall time (s)
    
    Returns:
        SwitchingEvent
    """
    return SwitchingEvent(
        current_start=0.0,
        current_end=peak_current,
        rise_time=rise_time,
        fall_time=fall_time,
        duty_cycle=pulse_width / (pulse_width + rise_time + fall_time),
    )

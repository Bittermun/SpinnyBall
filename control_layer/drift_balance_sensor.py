"""
Drift balance sensor architecture for stream balance ε measurement.

Provides sensor model, error budget, and measurement interface for
maintaining ε < 10⁻⁴ requirement.

Error Budget Breakdown:
- Mass flow sensor: ±5e-5 (50% of budget)
- Timing sensor: ±3e-5 (30% of budget)
- Position sensor: ±1e-5 (10% of budget)
- Velocity sensor: ±1e-5 (10% of budget)
Total: ±1e-4 (100% of budget)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from enum import Enum


class SensorType(Enum):
    """Types of sensors for drift balance measurement."""
    MASS_FLOW = "mass_flow"
    TIMING = "timing"
    POSITION = "position"
    VELOCITY = "velocity"


@dataclass
class SensorErrorBudget:
    """Error budget for a single sensor."""
    sensor_type: SensorType
    nominal_error: float  # Nominal error (1-sigma)
    max_error: float  # Maximum allowable error
    drift_rate: float  # Error drift rate per second
    calibration_interval: float  # Calibration interval (s)
    last_calibration: float = 0.0  # Time of last calibration
    
    def current_error(self, current_time: float) -> float:
        """Calculate current error including drift since calibration."""
        time_since_cal = current_time - self.last_calibration
        drift_error = self.drift_rate * time_since_cal
        return self.nominal_error + drift_error
    
    def needs_calibration(self, current_time: float) -> bool:
        """Check if sensor needs calibration."""
        return (current_time - self.last_calibration) >= self.calibration_interval


@dataclass
class DriftBalanceSensorConfig:
    """Configuration for drift balance sensor architecture."""
    target_epsilon: float = 1e-4  # Target balance ε < 10⁻⁴
    sampling_rate: float = 1000.0  # Hz (1 kHz sampling)
    measurement_window: int = 100  # Number of samples for averaging
    enable_error_budgeting: bool = True
    enable_sensor_fusion: bool = True
    
    # Individual sensor error budgets (allocated to meet target_epsilon)
    mass_flow_error: float = 5e-5  # 50% of budget
    timing_error: float = 3e-5  # 30% of budget
    position_error: float = 1e-5  # 10% of budget
    velocity_error: float = 1e-5  # 10% of budget
    
    # Sensor drift rates
    mass_flow_drift: float = 1e-8  # Error drift per second
    timing_drift: float = 5e-9
    position_drift: float = 1e-9
    velocity_drift: float = 1e-9
    
    # Calibration intervals
    mass_flow_cal_interval: float = 3600.0  # 1 hour
    timing_cal_interval: float = 1800.0  # 30 minutes
    position_cal_interval: float = 7200.0  # 2 hours
    velocity_cal_interval: float = 7200.0  # 2 hours


class DriftBalanceSensor:
    """
    Drift balance sensor architecture for ε measurement.
    
    Provides fused measurement from multiple sensors with error budgeting.
    Maintains ε < 10⁻⁴ through sensor fusion and calibration management.
    """
    
    def __init__(self, config: DriftBalanceSensorConfig = None):
        """
        Initialize drift balance sensor.
        
        Args:
            config: DriftBalanceSensorConfig instance
        """
        self.config = config or DriftBalanceSensorConfig()
        self.current_time = 0.0
        
        # Initialize sensor error budgets
        self.sensors = {
            SensorType.MASS_FLOW: SensorErrorBudget(
                sensor_type=SensorType.MASS_FLOW,
                nominal_error=self.config.mass_flow_error,
                max_error=self.config.mass_flow_error * 2.0,
                drift_rate=self.config.mass_flow_drift,
                calibration_interval=self.config.mass_flow_cal_interval,
                last_calibration=0.0,
            ),
            SensorType.TIMING: SensorErrorBudget(
                sensor_type=SensorType.TIMING,
                nominal_error=self.config.timing_error,
                max_error=self.config.timing_error * 2.0,
                drift_rate=self.config.timing_drift,
                calibration_interval=self.config.timing_cal_interval,
                last_calibration=0.0,
            ),
            SensorType.POSITION: SensorErrorBudget(
                sensor_type=SensorType.POSITION,
                nominal_error=self.config.position_error,
                max_error=self.config.position_error * 2.0,
                drift_rate=self.config.position_drift,
                calibration_interval=self.config.position_cal_interval,
                last_calibration=0.0,
            ),
            SensorType.VELOCITY: SensorErrorBudget(
                sensor_type=SensorType.VELOCITY,
                nominal_error=self.config.velocity_error,
                max_error=self.config.velocity_error * 2.0,
                drift_rate=self.config.velocity_drift,
                calibration_interval=self.config.velocity_cal_interval,
                last_calibration=0.0,
            ),
        }
        
        # Measurement buffer
        self.measurement_buffer = np.zeros(self.config.measurement_window)
        self.buffer_idx = 0
        self.valid_samples = 0
    
    def measure_imbalance(
        self,
        mass_flow_rate: float,
        timing_error: float,
        position_error: float,
        velocity_error: float,
    ) -> Tuple[float, dict]:
        """
        Measure stream imbalance ε from sensor inputs.
        
        Args:
            mass_flow_rate: Measured mass flow rate (kg/s)
            timing_error: Timing error (s)
            position_error: Position error (m)
            velocity_error: Velocity error (m/s)
        
        Returns:
            (epsilon, error_breakdown) where:
            - epsilon: Measured imbalance ε
            - error_breakdown: Dictionary of individual sensor contributions
        """
        # Validate inputs (use absolute values for errors, but validate they're finite)
        if not np.isfinite(mass_flow_rate):
            raise ValueError(f"mass_flow_rate must be finite, got {mass_flow_rate}")
        if not np.isfinite(timing_error):
            raise ValueError(f"timing_error must be finite, got {timing_error}")
        if not np.isfinite(position_error):
            raise ValueError(f"position_error must be finite, got {position_error}")
        if not np.isfinite(velocity_error):
            raise ValueError(f"velocity_error must be finite, got {velocity_error}")
        
        # Calculate individual sensor contributions
        mass_contribution = self._measure_mass_flow(mass_flow_rate)
        timing_contribution = self._measure_timing(timing_error)
        position_contribution = self._measure_position(position_error)
        velocity_contribution = self._measure_velocity(velocity_error)
        
        # Sensor fusion (weighted sum based on error budget)
        if self.config.enable_sensor_fusion:
            epsilon = (
                mass_contribution +
                timing_contribution +
                position_contribution +
                velocity_contribution
            )
        else:
            # Simple average (less accurate)
            epsilon = (
                mass_contribution + timing_contribution +
                position_contribution + velocity_contribution
            ) / 4.0
        
        # Update measurement buffer
        self.measurement_buffer[self.buffer_idx] = epsilon
        self.buffer_idx = (self.buffer_idx + 1) % self.config.measurement_window
        self.valid_samples = min(self.valid_samples + 1, self.config.measurement_window)
        
        # Error breakdown
        error_breakdown = {
            'mass_flow': mass_contribution,
            'timing': timing_contribution,
            'position': position_contribution,
            'velocity': velocity_contribution,
            'total': epsilon,
            'within_budget': epsilon <= self.config.target_epsilon,
        }
        
        return epsilon, error_breakdown
    
    def _measure_mass_flow(self, mass_flow_rate: float) -> float:
        """Measure mass flow contribution to ε."""
        sensor = self.sensors[SensorType.MASS_FLOW]
        current_error = sensor.current_error(self.current_time)
        
        # Normalize mass flow rate to dimensionless ε contribution
        # Assuming nominal flow rate of 1 kg/s
        nominal_flow = 1.0
        normalized_flow = mass_flow_rate / nominal_flow if nominal_flow > 0 else 0.0
        
        contribution = abs(normalized_flow - 1.0) * current_error
        return contribution
    
    def _measure_timing(self, timing_error: float) -> float:
        """Measure timing contribution to ε."""
        sensor = self.sensors[SensorType.TIMING]
        current_error = sensor.current_error(self.current_time)
        
        # Normalize timing error to dimensionless ε contribution
        # Assuming target timing accuracy of 1 μs
        target_timing = 1e-6
        normalized_timing = timing_error / target_timing if target_timing > 0 else 0.0
        
        contribution = abs(normalized_timing) * current_error
        return contribution
    
    def _measure_position(self, position_error: float) -> float:
        """Measure position contribution to ε."""
        sensor = self.sensors[SensorType.POSITION]
        current_error = sensor.current_error(self.current_time)
        
        # Normalize position error to dimensionless ε contribution
        # Assuming target position accuracy of 1 mm
        target_position = 1e-3
        normalized_position = position_error / target_position if target_position > 0 else 0.0
        
        contribution = abs(normalized_position) * current_error
        return contribution
    
    def _measure_velocity(self, velocity_error: float) -> float:
        """Measure velocity contribution to ε."""
        sensor = self.sensors[SensorType.VELOCITY]
        current_error = sensor.current_error(self.current_time)
        
        # Normalize velocity error to dimensionless ε contribution
        # Assuming target velocity accuracy of 0.1 m/s
        target_velocity = 0.1
        normalized_velocity = velocity_error / target_velocity if target_velocity > 0 else 0.0
        
        contribution = abs(normalized_velocity) * current_error
        return contribution
    
    def get_averaged_measurement(self) -> float:
        """Get averaged measurement over the measurement window."""
        if self.valid_samples == 0:
            return 0.0
        
        # Handle circular buffer correctly
        if self.valid_samples < self.config.measurement_window:
            # Buffer not full, samples are at the start
            measurements = self.measurement_buffer[:self.valid_samples]
        else:
            # Buffer is full, need to handle circular wrap
            # Roll buffer so that oldest samples are at the start
            measurements = np.roll(self.measurement_buffer, -self.buffer_idx)
        
        return np.mean(measurements)
    
    def calibrate_sensor(self, sensor_type: SensorType):
        """Calibrate a specific sensor."""
        if sensor_type in self.sensors:
            self.sensors[sensor_type].last_calibration = self.current_time
    
    def calibrate_all_sensors(self):
        """Calibrate all sensors."""
        for sensor_type in self.sensors:
            self.calibrate_sensor(sensor_type)
    
    def check_calibration_status(self) -> dict:
        """Check calibration status of all sensors."""
        status = {}
        for sensor_type, sensor in self.sensors.items():
            status[sensor_type.value] = {
                'needs_calibration': sensor.needs_calibration(self.current_time),
                'time_since_calibration': self.current_time - sensor.last_calibration,
                'current_error': sensor.current_error(self.current_time),
            }
        return status
    
    def get_error_budget_summary(self) -> dict:
        """Get summary of error budget allocation."""
        total_budget = sum([
            self.config.mass_flow_error,
            self.config.timing_error,
            self.config.position_error,
            self.config.velocity_error,
        ])
        
        # Avoid division by zero
        if total_budget > 0:
            mass_pct = self.config.mass_flow_error / total_budget * 100
            timing_pct = self.config.timing_error / total_budget * 100
            position_pct = self.config.position_error / total_budget * 100
            velocity_pct = self.config.velocity_error / total_budget * 100
        else:
            mass_pct = timing_pct = position_pct = velocity_pct = 0.0
        
        return {
            'target_epsilon': self.config.target_epsilon,
            'total_allocated': total_budget,
            'mass_flow': {
                'allocated': self.config.mass_flow_error,
                'percentage': mass_pct,
            },
            'timing': {
                'allocated': self.config.timing_error,
                'percentage': timing_pct,
            },
            'position': {
                'allocated': self.config.position_error,
                'percentage': position_pct,
            },
            'velocity': {
                'allocated': self.config.velocity_error,
                'percentage': velocity_pct,
            },
            'within_budget': total_budget <= self.config.target_epsilon,
        }
    
    def step(self, dt: float):
        """Advance sensor time by dt."""
        self.current_time += dt


def create_default_sensor() -> DriftBalanceSensor:
    """Create drift balance sensor with default configuration."""
    config = DriftBalanceSensorConfig()
    return DriftBalanceSensor(config)

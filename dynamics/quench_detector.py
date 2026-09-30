"""
Quench detection and emergency shutdown logic for GdBCO superconductors.

Monitors temperature and heating rates to detect quench events
and trigger emergency shutdown procedures.
"""

from dataclasses import dataclass


@dataclass
class QuenchThresholds:
    """Quench detection thresholds."""
    temperature_critical: float = 90.0  # K (GdBCO T_c ≈ 92K)
    temperature_warning: float = 85.0  # K (13K margin)
    temperature_rate_limit: float = 10.0  # K/s (rapid heating)
    hysteresis: float = 2.0  # K (prevent chatter)
    immediate_quench: bool = True
    min_time_above_threshold_s: float = 0.0


class QuenchDetector:
    """Quench detection and emergency shutdown logic."""

    def __init__(self, thresholds: QuenchThresholds | None = None, initial_temperature: float = 77.0):
        """Initialize quench detector.

        Args:
            thresholds: QuenchThresholds with detection parameters
            initial_temperature: Initial temperature for heating rate calculation (K)
        """
        self.thresholds = thresholds or QuenchThresholds()
        self.quenched = False
        self.warning_state = False
        self.prev_temperature = initial_temperature  # K
        self.quench_time = None
        self._time_above: float = 0.0

    def check_temperature(self, temperature: float, dt: float) -> dict:
        """Check temperature for quench conditions.

        Args:
            temperature: Current temperature (K)
            dt: Time step (s)

        Returns:
            Dictionary with status and alerts

        Raises:
            ValueError: If dt <= 0
        """
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        # Compute heating rate
        heating_rate = (temperature - self.prev_temperature) / dt
        self.prev_temperature = temperature

        # Check critical threshold (with hysteresis)
        if self.quenched:
            # Stay quenched until below warning - hysteresis
            if temperature < self.thresholds.temperature_warning - self.thresholds.hysteresis:
                self.quenched = False
                self.quench_time = None
        else:
            # Trigger quench if above critical
            if temperature > self.thresholds.temperature_critical:
                self.quenched = True
                self.quench_time = 0.0

        # Check warning threshold
        if temperature > self.thresholds.temperature_warning:
            self.warning_state = True
        else:
            self.warning_state = False

        # Check heating rate limit
        rate_violation = heating_rate > self.thresholds.temperature_rate_limit

        return {
            "quenched": self.quenched,
            "warning": self.warning_state,
            "rate_violation": rate_violation,
            "heating_rate": heating_rate,
            "emergency_shutdown": self.quenched or rate_violation,
        }

    def check_quench(
        self,
        temperature: float,
        critical_temp: float | None = None,
        current_time: float | None = None,
        dt: float = 0.0,
    ) -> bool:
        """Check whether temperature triggers a quench event (Monte Carlo runner API)."""
        threshold = critical_temp if critical_temp is not None else self.thresholds.temperature_critical
        exceeded = temperature >= threshold

        if self.thresholds.immediate_quench:
            if exceeded:
                self.quenched = True
            return bool(exceeded)

        if exceeded:
            self._time_above += float(dt)
        else:
            self._time_above = 0.0

        is_quenched = self._time_above >= float(self.thresholds.min_time_above_threshold_s)
        if is_quenched:
            self.quenched = True
        return is_quenched

    def increment_quench_time(self, dt: float):
        """Increment quench time tracking.

        Args:
            dt: Time step (s) to add to quench time
        """
        if self.quenched and self.quench_time is not None:
            self.quench_time += dt

    def reset(self):
        """Reset quench detector."""
        self.quenched = False
        self.warning_state = False
        self.quench_time = None
        self._time_above = 0.0

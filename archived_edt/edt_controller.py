"""
EDT Controller for current modulation and libration damping.

Implements current setpoint tracking, power generation mode, and libration damping.
Thrust vectoring is deferred to Phase 5.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class EDTController:
    """
    EDT controller for current modulation and libration damping.

    Focuses on:
    - Current setpoint tracking: I_cmd → I_actual (±5% accuracy)
    - Power generation mode: maximize P_gen = V_emf × I
    - Libration damping: current modulation to suppress oscillations
    - Rate limiting on current changes
    - Libration feedback control

    Thrust vectoring is deferred to Phase 5.
    """

    def __init__(
        self,
        max_current: float = 10.0,  # A
        max_rate: float = 1.0,  # A/s
        libration_damping_gain: float = 0.5,
        libration_derivative_gain: float = 0.1,
        power_generation_gain: float = 0.8,
        voltage_estimate: float = 100.0,  # V
    ):
        """
        Initialize EDT controller.

        Args:
            max_current: Maximum current (A)
            max_rate: Maximum rate of current change (A/s)
            libration_damping_gain: Proportional gain for libration damping control
            libration_derivative_gain: Derivative gain for libration damping control
            power_generation_gain: Gain for power generation mode
            voltage_estimate: Estimated EMF voltage for power generation (V)
        """
        self.max_current = max_current
        self.max_rate = max_rate
        self.libration_damping_gain = libration_damping_gain
        self.libration_derivative_gain = libration_derivative_gain
        self.power_generation_gain = power_generation_gain
        self.voltage_estimate = voltage_estimate

        self.current_actual = 0.0
        self.current_command = 0.0

        logger.info(
            f"EDTController initialized: max_current={max_current} A, "
            f"max_rate={max_rate} A/s, voltage_estimate={voltage_estimate} V"
        )

    def compute_current_setpoint(
        self,
        power_demand: float,
        libration_angle: float,
        libration_rate: float = 0.0,
    ) -> float:
        """
        Compute optimal current for power generation and libration damping.

        Args:
            power_demand: Desired power generation (W)
            libration_angle: Current libration angle (rad)
            libration_rate: Libration angular rate (rad/s)

        Returns:
            Current setpoint (A)
        """
        # Power generation component
        # I_power = P_demand / V_emf
        i_power = self.power_generation_gain * power_demand / self.voltage_estimate

        # Libration damping component
        # I_damping = -gain * libration_angle (proportional control)
        i_damping = -self.libration_damping_gain * libration_angle

        # Combine components
        i_setpoint = i_power + i_damping

        # Clamp to limits
        i_setpoint = np.clip(i_setpoint, -self.max_current, self.max_current)

        logger.debug(
            f"Current setpoint: i_power={i_power:.3f} A, "
            f"i_damping={i_damping:.3f} A, i_setpoint={i_setpoint:.3f} A"
        )

        return i_setpoint

    def libration_damping_current(self, libration_state: np.ndarray) -> float:
        """
        Compute damping current for libration suppression.

        Args:
            libration_state: Libration state [angle, angular_rate] (rad, rad/s)

        Returns:
            Damping current (A)
        """
        libration_angle = libration_state[0]
        libration_rate = libration_state[1]

        # PD controller for libration damping
        i_damping = -self.libration_damping_gain * libration_angle - self.libration_derivative_gain * libration_rate

        # Clamp to limits
        i_damping = np.clip(i_damping, -self.max_current, self.max_current)

        return i_damping

    def rate_limit_current(
        self,
        current_new: float,
        current_old: float,
        dt: float,
    ) -> float:
        """
        Apply rate limiting to current changes.

        Args:
            current_new: Desired new current (A)
            current_old: Current actual current (A)
            dt: Time step (s)

        Returns:
            Rate-limited current (A)
        """
        delta = current_new - current_old
        max_delta = self.max_rate * dt

        if abs(delta) > max_delta:
            delta = np.sign(delta) * max_delta
            logger.debug(
                f"Rate limiting applied: delta={delta:.3f} A (max={max_delta:.3f} A)"
            )

        current_limited = current_old + delta
        return current_limited

    def update(
        self,
        power_demand: float,
        libration_angle: float,
        libration_rate: float = 0.0,
        dt: float = 0.01,
    ) -> dict:
        """
        Update controller with new measurements and compute current command.

        Args:
            power_demand: Desired power generation (W)
            libration_angle: Current libration angle (rad)
            libration_rate: Libration angular rate (rad/s)
            dt: Time step (s)

        Returns:
            Dictionary with controller state
        """
        # Compute current setpoint
        i_setpoint = self.compute_current_setpoint(
            power_demand, libration_angle, libration_rate
        )

        # Apply rate limiting
        i_actual = self.rate_limit_current(i_setpoint, self.current_actual, dt)

        # Update state
        self.current_command = i_setpoint
        self.current_actual = i_actual

        # Compute tracking error with minimum threshold to avoid division by near-zero
        min_setpoint = 1e-6
        if abs(i_setpoint) < min_setpoint:
            tracking_error = abs(i_actual - i_setpoint) / min_setpoint
        else:
            tracking_error = abs(i_actual - i_setpoint) / abs(i_setpoint)

        return {
            "current_command": self.current_command,
            "current_actual": self.current_actual,
            "tracking_error": tracking_error,
            "libration_angle": libration_angle,
            "power_demand": power_demand,
        }

    def reset(self):
        """Reset controller state."""
        self.current_actual = 0.0
        self.current_command = 0.0
        logger.info("EDTController reset")

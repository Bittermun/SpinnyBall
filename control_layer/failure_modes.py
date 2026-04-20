"""
Synthetic failure mode library for Phase 3 anomaly detection.

Implements 5-10 failure modes with realistic physics for ML training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failure modes for synthetic data generation."""
    DEBRIS_IMPACT = "debris_impact"
    THERMAL_RUNAWAY = "thermal_runaway"
    MAGNETIC_QUENCH = "magnetic_quench"
    SENSOR_FAILURE = "sensor_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    PACKET_CAPTURE_FAILURE = "packet_capture_failure"
    PACKET_RELEASE_FAILURE = "packet_release_failure"
    VELOCITY_PERTURBATION = "velocity_perturbation"
    SPIN_RATE_PERTURBATION = "spin_rate_perturbation"
    POSITION_PERTURBATION = "position_perturbation"


@dataclass
class FailureEvent:
    """Represents a single failure event."""
    failure_type: FailureType
    severity: float  # 0.0 to 1.0
    timestamp: float  # Time of failure in seconds
    affected_packet_id: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


class FailureModeLibrary:
    """
    Library of failure modes with realistic physics parameters.

    Implements 10 failure modes with configurable severity levels
    for synthetic data generation.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize failure mode library.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Failure mode parameters
        self.failure_params = {
            FailureType.DEBRIS_IMPACT: {
                "min_momentum": 0.01,  # N·s
                "max_momentum": 0.5,  # N·s
                "direction_variance": 0.5,
            },
            FailureType.THERMAL_RUNAWAY: {
                "min_temp_increase": 50.0,  # K
                "max_temp_increase": 200.0,  # K
                "eta_ind_reduction": 0.3,  # 30% efficiency reduction
            },
            FailureType.MAGNETIC_QUENCH: {
                "min_efficiency_drop": 0.1,  # 10% drop
                "max_efficiency_drop": 0.5,  # 50% drop
            },
            FailureType.SENSOR_FAILURE: {
                "noise_std": 0.1,  # Standard deviation of sensor noise
                "bias_range": 0.2,  # Sensor bias range
            },
            FailureType.ACTUATOR_FAILURE: {
                "signal_loss_prob": 0.5,  # Probability of complete signal loss
                "signal_reduction": 0.8,  # Signal reduction factor
            },
            FailureType.PACKET_CAPTURE_FAILURE: {
                "miss_probability": 0.3,  # Probability of missed capture
            },
            FailureType.PACKET_RELEASE_FAILURE: {
                "stuck_probability": 0.2,  # Probability of stuck packet
            },
            FailureType.VELOCITY_PERTURBATION: {
                "min_kick": 0.1,  # m/s
                "max_kick": 2.0,  # m/s
            },
            FailureType.SPIN_RATE_PERTURBATION: {
                "min_delta_omega": 0.5,  # rad/s
                "max_delta_omega": 5.0,  # rad/s
            },
            FailureType.POSITION_PERTURBATION: {
                "min_offset": 0.01,  # m
                "max_offset": 0.5,  # m
            },
        }

    def apply_failure(
        self,
        failure_event: FailureEvent,
        packet,
        current_time: float,
    ) -> None:
        """
        Apply failure event to a packet.

        Args:
            failure_event: Failure event to apply
            packet: Packet to apply failure to
            current_time: Current simulation time
        """
        params = self.failure_params[failure_event.failure_type]
        severity = failure_event.severity

        if failure_event.failure_type == FailureType.DEBRIS_IMPACT:
            # Momentum kick in random direction
            momentum = params["min_momentum"] + severity * (params["max_momentum"] - params["min_momentum"])
            direction = np.random.randn(3)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            impulse = momentum * direction
            packet.body.velocity += impulse / packet.body.mass

        elif failure_event.failure_type == FailureType.THERMAL_RUNAWAY:
            # Temperature increase
            temp_increase = params["min_temp_increase"] + severity * (params["max_temp_increase"] - params["min_temp_increase"])
            packet.temperature += temp_increase
            # Reduce induction efficiency
            packet.eta_ind *= (1.0 - severity * params["eta_ind_reduction"])

        elif failure_event.failure_type == FailureType.MAGNETIC_QUENCH:
            # Efficiency drop
            efficiency_drop = params["min_efficiency_drop"] + severity * (params["max_efficiency_drop"] - params["min_efficiency_drop"])
            packet.eta_ind *= (1.0 - efficiency_drop)

        elif failure_event.failure_type == FailureType.SENSOR_FAILURE:
            # Add sensor noise (simulated by perturbing state)
            noise = np.random.randn(7) * params["noise_std"] * severity
            bias = (np.random.rand(7) - 0.5) * 2 * params["bias_range"] * severity
            # Apply to state (simulated)
            packet.body.quaternion += noise[:4] * 0.01
            packet.body.angular_velocity += noise[4:] * 0.1

        elif failure_event.failure_type == FailureType.ACTUATOR_FAILURE:
            # Reduce control signal (simulated by reducing angular velocity)
            reduction = 1.0 - severity * params["signal_reduction"]
            if np.random.random() < params["signal_loss_prob"]:
                # Complete signal loss
                packet.body.angular_velocity *= 0.1
            else:
                packet.body.angular_velocity *= reduction

        elif failure_event.failure_type == FailureType.PACKET_CAPTURE_FAILURE:
            # Missed capture (simulated by reducing eta_ind)
            if np.random.random() < params["miss_probability"]:
                packet.eta_ind *= 0.5  # Reduced capture efficiency

        elif failure_event.failure_type == FailureType.PACKET_RELEASE_FAILURE:
            # Stuck packet (simulated by stopping motion)
            if np.random.random() < params["stuck_probability"]:
                packet.body.velocity *= 0.1

        elif failure_event.failure_type == FailureType.VELOCITY_PERTURBATION:
            # Velocity kick
            kick = params["min_kick"] + severity * (params["max_kick"] - params["min_kick"])
            direction = np.random.randn(3)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            packet.body.velocity += kick * direction

        elif failure_event.failure_type == FailureType.SPIN_RATE_PERTURBATION:
            # Angular velocity perturbation
            delta_omega = params["min_delta_omega"] + severity * (params["max_delta_omega"] - params["min_delta_omega"])
            direction = np.random.randn(3)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            packet.body.angular_velocity += delta_omega * direction

        elif failure_event.failure_type == FailureType.POSITION_PERTURBATION:
            # Position offset
            offset = params["min_offset"] + severity * (params["max_offset"] - params["min_offset"])
            direction = np.random.randn(3)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            packet.body.position += offset * direction

        logger.debug(f"Applied {failure_event.failure_type.value} (severity={severity:.2f}) to packet {packet.id}")

    def generate_failure_sequence(
        self,
        time_horizon: float,
        max_failures: int = 3,
        failure_types: Optional[List[FailureType]] = None,
    ) -> List[FailureEvent]:
        """
        Generate a sequence of failure events.

        Args:
            time_horizon: Simulation time horizon
            max_failures: Maximum number of failures
            failure_types: List of failure types to include (None = all)

        Returns:
            List of failure events
        """
        if failure_types is None:
            failure_types = list(FailureType)

        num_failures = np.random.randint(1, max_failures + 1)
        events = []

        for _ in range(num_failures):
            failure_type = np.random.choice(failure_types)
            severity = np.random.uniform(0.1, 1.0)
            timestamp = np.random.uniform(0, time_horizon)
            packet_id = np.random.randint(0, 10)  # Assume 10 packets

            event = FailureEvent(
                failure_type=failure_type,
                severity=severity,
                timestamp=timestamp,
                affected_packet_id=packet_id,
                metadata={"applied": False},
            )
            events.append(event)

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        return events


def create_failure_library(random_seed: int = 42) -> FailureModeLibrary:
    """
    Factory function to create failure mode library.

    Args:
        random_seed: Random seed for reproducibility

    Returns:
        FailureModeLibrary instance
    """
    return FailureModeLibrary(random_seed=random_seed)

"""
Synthetic failure data generation pipeline for Phase 3.

Generates labeled failure data using the failure mode library and
high-fidelity simulator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

from control_layer.failure_modes import (
    FailureEvent,
    FailureModeLibrary,
    FailureType,
    create_failure_library,
)
from control_layer.training_data_generator import TrainingDataGenerator, GeneratorConfig
from dynamics.multi_body import MultiBodyStream, Packet, SNode
from dynamics.rigid_body import RigidBody
from control_layer.stream_balance import StreamBalanceController, StreamBalanceConfig

logger = logging.getLogger(__name__)

# Constants for data generation
FAILURE_DURATION_TIMESTEPS = 50  # Number of timesteps to label as failure after event


def generate_packet_loss_perturbation(
    n_packets: int,
    loss_rate: float = 0.01,
    random_seed: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate packet loss perturbation for counter-streams.
    
    Args:
        n_packets: Number of packets in stream
        loss_rate: Packet loss probability (0-1)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (loss_plus, loss_minus) - binary arrays indicating lost packets
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Independent loss for each stream
    loss_plus = np.random.random(n_packets) < loss_rate
    loss_minus = np.random.random(n_packets) < loss_rate
    
    return loss_plus.astype(int), loss_minus.astype(int)


def generate_timing_jitter_perturbation(
    n_packets: int,
    jitter_std: float = 1e-6,
    packet_period: float = 0.01,
    random_seed: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate timing jitter perturbation for counter-streams.
    
    Args:
        n_packets: Number of packets in stream
        jitter_std: RMS timing jitter (s)
        packet_period: Nominal packet period (s)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (jitter_plus, jitter_minus) - timing jitter arrays (s)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Gaussian jitter for each stream
    jitter_plus = np.random.normal(0, jitter_std, n_packets)
    jitter_minus = np.random.normal(0, jitter_std, n_packets)
    
    return jitter_plus, jitter_minus


def generate_mass_drift_perturbation(
    n_packets: int,
    drift_rate: float = 0.001,
    nominal_mass: float = 0.05,
    random_seed: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mass drift perturbation for counter-streams.
    
    Args:
        n_packets: Number of packets in stream
        drift_rate: Mass drift rate (fraction per packet)
        nominal_mass: Nominal packet mass (kg)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (mass_plus, mass_minus) - mass arrays (kg)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Random walk mass drift
    drift_plus = np.cumsum(np.random.normal(0, drift_rate * nominal_mass, n_packets))
    drift_minus = np.cumsum(np.random.normal(0, drift_rate * nominal_mass, n_packets))
    
    mass_plus = nominal_mass + drift_plus
    mass_minus = nominal_mass + drift_minus
    
    # Ensure positive mass
    mass_plus = np.maximum(mass_plus, 0.1 * nominal_mass)
    mass_minus = np.maximum(mass_minus, 0.1 * nominal_mass)
    
    return mass_plus, mass_minus


@dataclass
class DataGenerationConfig:
    """Configuration for synthetic failure data generation."""
    n_samples: int = 10000
    time_horizon: float = 10.0
    dt: float = 0.01
    n_packets: int = 5
    max_failures_per_sample: int = 3
    random_seed: int = 42
    output_dir: str = "control_layer/data"
    dataset_name: str = "synthetic_failure_data"


class SyntheticDataGenerator:
    """
    Generate synthetic failure data for ML training.

    Uses the failure mode library to inject realistic failures into
    high-fidelity simulations and generates labeled datasets.
    """

    def __init__(
        self,
        config: DataGenerationConfig = None,
        failure_library: FailureModeLibrary = None,
    ):
        """
        Initialize synthetic data generator.

        Args:
            config: Data generation configuration
            failure_library: Failure mode library instance
        """
        self.config = config or DataGenerationConfig()
        self.failure_library = failure_library or create_failure_library(self.config.random_seed)
        self.base_generator = TrainingDataGenerator(
            GeneratorConfig(random_seed=self.config.random_seed)
        )

        # Set random seed
        np.random.seed(self.config.random_seed)

        logger.info(
            f"Synthetic data generator initialized with {self.config.n_samples} samples, "
            f"max {self.config.max_failures_per_sample} failures per sample"
        )

    def generate_sample(
        self,
        sample_id: int,
    ) -> Tuple[np.ndarray, List[FailureEvent], np.ndarray]:
        """
        Generate a single sample with failure events.

        Args:
            sample_id: Sample identifier

        Returns:
            trajectory: [n_timesteps × n_packets × state_dim]
            failure_events: List of failure events
            labels: [n_timesteps × n_packets] (1 = failure, 0 = normal)
        """
        # Create stream
        stream = self.base_generator._create_stream(self.config.n_packets)

        # Generate failure sequence
        failure_events = self.failure_library.generate_failure_sequence(
            time_horizon=self.config.time_horizon,
            max_failures=self.config.max_failures_per_sample,
        )

        # Simulate with failure injection
        n_timesteps = int(self.config.time_horizon / self.config.dt)
        trajectory = np.zeros((n_timesteps, self.config.n_packets, 7))
        labels = np.zeros((n_timesteps, self.config.n_packets))

        for t in range(n_timesteps):
            current_time = t * self.config.dt

            # Apply failures at their timestamps
            for event in failure_events:
                if not event.metadata["applied"] and current_time >= event.timestamp:
                    # Validate packet ID before indexing
                    if (event.affected_packet_id is not None and
                        0 <= event.affected_packet_id < len(stream.packets)):
                        packet = stream.packets[event.affected_packet_id]
                        self.failure_library.apply_failure(event, packet, current_time)
                        event.metadata["applied"] = True

                        # Mark label as failure for this packet
                        failure_start_idx = int(event.timestamp / self.config.dt)
                        failure_end_idx = min(n_timesteps, failure_start_idx + FAILURE_DURATION_TIMESTEPS)
                        if 0 <= event.affected_packet_id < labels.shape[1]:
                            labels[failure_start_idx:failure_end_idx, event.affected_packet_id] = 1

            # Integrate dynamics
            def zero_torque(packet_id, t_sim, state):
                return np.array([0.0, 0.0, 0.0])

            stream.integrate(self.config.dt, zero_torque)

            # Record trajectory
            for i, packet in enumerate(stream.packets):
                trajectory[t, i, :4] = packet.body.quaternion
                trajectory[t, i, 4:] = packet.body.angular_velocity

        return trajectory, failure_events, labels

    def generate_dataset(self) -> dict:
        """
        Generate full synthetic failure dataset.

        Returns:
            Dictionary with dataset metadata
        """
        logger.info(f"Generating {self.config.n_samples} samples...")

        trajectories = []
        labels_list = []
        failure_events_list = []
        metadata_list = []

        for i in range(self.config.n_samples):
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{self.config.n_samples} samples")

            trajectory, failure_events, labels = self.generate_sample(i)

            trajectories.append(trajectory)
            labels_list.append(labels)
            failure_events_list.append(failure_events)

            # Metadata for this sample
            metadata = {
                "sample_id": i,
                "n_failures": len(failure_events),
                "failure_types": [e.failure_type.value for e in failure_events],
                "failure_severities": [e.severity for e in failure_events],
            }
            metadata_list.append(metadata)

        # Convert to numpy arrays
        trajectories_array = np.array(trajectories)
        labels_array = np.array(labels_list)

        logger.info(f"Dataset generation complete")
        logger.info(f"  Trajectories shape: {trajectories_array.shape}")
        logger.info(f"  Labels shape: {labels_array.shape}")
        logger.info(f"  Total failures: {sum(len(events) for events in failure_events_list)}")

        return {
            "trajectories": trajectories_array,
            "labels": labels_array,
            "failure_events": failure_events_list,
            "metadata": metadata_list,
        }

    def save_dataset_hdf5(
        self,
        dataset: dict,
        filepath: Optional[str] = None,
    ) -> str:
        """
        Save dataset to HDF5 format.

        Args:
            dataset: Dataset dictionary
            filepath: Output filepath (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filepath is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(output_dir / f"{self.config.dataset_name}.h5")

        logger.info(f"Saving dataset to {filepath}")

        with h5py.File(filepath, "w") as f:
            # Save trajectories
            f.create_dataset("trajectories", data=dataset["trajectories"], compression="gzip")

            # Save labels
            f.create_dataset("labels", data=dataset["labels"], compression="gzip")

            # Save metadata as attributes
            f.attrs["n_samples"] = self.config.n_samples
            f.attrs["time_horizon"] = self.config.time_horizon
            f.attrs["dt"] = self.config.dt
            f.attrs["n_packets"] = self.config.n_packets
            f.attrs["random_seed"] = self.config.random_seed

            # Save failure events as separate group
            events_group = f.create_group("failure_events")
            for i, events in enumerate(dataset["failure_events"]):
                if len(events) > 0:
                    event_data = [
                        {
                            "type": e.failure_type.value,
                            "severity": e.severity,
                            "timestamp": e.timestamp,
                            "packet_id": e.affected_packet_id,
                        }
                        for e in events
                    ]
                    events_group.create_dataset(f"sample_{i}", data=str(event_data))

        logger.info(f"Dataset saved successfully")
        return filepath


def generate_synthetic_failure_dataset(
    n_samples: int = 10000,
    output_dir: str = "control_layer/data",
    random_seed: int = 42,
) -> str:
    """
    Convenience function to generate synthetic failure dataset.

    Args:
        n_samples: Number of samples to generate
        output_dir: Output directory
        random_seed: Random seed

    Returns:
        Path to generated dataset file
    """
    config = DataGenerationConfig(
        n_samples=n_samples,
        output_dir=output_dir,
        random_seed=random_seed,
    )

    generator = SyntheticDataGenerator(config)
    dataset = generator.generate_dataset()
    filepath = generator.save_dataset_hdf5(dataset)

    return filepath


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Generate dataset
    filepath = generate_synthetic_failure_dataset(
        n_samples=1000,  # Smaller for testing
        output_dir="control_layer/data",
        random_seed=42,
    )

    print(f"\nSynthetic failure dataset generated: {filepath}")

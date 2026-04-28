"""
Training data generator for ML models.

Generates synthetic training data from high-fidelity simulator for
wobble detection and trajectory prediction tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from dynamics.multi_body import MultiBodyStream
from dynamics.rigid_body import RigidBody
from dynamics.multi_body import Packet, SNode

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """Configuration for training data generator."""
    dt: float = 0.01
    stream_velocity: float = 1600.0
    packet_mass: float = 0.05
    random_seed: int = 42


class TrainingDataGenerator:
    """
    Generate synthetic training data from high-fidelity simulator.

    Uses the existing MultiBodyStream simulator to generate trajectories
    with various perturbations for ML model training.
    """

    def __init__(self, config: GeneratorConfig | None = None):
        """
        Initialize training data generator.

        Args:
            config: Generator configuration. If None, uses defaults.
        """
        self.config = config or GeneratorConfig()
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        logger.info(f"Training data generator initialized with seed {self.config.random_seed}")

    def generate_trajectory(
        self,
        n_packets: int = 10,
        n_timesteps: int = 1000,
        perturbation_types: List[str] | None = None,
    ) -> np.ndarray:
        """
        Generate trajectory from high-fidelity simulator.

        Args:
            n_packets: Number of packets in stream
            n_timesteps: Number of timesteps to simulate
            perturbation_types: List of perturbation types to apply

        Returns:
            trajectory: [n_timesteps × n_packets × state_dim]
            state_dim = 7 (quaternion [4] + angular velocity [3])
        """
        # Create multi-body stream
        stream = self._create_stream(n_packets)

        # Apply perturbations (debris, thermal, magnetic)
        if perturbation_types:
            self._apply_perturbations(stream, perturbation_types)

        # Integrate dynamics
        trajectory = np.zeros((n_timesteps, n_packets, 7))
        for t in range(n_timesteps):
            def zero_torque(packet_id: int, t_sim: float, state: np.ndarray) -> np.ndarray:
                return np.array([0.0, 0.0, 0.0])
            
            stream.integrate(self.config.dt, zero_torque)
            for i, packet in enumerate(stream.packets):
                trajectory[t, i, :4] = packet.body.quaternion
                trajectory[t, i, 4:] = packet.body.angular_velocity

        return trajectory

    def generate_wobble_dataset(
        self,
        n_samples: int = 1000,
        wobble_magnitude_range: Tuple[float, float] = (0.1, 0.5),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dataset for wobble detection training.

        Args:
            n_samples: Number of samples to generate
            wobble_magnitude_range: Range of wobble magnitudes

        Returns:
            signals: [n_samples × n_timesteps]
            labels: [n_samples] (1 = wobble, 0 = no wobble)
        """
        # Reset seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        signals = []
        labels = []
        n_timesteps = 1000

        for i in range(n_samples):
            # Generate trajectory
            trajectory = self.generate_trajectory(n_packets=1, n_timesteps=n_timesteps)
            
            # Extract angular velocity signal
            signal = trajectory[:, 0, 4:]  # [n_timesteps × 3]
            signal_magnitude = np.linalg.norm(signal, axis=1)
            
            # Randomly decide to add wobble
            if np.random.random() < 0.5:
                # Add wobble (high-frequency component)
                wobble_mag = np.random.uniform(*wobble_magnitude_range)
                wobble = wobble_mag * np.sin(2 * np.pi * 50 * np.arange(n_timesteps) / n_timesteps)
                signal_magnitude += wobble
                labels.append(1)
            else:
                labels.append(0)
            
            signals.append(signal_magnitude)

        return np.array(signals), np.array(labels)

    def generate_prediction_dataset(
        self,
        n_samples: int = 1000,
        prediction_horizon: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dataset for trajectory prediction training.

        Args:
            n_samples: Number of samples
            prediction_horizon: Prediction horizon

        Returns:
            inputs: [n_samples × history_length × state_dim]
            targets: [n_samples × prediction_horizon × state_dim]
        """
        # Reset seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        history_length = 100
        inputs = []
        targets = []

        for i in range(n_samples):
            # Generate long trajectory
            total_timesteps = history_length + prediction_horizon
            trajectory = self.generate_trajectory(n_timesteps=total_timesteps, n_packets=1)
            
            # Extract single packet
            packet_trajectory = trajectory[:, 0, :]  # [n_timesteps × 7]
            
            # Split into history and future
            inputs.append(packet_trajectory[:history_length, :])
            targets.append(packet_trajectory[history_length:, :])

        return np.array(inputs), np.array(targets)

    def _create_stream(self, n_packets: int) -> MultiBodyStream:
        """
        Create multi-body stream for simulation.

        Args:
            n_packets: Number of packets

        Returns:
            MultiBodyStream instance
        """
        packets = []
        for i in range(n_packets):
            I = np.diag([0.0001, 0.00011, 0.00009])
            position = np.array([i * 10.0, 0.0, 0.0])
            velocity = np.array([self.config.stream_velocity, 0.0, 0.0])
            body = RigidBody(self.config.packet_mass, I, position=position, velocity=velocity)
            packets.append(Packet(id=i, body=body))

        nodes = [SNode(id=i, position=np.array([i * 20.0, 0.0, 0.0])) for i in range(3)]

        return MultiBodyStream(packets=packets, nodes=nodes, stream_velocity=self.config.stream_velocity)

    def _apply_perturbations(self, stream: MultiBodyStream, perturbation_types: List[str]):
        """
        Apply perturbations to stream.

        Args:
            stream: MultiBodyStream instance
            perturbation_types: List of perturbation types
        """
        for perturbation_type in perturbation_types:
            if perturbation_type == "debris":
                # Apply debris impact
                for packet in stream.packets:
                    impulse = np.random.randn(3) * 0.1
                    packet.body.velocity += impulse / packet.body.mass
            elif perturbation_type == "thermal":
                # Apply thermal transient
                for packet in stream.packets:
                    packet.temperature += np.random.uniform(0, 50)
            elif perturbation_type == "magnetic":
                # Apply magnetic noise
                for packet in stream.packets:
                    packet.eta_ind *= np.random.uniform(0.9, 1.0)

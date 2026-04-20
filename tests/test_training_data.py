"""
Unit tests for training data generator.
"""

import numpy as np
import pytest

from control_layer.training_data_generator import (
    GeneratorConfig,
    TrainingDataGenerator,
)


class TestGeneratorConfig:
    """Test generator configuration dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        config = GeneratorConfig()
        assert config.dt == 0.01
        assert config.stream_velocity == 1600.0
        assert config.packet_mass == 0.05
        assert config.random_seed == 42

    def test_custom_parameters(self):
        """Test custom parameter values."""
        config = GeneratorConfig(
            dt=0.02,
            stream_velocity=2000.0,
            packet_mass=0.1,
            random_seed=123,
        )
        assert config.dt == 0.02
        assert config.stream_velocity == 2000.0
        assert config.packet_mass == 0.1
        assert config.random_seed == 123


class TestTrainingDataGenerator:
    """Test training data generator implementation."""

    def test_initialization(self):
        """Test generator initialization."""
        config = GeneratorConfig(random_seed=42)
        generator = TrainingDataGenerator(config)
        assert generator.config.dt == 0.01
        assert generator.config.random_seed == 42

    def test_initialization_default_config(self):
        """Test generator with default configuration."""
        generator = TrainingDataGenerator()
        assert generator.config.dt == 0.01
        assert generator.config.random_seed == 42

    def test_generate_trajectory_shape(self):
        """Test trajectory generation output shape."""
        generator = TrainingDataGenerator()
        trajectory = generator.generate_trajectory(n_packets=2, n_timesteps=100)
        assert trajectory.shape == (100, 2, 7)

    def test_generate_trajectory_single_packet(self):
        """Test trajectory generation with single packet."""
        generator = TrainingDataGenerator()
        trajectory = generator.generate_trajectory(n_packets=1, n_timesteps=100)
        assert trajectory.shape == (100, 1, 7)

    def test_generate_trajectory_multiple_packets(self):
        """Test trajectory generation with multiple packets."""
        generator = TrainingDataGenerator()
        trajectory = generator.generate_trajectory(n_packets=5, n_timesteps=100)
        assert trajectory.shape == (100, 5, 7)

    def test_generate_trajectory_no_perturbations(self):
        """Test trajectory generation without perturbations."""
        generator = TrainingDataGenerator()
        trajectory = generator.generate_trajectory(n_packets=2, n_timesteps=100, perturbation_types=None)
        assert trajectory.shape == (100, 2, 7)

    def test_generate_trajectory_with_debris_perturbation(self):
        """Test trajectory generation with debris perturbation."""
        generator = TrainingDataGenerator()
        trajectory = generator.generate_trajectory(
            n_packets=2,
            n_timesteps=100,
            perturbation_types=["debris"],
        )
        assert trajectory.shape == (100, 2, 7)

    def test_generate_trajectory_with_thermal_perturbation(self):
        """Test trajectory generation with thermal perturbation."""
        generator = TrainingDataGenerator()
        trajectory = generator.generate_trajectory(
            n_packets=2,
            n_timesteps=100,
            perturbation_types=["thermal"],
        )
        assert trajectory.shape == (100, 2, 7)

    def test_generate_trajectory_with_magnetic_perturbation(self):
        """Test trajectory generation with magnetic perturbation."""
        generator = TrainingDataGenerator()
        trajectory = generator.generate_trajectory(
            n_packets=2,
            n_timesteps=100,
            perturbation_types=["magnetic"],
        )
        assert trajectory.shape == (100, 2, 7)

    def test_generate_trajectory_with_multiple_perturbations(self):
        """Test trajectory generation with multiple perturbations."""
        generator = TrainingDataGenerator()
        trajectory = generator.generate_trajectory(
            n_packets=2,
            n_timesteps=100,
            perturbation_types=["debris", "thermal"],
        )
        assert trajectory.shape == (100, 2, 7)

    def test_generate_wobble_dataset_shape(self):
        """Test wobble dataset output shape."""
        generator = TrainingDataGenerator()
        signals, labels = generator.generate_wobble_dataset(n_samples=100)
        assert signals.shape == (100, 1000)
        assert labels.shape == (100,)

    def test_generate_wobble_dataset_labels(self):
        """Test wobble dataset labels are binary."""
        generator = TrainingDataGenerator()
        signals, labels = generator.generate_wobble_dataset(n_samples=100)
        assert set(labels) == {0, 1}

    def test_generate_wobble_dataset_no_nan(self):
        """Test wobble dataset has no NaN values."""
        generator = TrainingDataGenerator()
        signals, labels = generator.generate_wobble_dataset(n_samples=100)
        assert not np.any(np.isnan(signals))
        assert not np.any(np.isnan(labels))

    def test_generate_wobble_dataset_no_inf(self):
        """Test wobble dataset has no infinity values."""
        generator = TrainingDataGenerator()
        signals, labels = generator.generate_wobble_dataset(n_samples=100)
        assert not np.any(np.isinf(signals))
        assert not np.any(np.isinf(labels))

    def test_generate_wobble_dataset_different_magnitude_range(self):
        """Test wobble dataset with different magnitude range."""
        generator = TrainingDataGenerator()
        signals, labels = generator.generate_wobble_dataset(
            n_samples=50,
            wobble_magnitude_range=(0.5, 1.0),
        )
        assert signals.shape == (50, 1000)
        assert labels.shape == (50,)

    def test_generate_prediction_dataset_shape(self):
        """Test prediction dataset output shape."""
        generator = TrainingDataGenerator()
        inputs, targets = generator.generate_prediction_dataset(n_samples=100)
        assert inputs.shape == (100, 100, 7)
        assert targets.shape == (100, 10, 7)

    def test_generate_prediction_dataset_different_horizon(self):
        """Test prediction dataset with different horizon."""
        generator = TrainingDataGenerator()
        inputs, targets = generator.generate_prediction_dataset(
            n_samples=50,
            prediction_horizon=20,
        )
        assert inputs.shape == (50, 100, 7)
        assert targets.shape == (50, 20, 7)

    def test_generate_prediction_dataset_no_nan(self):
        """Test prediction dataset has no NaN values."""
        generator = TrainingDataGenerator()
        inputs, targets = generator.generate_prediction_dataset(n_samples=100)
        assert not np.any(np.isnan(inputs))
        assert not np.any(np.isnan(targets))

    def test_generate_prediction_dataset_no_inf(self):
        """Test prediction dataset has no infinity values."""
        generator = TrainingDataGenerator()
        inputs, targets = generator.generate_prediction_dataset(n_samples=100)
        assert not np.any(np.isinf(inputs))
        assert not np.any(np.isinf(targets))

    def test_reproducibility_same_seed(self):
        """Test same seed produces same dataset."""
        config1 = GeneratorConfig(random_seed=42)
        config2 = GeneratorConfig(random_seed=42)
        
        generator1 = TrainingDataGenerator(config1)
        generator2 = TrainingDataGenerator(config2)
        
        signals1, labels1 = generator1.generate_wobble_dataset(n_samples=50)
        signals2, labels2 = generator2.generate_wobble_dataset(n_samples=50)
        
        np.testing.assert_array_equal(signals1, signals2)
        np.testing.assert_array_equal(labels1, labels2)

    def test_reproducibility_different_seed(self):
        """Test different seed produces different dataset."""
        config1 = GeneratorConfig(random_seed=42)
        config2 = GeneratorConfig(random_seed=123)
        
        generator1 = TrainingDataGenerator(config1)
        generator2 = TrainingDataGenerator(config2)
        
        signals1, labels1 = generator1.generate_wobble_dataset(n_samples=50)
        signals2, labels2 = generator2.generate_wobble_dataset(n_samples=50)
        
        # Should be different
        assert not np.array_equal(signals1, signals2)

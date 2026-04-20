"""
Training script for VMD-IRCNN models.

Generates synthetic training data and trains wobble detection
and prediction models using the training pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from control_layer.training_data_generator import TrainingDataGenerator, GeneratorConfig
from control_layer.ircnn_predictor import IRCNNPredictor, IRCNNParameters
from control_layer.training_pipeline import TrainingPipeline, TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WobbleDataset(Dataset):
    """Dataset for wobble detection."""

    def __init__(self, signals, labels):
        if not isinstance(signals, np.ndarray):
            raise TypeError("signals must be numpy array")
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be numpy array")
        if signals.ndim != 2:
            raise ValueError(f"signals must be 2D array, got shape {signals.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D array, got shape {labels.shape}")
        if len(signals) != len(labels):
            raise ValueError(f"signals and labels must have same length, got {len(signals)} and {len(labels)}")
        
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


class PredictionDataset(Dataset):
    """Dataset for trajectory prediction."""

    def __init__(self, inputs, targets):
        if not isinstance(inputs, np.ndarray):
            raise TypeError("inputs must be numpy array")
        if not isinstance(targets, np.ndarray):
            raise TypeError("targets must be numpy array")
        if inputs.ndim != 3:
            raise ValueError(f"inputs must be 3D array, got shape {inputs.shape}")
        if targets.ndim != 3:
            raise ValueError(f"targets must be 3D array, got shape {targets.shape}")
        if len(inputs) != len(targets):
            raise ValueError(f"inputs and targets must have same length, got {len(inputs)} and {len(targets)}")
        
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class SimpleWobbleDetector(torch.nn.Module):
    """Simple wobble detector for training."""

    def __init__(self, input_dim=1000, hidden_dim=64):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class SimplePredictor(torch.nn.Module):
    """Simple predictor for training."""

    def __init__(self, input_dim=100 * 7, hidden_dim=64, output_dim=10 * 7):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        expected_input_features = 100 * 7  # 700
        if x.shape[-1] != expected_input_features:
            raise ValueError(f"Expected input with {expected_input_features} features, got {x.shape[-1]}")
        
        x_flat = x.view(batch_size, -1)
        output = self.network(x_flat)
        return output.view(batch_size, 10, 7)


def main():
    """Main training function."""
    logger.info("Starting VMD-IRCNN training pipeline")

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate training data
    logger.info("Generating training data...")
    generator = TrainingDataGenerator(GeneratorConfig(random_seed=42))

    # Generate wobble dataset
    logger.info("Generating wobble detection dataset...")
    signals, labels = generator.generate_wobble_dataset(n_samples=1000)
    
    # Split into train/val
    split_idx = int(0.8 * len(signals))
    train_signals = signals[:split_idx]
    train_labels = labels[:split_idx]
    val_signals = signals[split_idx:]
    val_labels = labels[split_idx:]

    train_wobble_dataset = WobbleDataset(train_signals, train_labels)
    val_wobble_dataset = WobbleDataset(val_signals, val_labels)

    logger.info(f"Wobble dataset: Train={len(train_wobble_dataset)}, Val={len(val_wobble_dataset)}")

    # Generate prediction dataset
    logger.info("Generating prediction dataset...")
    inputs, targets = generator.generate_prediction_dataset(n_samples=1000)

    # Split into train/val
    split_idx = int(0.8 * len(inputs))
    train_inputs = inputs[:split_idx]
    train_targets = targets[:split_idx]
    val_inputs = inputs[split_idx:]
    val_targets = targets[split_idx:]

    train_pred_dataset = PredictionDataset(train_inputs, train_targets)
    val_pred_dataset = PredictionDataset(val_inputs, val_targets)

    logger.info(f"Prediction dataset: Train={len(train_pred_dataset)}, Val={len(val_pred_dataset)}")

    # Save datasets
    np.savez(data_dir / "wobble_train.npz", signals=train_signals, labels=train_labels)
    np.savez(data_dir / "wobble_val.npz", signals=val_signals, labels=val_labels)
    np.savez(data_dir / "prediction_train.npz", inputs=train_inputs, targets=train_targets)
    np.savez(data_dir / "prediction_val.npz", inputs=val_inputs, targets=val_targets)
    logger.info("Datasets saved to data/")

    # Setup training pipeline
    logger.info("Setting up training pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    config = TrainingConfig(
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5,
        epochs=50,
        patience=10,
        checkpoint_interval=5,
        device=device,
    )

    pipeline = TrainingPipeline(config)
    logger.info(f"Training on device: {pipeline.device}")

    # Train wobble detector
    logger.info("Training wobble detector...")
    wobble_model = SimpleWobbleDetector(input_dim=1000, hidden_dim=64)

    wobble_success = False
    try:
        wobble_metrics = pipeline.train_wobble_detector(
            wobble_model,
            train_wobble_dataset,
            val_wobble_dataset,
            save_dir="data/models/wobble_detector/v1.0.0",
        )
        logger.info(f"Wobble detector training complete. Best val loss: {wobble_metrics['best_val_loss']:.6f}")
        wobble_success = True
    except Exception as e:
        logger.error(f"Wobble detector training failed: {e}")
        logger.error("Skipping predictor training due to wobble detector failure")
        logger.info("Training pipeline complete with errors")
        return

    # Train predictor
    logger.info("Training predictor...")
    predictor_model = SimplePredictor(input_dim=100 * 7, hidden_dim=64, output_dim=10 * 7)

    try:
        pred_metrics = pipeline.train_predictor(
            predictor_model,
            train_pred_dataset,
            val_pred_dataset,
            save_dir="data/models/thermal_predictor/v1.0.0",
        )
        logger.info(f"Predictor training complete. Best val loss: {pred_metrics['best_val_loss']:.6f}")
    except Exception as e:
        logger.error(f"Predictor training failed: {e}")

    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()

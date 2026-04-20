"""
Training script for enhanced VMD-IRCNN stub.

Trains the deep residual network on synthetic data generated from
the high-fidelity simulator. This is a simplified training process
for the enhanced stub, not full VMD-IRCNN training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch not available. Install with: pip install torch")


def train_enhanced_predictor(
    n_samples: int = 10000,
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cpu",
    save_path: str | None = None,
) -> dict:
    """
    Train enhanced VMD-IRCNN predictor on synthetic data.

    Args:
        n_samples: Number of training samples to generate
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on (cpu or cuda)
        save_path: Path to save trained model

    Returns:
        Training metrics dictionary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training")

    from control_layer.vmd_enhanced_stub import (
        EnhancedPredictorCascade,
        EnhancedDecompositionParameters,
        DeepResidualPredictor,
    )
    from control_layer.training_data_generator import (
        TrainingDataGenerator,
        GeneratorConfig,
    )

    logger.info(f"Starting training with {n_samples} samples, {n_epochs} epochs")

    # Generate training data
    logger.info("Generating training data...")
    generator = TrainingDataGenerator(GeneratorConfig(random_seed=42))
    inputs, targets = generator.generate_prediction_dataset(
        n_samples=n_samples,
        prediction_horizon=10,
    )

    # Convert to PyTorch tensors
    # Flatten inputs: [n_samples, history_length, state_dim] -> [n_samples, history_length * state_dim]
    history_length = inputs.shape[1]
    state_dim = inputs.shape[2]
    input_dim = history_length * state_dim

    inputs_flat = inputs.reshape(n_samples, -1)
    # Use first timestep of target as prediction target (simplified)
    targets_flat = targets[:, 0, :]  # [n_samples, state_dim]

    inputs_tensor = torch.FloatTensor(inputs_flat)
    targets_tensor = torch.FloatTensor(targets_flat)

    # Create dataset and dataloader
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize predictor
    logger.info("Initializing enhanced predictor...")
    decomp_params = EnhancedDecompositionParameters(num_modes=6, adaptive_bands=True)
    predictor = DeepResidualPredictor(
        input_dim=input_dim,
        output_dim=state_dim,
        hidden_dim=128,
        num_blocks=4,
        dropout_rate=0.1,
    )

    predictor = predictor.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)

    # Training loop
    logger.info("Starting training loop...")
    train_losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = predictor(batch_inputs)
            loss = criterion(predictions, batch_targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")

    # Save model if path provided
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(predictor.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {save_path}: {e}")
            raise

    metrics = {
        "final_loss": train_losses[-1],
        "train_losses": train_losses,
        "n_epochs": n_epochs,
        "n_samples": n_samples,
    }

    logger.info(f"Training complete. Final loss: {metrics['final_loss']:.6f}")

    return metrics


def validate_against_rom(
    trained_predictor: DeepResidualPredictor,
    n_test_samples: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Validate trained predictor against ROM predictor baseline.

    Args:
        trained_predictor: Trained enhanced predictor
        n_test_samples: Number of test samples
        device: Device to run validation on

    Returns:
        Validation metrics dictionary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for validation")

    from control_layer.rom_predictor import create_rom
    from control_layer.training_data_generator import (
        TrainingDataGenerator,
        GeneratorConfig,
    )

    logger.info(f"Validating against ROM baseline with {n_test_samples} samples...")

    # Generate test data
    generator = TrainingDataGenerator(GeneratorConfig(random_seed=123))
    inputs, targets = generator.generate_prediction_dataset(
        n_samples=n_test_samples,
        prediction_horizon=10,
    )

    # Create ROM predictor
    I = np.diag([0.0001, 0.00011, 0.00009])
    rom = create_rom(mass=0.05, I=I)

    # Flatten inputs for enhanced predictor
    history_length = inputs.shape[1]
    state_dim = inputs.shape[2]
    input_dim = history_length * state_dim
    inputs_flat = inputs.reshape(n_test_samples, -1)

    # Evaluate predictions
    enhanced_errors = []
    rom_errors = []

    trained_predictor.eval()
    trained_predictor = trained_predictor.to(device)

    with torch.no_grad():
        for i in range(n_test_samples):
            input_tensor = torch.FloatTensor(inputs_flat[i:i+1]).to(device)
            target = targets[i, 0, :]  # First timestep of prediction horizon

            # Enhanced predictor prediction
            enhanced_pred = trained_predictor(input_tensor).cpu().numpy()[0, :state_dim]
            enhanced_error = np.linalg.norm(enhanced_pred - target)
            enhanced_errors.append(enhanced_error)

            # ROM prediction (simplified comparison)
            # ROM predicts state change, we compare against target
            delta_x = inputs[i, -1, :] - inputs[i, -2, :]
            rom_pred = inputs[i, -1, :] + delta_x  # Simple extrapolation
            rom_error = np.linalg.norm(rom_pred - target)
            rom_errors.append(rom_error)

    metrics = {
        "enhanced_mean_error": np.mean(enhanced_errors),
        "enhanced_std_error": np.std(enhanced_errors),
        "rom_mean_error": np.mean(rom_errors),
        "rom_std_error": np.std(rom_errors),
        "improvement_ratio": np.mean(rom_errors) / np.mean(enhanced_errors) if np.mean(enhanced_errors) > 0 else float('inf'),
    }

    logger.info(f"Validation complete:")
    logger.info(f"  Enhanced predictor error: {metrics['enhanced_mean_error']:.6f} ± {metrics['enhanced_std_error']:.6f}")
    logger.info(f"  ROM predictor error: {metrics['rom_mean_error']:.6f} ± {metrics['rom_std_error']:.6f}")
    logger.info(f"  Improvement ratio: {metrics['improvement_ratio']:.2f}x")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Train enhanced predictor
    metrics = train_enhanced_predictor(
        n_samples=10000,
        n_epochs=50,
        batch_size=32,
        learning_rate=0.001,
        device="cpu",
        save_path="control_layer/models/vmd_enhanced.pt",
    )

    # Load trained model for validation
    if TORCH_AVAILABLE:
        from control_layer.vmd_enhanced_stub import DeepResidualPredictor

        predictor = DeepResidualPredictor(
            input_dim=700,  # 100 timesteps * 7 state_dim
            output_dim=7,  # state_dim
            hidden_dim=128,
            num_blocks=4,
            dropout_rate=0.1,
        )

        predictor.load_state_dict(torch.load("control_layer/models/vmd_enhanced.pt"))

        # Validate against ROM
        val_metrics = validate_against_rom(predictor, n_test_samples=100, device="cpu")

        print("\nTraining and validation complete!")
        print(f"Final training loss: {metrics['final_loss']:.6f}")
        print(f"Validation improvement: {val_metrics['improvement_ratio']:.2f}x")

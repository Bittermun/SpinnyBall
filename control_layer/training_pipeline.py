"""
Training pipeline with checkpointing for ML models.

Implements training infrastructure with checkpointing, learning rate scheduling,
early stopping, and GPU/CPU auto-detection.
"""

from __future__ import annotations

import logging
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    Dataset = None
    DataLoader = None

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 20
    checkpoint_interval: int = 10
    num_workers: Optional[int] = None  # None for auto-detect (min(4, os.cpu_count()))
    device: Optional[str] = None  # 'cuda', 'cpu', or None for auto-detect


class TrainingPipeline:
    """
    Training pipeline with checkpointing and early stopping.

    Supports:
    - Checkpointing (best + periodic)
    - Learning rate scheduling (ReduceLROnPlateau)
    - Early stopping
    - Gradient clipping
    - GPU/CPU auto-detection
    """

    def __init__(self, config: TrainingConfig | None = None):
        """
        Initialize training pipeline.

        Args:
            config: Training configuration. If None, uses defaults.
        """
        self.config = config or TrainingConfig()
        self.device = self._get_device()
        logger.info(f"Training pipeline initialized on device: {self.device}")

    def _get_device(self) -> str:
        """Auto-detect device (GPU if available, else CPU)."""
        if not TORCH_AVAILABLE:
            return "cpu"

        if self.config.device is not None:
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def train_wobble_detector(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        save_dir: str = "data/models/wobble_detector/v1.0.0",
    ) -> dict:
        """
        Train wobble detection model.

        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            save_dir: Directory to save checkpoints

        Returns:
            Training metrics dict
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training. Install with: pip install torch")

        # Validate datasets are not empty
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Determine num_workers
        num_workers = self.config.num_workers if self.config.num_workers is not None else min(4, os.cpu_count() or 1)

        # Setup data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        # Loss function (binary classification)
        criterion = nn.BCEWithLogitsLoss()

        # Move model to device
        model = model.to(self.device)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            try:
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target.float())
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    train_loss += loss.item()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.error(f"CUDA out of memory at epoch {epoch+1}. Try reducing batch size.")
                    raise
                else:
                    raise

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            try:
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        loss = criterion(output, target.float())
                        val_loss += loss.item()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.error(f"CUDA out of memory during validation at epoch {epoch+1}. Try reducing batch size.")
                    raise
                else:
                    raise

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best checkpoint
                self._save_checkpoint(model, optimizer, epoch, val_loss, save_path / "checkpoint_best.pth")
            else:
                patience_counter += 1

            # Periodic checkpointing
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(model, optimizer, epoch, val_loss, save_path / f"checkpoint_epoch_{epoch+1}.pth")

            logger.info(f"Epoch {epoch+1}/{self.config.epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            # Early stopping check
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save training metrics
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'config': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                      for k, v in self.config.__dict__.items()},
        }
        with open(save_path / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def train_predictor(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        save_dir: str = "data/models/thermal_predictor/v1.0.0",
    ) -> dict:
        """
        Train prediction model.

        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            save_dir: Directory to save checkpoints

        Returns:
            Training metrics dict
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training. Install with: pip install torch")

        # Validate datasets are not empty
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Determine num_workers
        num_workers = self.config.num_workers if self.config.num_workers is not None else min(4, os.cpu_count() or 1)

        # Setup data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )

        # Loss function (regression)
        criterion = nn.MSELoss()

        # Move model to device
        model = model.to(self.device)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            try:
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    train_loss += loss.item()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.error(f"CUDA out of memory at epoch {epoch+1}. Try reducing batch size.")
                    raise
                else:
                    raise

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            try:
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = model(data)
                        loss = criterion(output, target)
                        val_loss += loss.item()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.error(f"CUDA out of memory during validation at epoch {epoch+1}. Try reducing batch size.")
                    raise
                else:
                    raise

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best checkpoint
                self._save_checkpoint(model, optimizer, epoch, val_loss, save_path / "checkpoint_best.pth")
            else:
                patience_counter += 1

            # Periodic checkpointing
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(model, optimizer, epoch, val_loss, save_path / f"checkpoint_epoch_{epoch+1}.pth")

            logger.info(f"Epoch {epoch+1}/{self.config.epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            # Early stopping check
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save training metrics
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'config': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                      for k, v in self.config.__dict__.items()},
        }
        with open(save_path / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        path: str,
    ):
        """Save model checkpoint."""
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, path)
            logger.info(f"Saved checkpoint to {path}")
        except (OSError, IOError) as e:
            logger.error(f"Failed to save checkpoint to {path}: {e}")
            raise

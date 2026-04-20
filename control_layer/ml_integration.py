"""
ML integration layer with feature flags and async model loading.

Provides thread-safe model initialization with timeout and error handling,
supporting both stub and true ML implementations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MLIntegrationLayer:
    """
    ML integration layer with feature flags and async model loading.

    Supports:
    - Feature flags for stub vs true implementation switching
    - Thread-safe async model loading with timeout
    - Graceful fallback on errors
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize ML integration layer.

        Args:
            config_path: Path to ML config JSON file. If None, uses default.
        """
        if config_path is None:
            config_path = "config/ml_config.json"

        self.config_path = Path(config_path)
        self._lock = asyncio.Lock()
        self._config = None
        self._vmd_implementation = "stub"
        self._ircnn_implementation = "stub"
        self._enable_training = False
        self._model_paths = {}

        self.wobble_detector = None
        self.thermal_model = None
        self.use_stub = False  # Fallback flag

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load ML configuration from JSON file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            self._set_defaults()
            return

        try:
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)

            self._vmd_implementation = self._config.get("vmd_implementation", "stub")
            self._ircnn_implementation = self._config.get("ircnn_implementation", "stub")
            self._enable_training = self._config.get("enable_training", False)
            self._model_paths = self._config.get("model_paths", {})

            logger.info(f"Loaded ML config: VMD={self._vmd_implementation}, "
                       f"IRCNN={self._ircnn_implementation}, training={self._enable_training}")

        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}, using defaults")
            self._set_defaults()

    def _set_defaults(self):
        """Set default configuration values."""
        self._vmd_implementation = "stub"
        self._ircnn_implementation = "stub"
        self._enable_training = False
        self._model_paths = {}

    async def initialize(self, timeout: float = 30.0):
        """
        Async initialization with timeout and error handling.

        Args:
            timeout: Timeout in seconds for model loading
        """
        async with self._lock:
            if self.wobble_detector is None and not self.use_stub:
                try:
                    # Wait for initialization with timeout
                    await asyncio.wait_for(self._load_wobble_detector(), timeout=timeout)
                    logger.info("Wobble detector initialized successfully")
                except asyncio.TimeoutError:
                    logger.error("Wobble detector initialization timed out, falling back to stub")
                    self.use_stub = True
                except Exception as e:
                    logger.error(f"Wobble detector initialization failed: {e}, falling back to stub")
                    self.use_stub = True

            if self.thermal_model is None and not self.use_stub:
                try:
                    await asyncio.wait_for(self._load_thermal_model(), timeout=timeout)
                    logger.info("Thermal model initialized successfully")
                except asyncio.TimeoutError:
                    logger.error("Thermal model initialization timed out, falling back to stub")
                    self.use_stub = True
                except Exception as e:
                    logger.error(f"Thermal model initialization failed: {e}, falling back to stub")
                    self.use_stub = True

    async def _load_wobble_detector(self):
        """Load wobble detector model (async wrapper)."""
        # Run synchronous loading in thread pool
        await asyncio.to_thread(self._load_wobble_detector_sync)

    def _load_wobble_detector_sync(self):
        """Load wobble detector model synchronously."""
        try:
            if self._vmd_implementation == "true":
                # Load true VMD-IRCNN model
                from control_layer import VMD_AVAILABLE, VMDDecomposer
                from control_layer import IRCNN_AVAILABLE, IRCNNPredictor

                if not VMD_AVAILABLE or not IRCNN_AVAILABLE:
                    logger.warning("VMD or IRCNN not available, using stub")
                    return

                # Initialize models (would load checkpoints here)
                self.wobble_detector = VMDDecomposer()
                logger.info("Loaded true VMD-IRCNN wobble detector")
            else:
                # Use stub
                from control_layer import VMDIRCNNDetector
                self.wobble_detector = VMDIRCNNDetector()
                logger.info("Using stub VMD-IRCNN wobble detector")

        except FileNotFoundError:
            logger.warning("Model checkpoint not found, using stub")
            raise
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    async def _load_thermal_model(self):
        """Load thermal model (async wrapper)."""
        await asyncio.to_thread(self._load_thermal_model_sync)

    def _load_thermal_model_sync(self):
        """Load thermal model synchronously."""
        try:
            # Load thermal model (would load checkpoint here)
            # For now, use JAX thermal model as baseline
            from dynamics.jax_thermal import JAXThermalModel
            self.thermal_model = JAXThermalModel()
            logger.info("Loaded JAX thermal model")
        except Exception as e:
            logger.error(f"Thermal model loading failed: {e}")
            raise

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            "vmd_implementation": self._vmd_implementation,
            "ircnn_implementation": self._ircnn_implementation,
            "enable_training": self._enable_training,
            "model_paths": self._model_paths,
            "use_stub": self.use_stub,
        }

    def reload_config(self):
        """Reload configuration from file."""
        self._load_config()
        logger.info("Configuration reloaded")


# Global ML integration instance
_ml_integration: Optional[MLIntegrationLayer] = None


def get_ml_integration() -> MLIntegrationLayer:
    """Get global ML integration instance."""
    global _ml_integration
    if _ml_integration is None:
        _ml_integration = MLIntegrationLayer()
    return _ml_integration

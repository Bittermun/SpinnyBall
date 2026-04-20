"""
ML integration layer for dashboard and control.

Integrates VMD-IRCNN wobble detection and JAX thermal models
with the backend API.
"""

import logging

import numpy as np

from control_layer.vmd_ircnn_stub import VMDIRCNNDetector

try:
    from dynamics.jax_thermal import JAXThermalModel
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    JAXThermalModel = None

logger = logging.getLogger(__name__)


class MLIntegrationLayer:
    """
    Integration layer for ML models.

    Provides unified interface for wobble detection and
    thermal prediction with fallback logic.
    """

    def __init__(
        self,
        enable_wobble_detection: bool = True,
        enable_thermal_prediction: bool = True,
    ):
        """
        Initialize ML integration layer.

        Args:
            enable_wobble_detection: Enable VMD-IRCNN detector
            enable_thermal_prediction: Enable JAX thermal model
        """
        self.enable_wobble_detection = enable_wobble_detection
        self.enable_thermal_prediction = enable_thermal_prediction

        # Initialize models
        self.wobble_detector = None
        self.thermal_model = None

        try:
            if enable_wobble_detection:
                self.wobble_detector = VMDIRCNNDetector()
                logger.info("Wobble detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize wobble detector: {e}")

        try:
            if enable_thermal_prediction and JAX_AVAILABLE:
                self.thermal_model = JAXThermalModel()
                logger.info("Thermal model initialized")
            elif enable_thermal_prediction:
                logger.warning("Thermal model requested but JAX not available")
        except Exception as e:
            logger.warning(f"Failed to initialize thermal model: {e}")

    def detect_wobble_batch(
        self,
        signals: list[np.ndarray],
        threshold: float = 0.1,
    ) -> list[dict]:
        """
        Detect wobble in batch of signals.

        Args:
            signals: List of signals
            threshold: Detection threshold

        Returns:
            List of detection results
        """
        results = []

        for i, signal in enumerate(signals):
            if self.wobble_detector is not None:
                is_wobble, confidence, metadata = self.wobble_detector.detect_wobble(
                    signal, threshold
                )
                results.append({
                    'signal_id': i,
                    'is_wobble': is_wobble,
                    'confidence': confidence,
                    'metadata': metadata,
                })
            else:
                results.append({
                    'signal_id': i,
                    'is_wobble': False,
                    'confidence': 0.0,
                    'metadata': {'error': 'Wobble detector unavailable'},
                })

        return results

    def predict_thermal_batch(
        self,
        T_initial: np.ndarray,  # noqa: N803
        Q_in: np.ndarray,  # noqa: N803
        T_amb: float = 293.15,  # noqa: N803
    ) -> dict:
        """
        Predict thermal evolution for batch.

        Args:
            T_initial: Initial temperatures
            Q_in: Heat input rates
            T_amb: Ambient temperature

        Returns:
            Prediction results
        """
        if self.thermal_model is not None:
            temperatures, metadata = self.thermal_model.predict_temperatures(
                T_initial, Q_in, T_amb
            )
            return {
                'success': True,
                'temperatures': temperatures.tolist(),
                'metadata': metadata,
            }
        else:
            return {
                'success': False,
                'error': 'Thermal model unavailable',
            }

    def get_model_status(self) -> dict:
        """Get status of all ML models."""
        return {
            'wobble_detector': {
                'available': self.wobble_detector is not None,
                'info': self.wobble_detector.get_model_info() if self.wobble_detector else None,
            },
            'thermal_model': {
                'available': self.thermal_model is not None,
                'info': self.thermal_model.get_model_info() if self.thermal_model else None,
            },
        }

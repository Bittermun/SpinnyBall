"""
ML integration layer for dashboard and control.

Integrates VMD-IRCNN wobble detection and JAX thermal models
with the backend API.
"""

import logging
import time

import numpy as np

# Try to import true VMD and IRCNN implementations
try:
    from control_layer.vmd_decomposition import VMDDecomposer, VMDParameters
    from control_layer.ircnn_predictor import IRCNNPredictor, IRCNNParameters
    TRUE_VMD_AVAILABLE = True
except ImportError:
    TRUE_VMD_AVAILABLE = False
    VMDDecomposer = None
    VMDParameters = None
    IRCNNPredictor = None
    IRCNNParameters = None

# Keep stub as fallback
from control_layer.vmd_ircnn_stub import VMDIRCNNDetector as StubDetector

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
        use_true_vmd: bool = False,  # Default to stub for real-time performance
    ):
        """
        Initialize ML integration layer.

        Args:
            enable_wobble_detection: Enable VMD-IRCNN detector
            enable_thermal_prediction: Enable JAX thermal model
            use_true_vmd: Use true VMD/IRCNN if available (slower, for offline analysis).
                          Default False uses stub (fast, for real-time detection).
        """
        self.enable_wobble_detection = enable_wobble_detection
        self.enable_thermal_prediction = enable_thermal_prediction
        self.use_true_vmd = use_true_vmd

        # Initialize models
        self.wobble_detector = None
        self.thermal_model = None
        self.vmd_decomposer = None
        self.ircnn_predictor = None

        # Try to use true VMD/IRCNN if available and requested
        if enable_wobble_detection and use_true_vmd and TRUE_VMD_AVAILABLE:
            try:
                self.vmd_decomposer = VMDDecomposer(VMDParameters())
                self.ircnn_predictor = IRCNNPredictor()
                logger.info("True VMD/IRCNN initialized (for offline analysis)")
            except Exception as e:
                logger.warning(f"Failed to initialize true VMD/IRCNN: {e}, falling back to stub")
                self._init_stub_detector()

        # Fallback to stub (default for real-time use)
        if self.vmd_decomposer is None and enable_wobble_detection:
            self._init_stub_detector()

        # Initialize thermal model
        try:
            if enable_thermal_prediction and JAX_AVAILABLE:
                self.thermal_model = JAXThermalModel()
                logger.info("Thermal model initialized")
            elif enable_thermal_prediction:
                logger.warning("Thermal model requested but JAX not available")
        except Exception as e:
            logger.warning(f"Failed to initialize thermal model: {e}")

    def _init_stub_detector(self):
        """Initialize stub detector as fallback."""
        try:
            self.wobble_detector = StubDetector()
            logger.info("Stub VMD-IRCNN detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize stub detector: {e}")

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
        start_time = time.perf_counter()

        for i, signal in enumerate(signals):
            signal_start = time.perf_counter()

            # Use true VMD/IRCNN if available
            if self.vmd_decomposer is not None and self.ircnn_predictor is not None:
                try:
                    # VMD decomposition
                    modes = self.vmd_decomposer.decompose(signal)

                    # IRCNN prediction (simplified: use mode energy as feature)
                    mode_energy = np.sum(modes ** 2, axis=1)
                    features = mode_energy[:7] if len(mode_energy) >= 7 else np.pad(mode_energy, (0, 7 - len(mode_energy)))

                    # Simple threshold-based detection (true IRCNN would use trained model)
                    is_wobble = bool(np.max(mode_energy) > threshold)
                    confidence = np.max(mode_energy) / (np.sum(mode_energy) + 1e-10)

                    signal_latency = (time.perf_counter() - signal_start) * 1000  # ms

                    # Warn if latency > 5 ms
                    if signal_latency > 5.0:
                        logger.warning(f"VMD detection latency high: {signal_latency:.1f} ms")

                    results.append({
                        'signal_id': i,
                        'is_wobble': is_wobble,
                        'confidence': float(confidence),
                        'metadata': {
                            'method': 'true_vmd_ircnn',
                            'latency_ms': signal_latency,
                            'mode_energy': mode_energy.tolist(),
                        },
                    })
                except Exception as e:
                    logger.warning(f"True VMD/IRCNN detection failed for signal {i}: {e}, falling back to stub")
                    # Fall back to stub for this signal
                    if self.wobble_detector is not None:
                        is_wobble, confidence, metadata = self.wobble_detector.detect_wobble(signal, threshold)
                        results.append({
                            'signal_id': i,
                            'is_wobble': is_wobble,
                            'confidence': confidence,
                            'metadata': {**metadata, 'method': 'stub_fallback'},
                        })
                    else:
                        results.append({
                            'signal_id': i,
                            'is_wobble': False,
                            'confidence': 0.0,
                            'metadata': {'error': 'All detectors unavailable'},
                        })

            # Use stub detector
            elif self.wobble_detector is not None:
                is_wobble, confidence, metadata = self.wobble_detector.detect_wobble(signal, threshold)
                signal_latency = (time.perf_counter() - signal_start) * 1000  # ms
                results.append({
                    'signal_id': i,
                    'is_wobble': is_wobble,
                    'confidence': confidence,
                    'metadata': {**metadata, 'method': 'stub', 'latency_ms': signal_latency},
                })

            # No detector available
            else:
                results.append({
                    'signal_id': i,
                    'is_wobble': False,
                    'confidence': 0.0,
                    'metadata': {'error': 'Wobble detector unavailable'},
                })

        total_latency = (time.perf_counter() - start_time) * 1000  # ms
        avg_latency = total_latency / len(signals) if signals else 0

        # Log average latency
        logger.info(f"Batch detection: {len(signals)} signals, avg latency: {avg_latency:.1f} ms")

        # Warn if average latency > 10 ms
        if avg_latency > 10.0:
            logger.warning(f"Average detection latency too high: {avg_latency:.1f} ms (> 10 ms target)")

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
        status = {
            'wobble_detector': {
                'available': self.wobble_detector is not None or self.vmd_decomposer is not None,
                'method': 'true_vmd_ircnn' if self.vmd_decomposer is not None else 'stub' if self.wobble_detector is not None else 'unavailable',
                'vmd_available': TRUE_VMD_AVAILABLE,
                'ircnn_available': TRUE_VMD_AVAILABLE,
            },
            'thermal_model': {
                'available': self.thermal_model is not None,
                'jax_available': JAX_AVAILABLE,
            },
        }

        # Add model info if available
        if self.vmd_decomposer is not None:
            status['wobble_detector']['vmd_info'] = {
                'num_modes': self.vmd_decomposer.params.num_modes,
                'alpha': self.vmd_decomposer.params.alpha,
            }
        if self.ircnn_predictor is not None:
            status['wobble_detector']['ircnn_info'] = self.ircnn_predictor.get_model_info()
        if self.wobble_detector is not None:
            status['wobble_detector']['stub_info'] = self.wobble_detector.get_model_info()

        return status

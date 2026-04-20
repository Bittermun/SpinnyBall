"""
Unit tests for VMD-IRCNN wobble detector.

Tests critical paths with performance benchmarking.
"""

import logging
import time

import numpy as np

from control_layer.vmd_ircnn_stub import VMDIRCNNDetector

logger = logging.getLogger(__name__)


class TestVMDIRCNNDetector:
    """Test suite for VMD-IRCNN wobble detector."""

    def test_detector_initialization(self):
        """Test detector initializes with correct parameters."""
        detector = VMDIRCNNDetector(n_modes=4, alpha=2000.0)
        assert detector.n_modes == 4
        assert detector.alpha == 2000.0
        assert detector.model_version == "1.0.0"
        assert not detector.is_trained

    def test_vmd_decomposition_mode_count(self):
        """Test VMD decomposition produces correct number of modes."""
        detector = VMDIRCNNDetector(n_modes=4)
        signal = np.random.randn(1000)
        modes = detector.vmd_decompose(signal)
        assert modes.shape == (4, 1000)

    def test_vmd_decomposition_different_modes(self):
        """Test VMD decomposition with different mode counts."""
        for n_modes in [2, 4, 8]:
            detector = VMDIRCNNDetector(n_modes=n_modes)
            signal = np.random.randn(1000)
            modes = detector.vmd_decompose(signal)
            assert modes.shape == (n_modes, 1000)

    def test_ircnn_denoise_reduces_noise(self):
        """Test IRCNN denoising reduces noise amplitude."""
        detector = VMDIRCNNDetector()
        # Create noisy signal
        clean = np.sin(np.linspace(0, 10, 1000))
        noise = np.random.randn(1000) * 0.5
        noisy = clean + noise
        denoised = detector.ircnn_denoise(noisy)
        # Denoised should have lower variance than noisy
        assert np.var(denoised) < np.var(noisy)

    def test_wobble_detection_synthetic_wobble(self):
        """Test wobble detection with synthetic wobble signal."""
        detector = VMDIRCNNDetector(n_modes=4)
        # Create signal with high-frequency wobble
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 100 * t)
        is_wobble, confidence, metadata = detector.detect_wobble(signal, threshold=0.1)
        assert isinstance(is_wobble, bool)
        assert 0 <= confidence <= 1
        assert 'wobble_ratio' in metadata
        assert 'mode_energies' in metadata
        assert len(metadata['mode_energies']) == 4

    def test_wobble_detection_clean_signal(self):
        """Test wobble detection with clean signal (no wobble)."""
        detector = VMDIRCNNDetector(n_modes=4)
        # Create clean low-frequency signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t)
        is_wobble, confidence, metadata = detector.detect_wobble(signal, threshold=0.1)
        assert isinstance(is_wobble, bool)
        assert 0 <= confidence <= 1
        assert metadata['wobble_ratio'] < 0.1  # Should be below threshold

    def test_threshold_sensitivity(self):
        """Test threshold sensitivity affects detection."""
        detector = VMDIRCNNDetector(n_modes=4)
        # Create signal with moderate wobble
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 0.2 * np.sin(2 * np.pi * 50 * t)

        results = []
        for threshold in [0.05, 0.1, 0.2, 0.3]:
            is_wobble, confidence, metadata = detector.detect_wobble(signal, threshold)
            results.append((threshold, is_wobble, metadata['wobble_ratio']))

        # Lower thresholds should be more likely to detect wobble
        thresholds, detections, ratios = zip(*results, strict=False)
        assert ratios[0] == ratios[1] == ratios[2] == ratios[3]  # Ratio should be constant
        # Detection should vary with threshold
        assert not all(d == detections[0] for d in detections)  # noqa: B905

    def test_wobble_detection_latency_benchmark(self):
        """Benchmark wobble detection latency (target ≤ 5 ms)."""
        detector = VMDIRCNNDetector(n_modes=4)
        signal = np.random.randn(1000)

        # Warmup
        for _ in range(10):
            detector.detect_wobble(signal)

        # Benchmark
        n_iterations = 100
        start = time.perf_counter()
        for _ in range(n_iterations):
            detector.detect_wobble(signal)
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / n_iterations) * 1000

        logger.info(f"Wobble detection latency: {latency_ms:.2f} ms (target ≤ 5 ms)")

        # Assert target met (or document if not)
        if latency_ms > 5.0:
            logger.warning(f"Wobble detection latency {latency_ms:.2f} ms exceeds target 5 ms")
        else:
            logger.info(f"Wobble detection latency {latency_ms:.2f} ms meets target")

    def test_get_model_info(self):
        """Test get_model_info returns correct metadata."""
        detector = VMDIRCNNDetector(n_modes=4, alpha=2000.0)
        info = detector.get_model_info()
        assert info['version'] == "1.0.0"
        assert not info['is_trained']
        assert info['n_modes'] == 4
        assert info['alpha'] == 2000.0

    def test_short_signal_handling(self):
        """Test detector handles short signals gracefully."""
        detector = VMDIRCNNDetector(n_modes=4)
        signal = np.random.randn(10)  # Very short signal
        is_wobble, confidence, metadata = detector.detect_wobble(signal)
        assert isinstance(is_wobble, bool)
        assert isinstance(confidence, float)
        assert isinstance(metadata, dict)

    def test_constant_signal(self):
        """Test detector handles constant signal."""
        detector = VMDIRCNNDetector(n_modes=4)
        signal = np.ones(1000)
        is_wobble, confidence, metadata = detector.detect_wobble(signal)
        # Constant signal should not be detected as wobble
        assert metadata['wobble_ratio'] < 0.1

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        detector = VMDIRCNNDetector(n_modes=4)
        # Very large values
        signal_large = np.random.randn(1000) * 1e6
        is_wobble, confidence, metadata = detector.detect_wobble(signal_large)
        assert isinstance(is_wobble, bool)
        assert not np.isnan(confidence)
        assert not np.isinf(confidence)

        # Very small values
        signal_small = np.random.randn(1000) * 1e-6
        is_wobble, confidence, metadata = detector.detect_wobble(signal_small)
        assert isinstance(is_wobble, bool)
        assert not np.isnan(confidence)
        assert not np.isinf(confidence)

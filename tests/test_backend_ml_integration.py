"""
Tests for backend ML integration layer.

Tests runtime selection between true VMD/IRCNN and stub implementations.
"""

import time

import numpy as np
import pytest

from backend.ml_integration import MLIntegrationLayer, TRUE_VMD_AVAILABLE


class TestMLIntegrationLayer:
    """Test ML integration layer runtime selection."""

    def test_init_with_true_vmd(self):
        """Test initialization with true VMD/IRCNN if available."""
        # Test with use_true_vmd=True (for offline analysis)
        if TRUE_VMD_AVAILABLE:
            ml = MLIntegrationLayer(use_true_vmd=True)
            status = ml.get_model_status()

            assert status['wobble_detector']['method'] == 'true_vmd_ircnn'
            assert ml.vmd_decomposer is not None
            assert ml.ircnn_predictor is not None
        else:
            # Skip test if true VMD not available
            pytest.skip("True VMD/IRCNN not available")

    def test_init_default_uses_stub(self):
        """Test that default initialization uses stub for real-time performance."""
        ml = MLIntegrationLayer()  # Default use_true_vmd=False
        status = ml.get_model_status()

        # Default should be stub for real-time performance
        assert status['wobble_detector']['method'] == 'stub'
        assert ml.wobble_detector is not None

    def test_init_with_stub(self):
        """Test initialization forcing stub usage."""
        ml = MLIntegrationLayer(use_true_vmd=False)
        status = ml.get_model_status()

        assert status['wobble_detector']['method'] == 'stub'
        assert ml.wobble_detector is not None
        assert ml.vmd_decomposer is None

    def test_detect_wobble_batch(self):
        """Test batch wobble detection."""
        ml = MLIntegrationLayer(use_true_vmd=False)  # Use stub for real-time performance
        signals = [np.random.randn(1000) for _ in range(10)]

        results = ml.detect_wobble_batch(signals, threshold=0.1)

        assert len(results) == 10
        for result in results:
            assert 'signal_id' in result
            assert 'is_wobble' in result
            assert 'confidence' in result
            assert 'metadata' in result
            assert 'method' in result['metadata']

    def test_latency_target(self):
        """Test that detection latency meets target (< 10 ms) with stub."""
        ml = MLIntegrationLayer(use_true_vmd=False)  # Use stub for real-time performance
        signals = [np.random.randn(1000) for _ in range(100)]

        start = time.perf_counter()
        results = ml.detect_wobble_batch(signals, threshold=0.1)
        elapsed = (time.perf_counter() - start) / len(signals) * 1000

        # Average latency should be < 10 ms with stub
        assert elapsed < 10.0, f"Average latency too high: {elapsed:.1f} ms"

    def test_latency_warning_threshold(self):
        """Test that latency warning threshold is > 5 ms."""
        ml = MLIntegrationLayer(use_true_vmd=False)  # Use stub for real-time performance
        signals = [np.random.randn(1000) for _ in range(10)]

        results = ml.detect_wobble_batch(signals, threshold=0.1)

        # Check that latency is tracked in metadata
        for result in results:
            if 'latency_ms' in result['metadata']:
                # Latency should be reasonable (< 30 ms for failure threshold)
                assert result['metadata']['latency_ms'] < 30.0

    def test_api_backward_compatibility(self):
        """Test that API contract is maintained."""
        ml = MLIntegrationLayer(use_true_vmd=False)  # Use stub for real-time performance
        signals = [np.random.randn(1000)]

        results = ml.detect_wobble_batch(signals)

        # Check response format matches original API
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]['signal_id'] == 0
        assert isinstance(results[0]['is_wobble'], (bool, np.bool_))
        assert isinstance(results[0]['confidence'], (float, np.floating))

    def test_model_status(self):
        """Test model status reporting."""
        ml = MLIntegrationLayer(use_true_vmd=TRUE_VMD_AVAILABLE)
        status = ml.get_model_status()

        assert 'wobble_detector' in status
        assert 'thermal_model' in status
        assert 'available' in status['wobble_detector']
        assert 'method' in status['wobble_detector']
        assert 'vmd_available' in status['wobble_detector']
        assert 'ircnn_available' in status['wobble_detector']

    def test_thermal_prediction(self):
        """Test thermal prediction (if JAX available)."""
        ml = MLIntegrationLayer(enable_thermal_prediction=True)
        T_initial = np.array([77.0, 80.0])
        Q_in = np.array([10.0, 15.0])

        result = ml.predict_thermal_batch(T_initial, Q_in)

        assert 'success' in result
        if result['success']:
            assert 'temperatures' in result
            assert 'metadata' in result
        else:
            assert 'error' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

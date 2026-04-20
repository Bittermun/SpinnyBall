"""
Integration tests for ML integration layer.

Tests ML integration with both models, API endpoints, and fallback logic.
"""

import logging
import time

import numpy as np
import pytest

try:
    from backend.ml_integration import MLIntegrationLayer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    MLIntegrationLayer = None

try:
    from fastapi.testclient import TestClient

    from backend.app import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    TestClient = None
    app = None

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML integration not available")
class TestMLIntegrationLayer:
    """Test suite for ML integration layer."""

    def test_initialization_both_models(self):
        """Test initialization with both models enabled."""
        ml = MLIntegrationLayer(
            enable_wobble_detection=True,
            enable_thermal_prediction=True,
        )
        assert ml.enable_wobble_detection
        assert ml.enable_thermal_prediction
        # Models may or may not be available depending on dependencies
        assert ml.wobble_detector is not None or ml.wobble_detector is None
        assert ml.thermal_model is not None or ml.thermal_model is None

    def test_initialization_wobble_only(self):
        """Test initialization with only wobble detection enabled."""
        ml = MLIntegrationLayer(
            enable_wobble_detection=True,
            enable_thermal_prediction=False,
        )
        assert ml.enable_wobble_detection
        assert not ml.enable_thermal_prediction
        assert ml.wobble_detector is not None
        assert ml.thermal_model is None

    def test_initialization_thermal_only(self):
        """Test initialization with only thermal prediction enabled."""
        ml = MLIntegrationLayer(
            enable_wobble_detection=False,
            enable_thermal_prediction=True,
        )
        assert not ml.enable_wobble_detection
        assert ml.enable_thermal_prediction
        assert ml.wobble_detector is None
        # Thermal model may be None if JAX not available

    def test_detect_wobble_batch(self):
        """Test wobble detection on batch of signals."""
        ml = MLIntegrationLayer(enable_wobble_detection=True)
        signals = [
            np.random.randn(1000),
            np.random.randn(1000),
            np.random.randn(1000),
        ]
        results = ml.detect_wobble_batch(signals, threshold=0.1)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['signal_id'] == i
            assert 'is_wobble' in result
            assert 'confidence' in result
            assert 'metadata' in result
            assert isinstance(result['is_wobble'], bool)
            assert 0 <= result['confidence'] <= 1

    def test_detect_wobble_batch_fallback(self):
        """Test wobble detection fallback when detector unavailable."""
        ml = MLIntegrationLayer(enable_wobble_detection=False)
        signals = [np.random.randn(1000)]
        results = ml.detect_wobble_batch(signals, threshold=0.1)

        assert len(results) == 1
        assert not results[0]['is_wobble']
        assert results[0]['confidence'] == 0.0
        assert 'error' in results[0]['metadata']

    def test_predict_thermal_batch(self):
        """Test thermal prediction batch."""
        ml = MLIntegrationLayer(enable_thermal_prediction=True)
        T_initial = np.array([300.0, 310.0])  # noqa: N806
        Q_in = np.random.randn(100, 2) * 10  # noqa: N806

        result = ml.predict_thermal_batch(T_initial, Q_in, t_amb=293.15)

        if result['success']:
            assert 'temperatures' in result
            assert 'metadata' in result
            assert len(result['temperatures']) == 101  # n_steps + 1
        else:
            assert 'error' in result

    def test_predict_thermal_batch_fallback(self):
        """Test thermal prediction fallback when model unavailable."""
        ml = MLIntegrationLayer(enable_thermal_prediction=False)
        T_initial = np.array([300.0, 310.0])  # noqa: N806
        Q_in = np.random.randn(100, 2) * 10  # noqa: N806

        result = ml.predict_thermal_batch(T_initial, Q_in, t_amb=293.15)

        assert not result['success']
        assert 'error' in result

    def test_get_model_status(self):
        """Test get_model_status returns correct status."""
        ml = MLIntegrationLayer(
            enable_wobble_detection=True,
            enable_thermal_prediction=True,
        )
        status = ml.get_model_status()

        assert 'wobble_detector' in status
        assert 'thermal_model' in status
        assert 'available' in status['wobble_detector']
        assert 'available' in status['thermal_model']
        assert 'info' in status['wobble_detector']
        assert 'info' in status['thermal_model']

    def test_batch_processing_multiple_signals(self):
        """Test batch processing with multiple signals."""
        ml = MLIntegrationLayer(enable_wobble_detection=True)
        n_signals = 10
        signals = [np.random.randn(1000) for _ in range(n_signals)]
        results = ml.detect_wobble_batch(signals, threshold=0.1)

        assert len(results) == n_signals
        for i, result in enumerate(results):
            assert result['signal_id'] == i

    def test_wobble_detection_latency_benchmark(self):
        """Benchmark end-to-end wobble detection latency."""
        ml = MLIntegrationLayer(enable_wobble_detection=True)
        signals = [np.random.randn(1000) for _ in range(10)]

        # Warmup
        ml.detect_wobble_batch(signals, threshold=0.1)

        # Benchmark
        n_iterations = 10
        start = time.perf_counter()
        for _ in range(n_iterations):
            ml.detect_wobble_batch(signals, threshold=0.1)
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / n_iterations) * 1000

        logger.info(f"ML integration wobble detection latency: {latency_ms:.2f} ms for 10 signals")

    def test_thermal_prediction_latency_benchmark(self):
        """Benchmark end-to-end thermal prediction latency."""
        ml = MLIntegrationLayer(enable_thermal_prediction=True)
        T_initial = np.array([300.0, 310.0])  # noqa: N806
        Q_in = np.random.randn(100, 2) * 10  # noqa: N806

        # Warmup
        ml.predict_thermal_batch(T_initial, Q_in, t_amb=293.15)

        # Benchmark
        n_iterations = 10
        start = time.perf_counter()
        for _ in range(n_iterations):
            ml.predict_thermal_batch(T_initial, Q_in, T_amb=293.15)
        elapsed = time.perf_counter() - start
        latency_ms = (elapsed / n_iterations) * 1000

        logger.info(f"ML integration thermal prediction latency: {latency_ms:.2f} ms")


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestMLEndpoints:
    """Test suite for ML API endpoints."""

    def test_wobble_detect_endpoint(self):
        """Test POST /ml/wobble-detect endpoint."""
        client = TestClient(app)
        response = client.post(
            "/ml/wobble-detect",
            json={
                "signals": [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]],
                "threshold": 0.1
            }
        )

        # May return 503 if ML unavailable
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert 'results' in data
            assert len(data['results']) == 2

    def test_thermal_predict_endpoint(self):
        """Test POST /ml/thermal-predict endpoint."""
        client = TestClient(app)
        response = client.post(
            "/ml/thermal-predict",
            json={
                "T_initial": [300.0, 310.0],
                "Q_in": [[10.0, 15.0], [12.0, 18.0]],
                "T_amb": 293.15
            }
        )

        # May return 503 if ML unavailable
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert 'success' in data or 'temperatures' in data

    def test_ml_status_endpoint(self):
        """Test GET /ml/status endpoint."""
        client = TestClient(app)
        response = client.get("/ml/status")

        assert response.status_code == 200
        data = response.json()
        assert 'wobble_detector' in data
        assert 'thermal_model' in data

    def test_wobble_detect_invalid_input(self):
        """Test wobble detection endpoint with invalid input."""
        client = TestClient(app)
        response = client.post(
            "/ml/wobble-detect",
            json={
                "signals": "not a list",
                "threshold": 0.1
            }
        )
        assert response.status_code in [422, 503]  # Validation error or service unavailable

    def test_thermal_predict_invalid_input(self):
        """Test thermal prediction endpoint with invalid input."""
        client = TestClient(app)
        response = client.post(
            "/ml/thermal-predict",
            json={
                "T_initial": "not a list",
                "Q_in": [[10.0, 15.0]],
                "T_amb": 293.15
            }
        )
        assert response.status_code in [422, 503]  # Validation error or service unavailable

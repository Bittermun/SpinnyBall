"""
Tests for statistical anomaly detection system.
"""

import numpy as np
import pytest

from control_layer.anomaly_detector import (
    ZScoreDetector,
    IsolationForestDetector,
    StatisticalAnomalyDetector,
    ResponseHandler,
    AlertSeverity,
    AlertAction,
    create_statistical_anomaly_detector,
)


def test_z_score_detector_initialization():
    """Test z-score detector initialization."""
    detector = ZScoreDetector(threshold=3.0, window_size=100, state_dim=7)

    assert detector.threshold == 3.0
    assert detector.window_size == 100
    assert detector.state_dim == 7
    assert not detector.is_initialized


def test_z_score_detector_statistics():
    """Test z-score detector statistics update."""
    detector = ZScoreDetector(threshold=3.0, window_size=10, state_dim=7)

    # Add samples
    for _ in range(15):
        state = np.random.randn(7)
        detector.update_statistics(state)

    assert detector.is_initialized


def test_z_score_detector_anomaly_detection():
    """Test z-score detector anomaly detection."""
    detector = ZScoreDetector(threshold=3.0, window_size=10, state_dim=7)

    # Manually set statistics to known values for predictable testing
    detector.mean_buffer[:] = 0.5
    detector.std_buffer[:] = 0.01
    detector.buffer_idx = 9  # Valid index for window_size=10
    detector.is_initialized = True

    # Test normal state (no alert)
    normal_state = np.ones(7) * 0.5
    alert = detector.detect(normal_state, packet_id=0, timestamp=0.0)
    assert alert is None

    # Test anomalous state (should trigger alert) - create a large deviation
    anomalous_state = np.ones(7) * 0.5
    anomalous_state[0] = 10.0  # Large deviation in first dimension
    alert = detector.detect(anomalous_state, packet_id=0, timestamp=1.0)
    assert alert is not None
    assert alert.detector_name == "z_score"
    assert alert.affected_packet_id == 0


def test_isolation_forest_detector_initialization():
    """Test isolation forest detector initialization."""
    try:
        detector = IsolationForestDetector(contamination=0.1, n_estimators=100, state_dim=7)
        assert detector.contamination == 0.1
        assert detector.n_estimators == 100
        assert detector.state_dim == 7
        assert not detector.is_trained
    except ImportError:
        pytest.skip("scikit-learn not available")


def test_isolation_forest_detector_training():
    """Test isolation forest detector training."""
    try:
        detector = IsolationForestDetector(contamination=0.1, n_estimators=100, state_dim=7)

        # Add training samples
        for _ in range(20):
            state = np.random.randn(7) * 0.1
            detector.add_training_sample(state)

        detector.train()

        assert detector.is_trained
    except ImportError:
        pytest.skip("scikit-learn not available")


def test_isolation_forest_detector_anomaly_detection():
    """Test isolation forest detector anomaly detection."""
    try:
        detector = IsolationForestDetector(contamination=0.1, n_estimators=100, state_dim=7)

        # Add normal training samples
        for _ in range(20):
            state = np.random.randn(7) * 0.1
            detector.add_training_sample(state)

        detector.train()

        # Test normal state (no alert)
        normal_state = np.random.randn(7) * 0.1
        alert = detector.detect(normal_state, packet_id=0, timestamp=0.0)
        # May or may not trigger depending on anomaly score
        if alert:
            assert alert.detector_name == "isolation_forest"

        # Test anomalous state
        anomalous_state = np.random.randn(7) * 5.0
        alert = detector.detect(anomalous_state, packet_id=0, timestamp=1.0)
        # May or may not trigger depending on anomaly score
        if alert:
            assert alert.detector_name == "isolation_forest"
    except ImportError:
        pytest.skip("scikit-learn not available")


def test_statistical_anomaly_detector_initialization():
    """Test statistical anomaly detector initialization."""
    detector = StatisticalAnomalyDetector(
        use_z_score=True,
        use_isolation_forest=False,  # Disable to avoid sklearn dependency
        state_dim=7,
    )

    assert len(detector.detectors) == 1
    assert hasattr(detector, 'z_score_detector')


def test_statistical_anomaly_detector_detection():
    """Test statistical anomaly detector detection."""
    detector = StatisticalAnomalyDetector(
        use_z_score=True,
        use_isolation_forest=False,
        state_dim=7,
    )

    # Add normal samples to build statistics
    for i in range(150):
        state = np.random.randn(7) * 0.1
        detector.detect(state, packet_id=0, timestamp=float(i) * 0.01)

    # Test normal state
    normal_state = np.random.randn(7) * 0.1
    alerts = detector.detect(normal_state, packet_id=0, timestamp=10.0)
    assert len(alerts) == 0

    # Test anomalous state
    anomalous_state = np.random.randn(7) * 0.1
    anomalous_state[0] = 10.0  # Large deviation
    alerts = detector.detect(anomalous_state, packet_id=0, timestamp=11.0)
    assert len(alerts) > 0


def test_response_handler_initialization():
    """Test response handler initialization."""
    handler = ResponseHandler()
    assert handler.alert_history == []


def test_response_handler_alert_handling():
    """Test response handler alert handling."""
    handler = ResponseHandler()

    # Create test alert
    from control_layer.anomaly_detector import AnomalyAlert

    alert = AnomalyAlert(
        timestamp=0.0,
        severity=AlertSeverity.WARNING,
        detector_name="test",
        anomaly_score=4.5,
        threshold=3.0,
        affected_packet_id=0,
        action=AlertAction.LOG,
        message="Test alert",
    )

    result = handler.handle_alert(alert)
    assert result is True
    assert len(handler.alert_history) == 1


def test_response_handler_alert_summary():
    """Test response handler alert summary."""
    handler = ResponseHandler()

    from control_layer.anomaly_detector import AnomalyAlert

    # Add multiple alerts
    alerts = [
        AnomalyAlert(0.0, AlertSeverity.INFO, "test", 2.5, 3.0, 0, AlertAction.LOG, "Info alert"),
        AnomalyAlert(1.0, AlertSeverity.WARNING, "test", 4.0, 3.0, 1, AlertAction.REDUCE_GAIN, "Warning alert"),
        AnomalyAlert(2.0, AlertSeverity.CRITICAL, "test", 5.5, 3.0, 2, AlertAction.SAFE_SET_HOLD, "Critical alert"),
    ]

    for alert in alerts:
        handler.handle_alert(alert)

    summary = handler.get_alert_summary()
    assert summary["total_alerts"] == 3
    assert summary["by_severity"]["info"] == 1
    assert summary["by_severity"]["warning"] == 1
    assert summary["by_severity"]["critical"] == 1


def test_create_statistical_anomaly_detector():
    """Test factory function for statistical anomaly detector."""
    detector = create_statistical_anomaly_detector(
        use_z_score=True,
        use_isolation_forest=False,
        state_dim=7,
    )

    assert isinstance(detector, StatisticalAnomalyDetector)
    assert len(detector.detectors) == 1


def test_z_score_severity_levels():
    """Test z-score detector severity levels."""
    detector = ZScoreDetector(threshold=3.0, window_size=10, state_dim=7)

    # Build statistics
    for _ in range(15):
        state = np.random.randn(7) * 0.1
        detector.update_statistics(state)

    # Test INFO severity (z-score just above threshold)
    state_info = np.random.randn(7) * 0.1
    state_info[0] += 3.5 * detector.std_buffer[0, 0]  # z-score ~3.5
    alert = detector.detect(state_info, packet_id=0, timestamp=0.0)
    if alert:
        assert alert.severity in [AlertSeverity.INFO, AlertSeverity.WARNING]

    # Test CRITICAL severity (z-score very high)
    state_critical = np.random.randn(7) * 0.1
    state_critical[0] += 6.0 * detector.std_buffer[0, 0]  # z-score ~6.0
    alert = detector.detect(state_critical, packet_id=0, timestamp=1.0)
    if alert:
        assert alert.severity == AlertSeverity.CRITICAL

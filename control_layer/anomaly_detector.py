"""
Statistical anomaly detection system for Phase 3.

Implements z-score and isolation forest detection methods for
real-time anomaly detection in SpinnyBall system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Constants for anomaly detection
Z_SCORE_THRESHOLD_CRITICAL = 5.0
Z_SCORE_THRESHOLD_WARNING = 4.0
ISOLATION_FOREST_CRITICAL_SCORE = 0.5
ISOLATION_FOREST_WARNING_SCORE = 0.3
NORM_EPSILON = 1e-8

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")


class AlertSeverity(Enum):
    """Severity levels for anomaly alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertAction(Enum):
    """Response actions for anomaly alerts."""
    LOG = "log"
    REDUCE_GAIN = "reduce_gain"
    SAFE_SET_HOLD = "safe_set_hold"


@dataclass
class AnomalyAlert:
    """Represents an anomaly detection alert."""
    timestamp: float
    severity: AlertSeverity
    detector_name: str
    anomaly_score: float
    threshold: float
    affected_packet_id: int
    action: AlertAction
    message: str


class ZScoreDetector:
    """
    Z-score based anomaly detector.

    Uses statistical z-score to detect anomalies in state variables.
    Simple, fast, and interpretable.
    """

    def __init__(
        self,
        threshold: float = 3.0,
        window_size: int = 100,
        state_dim: int = 7,
    ):
        """
        Initialize z-score detector.

        Args:
            threshold: Z-score threshold for anomaly detection
            window_size: Window size for moving statistics
            state_dim: State vector dimension
        """
        self.threshold = threshold
        self.window_size = window_size
        self.state_dim = state_dim

        # Statistics buffers
        self.mean_buffer = np.zeros((window_size, state_dim))
        self.std_buffer = np.zeros((window_size, state_dim))
        self.buffer_idx = 0
        self.samples_added = 0  # Track number of samples added
        self.is_initialized = False

    def update_statistics(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update rolling statistics with new state.

        Args:
            state: Current state vector [state_dim]

        Returns:
            mean: Current mean vector
            std: Current std vector
        """
        # Update buffers
        self.mean_buffer[self.buffer_idx] = state
        self.std_buffer[self.buffer_idx] = state ** 2
        self.buffer_idx = (self.buffer_idx + 1) % self.window_size
        self.samples_added += 1

        # Initialize after filling buffer once
        if self.samples_added >= self.window_size:
            self.is_initialized = True

        # Compute rolling statistics
        if self.is_initialized:
            mean = np.mean(self.mean_buffer, axis=0)
            std = np.sqrt(np.maximum(np.mean(self.std_buffer, axis=0) - mean ** 2, NORM_EPSILON))
        else:
            mean = np.zeros_like(state)
            std = np.ones_like(state)

        return mean, std

    def detect(self, state: np.ndarray, packet_id: int, timestamp: float) -> Optional[AnomalyAlert]:
        """
        Detect anomalies using z-score.

        Args:
            state: Current state vector [state_dim]
            packet_id: Packet identifier
            timestamp: Current timestamp

        Returns:
            AnomalyAlert if anomaly detected, None otherwise
        """
        mean, std = self.update_statistics(state)

        if not self.is_initialized:
            return None

        # Compute z-scores
        z_scores = np.abs((state - mean) / (std + NORM_EPSILON))
        max_z_score = np.max(z_scores)

        if max_z_score > self.threshold:
            # Determine severity based on z-score magnitude
            if max_z_score > Z_SCORE_THRESHOLD_CRITICAL:
                severity = AlertSeverity.CRITICAL
                action = AlertAction.SAFE_SET_HOLD
            elif max_z_score > Z_SCORE_THRESHOLD_WARNING:
                severity = AlertSeverity.WARNING
                action = AlertAction.REDUCE_GAIN
            else:
                severity = AlertSeverity.INFO
                action = AlertAction.LOG

            alert = AnomalyAlert(
                timestamp=timestamp,
                severity=severity,
                detector_name="z_score",
                anomaly_score=float(max_z_score),
                threshold=self.threshold,
                affected_packet_id=packet_id,
                action=action,
                message=f"Z-score anomaly detected: {max_z_score:.2f} (threshold: {self.threshold:.2f})",
            )
            return alert

        return None


class IsolationForestDetector:
    """
    Isolation Forest based anomaly detector.

    Uses ensemble-based isolation forest for anomaly detection.
    More sophisticated than z-score, can detect complex patterns.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        state_dim: int = 7,
    ):
        """
        Initialize isolation forest detector.

        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in forest
            state_dim: State vector dimension
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for IsolationForestDetector")

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.state_dim = state_dim

        # Initialize model
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        self.scaler = StandardScaler()

        # Training data buffer
        self.training_data = []
        self.is_trained = False

    def add_training_sample(self, state: np.ndarray):
        """
        Add training sample.

        Args:
            state: State vector [state_dim]
        """
        self.training_data.append(state)

    def train(self):
        """Train the isolation forest model."""
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for isolation forest")
            return

        training_array = np.array(self.training_data)
        scaled_data = self.scaler.fit_transform(training_array)
        self.model.fit(scaled_data)
        self.is_trained = True

        logger.info(f"Isolation forest trained on {len(self.training_data)} samples")

    def detect(self, state: np.ndarray, packet_id: int, timestamp: float) -> Optional[AnomalyAlert]:
        """
        Detect anomalies using isolation forest.

        Args:
            state: Current state vector [state_dim]
            packet_id: Packet identifier
            timestamp: Current timestamp

        Returns:
            AnomalyAlert if anomaly detected, None otherwise
        """
        if not self.is_trained:
            # Auto-train if we have enough data
            if len(self.training_data) >= 10:
                self.train()
            else:
                return None

        # Scale and predict
        state_scaled = self.scaler.transform(state.reshape(1, -1))
        anomaly_score = self.model.decision_function(state_scaled)[0]

        # Isolation forest returns negative scores for anomalies
        if anomaly_score < 0:
            # Determine severity based on score magnitude
            score_magnitude = abs(anomaly_score)
            if score_magnitude > ISOLATION_FOREST_CRITICAL_SCORE:
                severity = AlertSeverity.CRITICAL
                action = AlertAction.SAFE_SET_HOLD
            elif score_magnitude > ISOLATION_FOREST_WARNING_SCORE:
                severity = AlertSeverity.WARNING
                action = AlertAction.REDUCE_GAIN
            else:
                severity = AlertSeverity.INFO
                action = AlertAction.LOG

            alert = AnomalyAlert(
                timestamp=timestamp,
                severity=severity,
                detector_name="isolation_forest",
                anomaly_score=float(anomaly_score),
                threshold=0.0,
                affected_packet_id=packet_id,
                action=action,
                message=f"Isolation forest anomaly detected: score={anomaly_score:.3f}",
            )
            return alert

        return None


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection system.

    Combines z-score and isolation forest detectors for comprehensive
    anomaly detection with real-time scoring.
    """

    def __init__(
        self,
        use_z_score: bool = True,
        use_isolation_forest: bool = True,
        state_dim: int = 7,
    ):
        """
        Initialize statistical anomaly detector.

        Args:
            use_z_score: Enable z-score detector
            use_isolation_forest: Enable isolation forest detector
            state_dim: State vector dimension
        """
        self.state_dim = state_dim
        self.detectors = []

        if use_z_score:
            self.z_score_detector = ZScoreDetector(
                threshold=3.0,
                window_size=100,
                state_dim=state_dim,
            )
            self.detectors.append(self.z_score_detector)

        if use_isolation_forest and SKLEARN_AVAILABLE:
            self.isolation_forest_detector = IsolationForestDetector(
                contamination=0.1,
                n_estimators=100,
                state_dim=state_dim,
            )
            self.detectors.append(self.isolation_forest_detector)

        logger.info(f"Initialized statistical anomaly detector with {len(self.detectors)} detectors")

    def detect(
        self,
        state: np.ndarray,
        packet_id: int,
        timestamp: float,
    ) -> List[AnomalyAlert]:
        """
        Detect anomalies using all enabled detectors.

        Args:
            state: Current state vector [state_dim]
            packet_id: Packet identifier
            timestamp: Current timestamp

        Returns:
            List of anomaly alerts
        """
        alerts = []

        for detector in self.detectors:
            # Add training data for isolation forest
            if isinstance(detector, IsolationForestDetector):
                detector.add_training_sample(state)

            # Detect anomaly
            alert = detector.detect(state, packet_id, timestamp)
            if alert is not None:
                alerts.append(alert)

        return alerts

    def train_isolation_forest(self):
        """Train isolation forest detector if enabled."""
        if hasattr(self, 'isolation_forest_detector'):
            self.isolation_forest_detector.train()


class ResponseHandler:
    """
    Handle anomaly detection responses.

    Executes actions based on anomaly alerts.
    """

    def __init__(self):
        """Initialize response handler."""
        self.alert_history = []

    def handle_alert(self, alert: AnomalyAlert) -> bool:
        """
        Handle an anomaly alert.

        Args:
            alert: Anomaly alert to handle

        Returns:
            True if action was executed, False otherwise
        """
        self.alert_history.append(alert)

        if alert.action == AlertAction.LOG:
            logger.info(f"[{alert.severity.value.upper()}] {alert.message}")
            return True

        elif alert.action == AlertAction.REDUCE_GAIN:
            logger.warning(f"[WARNING] Reducing gain for packet {alert.affected_packet_id}: {alert.message}")
            # In a real implementation, this would reduce control gain
            return True

        elif alert.action == AlertAction.SAFE_SET_HOLD:
            logger.error(f"[CRITICAL] Safe-set hold for packet {alert.affected_packet_id}: {alert.message}")
            # In a real implementation, this would trigger safe-set hold
            return True

        return False

    def get_alert_summary(self) -> dict:
        """
        Get summary of alert history.

        Returns:
            Dictionary with alert statistics
        """
        if not self.alert_history:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "by_detector": {},
            }

        by_severity = {}
        by_detector = {}

        for alert in self.alert_history:
            severity = alert.severity.value
            detector = alert.detector_name

            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_detector[detector] = by_detector.get(detector, 0) + 1

        return {
            "total_alerts": len(self.alert_history),
            "by_severity": by_severity,
            "by_detector": by_detector,
        }


def create_statistical_anomaly_detector(
    use_z_score: bool = True,
    use_isolation_forest: bool = True,
    state_dim: int = 7,
) -> StatisticalAnomalyDetector:
    """
    Factory function to create statistical anomaly detector.

    Args:
        use_z_score: Enable z-score detector
        use_isolation_forest: Enable isolation forest detector
        state_dim: State vector dimension

    Returns:
        StatisticalAnomalyDetector instance
    """
    return StatisticalAnomalyDetector(
        use_z_score=use_z_score,
        use_isolation_forest=use_isolation_forest,
        state_dim=state_dim,
    )

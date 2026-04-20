"""
Unit tests for quench detection.
"""

import pytest

from dynamics.quench_detector import (
    QuenchDetector,
    QuenchThresholds,
)


def test_quench_thresholds():
    """Test QuenchThresholds dataclass."""
    thresholds = QuenchThresholds(
        temperature_critical=90.0,
        temperature_warning=85.0,
        temperature_rate_limit=10.0,
        hysteresis=2.0,
    )
    assert thresholds.temperature_critical == 90.0
    assert thresholds.temperature_warning == 85.0
    assert thresholds.temperature_rate_limit == 10.0
    assert thresholds.hysteresis == 2.0


def test_quench_detector_initialization():
    """Test QuenchDetector initialization."""
    thresholds = QuenchThresholds()
    detector = QuenchDetector(thresholds)
    assert detector.quenched == False
    assert detector.warning_state == False
    assert detector.prev_temperature == 70.0
    assert detector.quench_time is None


def test_quench_detection_critical():
    """Test quench detection at critical temperature."""
    thresholds = QuenchThresholds(temperature_critical=90.0)
    detector = QuenchDetector(thresholds)
    
    # Below critical - no quench
    status = detector.check_temperature(80.0, dt=0.1)
    assert status["quenched"] == False
    assert status["emergency_shutdown"] == False
    
    # Above critical - quench
    status = detector.check_temperature(95.0, dt=0.1)
    assert status["quenched"] == True
    assert status["emergency_shutdown"] == True


def test_quench_warning_detection():
    """Test warning threshold detection."""
    thresholds = QuenchThresholds(
        temperature_critical=90.0,
        temperature_warning=85.0,
    )
    detector = QuenchDetector(thresholds)
    
    # Below warning
    status = detector.check_temperature(80.0, dt=0.1)
    assert status["warning"] == False
    
    # Above warning
    status = detector.check_temperature(87.0, dt=0.1)
    assert status["warning"] == True


def test_heating_rate_violation():
    """Test heating rate violation detection."""
    thresholds = QuenchThresholds(temperature_rate_limit=10.0)
    detector = QuenchDetector(thresholds)
    
    # Normal heating rate
    status = detector.check_temperature(75.0, dt=0.1)  # 50 K/s
    assert status["rate_violation"] == False
    
    # Reset for clean test
    detector = QuenchDetector(thresholds)
    
    # Rapid heating rate
    status = detector.check_temperature(80.0, dt=0.01)  # 500 K/s
    assert status["rate_violation"] == True
    assert status["emergency_shutdown"] == True


def test_quench_hysteresis():
    """Test quench hysteresis to prevent chatter."""
    thresholds = QuenchThresholds(
        temperature_critical=90.0,
        temperature_warning=85.0,
        hysteresis=2.0,
    )
    detector = QuenchDetector(thresholds)
    
    # Trigger quench
    status = detector.check_temperature(95.0, dt=0.1)
    assert status["quenched"] == True
    
    # Cool to warning threshold - should still be quenched
    status = detector.check_temperature(85.0, dt=0.1)
    assert status["quenched"] == True
    
    # Cool below warning - hysteresis - should clear
    status = detector.check_temperature(82.0, dt=0.1)
    assert status["quenched"] == False


def test_quench_reset():
    """Test quench detector reset."""
    thresholds = QuenchThresholds()
    detector = QuenchDetector(thresholds)
    
    # Trigger quench
    detector.check_temperature(95.0, dt=0.1)
    assert detector.quenched == True
    
    # Reset
    detector.reset()
    assert detector.quenched == False
    assert detector.warning_state == False
    assert detector.prev_temperature == 70.0
    assert detector.quench_time is None


def test_heating_rate_calculation():
    """Test heating rate calculation."""
    thresholds = QuenchThresholds()
    detector = QuenchDetector(thresholds)
    
    # Initial temperature
    detector.check_temperature(70.0, dt=0.1)
    
    # Temperature change
    status = detector.check_temperature(75.0, dt=0.1)
    assert status["heating_rate"] == 50.0  # (75-70)/0.1


def test_emergency_shutdown_conditions():
    """Test emergency shutdown triggers."""
    thresholds = QuenchThresholds(
        temperature_critical=90.0,
        temperature_rate_limit=10.0,
    )
    detector = QuenchDetector(thresholds)
    
    # Emergency from quench
    status = detector.check_temperature(95.0, dt=0.1)
    assert status["emergency_shutdown"] == True
    
    # Reset
    detector.reset()
    
    # Emergency from rate violation
    detector.check_temperature(70.0, dt=0.001)
    status = detector.check_temperature(80.0, dt=0.001)
    assert status["emergency_shutdown"] == True


def test_default_thresholds():
    """Test default threshold values."""
    thresholds = QuenchThresholds()
    assert thresholds.temperature_critical == 90.0
    assert thresholds.temperature_warning == 85.0
    assert thresholds.temperature_rate_limit == 10.0
    assert thresholds.hysteresis == 2.0

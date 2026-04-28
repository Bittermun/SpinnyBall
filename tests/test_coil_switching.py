"""
Tests for coil switching loss model.
"""

import pytest
import numpy as np
from dynamics.coil_switching import (
    CoilSpecs,
    SwitchingEvent,
    CoilSwitchingModel,
    DEFAULT_COIL_SPECS,
    create_pulsed_switching_event,
)


def test_coil_specs_validation():
    """Test that invalid coil specs raise errors."""
    with pytest.raises(ValueError):
        CoilSpecs(
            length=0.1,
            radius=0.05,
            turns=100,
            resistance=0.0,  # Invalid: must be > 0
            inductance=1e-3,
            conductivity=1e8,
            permeability=4*np.pi*1e-7,
            operating_temp=77.0,
            skin_depth=1e-4,
        )
    
    with pytest.raises(ValueError):
        CoilSpecs(
            length=0.1,
            radius=0.05,
            turns=0,  # Invalid: must be > 0
            resistance=0.01,
            inductance=1e-3,
            conductivity=1e8,
            permeability=4*np.pi*1e-7,
            operating_temp=77.0,
            skin_depth=1e-4,
        )


def test_i2r_loss():
    """Test I²R loss calculation."""
    model = CoilSwitchingModel(DEFAULT_COIL_SPECS)
    
    # P = I²R
    current = 1000.0  # A
    duration = 1.0  # s
    expected_power = current**2 * DEFAULT_COIL_SPECS.resistance
    expected_energy = expected_power * duration
    
    loss = model.i2r_loss(current, duration)
    
    assert np.isclose(loss, expected_energy, rtol=1e-6)


def test_eddy_current_loss():
    """Test eddy current loss calculation."""
    model = CoilSwitchingModel(DEFAULT_COIL_SPECS)
    
    # Eddy loss should scale with (dI/dt)²
    current_change = 1000.0  # A
    switching_time = 1e-5  # s
    
    loss1 = model.eddy_current_loss(current_change, switching_time)
    loss2 = model.eddy_current_loss(current_change * 2, switching_time)
    
    # Doubling current change should quadruple loss (square relationship)
    assert np.isclose(loss2, 4 * loss1, rtol=0.1)


def test_switching_loss():
    """Test total switching loss calculation."""
    model = CoilSwitchingModel(DEFAULT_COIL_SPECS)
    
    event = create_pulsed_switching_event(
        peak_current=1000.0,
        pulse_width=0.1,
        rise_time=1e-5,
        fall_time=1e-5,
    )
    
    total_loss, breakdown = model.switching_loss(event)
    
    # Total loss should be sum of components
    expected_total = (
        breakdown['i2r_rise_J'] +
        breakdown['i2r_fall_J'] +
        breakdown['i2r_steady_J'] +
        breakdown['eddy_rise_J'] +
        breakdown['eddy_fall_J']
    )
    
    assert np.isclose(total_loss, expected_total, rtol=1e-6)
    assert total_loss > 0
    assert 'total_i2r_J' in breakdown
    assert 'total_eddy_J' in breakdown


def test_average_power_loss():
    """Test average power loss calculation."""
    model = CoilSwitchingModel(DEFAULT_COIL_SPECS)
    
    # Create multiple switching events
    events = [
        create_pulsed_switching_event(1000.0, 0.1, 1e-5, 1e-5),
        create_pulsed_switching_event(1000.0, 0.1, 1e-5, 1e-5),
        create_pulsed_switching_event(1000.0, 0.1, 1e-5, 1e-5),
    ]
    
    period = 1.0  # s
    avg_power, breakdown = model.average_power_loss(events, period)
    
    assert avg_power > 0
    assert breakdown['num_events'] == 3
    assert np.isclose(breakdown['avg_power_W'], avg_power, rtol=1e-6)


def test_create_pulsed_switching_event():
    """Test creation of pulsed switching event."""
    event = create_pulsed_switching_event(
        peak_current=1000.0,
        pulse_width=0.1,
        rise_time=1e-5,
        fall_time=1e-5,
    )
    
    assert event.current_start == 0.0
    assert event.current_end == 1000.0
    assert event.rise_time == 1e-5
    assert event.fall_time == 1e-5
    assert 0 < event.duty_cycle < 1


def test_switching_loss_scaling():
    """Test that switching losses scale correctly with current."""
    model = CoilSwitchingModel(DEFAULT_COIL_SPECS)
    
    event1 = create_pulsed_switching_event(500.0, 0.1, 1e-5, 1e-5)
    event2 = create_pulsed_switching_event(1000.0, 0.1, 1e-5, 1e-5)
    
    loss1, _ = model.switching_loss(event1)
    loss2, _ = model.switching_loss(event2)
    
    # I²R loss should scale with I²
    # Eddy loss should scale with (dI/dt)² ~ I²
    # So total loss should scale roughly with I²
    ratio = loss2 / loss1
    expected_ratio = (1000.0 / 500.0)**2
    
    # Allow some tolerance due to eddy current nonlinearities
    assert 0.5 * expected_ratio < ratio < 2.0 * expected_ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

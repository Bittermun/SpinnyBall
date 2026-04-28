"""
Tests for GdBCO material enhancements (field gradient checks, fringe correction, thermal feedback).
"""

import pytest
import numpy as np
from dynamics.gdBCO_material import (
    GdBCOProperties,
    GdBCOMaterial,
)


def test_field_gradient_check():
    """Test field gradient safety checks."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    
    # Safe gradient
    is_safe, msg = material.check_field_gradient(30.0)
    assert is_safe
    assert "within safe limits" in msg
    
    # At safety limit (80% of max)
    safe_limit = props.max_field_gradient * props.safety_margin
    is_safe, msg = material.check_field_gradient(safe_limit)
    assert is_safe
    
    # Exceeds safety limit but below absolute max
    is_safe, msg = material.check_field_gradient(45.0)
    assert not is_safe
    assert "exceeds safety limit" in msg
    
    # Exceeds absolute max
    is_safe, msg = material.check_field_gradient(60.0)
    assert not is_safe
    assert "exceeds absolute limit" in msg


def test_current_density_check():
    """Test current density safety checks."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    
    # Safe current density
    is_safe, msg = material.check_current_density(1e10)
    assert is_safe
    assert "within safe limits" in msg
    
    # At safety limit
    safe_limit = props.max_current_density * props.safety_margin
    is_safe, msg = material.check_current_density(safe_limit)
    assert is_safe
    
    # Exceeds safety limit
    is_safe, msg = material.check_current_density(4.5e10)
    assert not is_safe
    assert "exceeds safety limit" in msg
    
    # Exceeds absolute max
    is_safe, msg = material.check_current_density(6e10)
    assert not is_safe
    assert "exceeds absolute limit" in msg


def test_fringe_correction():
    """Test fringe field correction."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    
    B_center = 1.0  # T
    
    # At coil center (distance = 0), correction should be 1.0
    B_corrected = material.apply_fringe_correction(B_center, 0.0)
    assert np.isclose(B_corrected, B_center)
    
    # At distance, field should be reduced
    B_corrected = material.apply_fringe_correction(B_center, 0.01)
    assert B_corrected < B_center
    assert B_corrected > 0
    
    # At large distance, field should be very small
    B_corrected = material.apply_fringe_correction(B_center, 0.1)
    assert B_corrected < 0.1 * B_center


def test_thermal_degradation_factor():
    """Test thermal degradation factor calculation."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    
    # Well below Tc - no degradation (use 70K, well within safe range)
    degradation = material.compute_thermal_degradation_factor(70.0)
    assert np.isclose(degradation, 1.0)
    
    # Near Tc - some degradation
    T_near = props.Tc - 0.05 * props.Tc  # Within 5% of Tc
    degradation = material.compute_thermal_degradation_factor(T_near)
    assert 0 < degradation < 1
    
    # At Tc - complete loss
    degradation = material.compute_thermal_degradation_factor(props.Tc)
    assert degradation == 0.0
    
    # Above Tc - complete loss
    degradation = material.compute_thermal_degradation_factor(props.Tc + 10)
    assert degradation == 0.0


def test_thermal_feedback():
    """Test thermal feedback from switching losses."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    
    B = 0.1  # T
    T = 77.0  # K
    switching_power = 10.0  # W
    thermal_mass = 10.0  # J/K
    dt = 1.0  # s
    
    I_c, T_updated, feedback = material.critical_current_with_thermal_feedback(
        B, T, switching_power, thermal_mass, dt
    )
    
    # Temperature should increase
    assert T_updated > T
    assert feedback['temperature_rise_K'] > 0
    
    # I_c should be computed
    assert I_c > 0
    
    # Feedback dict should contain all keys
    assert 'I_c_base_A' in feedback
    assert 'degradation_initial' in feedback
    assert 'I_c_degraded_A' in feedback
    assert 'T_updated_K' in feedback
    assert 'degradation_updated' in feedback
    assert 'I_c_final_A' in feedback


def test_thermal_feedback_no_switching():
    """Test thermal feedback with no switching losses."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    
    B = 0.1  # T
    T = 70.0  # K (well within safe range)
    
    I_c, T_updated, feedback = material.critical_current_with_thermal_feedback(
        B, T, switching_power=0.0
    )
    
    # Temperature should not change
    assert np.isclose(T_updated, T)
    assert feedback['temperature_rise_K'] == 0
    
    # I_c should equal base I_c (no degradation at 70K)
    I_c_base = material.critical_current(B, T)
    assert np.isclose(I_c, I_c_base)


def test_thermal_feedback_high_power():
    """Test thermal feedback with high switching power."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    
    B = 0.1  # T
    T = 85.0  # K (closer to Tc)
    switching_power = 100.0  # W (high power)
    thermal_mass = 10.0  # J/K
    dt = 1.0  # s
    
    I_c, T_updated, feedback = material.critical_current_with_thermal_feedback(
        B, T, switching_power, thermal_mass, dt
    )
    
    # Significant temperature rise
    assert feedback['temperature_rise_K'] > 5.0
    
    # Degradation should increase with temperature
    assert feedback['degradation_updated'] <= feedback['degradation_initial']
    
    # Final I_c should be lower than base I_c
    assert feedback['I_c_final_A'] <= feedback['I_c_base_A']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

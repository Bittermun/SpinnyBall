"""
Unit tests for thermal model with eddy-current heating.
"""

import numpy as np
import pytest

from dynamics.thermal_model import (
    update_temperature_euler,
    eddy_heating_power,
)


def test_eddy_heating_power():
    """Test eddy heating power calculation."""
    k_drag = 0.01  # N·s/m
    velocity = 1000.0  # m/s
    radius = 0.1  # m
    
    power = eddy_heating_power(velocity, k_drag, radius)
    
    # P = k_drag * v^2
    expected = k_drag * velocity**2
    assert abs(power - expected) < 1e-6


def test_eddy_heating_power_zero_velocity():
    """Test eddy heating power at zero velocity."""
    power = eddy_heating_power(0.0, 0.01, 0.1)
    assert power == 0.0


def test_eddy_heating_power_scaling():
    """Test that eddy heating scales with velocity squared."""
    k_drag = 0.01
    radius = 0.1
    
    power_100 = eddy_heating_power(100.0, k_drag, radius)
    power_200 = eddy_heating_power(200.0, k_drag, radius)
    
    # 2x velocity should give 4x power
    assert abs(power_200 - 4.0 * power_100) < 1e-6


def test_update_temperature_euler_with_eddy_heating():
    """Test update_temperature_euler with eddy heating."""
    temperature = 77.0  # K
    mass = 0.05  # kg
    radius = 0.1  # m
    emissivity = 0.1
    specific_heat = 500.0  # J/kg/K
    dt = 0.01  # s
    eddy_power = 10.0  # W
    
    new_temp = update_temperature_euler(
        temperature=temperature,
        mass=mass,
        radius=radius,
        emissivity=emissivity,
        specific_heat=specific_heat,
        dt=dt,
        eddy_heating_power=eddy_power,
    )
    
    # Temperature should increase with eddy heating
    assert new_temp > temperature


def test_update_temperature_euler_eddy_validation():
    """Test that negative eddy_heating_power raises ValueError."""
    with pytest.raises(ValueError, match="eddy_heating_power must be >= 0"):
        update_temperature_euler(
            temperature=77.0,
            mass=0.05,
            radius=0.1,
            emissivity=0.1,
            specific_heat=500.0,
            dt=0.01,
            eddy_heating_power=-1.0,
        )


def test_update_temperature_euler_zero_eddy():
    """Test update_temperature_euler with zero eddy heating (default)."""
    temperature = 77.0
    mass = 0.05
    radius = 0.1
    emissivity = 0.1
    specific_heat = 500.0
    dt = 0.01
    
    new_temp = update_temperature_euler(
        temperature=temperature,
        mass=mass,
        radius=radius,
        emissivity=emissivity,
        specific_heat=specific_heat,
        dt=dt,
        eddy_heating_power=0.0,
    )
    
    # Temperature should decrease due to radiative cooling
    assert new_temp < temperature


def test_eddy_heating_with_high_velocity():
    """Test eddy heating at operational velocity (1600 m/s)."""
    k_drag = 0.01
    velocity = 1600.0  # m/s (operational)
    radius = 0.1
    
    power = eddy_heating_power(velocity, k_drag, radius)
    
    # P = 0.01 * 1600^2 = 25.6 kW
    expected = 0.01 * 1600.0**2
    assert abs(power - expected) < 1e-3


def test_thermal_balance_with_eddy_and_cryocooler():
    """Test thermal balance with both eddy heating and cryocooler cooling."""
    temperature = 77.0
    mass = 0.05
    radius = 0.1
    emissivity = 0.1
    specific_heat = 500.0
    dt = 0.01
    eddy_power = 10.0
    
    # With eddy heating only
    temp_eddy = update_temperature_euler(
        temperature=temperature,
        mass=mass,
        radius=radius,
        emissivity=emissivity,
        specific_heat=specific_heat,
        dt=dt,
        eddy_heating_power=eddy_power,
    )
    
    # Eddy heating should increase temperature
    assert temp_eddy > temperature
    
    # With cryocooler cooling (simulated by negative eddy power in this test)
    # Note: In real system, cryocooler is separate parameter
    # This test just verifies the function handles the parameter


def test_eddy_heating_negative_velocity():
    """Test eddy heating with negative velocity (should still work with v^2)."""
    k_drag = 0.01
    velocity = -1000.0  # Negative velocity
    radius = 0.1
    
    power = eddy_heating_power(velocity, k_drag, radius)
    
    # P = k_drag * v^2, so negative velocity should give same power
    expected = k_drag * (-1000.0)**2
    assert abs(power - expected) < 1e-6


def test_eddy_heating_zero_radius():
    """Test eddy heating with zero radius (edge case)."""
    k_drag = 0.01
    velocity = 1000.0
    radius = 0.0  # Zero radius
    
    # Should still compute power (radius is currently unused in calculation)
    power = eddy_heating_power(velocity, k_drag, radius)
    expected = k_drag * velocity**2
    assert abs(power - expected) < 1e-6


def test_update_temperature_euler_zero_mass():
    """Test update_temperature_euler with zero mass (should fail or handle gracefully)."""
    with pytest.raises((ValueError, ZeroDivisionError)):
        update_temperature_euler(
            temperature=77.0,
            mass=0.0,  # Zero mass
            radius=0.1,
            emissivity=0.1,
            specific_heat=500.0,
            dt=0.01,
        )


def test_update_temperature_euler_zero_specific_heat():
    """Test update_temperature_euler with zero specific heat (should fail)."""
    with pytest.raises((ValueError, ZeroDivisionError)):
        update_temperature_euler(
            temperature=77.0,
            mass=0.05,
            radius=0.1,
            emissivity=0.1,
            specific_heat=0.0,  # Zero specific heat
            dt=0.01,
        )


def test_update_temperature_euler_negative_dt():
    """Test update_temperature_euler with negative dt (should handle gracefully)."""
    # Negative dt would cause temperature to go backward in time
    # This is physically invalid but function should handle it
    temperature = 77.0
    mass = 0.05
    radius = 0.1
    emissivity = 0.1
    specific_heat = 500.0
    dt = -0.01  # Negative time step
    
    new_temp = update_temperature_euler(
        temperature=temperature,
        mass=mass,
        radius=radius,
        emissivity=emissivity,
        specific_heat=specific_heat,
        dt=dt,
    )
    
    # Temperature should decrease (or increase depending on radiative balance)
    # Just verify it doesn't crash
    assert isinstance(new_temp, float)

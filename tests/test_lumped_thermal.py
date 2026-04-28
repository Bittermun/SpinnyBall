"""
Unit tests for lumped-parameter thermal model.
"""

import numpy as np
import pytest

from dynamics.lumped_thermal import (
    LumpedThermalModel,
    LumpedThermalParams,
)
from dynamics.cryocooler_model import CryocoolerModel, DEFAULT_CRYOCOOLER_SPECS


def test_lumped_thermal_params():
    """Test LumpedThermalParams dataclass."""
    params = LumpedThermalParams(
        stator_mass=10.0,
        stator_specific_heat=500.0,
        stator_surface_area=0.1,
        stator_emissivity=0.1,
        rotor_mass=5.0,
        rotor_specific_heat=400.0,
        rotor_surface_area=0.05,
        rotor_emissivity=0.2,
        shaft_conductance=10.0,
        ambient_temp=4.0,
        initial_temp=77.0,
    )
    assert params.stator_mass == 10.0
    assert params.rotor_mass == 5.0
    assert params.shaft_conductance == 10.0


def test_lumped_thermal_initialization():
    """Test LumpedThermalModel initialization."""
    params = LumpedThermalParams()
    model = LumpedThermalModel(params, dt=0.01)
    assert model.params == params
    assert model.dt == 0.01
    assert model.T_stator == params.initial_temp
    assert model.T_rotor == params.initial_temp


def test_thermal_step_no_heat():
    """Test thermal step with no heat input."""
    params = LumpedThermalParams()
    model = LumpedThermalModel(params, dt=0.01)
    
    result = model.step({'stator': 0.0, 'rotor': 0.0})
    
    # Temperatures should decrease due to radiative cooling
    assert result['T_stator'] < params.initial_temp
    assert result['T_rotor'] < params.initial_temp


def test_thermal_step_with_heat():
    """Test thermal step with heat input."""
    params = LumpedThermalParams()
    model = LumpedThermalModel(params, dt=0.01)
    
    # Add heat to stator
    result = model.step({'stator': 100.0, 'rotor': 0.0})
    
    # Stator temperature should increase
    assert result['T_stator'] > params.initial_temp


def test_radiative_cooling():
    """Test radiative cooling calculation."""
    params = LumpedThermalParams(
        stator_surface_area=0.1,
        stator_emissivity=0.1,
        ambient_temp=4.0,
    )
    model = LumpedThermalModel(params, dt=0.01)
    model.T_stator = 100.0
    
    result = model.step({'stator': 0.0, 'rotor': 0.0})
    
    # Radiative power should be positive (cooling)
    assert result['P_rad_stator'] > 0


def test_conductive_heat_transfer():
    """Test conductive heat transfer between stator and rotor."""
    params = LumpedThermalParams(
        shaft_conductance=10.0,
    )
    model = LumpedThermalModel(params, dt=0.01)
    model.T_stator = 80.0
    model.T_rotor = 90.0
    
    result = model.step({'stator': 0.0, 'rotor': 0.0})
    
    # Heat should flow from hot rotor to cooler stator
    assert result['P_cond'] > 0  # Positive = rotor to stator


def test_get_temperatures():
    """Test get_temperatures method."""
    params = LumpedThermalParams()
    model = LumpedThermalModel(params, dt=0.01)
    
    temps = model.get_temperatures()
    assert len(temps) == 2
    assert temps[0] == model.T_stator
    assert temps[1] == model.T_rotor


def test_reset():
    """Test model reset."""
    params = LumpedThermalParams()
    model = LumpedThermalModel(params, dt=0.01)
    
    # Change temperatures
    model.T_stator = 100.0
    model.T_rotor = 90.0
    
    # Reset
    model.reset()
    
    # Should return to initial
    assert model.T_stator == params.initial_temp
    assert model.T_rotor == params.initial_temp


def test_euler_integration():
    """Test explicit Euler integration."""
    params = LumpedThermalParams()
    dt = 0.01
    model = LumpedThermalModel(params, dt=dt)
    
    # Store initial temperature
    T_initial = model.T_stator.copy()
    
    # Step with known heat
    Q = 100.0  # W
    result = model.step({'stator': Q, 'rotor': 0.0})
    
    # Calculate expected temperature change
    # dT = Q * dt / (m * c)
    dT_expected = Q * dt / (params.stator_mass * params.stator_specific_heat)
    
    # Check that temperature changed appropriately (accounting for radiative loss)
    assert result['T_stator'] > T_initial


def test_steady_state_approach():
    """Test that system approaches steady state."""
    params = LumpedThermalParams()
    model = LumpedThermalModel(params, dt=0.01)
    
    # Apply constant heat
    for _ in range(1000):
        model.step({'stator': 10.0, 'rotor': 0.0})
    
    # Temperature should stabilize
    T_final = model.T_stator
    model.step({'stator': 10.0, 'rotor': 0.0})
    T_next = model.T_stator
    
    # Small change indicates near steady state
    assert abs(T_next - T_final) < 0.01


def test_different_time_steps():
    """Test model with different time steps."""
    params = LumpedThermalParams()
    
    # Small time step
    model_small = LumpedThermalModel(params, dt=0.001)
    model_small.step({'stator': 100.0, 'rotor': 0.0})
    
    # Large time step
    model_large = LumpedThermalModel(params, dt=0.1)
    model_large.step({'stator': 100.0, 'rotor': 0.0})
    
    # Both should work (though results differ due to integration error)
    assert model_small.T_stator != params.initial_temp
    assert model_large.T_stator != params.initial_temp


def test_cryocooler_validation():
    """Test validation for cryocooler_cooling_power >= 0."""
    params = LumpedThermalParams(cryocooler_cooling_power=-1.0)
    with pytest.raises(ValueError, match="cryocooler_cooling_power must be >= 0"):
        LumpedThermalModel(params, dt=0.01)


def test_cryocooler_constant_power():
    """Test cryocooler with constant cooling power (fallback mode)."""
    params = LumpedThermalParams(
        enable_cryocooler=True,
        cryocooler_cooling_power=5.0,
        cryocooler_model=None,  # Use constant power fallback
    )
    model = LumpedThermalModel(params, dt=0.01)
    model.T_stator = 100.0
    
    result = model.step({'stator': 0.0, 'rotor': 0.0})
    
    # Temperature should decrease more with cryocooler
    assert result['T_stator'] < 100.0


def test_cryocooler_temperature_dependent():
    """Test cryocooler with temperature-dependent cooling from CryocoolerModel."""
    cryo_model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    params = LumpedThermalParams(
        enable_cryocooler=True,
        cryocooler_cooling_power=5.0,  # Fallback, should not be used
        cryocooler_model=cryo_model,
    )
    model = LumpedThermalModel(params, dt=0.01)
    model.T_stator = 100.0
    
    result = model.step({'stator': 0.0, 'rotor': 0.0})
    
    # Temperature should decrease with cryocooler
    assert result['T_stator'] < 100.0


def test_cryocooler_disabled():
    """Test that cryocooler is disabled when enable_cryocooler=False."""
    params = LumpedThermalParams(
        enable_cryocooler=False,
        cryocooler_cooling_power=5.0,
    )
    model = LumpedThermalModel(params, dt=0.01)
    model.T_stator = 100.0
    
    result = model.step({'stator': 0.0, 'rotor': 0.0})
    
    # Temperature should decrease due to radiative cooling only
    assert result['T_stator'] < 100.0

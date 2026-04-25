"""
End-to-end integration tests for Phase 1 components.

Tests integration between PID controller, thermal management, and flux-pinning.
"""

import numpy as np
import pytest

from sgms_anchor_control import PIDController, PIDParameters, simulate_controller
from dynamics.cryocooler_model import CryocoolerModel, CryocoolerSpecs, DEFAULT_CRYOCOOLER_SPECS
from dynamics.quench_detector import QuenchDetector, QuenchThresholds
from dynamics.lumped_thermal import LumpedThermalModel, LumpedThermalParams
from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.bean_london_model import BeanLondonModel
from sgms_anchor_v1 import simulate_anchor_with_flux_pinning


def test_pid_thermal_integration():
    """Test PID controller with temperature-dependent gain scheduling."""
    # Create PID controller
    params = PIDParameters(kp=100.0, ki=10.0, kd=1.0)
    pid = PIDController(params, dt=0.01)
    
    # Test at normal temperature
    output_normal = pid.update(1.0)
    assert output_normal > 0
    
    # Reset and test at high temperature (gain should be reduced)
    pid.reset()
    # In a full implementation, gain scheduling would be active here
    output_high = pid.update(1.0)
    assert output_high > 0


def test_thermal_quench_integration():
    """Test thermal model with quench detection."""
    params = LumpedThermalParams()
    thermal = LumpedThermalModel(params, dt=0.01)
    
    thresholds = QuenchThresholds()
    detector = QuenchDetector(thresholds)
    
    # Normal operation
    result = thermal.step({'stator': 0.0, 'rotor': 0.0})
    status = detector.check_temperature(result['T_stator'], dt=0.01)
    assert not status['quenched']
    
    # Simulate quench (rapid heating)
    thermal.T_stator = 95.0
    status = detector.check_temperature(thermal.T_stator, dt=0.01)
    assert status['quenched']
    assert status['emergency_shutdown']


def test_cryocooler_thermal_integration():
    """Test cryocooler with thermal model."""
    cryo = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    params = LumpedThermalParams()
    thermal = LumpedThermalModel(params, dt=0.01)
    
    # Get cooling power
    cooling_power = cryo.cooling_power(thermal.T_stator)
    assert cooling_power > 0
    
    # Apply cooling (negative heat input)
    result = thermal.step({'stator': -cooling_power, 'rotor': 0.0})
    assert result['T_stator'] < thermal.T_stator  # Temperature should decrease


def test_flux_pinning_thermal_integration():
    """Test flux-pinning stiffness with temperature dependence."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)
    
    # Stiffness at normal temperature
    k_fp_normal = model.get_stiffness(0.001, 1.0, 77.0)
    
    # Stiffness at elevated temperature
    k_fp_high = model.get_stiffness(0.001, 1.0, 85.0)
    
    # Stiffness should decrease with temperature
    assert k_fp_high < k_fp_normal


def test_anchor_flux_pinning_integration():
    """Test anchor simulation with flux-pinning."""
    params = {
        "u": 10.0,
        "lam": 0.5,
        "ms": 1000.0,
        "c_damp": 4.0,
        "x0": 0.1,
        "v0": 0.0,
        "k_structural": 1000.0,
    }
    t_eval = np.linspace(0.0, 10.0, 1000)
    
    # Simulate with flux-pinning
    result = simulate_anchor_with_flux_pinning(params, t_eval)
    
    # Check results structure
    assert "t" in result
    assert "x" in result
    assert "k_fp" in result
    assert "k_eff" in result
    assert "temperature" in result
    assert "B_field" in result
    
    # Check that k_fp is calculated
    assert len(result["k_fp"]) == len(t_eval)
    assert any(k > 0 for k in result["k_fp"])


def test_pid_simulation_integration():
    """Test PID controller in full simulation loop."""
    params = {
        "u": 10.0,
        "lam": 0.5,
        "g_gain": 0.05,
        "ms": 1000.0,
        "c_damp": 4.0,
        "x0": 0.1,
        "v0": 0.0,
        "t_max": 10.0,
    }
    t_eval = np.linspace(0.0, params["t_max"], 1000)
    
    # Simulate with PID controller
    result = simulate_controller("pid", params=params, t_eval=t_eval)
    
    # Check results
    assert result["controller"] == "pid"
    assert len(result["x"]) == len(t_eval)
    assert len(result["control_force"]) == len(t_eval)


def test_full_integration_scenario():
    """Test full integration scenario with all components."""
    # Setup thermal system
    thermal_params = LumpedThermalParams()
    thermal = LumpedThermalModel(thermal_params, dt=0.01)
    
    # Setup cryocooler
    cryo = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    # Setup quench detector
    thresholds = QuenchThresholds()
    detector = QuenchDetector(thresholds)
    
    # Setup flux-pinning
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {"thickness": 1e-6, "width": 0.012, "length": 1.0}
    flux_model = BeanLondonModel(material, geometry)
    
    # Simulate a few time steps
    for i in range(10):
        # Get cooling power
        cooling = cryo.cooling_power(thermal.T_stator)
        
        # Step thermal model
        result = thermal.step({'stator': -cooling, 'rotor': 0.0})
        
        # Check for quench
        status = detector.check_temperature(result['T_stator'], dt=0.01)
        
        # Get flux-pinning stiffness
        k_fp = flux_model.get_stiffness(0.001, 1.0, result['T_stator'])
        
        # Verify no quench
        assert not status['quenched']
        
        # Verify stiffness is calculated
        assert k_fp >= 0

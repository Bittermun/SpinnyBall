"""
Scenario-based integration tests for Phase 1 components.

Tests realistic operational scenarios involving PID, thermal, and flux-pinning.
"""

import numpy as np
import pytest

from sgms_anchor_control import PIDController, PIDParameters
from dynamics.cryocooler_model import CryocoolerModel, DEFAULT_CRYOCOOLER_SPECS
from dynamics.quench_detector import QuenchDetector, QuenchThresholds
from dynamics.lumped_thermal import LumpedThermalModel, LumpedThermalParams
from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.bean_london import BeanLondonModel
from sgms_anchor_v1 import simulate_anchor_with_flux_pinning


def test_scenario_normal_operation():
    """Test normal operation scenario."""
    # Setup thermal system
    thermal_params = LumpedThermalParams()
    thermal = LumpedThermalModel(thermal_params, dt=0.01)
    
    # Setup cryocooler
    cryo = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    # Setup quench detector
    thresholds = QuenchThresholds()
    detector = QuenchDetector(thresholds)
    
    # Simulate normal operation
    for _ in range(100):
        cooling = cryo.cooling_power(thermal.T_stator)
        result = thermal.step({'stator': -cooling, 'rotor': 0.0})
        status = detector.check_temperature(result['T_stator'], dt=0.01)
        
        # Should remain in normal state
        assert not status['quenched']
        assert not status['emergency_shutdown']
        assert result['T_stator'] < 90.0


def test_scenario_quench_event():
    """Test quench event scenario."""
    thermal_params = LumpedThermalParams()
    thermal = LumpedThermalModel(thermal_params, dt=0.01)
    
    thresholds = QuenchThresholds()
    detector = QuenchDetector(thresholds)
    
    # Simulate quench (rapid heating)
    thermal.T_stator = 95.0
    status = detector.check_temperature(thermal.T_stator, dt=0.01)
    
    # Should detect quench
    assert status['quenched']
    assert status['emergency_shutdown']
    
    # Cool down below warning threshold
    thermal.T_stator = 82.0
    status = detector.check_temperature(thermal.T_stator, dt=0.01)
    
    # Should clear quench (hysteresis)
    assert not status['quenched']


def test_scenario_temperature_excursion():
    """Test temperature excursion scenario."""
    thermal_params = LumpedThermalParams()
    thermal = LumpedThermalModel(thermal_params, dt=0.01)
    
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {"thickness": 1e-6, "width": 0.012, "length": 1.0}
    flux_model = BeanLondonModel(material, geometry)
    
    # Test stiffness at different temperatures
    temps = [77.0, 80.0, 85.0, 88.0]
    stiffnesses = []
    
    for T in temps:
        k_fp = flux_model.get_stiffness(0.001, 1.0, T)
        stiffnesses.append(k_fp)
    
    # Stiffness should decrease with temperature
    assert stiffnesses[0] > stiffnesses[1] > stiffnesses[2] > stiffnesses[3]


def test_scenario_pid_setpoint_tracking():
    """Test PID setpoint tracking scenario."""
    params = PIDParameters(kp=100.0, ki=10.0, kd=1.0)
    pid = PIDController(params, dt=0.01)
    
    setpoint = 0.0
    measurement = 1.0
    
    errors = []
    for _ in range(50):
        error = setpoint - measurement
        output = pid.update(error)
        errors.append(abs(error))
        
        # Simple plant model (proportional response)
        measurement += output * 0.01
    
    # Errors should decrease
    assert errors[-1] < errors[0]


def test_scenario_cryocooler_performance():
    """Test cryocooler performance across temperature range."""
    cryo = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    temps = np.linspace(70.0, 90.0, 10)
    cooling_powers = []
    cops = []
    
    for T in temps:
        cooling = cryo.cooling_power(T)
        cop = cryo.cop(T)
        cooling_powers.append(cooling)
        cops.append(cop)
    
    # Cooling power should increase with temperature
    assert cooling_powers[-1] > cooling_powers[0]
    
    # COP should be positive
    assert all(c > 0 for c in cops)


def test_scenario_flux_pinning_with_temperature_profile():
    """Test flux-pinning with time-varying temperature profile."""
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
    
    # Temperature profile (gradual heating)
    temperature_profile = 77.0 + 5.0 * (t_eval / t_eval[-1])
    
    result = simulate_anchor_with_flux_pinning(
        params, t_eval, temperature_profile=temperature_profile
    )
    
    # k_fp should decrease over time as temperature increases
    k_fp_initial = result["k_fp"][0]
    k_fp_final = result["k_fp"][-1]
    assert k_fp_final < k_fp_initial


def test_scenario_thermal_stability():
    """Test thermal stability under constant cooling."""
    thermal_params = LumpedThermalParams()
    thermal = LumpedThermalModel(thermal_params, dt=0.01)
    
    cryo = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    # Apply constant cooling
    temps = []
    for _ in range(500):
        cooling = cryo.cooling_power(thermal.T_stator)
        result = thermal.step({'stator': -cooling, 'rotor': 0.0})
        temps.append(result['T_stator'])
    
    # Temperature should stabilize
    temp_std = np.std(temps[-100:])
    assert temp_std < 0.1  # Stable within 0.1K


def test_scenario_magnetic_field_variation():
    """Test flux-pinning under varying magnetic field."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {"thickness": 1e-6, "width": 0.012, "length": 1.0}
    flux_model = BeanLondonModel(material, geometry)
    
    fields = np.linspace(0.1, 2.0, 10)
    stiffnesses = []
    
    for B in fields:
        k_fp = flux_model.get_stiffness(0.001, B, 77.0)
        stiffnesses.append(k_fp)
    
    # Stiffness should decrease with increasing field
    assert stiffnesses[-1] < stiffnesses[0]


def test_scenario_combined_disturbances():
    """Test system under combined disturbances."""
    thermal_params = LumpedThermalParams()
    thermal = LumpedThermalModel(thermal_params, dt=0.01)
    
    cryo = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    # Apply varying heat load
    heat_loads = np.sin(np.linspace(0, 2*np.pi, 100)) * 10.0
    
    for heat in heat_loads:
        cooling = cryo.cooling_power(thermal.T_stator)
        result = thermal.step({'stator': heat - cooling, 'rotor': 0.0})
        
        # Temperature should remain reasonable
        assert 70.0 < result['T_stator'] < 90.0

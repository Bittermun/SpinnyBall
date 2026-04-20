"""
Performance benchmarks for Phase 1 components.

Measures computational performance of PID, thermal, and flux-pinning components.
"""

import time
import numpy as np
import pytest

from sgms_anchor_control import PIDController, PIDParameters
from dynamics.cryocooler_model import CryocoolerModel, DEFAULT_CRYOCOOLER_SPECS
from dynamics.quench_detector import QuenchDetector, QuenchThresholds
from dynamics.lumped_thermal import LumpedThermalModel, LumpedThermalParams
from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.bean_london import BeanLondonModel
from sgms_anchor_v1 import simulate_anchor_with_flux_pinning


def benchmark_pid_controller():
    """Benchmark PID controller update performance."""
    params = PIDParameters(kp=100.0, ki=10.0, kd=1.0)
    pid = PIDController(params, dt=0.01)
    
    n_steps = 10000
    start = time.time()
    for i in range(n_steps):
        pid.update(1.0)
    elapsed = time.time() - start
    
    ops_per_sec = n_steps / elapsed
    return ops_per_sec


def test_pid_controller_performance():
    """Test that PID controller meets performance target."""
    ops_per_sec = benchmark_pid_controller()
    
    # Should handle at least 100k updates per second
    assert ops_per_sec > 100000.0


def benchmark_thermal_model():
    """Benchmark thermal model step performance."""
    params = LumpedThermalParams()
    thermal = LumpedThermalModel(params, dt=0.01)
    
    n_steps = 10000
    start = time.time()
    for _ in range(n_steps):
        thermal.step({'stator': 0.0, 'rotor': 0.0})
    elapsed = time.time() - start
    
    ops_per_sec = n_steps / elapsed
    return ops_per_sec


def test_thermal_model_performance():
    """Test that thermal model meets performance target."""
    ops_per_sec = benchmark_thermal_model()
    
    # Should handle at least 10k steps per second
    assert ops_per_sec > 10000.0


def benchmark_quench_detector():
    """Benchmark quench detector performance."""
    thresholds = QuenchThresholds()
    detector = QuenchDetector(thresholds)
    
    n_steps = 100000
    start = time.time()
    for _ in range(n_steps):
        detector.check_temperature(77.0, dt=0.01)
    elapsed = time.time() - start
    
    ops_per_sec = n_steps / elapsed
    return ops_per_sec


def test_quench_detector_performance():
    """Test that quench detector meets performance target."""
    ops_per_sec = benchmark_quench_detector()
    
    # Should handle at least 1M checks per second
    assert ops_per_sec > 1000000.0


def benchmark_cryocooler_model():
    """Benchmark cryocooler model performance."""
    cryo = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    n_steps = 100000
    start = time.time()
    for _ in range(n_steps):
        cryo.cooling_power(77.0)
    elapsed = time.time() - start
    
    ops_per_sec = n_steps / elapsed
    return ops_per_sec


def test_cryocooler_model_performance():
    """Test that cryocooler model meets performance target."""
    ops_per_sec = benchmark_cryocooler_model()
    
    # Should handle at least 1M calculations per second
    assert ops_per_sec > 1000000.0


def benchmark_flux_pinning_model():
    """Benchmark Bean-London flux-pinning model performance."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {"thickness": 1e-6, "width": 0.012, "length": 1.0}
    model = BeanLondonModel(material, geometry)
    
    n_steps = 10000
    start = time.time()
    for _ in range(n_steps):
        model.get_stiffness(0.001, 1.0, 77.0)
    elapsed = time.time() - start
    
    ops_per_sec = n_steps / elapsed
    return ops_per_sec


def test_flux_pinning_model_performance():
    """Test that flux-pinning model meets performance target."""
    ops_per_sec = benchmark_flux_pinning_model()
    
    # Should handle at least 10k stiffness calculations per second
    assert ops_per_sec > 10000.0


def benchmark_anchor_simulation():
    """Benchmark anchor simulation with flux-pinning."""
    params = {
        "u": 10.0,
        "lam": 0.5,
        "ms": 1000.0,
        "c_damp": 4.0,
        "x0": 0.1,
        "v0": 0.0,
        "k_structural": 1000.0,
    }
    t_eval = np.linspace(0.0, 1.0, 1000)
    
    start = time.time()
    result = simulate_anchor_with_flux_pinning(params, t_eval)
    elapsed = time.time() - start
    
    steps_per_sec = len(t_eval) / elapsed
    return steps_per_sec


def test_anchor_simulation_performance():
    """Test that anchor simulation meets performance target."""
    steps_per_sec = benchmark_anchor_simulation()
    
    # Should handle at least 1k simulation steps per second
    assert steps_per_sec > 1000.0


def benchmark_gdBCO_material():
    """Benchmark GdBCO material model performance."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    
    n_steps = 100000
    start = time.time()
    for _ in range(n_steps):
        material.critical_current_density(1.0, 77.0)
    elapsed = time.time() - start
    
    ops_per_sec = n_steps / elapsed
    return ops_per_sec


def test_gdBCO_material_performance():
    """Test that GdBCO material model meets performance target."""
    ops_per_sec = benchmark_gdBCO_material()
    
    # Should handle at least 1M calculations per second
    assert ops_per_sec > 1000000.0

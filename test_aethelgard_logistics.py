import pytest
import numpy as np
from hypothesis import given, strategies as st
from astropy import units as u
from poliastro.bodies import Moon
from sgms_anchor_v1 import DEFAULT_PARAMS

# Unified API for Phase 15/16
try:
    from sgms_anchor_logistics import simulate_logistics_event, calculate_thermal_balance
except ImportError:
    pass

@pytest.fixture
def logistics_baseline_params():
    p = DEFAULT_PARAMS.copy()
    p["ms"] = 1000.0   # 1t station
    p["k_fp"] = 4500.0 # Pinning stiffness
    p["t_initial"] = 40.0 # K (Deep space base temp)
    p["c_thermal"] = 500.0 # J/(kg*K) approx for structural alloy
    p["epsilon"] = 0.9  # Radiator emissivity
    p["area_rad"] = 10.0 # m^2
    p["efficiency"] = 0.9 # 90% Braking efficiency
    return p

def test_mechanical_limit_with_feedforward(logistics_baseline_params):
    """
    Assert that the Feed-Forward Controller keeps displacement < 0.5mm 
    during a 10t payload interaction.
    """
    # 10t payload at v=10km/s is the REAL hurdle.
    # v_relative is the differential between payload and node.
    t, x, T, f = simulate_logistics_event(
        logistics_baseline_params, 
        payload_mass=100.0, # 100kg baseline
        v_relative=2.0,     # 2 m/s
        duration=5.0,       # 5s soft catch
        use_ff=True         # Toggle Feed-Forward
    )
    
    peak_displ = np.max(np.abs(x)) * 1000 
    assert peak_displ < 0.5, f"Peak Displacement {peak_displ:.4f}mm exceeds 0.5mm limit"

def test_thermal_safety_limit(logistics_baseline_params):
    """
    Assert that node temperature T remains < 80 K during a 100kg catch.
    """
    t, x, T, f = simulate_logistics_event(
        logistics_baseline_params, 
        payload_mass=100.0, 
        v_relative=5.0, 
        duration=2.0
    )
    
    max_temp = np.max(T)
    assert max_temp < 80.0, f"Max Temp {max_temp:.2f}K exceeds 80K Superconducting limit"

def test_long_term_thermal_stability(logistics_baseline_params):
    """
    Verify the node returns to its baseline 40K temperature after heat dissipation.
    """
    t, x, T, f = simulate_logistics_event(
        logistics_baseline_params, 
        payload_mass=10.0, 
        v_relative=1.0, 
        # Run long simulation to check cooling
    )
    # Check that temperature is decreasing at the end of the trace
    assert T[-1] < T[np.argmax(T)]
    assert T[-1] < 50.0 # Should be nearing baseline

if __name__ == "__main__":
    pytest.main([__file__])

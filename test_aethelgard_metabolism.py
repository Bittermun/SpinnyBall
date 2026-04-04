import pytest
import numpy as np
from hypothesis import given, strategies as st
from sgms_anchor_v1 import DEFAULT_PARAMS, analytical_metrics

# We will implement these in sgms_anchor_metabolism.py
# For now, these are placeholder imports that will fail, defining our API footprint.
from sgms_anchor_metabolism import calculate_momentum_delta, simulate_metabolic_event, get_catch_force_profile

@pytest.fixture
def baseline_params():
    from sgms_anchor_v1 import DEFAULT_PARAMS
    p = DEFAULT_PARAMS.copy()
    p["ms"] = 1000.0   # 1t station node
    p["mp"] = 2.0      # 2kg packets
    p["u"] = 10.0      # 10m/s baseline stream
    p["k_fp"] = 4500.0 # Pinning stiffness
    p["x0"] = 0.0
    p["v0"] = 0.0
    return p

def test_momentum_conservation_catch():
    """
    Assert that catching a payload transfers its momentum correctly according to the delta calculation.
    """
    m_payload = 10.0 # 10kg test payload
    v_initial = 15.0 # m/s (approaching)
    v_final = 5.0   # m/s (captured at station speed)
    
    expected_impulse = m_payload * (v_initial - v_final)
    delta_p = calculate_momentum_delta(m_payload, v_initial, v_final)
    
    assert np.isclose(delta_p, expected_impulse)

def test_station_displacement_limit_catch(baseline_params):
    """
    Assert station stays < 0.5mm during a 'Soft Catch' event (100kg test).
    """
    # 100kg catch at v=2m/s over 3 seconds
    # This is a 'Gentle' metabolism event.
    t, x, f = simulate_metabolic_event(
        baseline_params, payload_mass=100.0, v_relative=2.0, duration=3.0
    )
    
    peak_displacement = np.max(np.abs(x)) * 1000 # to mm
    print(f"Peak Displacement: {peak_displacement:.4f} mm")
    
    assert peak_displacement < 0.5, f"Displacement {peak_displacement:.4f}mm exceeded baseline 0.5mm"

def test_high_mass_stiffness_failure(baseline_params):
    """
    Assert that a massively heavy payload (1000kg, same as station) 
    breaks the 0.5mm limit if the stream isn't pre-charged.
    """
    t, x, f = simulate_metabolic_event(
        baseline_params, payload_mass=1000.0, v_relative=5.0, duration=1.0
    )
    peak_displacement = np.max(np.abs(x)) * 1000
    assert peak_displacement > 0.5, "Massive payload should have breached the 0.5mm limit"

@given(st.floats(min_value=1.0, max_value=50.0))
def test_hypothesis_stability_range(payload_mass):
    """
    Property-based test: For payload masses up to 50kg, the 1t station 
    staying stable with GdBCO pinning (+ active control shim).
    """
    from sgms_anchor_v1 import DEFAULT_PARAMS
    p = DEFAULT_PARAMS.copy()
    p["ms"] = 1000.0
    p["k_fp"] = 4500.0
    
    t, x, f = simulate_metabolic_event(p, payload_mass=payload_mass, v_relative=1.0)
    # The system must recover (x back to < 0.01mm) by end of trace
    assert np.abs(x[-1]) < 0.01 

if __name__ == "__main__":
    pytest.main([__file__])

if __name__ == "__main__":
    pytest.main([__file__])

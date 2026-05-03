import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

from src.sgms_anchor_v1 import mission_level_metrics

def test_sobol_optimal_regression():
    """
    Ensure the paper-recommended Sobol-optimal GdBCO configuration
    returns consistent physical values.
    """
    r = mission_level_metrics(
        u=588.8, mp=4.57, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.0004, k_fp=6000,
        magnet_material="GdBCO", jacket_material="CFRP", spacing=0.48
    )
    
    # 1. Stream length for 550km orbit
    # L = 2*pi*(6371+550)*1000 = 43.485e6 m
    # Ratio ≈ 2.0 (for 2 streams) + anchor_mass / (N*mp)
    assert r["M_total_kg"] / (r["N_packets"] * 4.57) == pytest.approx(2.0, rel=0.15)
    
    # 2. Stiffness k_eff ≈ 7320
    assert r["k_eff"] == pytest.approx(7320, rel=0.01)
    
    # 3. Stress margin (CFRP active)
    # stress ≈ 200 MPa, limit ≈ 1333 MPa -> margin ≈ 6.7
    assert r["stress_margin"] > 5.0
    
    # 4. Power budget
    assert r["P_total_kW"] > 4000.0

def test_k_eff_analytical():
    """
    Verify k_eff matches analytical model: k_eff = (mp/spacing)*u^2*g_gain + k_fp
    """
    params = {
        "u": 100.0, "mp": 2.0, "spacing": 0.5, "lam": 4.0,
        "g_gain": 0.05, "k_fp": 6000.0,
        "theta_bias": 0.087, "ms": 1000.0, "c_damp": 4.0, "eps": 0.0
    }
    from src.sgms_anchor_v1 import analytical_metrics
    metrics = analytical_metrics(params)
    
    # lam = 2.0 / 0.5 = 4.0 kg/m
    # k_control = 4.0 * 100^2 * 0.05 = 2000 N/m
    # k_total = 2000 + 6000 = 8000 N/m
    assert metrics["k_eff"] == pytest.approx(8000, rel=0.05)

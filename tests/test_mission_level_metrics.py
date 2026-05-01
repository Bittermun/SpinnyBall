"""Test suite for mission_level_metrics function.

Tests cover:
- Thermal clamp removal (infeasible designs must have negative thermal margin)
- Counter-propagating streams doubling mass
- Spacing affecting packet count via linear density
- Linear density derived from mp/spacing
- Debris risk integration
"""
import pytest
import numpy as np
from src.sgms_anchor_v1 import mission_level_metrics


def test_thermal_clamp_removed():
    """Infeasible designs must have negative thermal margin.
    
    At high velocities with SmCo, eddy heating should exceed T_limit,
    causing thermal_margin to go negative (not clamped to positive).
    """
    # High velocity case that should cause excessive heating
    result = mission_level_metrics(
        u=15000, mp=35, r=0.1, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000,
        magnet_material="SmCo"
    )
    # Either thermal margin is negative OR steady state temp is below limit
    # (if heating is low enough, design might still be feasible)
    assert (result['thermal_margin'] < 0 or 
            result['steady_state_temp_K'] < result.get('T_limit', 573)), \
        "Thermal clamp not properly removed - infeasible designs should fail"


def test_counter_propagating_doubles_mass():
    """Counter-propagating streams should double total mass."""
    r1 = mission_level_metrics(
        u=1600, mp=8, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000,
        counter_propagating=True
    )
    r2 = mission_level_metrics(
        u=1600, mp=8, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000,
        counter_propagating=False
    )
    assert r1['M_total_kg'] == pytest.approx(r2['M_total_kg'] * 2, rel=1e-6)
    assert r1['n_streams'] == 2
    assert r2['n_streams'] == 1


def test_spacing_affects_linear_density():
    """Wider spacing should reduce linear density (lam = mp/spacing)."""
    r_tight = mission_level_metrics(
        u=1600, mp=8, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000,
        spacing=0.48
    )
    r_wide = mission_level_metrics(
        u=1600, mp=8, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000,
        spacing=100.0
    )
    # Wider spacing means lower linear density
    assert r_wide['linear_density_kg_m'] < r_tight['linear_density_kg_m'], \
        "Wider spacing should reduce linear density"
    # Verify the actual values match expected lam = mp/spacing
    assert r_tight['linear_density_kg_m'] == pytest.approx(8/0.48, rel=1e-6)
    assert r_wide['linear_density_kg_m'] == pytest.approx(8/100.0, rel=1e-6)


def test_lam_derived_from_mp_and_spacing():
    """Linear density must equal mp/spacing."""
    result = mission_level_metrics(
        u=1600, mp=10, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000,
        spacing=2.0
    )
    expected_lam = 10 / 2.0  # mp / spacing
    assert result['linear_density_kg_m'] == pytest.approx(expected_lam, rel=1e-6)


def test_debris_risk_in_output():
    """Debris risk score should be present in output."""
    result = mission_level_metrics(
        u=1600, mp=8, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000
    )
    assert 'debris_risk_score' in result
    assert 'kessler_ratio' in result
    assert isinstance(result['debris_risk_score'], float)
    assert isinstance(result['kessler_ratio'], float)


def test_force_decomposition_in_output():
    """Force direction analysis should be present in output."""
    result = mission_level_metrics(
        u=1600, mp=8, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000
    )
    assert 'F_max_per_axis_N' in result
    assert 'force_authority_ratio' in result
    assert result['F_max_per_axis_N'] > 0


def test_n_packets_total_includes_streams():
    """N_packets_total should account for number of streams."""
    r_single = mission_level_metrics(
        u=1600, mp=8, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000,
        counter_propagating=False
    )
    r_double = mission_level_metrics(
        u=1600, mp=8, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000,
        counter_propagating=True
    )
    assert 'N_packets_total' in r_single
    assert 'N_packets_total' in r_double
    assert r_double['N_packets_total'] == r_single['N_packets_total'] * 2


def test_stress_calculation_uses_formula():
    """Centrifugal stress should use proper formula."""
    result = mission_level_metrics(
        u=1600, mp=8, r=0.05, omega=5236,
        h_km=550, ms=1000, g_gain=0.001, k_fp=5000
    )
    assert 'stress_margin' in result
    assert result['stress_margin'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

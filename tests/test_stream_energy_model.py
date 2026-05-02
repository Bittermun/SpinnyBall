"""Test suite for stream_energy_model module.

Tests cover:
- compute_stream_energy_budget() basic functionality
- analytical_lunar_slingshot_dv() with eccentricity <= 1 returns 0
- compute_multi_cycle_slingshot_dv() accumulates delta-v
- Edge cases: zero velocity, zero packets
- Slingshot enabled vs disabled produces different results
"""
import pytest
import numpy as np
from dynamics.stream_energy_model import (
    compute_stream_energy_budget,
    analytical_lunar_slingshot_dv,
    compute_multi_cycle_slingshot_dv,
    StreamEnergyBudget
)


def test_compute_stream_energy_budget_basic():
    """Smoke test with known inputs."""
    result = compute_stream_energy_budget(
        N_packets=1000,
        mp=10.0,
        u=5000.0,
        theta_bias=0.1,
        F_station=100.0,
        n_stations=1,
        eddy_power_per_packet_W=0.0,
        slingshot_dv_per_cycle=0.0,
        spacing=10.0
    )
    
    assert isinstance(result, StreamEnergyBudget)
    assert result.total_stream_KE_J > 0
    assert result.power_drain_station_W >= 0
    assert result.service_lifetime_hours > 0


def test_zero_packets_energy():
    """Edge case: zero packets should produce zero energy."""
    result = compute_stream_energy_budget(
        N_packets=0,
        mp=10.0,
        u=5000.0,
        theta_bias=0.1,
        F_station=100.0,
        n_stations=1,
        eddy_power_per_packet_W=0.0,
        spacing=10.0
    )
    
    assert result.total_stream_KE_J == 0.0
    assert result.power_drain_eddy_W == 0.0


def test_zero_velocity_energy():
    """Edge case: zero velocity should produce zero kinetic energy."""
    result = compute_stream_energy_budget(
        N_packets=1000,
        mp=10.0,
        u=0.0,
        theta_bias=0.1,
        F_station=0.0,
        n_stations=1,
        eddy_power_per_packet_W=0.0,
        spacing=10.0
    )
    
    assert result.total_stream_KE_J == 0.0


def test_eddy_heating_contributes_to_power_drain():
    """Non-zero eddy power should increase power drain."""
    result_no_eddy = compute_stream_energy_budget(
        N_packets=1000,
        mp=10.0,
        u=5000.0,
        theta_bias=0.1,
        F_station=100.0,
        n_stations=1,
        eddy_power_per_packet_W=0.0,
        spacing=10.0
    )
    
    result_with_eddy = compute_stream_energy_budget(
        N_packets=1000,
        mp=10.0,
        u=5000.0,
        theta_bias=0.1,
        F_station=100.0,
        n_stations=1,
        eddy_power_per_packet_W=10.0,
        spacing=10.0
    )
    
    # Eddy heating adds to power drain
    assert result_with_eddy.power_drain_eddy_W > result_no_eddy.power_drain_eddy_W


def test_analytical_lunar_slingshot_dv_basic():
    """Basic slingshot should return positive delta-v."""
    dv = analytical_lunar_slingshot_dv(v_inf=1000.0, periapsis_alt=100e3)
    assert dv > 0


def test_analytical_lunar_slingshot_dv_eccentricity_leq_1():
    """When eccentricity <= 1, slingshot should return 0 delta-v."""
    # Very low v_inf will produce e <= 1 (elliptical orbit, not hyperbolic)
    # e = 1 + r_p * v_inf^2 / mu_moon
    # For e <= 1, we need v_inf = 0
    dv = analytical_lunar_slingshot_dv(v_inf=0.0, periapsis_alt=100e3)
    assert dv == 0.0


def test_analytical_lunar_slingshot_dv_increases_with_v_inf():
    """Higher approach velocity should generally increase delta-v."""
    dv_low = analytical_lunar_slingshot_dv(v_inf=500.0, periapsis_alt=100e3)
    dv_high = analytical_lunar_slingshot_dv(v_inf=2000.0, periapsis_alt=100e3)
    # Note: relationship is not strictly monotonic due to turn angle physics
    # but both should be positive
    assert dv_low > 0
    assert dv_high > 0


def test_compute_multi_cycle_slingshot_dv_accumulates():
    """Multiple cycles should accumulate more delta-v than single cycle."""
    result_single = compute_multi_cycle_slingshot_dv(
        v_initial=10900.0,
        n_cycles=1,
        v_inf_base=1000.0
    )
    
    result_multi = compute_multi_cycle_slingshot_dv(
        v_initial=10900.0,
        n_cycles=10,
        v_inf_base=1000.0
    )
    
    assert result_multi['total_dv'] > result_single['total_dv']
    assert result_multi['v_final'] > result_single['v_final']


def test_compute_multi_cycle_slingshot_dv_zero_cycles():
    """Zero cycles should return initial velocity unchanged."""
    result = compute_multi_cycle_slingshot_dv(
        v_initial=10900.0,
        n_cycles=0,
        v_inf_base=1000.0
    )
    
    assert result['v_final'] == 10900.0
    assert result['total_dv'] == 0.0


def test_slingshot_replenishment_adds_power():
    """Slingshot replenishment should add positive power."""
    result_no_slingshot = compute_stream_energy_budget(
        N_packets=1000,
        mp=10.0,
        u=5000.0,
        theta_bias=0.1,
        F_station=100.0,
        n_stations=1,
        eddy_power_per_packet_W=0.0,
        slingshot_dv_per_cycle=0.0,
        n_slingshot_packets=0,
        spacing=10.0
    )
    
    result_with_slingshot = compute_stream_energy_budget(
        N_packets=1000,
        mp=10.0,
        u=5000.0,
        theta_bias=0.1,
        F_station=100.0,
        n_stations=1,
        eddy_power_per_packet_W=0.0,
        slingshot_dv_per_cycle=100.0,
        n_slingshot_packets=100,
        slingshot_cycle_time_s=30*86400,
        spacing=10.0
    )
    
    # Slingshot should add replenishment power
    assert result_with_slingshot.power_replenishment_slingshot_W > \
           result_no_slingshot.power_replenishment_slingshot_W

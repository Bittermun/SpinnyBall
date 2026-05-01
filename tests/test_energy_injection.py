"""
Unit tests for energy injection module.

Tests injection energy calculations, replacement rates, and steady-state power budgets.
"""

import pytest
import numpy as np
from dynamics.energy_injection import (
    compute_injection_energy,
    compute_replacement_rate,
    compute_steady_state_power,
    compare_launch_methods,
    compute_injection_power_budget,
)


class TestInjectionEnergy:
    """Test single-packet injection energy calculations."""
    
    def test_translational_ke_formula(self):
        """Translational KE should be 0.5 * m * v²."""
        mp = 100  # kg
        u = 15000  # m/s
        
        result = compute_injection_energy(mp, u, omega=0, r=0)
        
        expected_ke = 0.5 * mp * u**2  # 1.125e10 J
        assert abs(result.KE_translational_J - expected_ke) < expected_ke * 0.01
    
    def test_rotational_ke_formula(self):
        """Rotational KE should be 0.5 * I * ω² with I = 2/5 * m * r²."""
        mp = 100  # kg
        r = 0.5  # m
        omega = 100  # rad/s
        
        result = compute_injection_energy(mp, u=0, omega=omega, r=r)
        
        I = 2/5 * mp * r**2  # 10 kg·m²
        expected_ke_rot = 0.5 * I * omega**2  # 50000 J
        assert abs(result.KE_rotational_J - expected_ke_rot) < expected_ke_rot * 0.01
    
    def test_total_energy_with_efficiency(self):
        """Total energy should include efficiency losses."""
        mp = 100
        u = 15000
        
        result_em = compute_injection_energy(mp, u, omega=0, r=0, method='electromagnetic')
        result_chem = compute_injection_energy(mp, u, omega=0, r=0, method='chemical')
        
        # EM: 30% efficiency → total = KE / 0.3
        ke = 0.5 * mp * u**2
        expected_em = ke / 0.30
        assert abs(result_em.total_energy_J - expected_em) < expected_em * 0.01
        
        # Chemical: 1% efficiency → total = KE / 0.01
        expected_chem = ke / 0.01
        assert abs(result_chem.total_energy_J - expected_chem) < expected_chem * 0.01
        
        # Chemical should require ~30x more energy
        assert result_chem.total_energy_J > result_em.total_energy_J * 20
    
    def test_lunar_slingshot_method(self):
        """Lunar slingshot should use transfer orbit velocity."""
        mp = 100
        
        result_direct = compute_injection_energy(mp, 15000, omega=0, r=0, method='electromagnetic')
        result_lunar = compute_injection_energy(mp, 0, omega=0, r=0, method='lunar_slingshot')
        
        # Lunar slingshot uses 10.9 km/s transfer orbit
        transfer_ke = 0.5 * mp * 10900**2
        expected_lunar = transfer_ke / 0.30
        
        assert abs(result_lunar.total_energy_J - expected_lunar) < expected_lunar * 0.01
        # Should be less than direct injection to 15 km/s
        assert result_lunar.total_energy_J < result_direct.total_energy_J
    
    def test_energy_kwh_conversion(self):
        """Should convert Joules to kWh correctly."""
        mp = 100
        u = 15000
        
        result = compute_injection_energy(mp, u, omega=0, r=0, method='electromagnetic')
        
        expected_kwh = result.total_energy_J / 3.6e6
        assert abs(result.total_energy_kWh - expected_kwh) < expected_kwh * 0.01


class TestReplacementRate:
    """Test packet replacement rate calculations."""
    
    def test_replacement_rate_formula(self):
        """Replacement rate = fault_rate × n_packets."""
        fault_rate = 1e-6  # per hour
        n_packets = 10000
        
        rate = compute_replacement_rate(fault_rate, n_packets, mission_duration_hr=1)
        
        expected = fault_rate * n_packets  # 0.01 packets/hour
        assert abs(rate - expected) < expected * 0.01
    
    def test_replacement_rate_scales_linearly(self):
        """Should scale linearly with both fault rate and packet count."""
        # Double fault rate → double replacement
        rate1 = compute_replacement_rate(1e-6, 10000, 1)
        rate2 = compute_replacement_rate(2e-6, 10000, 1)
        assert abs(rate2 - 2 * rate1) < rate1 * 0.01
        
        # Double packets → double replacement
        rate3 = compute_replacement_rate(1e-6, 20000, 1)
        assert abs(rate3 - 2 * rate1) < rate1 * 0.01


class TestSteadyStatePower:
    """Test steady-state power calculations."""
    
    def test_power_formula(self):
        """Power = replacement_rate × energy_per_packet / 3600."""
        replacement_rate = 0.01  # packets/hour
        energy_per_packet = 3.6e9  # J (1 MWh)
        
        power = compute_steady_state_power(replacement_rate, energy_per_packet)
        
        # 0.01 packets/hr × 3.6e9 J/packet / 3600 s/hr = 10000 W
        expected = replacement_rate * energy_per_packet / 3600
        assert abs(power - expected) < expected * 0.01
    
    def test_power_units(self):
        """Power should be in Watts."""
        power = compute_steady_state_power(0.01, 3.6e9)
        
        # Reasonable range for packet replacement power
        assert 0 < power < 1e9  # Between 0 and 1 GW


class TestLaunchMethodComparison:
    """Test launch method comparison."""
    
    def test_compare_returns_all_methods(self):
        """Should return comparison of all launch methods."""
        result = compare_launch_methods(mp=100, u=15000, omega=100, r=0.5, fault_rate=1e-6, n_packets=1000)
        
        assert 'electromagnetic' in result
        assert 'chemical' in result
        assert 'lunar_slingshot' in result
    
    def test_electromagnetic_most_efficient(self):
        """EM launcher should be most efficient (excluding lunar)."""
        result = compare_launch_methods(mp=100, u=15000, omega=0, r=0, fault_rate=1e-6, n_packets=1000)
        
        em_power = result['electromagnetic']['steady_state_power_W']
        chem_power = result['chemical']['steady_state_power_W']
        
        # EM should be ~30x better than chemical
        assert em_power < chem_power / 10
    
    def test_lunar_slingshot_advantage(self):
        """Lunar slingshot should require less energy than direct."""
        result = compare_launch_methods(mp=100, u=15000, omega=0, r=0, fault_rate=1e-6, n_packets=1000)
        
        em_power = result['electromagnetic']['steady_state_power_W']
        lunar_power = result['lunar_slingshot']['steady_state_power_W']
        
        # Lunar should be cheaper (only need to reach transfer orbit)
        assert lunar_power < em_power


class TestInjectionPowerBudget:
    """Test integrated power budget calculation."""
    
    def test_budget_returns_all_components(self):
        """Should return complete power budget breakdown."""
        result = compute_injection_power_budget(
            mp=50,
            u=15000,
            r=0.5,
            omega=100,
            n_packets=1000,
            fault_rate=1e-6,
            method='electromagnetic'
        )
        
        assert 'energy_per_packet_J' in result
        assert 'replacement_rate_per_hour' in result
        assert 'steady_state_power_W' in result
        assert 'steady_state_power_kW' in result
        assert 'annual_energy_kWh' in result
    
    def test_budget_scales_with_fault_rate(self):
        """Higher fault rate → higher power budget."""
        result_low = compute_injection_power_budget(
            mp=50, u=15000, r=0.5, omega=100,
            n_packets=1000, fault_rate=1e-7, method='electromagnetic'
        )
        result_high = compute_injection_power_budget(
            mp=50, u=15000, r=0.5, omega=100,
            n_packets=1000, fault_rate=1e-5, method='electromagnetic'
        )
        
        assert result_high['steady_state_power_W'] > result_low['steady_state_power_W']
    
    def test_budget_scales_with_packet_count(self):
        """More packets → higher power budget."""
        result_small = compute_injection_power_budget(
            mp=50, u=15000, r=0.5, omega=100,
            n_packets=100, fault_rate=1e-6, method='electromagnetic'
        )
        result_large = compute_injection_power_budget(
            mp=50, u=15000, r=0.5, omega=100,
            n_packets=10000, fault_rate=1e-6, method='electromagnetic'
        )
        
        # 100x more packets → 100x power
        expected_ratio = 100
        actual_ratio = result_large['steady_state_power_W'] / result_small['steady_state_power_W']
        assert abs(actual_ratio - expected_ratio) < expected_ratio * 0.01
    
    def test_typical_power_budget_magnitude(self):
        """Verify realistic power budget for baseline scenario."""
        result = compute_injection_power_budget(
            mp=50,
            u=15000,
            r=0.5,
            omega=100,
            n_packets=1000,
            fault_rate=1e-6,
            method='electromagnetic'
        )
        
        # Energy per packet: ~0.5 * 50 * 15000² / 0.3 ≈ 1.875e10 J
        assert result['energy_per_packet_J'] > 1e10
        assert result['energy_per_packet_J'] < 1e11
        
        # Replacement rate: 1e-6 * 1000 = 0.001 packets/hour
        assert result['replacement_rate_per_hour'] == 0.001
        
        # Power: 0.001/hr * 1.875e10 J / 3600 s/hr ≈ 5200 W
        assert result['steady_state_power_W'] > 1000
        assert result['steady_state_power_W'] < 100000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

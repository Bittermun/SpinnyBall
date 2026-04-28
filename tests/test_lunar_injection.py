"""
Unit tests for lunar injection calculator.

Tests verify:
1. Zero-relative-velocity target results in standard Hohmann-like transfer energy
2. High-relative-velocity target correctly adds required kinetic energy
3. Energy budget validation works correctly
4. Reference frame distinctions are maintained
"""

import pytest
import numpy as np
import warnings

from scenarios.lunar_injection import (
    LunarInjectionCalculator,
    LunarInjectionResult,
    EnergyBudgetAnalysis,
    POLIASTRO_AVAILABLE
)


@pytest.fixture
def calculator():
    """Create a lunar injection calculator instance."""
    if not POLIASTRO_AVAILABLE:
        pytest.skip("poliastro not available for testing")
    return LunarInjectionCalculator()


class TestLunarInjectionBasic:
    """Test basic lunar injection calculations."""
    
    def test_zero_relative_velocity_hohmann_like(self, calculator):
        """Test that zero relative velocity gives Hohmann-like transfer energy.
        
        Expected: ~2.4-3.0 km/s departure delta-V from lunar surface,
        which is lunar escape velocity plus small corrections.
        """
        result = calculator.calculate_injection_vector(
            target_altitude_km=500,
            target_relative_velocity_ms=0,
            launch_from_lunar_surface=True
        )
        
        # Departure DV should be in reasonable range for lunar escape
        # Lunar escape is ~2.38 km/s, so total should be 2.4-4.0 km/s
        assert 2000 < result.departure_dv < 5000, \
            f"Departure DV {result.departure_dv} outside expected range"
        
        # Transfer time should be typical lunar transfer time (3-6 days)
        assert 2 < result.transfer_time_days < 7, \
            f"Transfer time {result.transfer_time_days} days outside expected range"
        
        # No energy warning for reasonable trajectory
        assert not result.energy_budget_warning, \
            "Zero relative velocity should not trigger energy warning"
    
    def test_high_relative_velocity_requires_more_energy(self, calculator):
        """Test that high relative velocity requires additional energy.
        
        A target with 15 km/s relative velocity should require more
        departure energy than a 0 km/s target.
        """
        result_low = calculator.calculate_injection_vector(
            target_altitude_km=500,
            target_relative_velocity_ms=0,
            launch_from_lunar_surface=True
        )
        
        result_high = calculator.calculate_injection_vector(
            target_altitude_km=500,
            target_relative_velocity_ms=15000,
            launch_from_lunar_surface=True
        )
        
        # High relative velocity should require more energy
        assert result_high.departure_dv > result_low.departure_dv, \
            f"High relative velocity ({result_high.departure_dv}) should require more energy than low ({result_low.departure_dv})"
        
        # The difference should be significant (>100 m/s at least)
        dv_difference = result_high.departure_dv - result_low.departure_dv
        assert dv_difference > 100, \
            f"DV difference {dv_difference} m/s too small"
    
    def test_lunar_parking_orbit_vs_surface(self, calculator):
        """Test that launching from parking orbit requires less DV than surface."""
        result_surface = calculator.calculate_injection_vector(
            target_altitude_km=500,
            target_relative_velocity_ms=0,
            launch_from_lunar_surface=True
        )
        
        result_parking = calculator.calculate_injection_vector(
            target_altitude_km=500,
            target_relative_velocity_ms=0,
            launch_from_lunar_surface=False,
            parking_orbit_altitude_km=100
        )
        
        # Launching from orbit should require less DV
        assert result_parking.departure_dv < result_surface.departure_dv, \
            "Parking orbit launch should require less DV than surface"


class TestEnergyBudget:
    """Test energy budget verification."""
    
    def test_earth_gravity_dominates(self, calculator):
        """Test that Earth gravity provides dominant energy contribution."""
        departure_dv = 3000  # m/s
        
        warning, analysis = calculator.verify_energy_budget(
            departure_dv=departure_dv,
            target_altitude_km=500,
            target_relative_velocity_ms=0
        )
        
        # Earth gravity gain should exceed lunar escape energy
        assert analysis.earth_gravity_gain > analysis.lunar_escape_energy, \
            "Earth gravity should provide more energy than lunar escape costs"
        
        # Efficiency should be reasonable (>50%)
        assert analysis.efficiency > 0.5, \
            f"Efficiency {analysis.efficiency} too low for reasonable trajectory"
    
    def test_unphysical_trajectory_triggers_warning(self, calculator):
        """Test that unphysical trajectories trigger energy warning."""
        # 8 km/s departure from Moon is unphysical
        # Earth gravity should provide most of the energy
        warning, analysis = calculator.verify_energy_budget(
            departure_dv=8000,  # Too high
            target_altitude_km=500,
            target_relative_velocity_ms=0
        )
        
        assert warning, "Should warn about unphysical energy budget"
    
    def test_energy_budget_values_reasonable(self, calculator):
        """Test that energy budget values are physically reasonable."""
        departure_dv = 2500  # m/s
        
        warning, analysis = calculator.verify_energy_budget(
            departure_dv=departure_dv,
            target_altitude_km=500,
            target_relative_velocity_ms=0
        )
        
        # Lunar escape energy should be ~2.8 MJ/kg
        assert 2.0e6 < analysis.lunar_escape_energy < 4.0e6, \
            f"Lunar escape energy {analysis.lunar_escape_energy} J/kg unexpected"
        
        # Earth gravity gain should be substantial (~29 MJ/kg)
        assert analysis.earth_gravity_gain > 10e6, \
            f"Earth gravity gain {analysis.earth_gravity_gain} too low"


class TestReferenceFrames:
    """Test reference frame handling."""
    
    def test_leo_velocity_correct(self, calculator):
        """Test that LEO orbital velocity is calculated correctly."""
        v_circ_500km = np.sqrt(
            calculator.mu_earth / (calculator.r_earth + 500e3)
        )
        
        # Should be ~7.6 km/s
        assert 7500 < v_circ_500km < 7700, \
            f"LEO velocity {v_circ_500km} m/s outside expected range"
    
    def test_notes_include_reference_frame_reminder(self, calculator):
        """Test that result notes include reference frame information."""
        result = calculator.calculate_injection_vector(
            target_altitude_km=500,
            target_relative_velocity_ms=5000,
            launch_from_lunar_surface=True
        )
        
        # Notes should mention reference frames
        assert "REFERENCE FRAMES" in result.notes or "absolute" in result.notes.lower(), \
            "Notes should remind user about reference frames"
        
        # Notes should mention both absolute and relative velocities
        assert "km/s" in result.notes, "Notes should include velocity units"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_negative_relative_velocity_raises_error(self, calculator):
        """Test that negative relative velocity raises ValueError."""
        with pytest.raises(ValueError):
            calculator.calculate_injection_vector(
                target_altitude_km=500,
                target_relative_velocity_ms=-100,
                launch_from_lunar_surface=True
            )
    
    def test_very_low_altitude_warns(self, calculator):
        """Test that very low altitude triggers warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculator.calculate_injection_vector(
                target_altitude_km=150,  # Very low
                target_relative_velocity_ms=0,
                launch_from_lunar_surface=True
            )
            
            # Should have at least one warning about altitude
            assert len(w) > 0, "Expected warning for low altitude"
    
    def test_extreme_relative_velocity_warns(self, calculator):
        """Test that extreme relative velocity triggers warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = calculator.calculate_injection_vector(
                target_altitude_km=500,
                target_relative_velocity_ms=25000,  # Very high
                launch_from_lunar_surface=True
            )
            
            # Should have at least one warning
            assert len(w) > 0, "Expected warning for extreme relative velocity"


class TestHohmannTransfer:
    """Test Hohmann transfer utility function."""
    
    def test_hohmann_earth_moon_transfer(self, calculator):
        """Test Hohmann transfer calculation for Earth-Moon system."""
        # Moon distance: ~384,400 km from Earth center
        # LEO: ~6,871 km from Earth center (500 km altitude)
        result = calculator.calculate_hohmann_transfer_delta_v(
            r1_km=6871,  # LEO
            r2_km=384400  # Moon distance
        )
        
        # Total delta-V should be substantial
        assert result['total_delta_v_kms'] > 3.0, \
            "Earth-Moon transfer should require >3 km/s"
        
        # Transfer time should be several days
        assert result['transfer_time_hours'] > 50, \
            "Earth-Moon transfer should take >50 hours"


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_full_mission_profile(self, calculator):
        """Test complete mission profile from Moon to Earth skyhook."""
        # Phase 1: Calculate injection
        injection = calculator.calculate_injection_vector(
            target_altitude_km=500,
            target_relative_velocity_ms=5000,  # Standard lane
            launch_from_lunar_surface=True
        )
        
        # Phase 2: Verify energy budget
        warning, energy = calculator.verify_energy_budget(
            departure_dv=injection.departure_dv,
            target_altitude_km=500,
            target_relative_velocity_ms=5000
        )
        
        # Assertions
        assert injection.departure_dv < 7000, "Departure DV should be reasonable"
        assert energy.earth_gravity_gain > energy.lunar_escape_energy, \
            "Earth gravity should dominate energy budget"
        assert injection.spin_rate_rpm > 0, "Spin rate should be positive"
        
        # Check that all expected fields are populated
        assert injection.arrival_eci_vector is not None
        assert injection.hyperbolic_excess_velocity >= 0
        assert len(injection.notes) > 0
    
    def test_multi_lane_comparison(self, calculator):
        """Test comparison across different velocity lanes."""
        lanes = {
            'slow': 1500,
            'standard': 5000,
            'fast': 12000
        }
        
        results = {}
        for lane_name, v_rel in lanes.items():
            results[lane_name] = calculator.calculate_injection_vector(
                target_altitude_km=500,
                target_relative_velocity_ms=v_rel,
                launch_from_lunar_surface=True
            )
        
        # Higher lanes should require more energy
        assert results['fast'].departure_dv > results['standard'].departure_dv
        assert results['standard'].departure_dv > results['slow'].departure_dv
        
        # All should be within reasonable bounds
        for name, result in results.items():
            assert result.departure_dv < 10000, \
                f"{name} lane DV {result.departure_dv} too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

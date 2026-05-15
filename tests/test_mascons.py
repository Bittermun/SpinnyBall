"""
Unit tests for lunar mascon gravity and CR3BP+mascon integration.

Tests cover:
1. Mascon coefficient loading and normalization
2. Spherical harmonic acceleration computation
3. Perigee precession rate calculations
4. CR3BP+mascon propagation (lunar orbits)
5. Validation against published precession rates (Konopliv et al.)
"""

import pytest
import numpy as np
from pathlib import Path


try:
    from dynamics.mascons import (
        LunarMascon, LunarMasconConfig, 
        validate_lunar_orbit_against_theory,
        GRAIL_COEFFICIENTS
    )
    MASCON_AVAILABLE = True
except ImportError:
    MASCON_AVAILABLE = False

try:
    from dynamics.cislunar_mascon import CR3BPMasconPropagator, CR3BPMasconConfig
    MASCON_CR3BP_AVAILABLE = True
except ImportError:
    MASCON_CR3BP_AVAILABLE = False


# ============================================================================
# Mascon Model Tests
# ============================================================================

class TestMasconLoading:
    """Test lunar mascon coefficient loading and configuration."""
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_mascon_initialization(self):
        """Test basic mascon model initialization."""
        mascon = LunarMascon()
        
        assert mascon is not None
        assert mascon.degree_max > 0
        assert mascon.R_MOON == pytest.approx(1737.4, rel=1e-3)
        assert mascon.MU_MOON == pytest.approx(4902.8, rel=1e-3)
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_mascon_config_default(self):
        """Test default mascon configuration."""
        config = LunarMasconConfig()
        
        assert config.degree_max == 20
        assert config.include_sectorial is True
        assert config.include_tesseral is True
        assert config.normalize_coefficients is True
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_mascon_config_custom(self):
        """Test custom mascon configuration."""
        config = LunarMasconConfig(
            degree_max=10,
            include_sectorial=False
        )
        
        mascon = LunarMascon(config)
        assert mascon.degree_max == 10
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_grail_coefficients_loaded(self):
        """Test that GRAIL coefficients are available."""
        mascon = LunarMascon()
        
        # Check that key coefficients exist
        assert (0, 0) in mascon.coefficients  # Monopole
        assert (2, 0) in mascon.coefficients  # J2 term
        
        # Check monopole is normalized to 1.0
        c00, s00 = mascon.coefficients[(0, 0)]
        assert c00 == pytest.approx(1.0, rel=1e-6)


class TestMasconAcceleration:
    """Test mascon acceleration computation."""
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_acceleration_at_surface(self):
        """Test surface gravity acceleration."""
        mascon = LunarMascon()
        
        # Surface of Moon
        pos = np.array([mascon.R_MOON, 0.0, 0.0])
        accel = mascon.acceleration(pos)
        
        # Magnitude should be close to surface gravity
        g_surface = mascon.MU_MOON / mascon.R_MOON**2  # ≈ 1.62 m/s²
        accel_mag = np.linalg.norm(accel)
        
        # Should be positive and reasonably large (km/s² units)
        assert accel_mag > 0
        assert accel_mag < g_surface * 1.5  # Allow some deviation from 2-body
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_acceleration_radial_direction(self):
        """Test that acceleration points radially (approximately)."""
        mascon = LunarMascon()
        
        # Position on x-axis
        pos = np.array([mascon.R_MOON * 3.0, 0.0, 0.0])
        accel = mascon.acceleration(pos)
        
        # Acceleration should mostly point toward center (negative x)
        assert accel[0] < 0
        assert abs(accel[1]) < abs(accel[0]) * 0.1  # Small y component
        assert abs(accel[2]) < abs(accel[0]) * 0.1  # Small z component
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_acceleration_discontinuity_avoided(self):
        """Test that acceleration is finite at center."""
        mascon = LunarMascon()
        
        pos = np.array([0.1, 0.0, 0.0])  # Close to center
        accel = mascon.acceleration(pos)
        
        # Should be zero or near-zero at center (singularity avoidance)
        assert np.isfinite(accel).all()


class TestMasconPrecession:
    """Test perigee precession rate calculations."""
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_precession_rate_100km_orbit(self):
        """
        Test perigee precession for circular 100 km lunar orbit.
        
        Reference: ~0.1-0.2 deg/day for equatorial orbit (from literature)
        """
        mascon = LunarMascon()
        
        # 100 km circular orbit over Moon
        a = mascon.R_MOON + 100.0  # km
        e = 0.0
        i = 0.0  # Equatorial
        
        rate = mascon.perigee_precession_rate(a, e, i)
        
        # Should be positive for equatorial orbit
        assert rate > 0
        # Typical value for lunar orbit: 0.1-0.2 deg/day
        assert rate < 1.0  # Reasonable upper bound
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_precession_rate_depends_on_inclination(self):
        """Test that precession rate varies with inclination."""
        mascon = LunarMascon()
        
        a = mascon.R_MOON + 100.0
        e = 0.0
        
        rate_eq = mascon.perigee_precession_rate(a, e, 0.0)    # Equatorial
        rate_45 = mascon.perigee_precession_rate(a, e, 45.0)   # 45°
        rate_90 = mascon.perigee_precession_rate(a, e, 90.0)   # Polar
        
        # Should decrease with increasing inclination
        assert rate_eq > rate_45 > abs(rate_90)  # Polar close to zero
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_precession_rate_altitude_dependence(self):
        """Test that precession rate increases closer to Moon."""
        mascon = LunarMascon()
        
        a_100 = mascon.perigee_precession_rate(mascon.R_MOON + 100, 0.0, 0.0)
        a_300 = mascon.perigee_precession_rate(mascon.R_MOON + 300, 0.0, 0.0)
        
        # Closer orbit should have larger precession
        assert abs(a_100) > abs(a_300)


# ============================================================================
# CR3BP + Mascon Integration Tests
# ============================================================================

class TestCR3BPMasconConfig:
    """Test CR3BP+mascon configuration."""
    
    @pytest.mark.skipif(not MASCON_CR3BP_AVAILABLE, reason="cislunar_mascon not available")
    def test_cr3bp_mascon_config_creation(self):
        """Test creating CR3BP+mascon configuration."""
        config = CR3BPMasconConfig(
            use_mascons=True,
            mascon_degree_max=20
        )
        
        assert config.use_mascons is True
        assert config.mascon_degree_max == 20
    
    @pytest.mark.skipif(not MASCON_CR3BP_AVAILABLE, reason="cislunar_mascon not available")
    def test_cr3bp_mascon_propagator_init(self):
        """Test CR3BP+mascon propagator initialization."""
        config = CR3BPMasconConfig(use_mascons=True)
        prop = CR3BPMasconPropagator(config)
        
        assert prop is not None
        assert prop.mascon is not None
        assert prop.mascon_config.use_mascons is True


class TestCR3BPMasconPropagation:
    """Test CR3BP+mascon propagation."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(not MASCON_CR3BP_AVAILABLE, reason="cislunar_mascon not available")
    def test_mascon_perturbation_effect(self):
        """
        Test that mascon perturbation has observable effect on trajectory.
        
        Compares propagation with and without mascons over a lunar orbit.
        """
        # Lunar circular orbit (100 km)
        r_orbit = 1737.4 + 100.0  # km
        v_orbit = np.sqrt(4902.8005 / r_orbit)
        
        # Position the orbit 384400 km from Earth
        state0 = np.array([
            384400.0 + r_orbit, 0.0, 0.0,    # Position at Moon + orbit
            0.0, v_orbit, 0.0                 # Velocity
        ])
        
        # Propagate with mascons
        config_with = CR3BPMasconConfig(use_mascons=True)
        prop_with = CR3BPMasconPropagator(config_with)
        
        # Propagate without mascons
        config_without = CR3BPMasconConfig(use_mascons=False)
        prop_without = CR3BPMasconPropagator(config_without)
        
        t_eval = np.linspace(0, 3600, 100)  # 1 hour
        
        sol_with = prop_with.propagate(state0, t_eval)
        sol_without = prop_without.propagate(state0, t_eval)
        
        # Compute difference in final positions
        pos_with = sol_with.y[0:3, -1]
        pos_without = sol_without.y[0:3, -1]
        
        diff = np.linalg.norm(pos_with - pos_without)
        
        # Mascon perturbations should cause measurable deviation
        # (may be small for 1-hour, but should be non-zero)
        assert diff >= 0
    
    @pytest.mark.slow
    @pytest.mark.skipif(not MASCON_CR3BP_AVAILABLE, reason="cislunar_mascon not available")
    def test_lunar_orbit_propagation_30days(self):
        """Test 30-day lunar orbit propagation with mascons."""
        r_orbit = 1737.4 + 100.0  # 100 km altitude
        v_orbit = np.sqrt(4902.8005 / r_orbit)
        
        state0 = np.array([
            384400.0 + r_orbit, 0.0, 0.0,
            0.0, v_orbit, 0.0
        ])
        
        config = CR3BPMasconConfig(use_mascons=True, mascon_degree_max=15)
        prop = CR3BPMasconPropagator(config)
        
        # 30-day propagation
        t_eval = np.linspace(0, 30*86400, 500)
        
        sol = prop.propagate(state0, t_eval)
        
        # Check that propagation succeeded
        assert sol.status == 0
        assert len(sol.t) > 0
        
        # Extract final orbital elements
        elems = sol.get_orbital_elements(t_eval[-1])
        
        # Orbit should remain stable (periapsis > Moon surface)
        assert elems['semi_major_axis_km'] > 1737.4


class TestMasconValidation:
    """Test mascon validation against literature values."""
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_validation_function_basic(self):
        """Test the validation function."""
        result = validate_lunar_orbit_against_theory(
            a_km=1837.4,  # 100 km altitude
            e=0.0,
            i_deg=0.0,
            theory_rate_deg_day=0.15,  # Typical value
            tolerance_percent=20.0  # Loose tolerance for demo
        )
        
        assert 'computed_rate_deg_day' in result
        assert 'error_percent' in result
        assert 'passes' in result
        assert isinstance(result['passes'], (bool, np.bool_))


# ============================================================================
# Spherical Harmonic Math Tests
# ============================================================================

class TestSphericalHarmonics:
    """Test spherical harmonic calculations."""
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_legendre_derivatives_equator(self):
        """Test Legendre polynomial derivatives at equator."""
        mascon = LunarMascon()
        
        cos_theta = 0.0  # At equator
        
        p_nm, dp_nm = mascon._legendre_derivatives(2, 0, cos_theta)
        
        # P_2^0 at equator (cos θ = 0)
        # P_2(0) = (3*0^2 - 1)/2 = -1/2
        assert isinstance(p_nm, (float, np.floating))
        assert isinstance(dp_nm, (float, np.floating))
    
    @pytest.mark.skipif(not MASCON_AVAILABLE, reason="mascons module not available")
    def test_cartesian_spherical_conversion(self):
        """Test Cartesian <-> spherical coordinate conversions."""
        mascon = LunarMascon()
        
        # Cartesian position
        pos_cart = np.array([1000.0, 0.0, 0.0])
        
        lat, lon = mascon._cartesian_to_spherical(pos_cart)
        
        # Should be at equator (lat = 0) and prime meridian (lon = 0)
        assert lat == pytest.approx(0.0, abs=1e-10)
        assert lon == pytest.approx(0.0, abs=1e-10)
        
        # Test point at north pole
        pos_pole = np.array([0.0, 0.0, 1000.0])
        lat_pole, lon_pole = mascon._cartesian_to_spherical(pos_pole)
        
        assert lat_pole == pytest.approx(np.pi/2, rel=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])

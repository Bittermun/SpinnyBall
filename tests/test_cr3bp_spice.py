"""
Unit tests for CR3BP and SPICE modules.

Tests cover:
1. SPICE wrapper kernel loading and body queries
2. CR3BP propagation in rotating and inertial frames
3. Lagrange point computations
4. Integration with cislunar scenarios
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime

# Conditional imports
try:
    from third_party.spice import SPICEWrapper, SPICEYPY_AVAILABLE, SPICEState
except ImportError:
    SPICEWrapper = None
    SPICEYPY_AVAILABLE = False
    SPICEState = None

try:
    from dynamics.cislunar import CR3BPPropagator, CR3BPConfig, CR3BPSolution
except ImportError:
    CR3BPPropagator = None
    CR3BPConfig = None
    CR3BPSolution = None


# ============================================================================
# SPICE Wrapper Tests
# ============================================================================

class TestSPICEWrapper:
    """Test suite for SPICEWrapper class."""
    
    @pytest.mark.skipif(not SPICEYPY_AVAILABLE, reason="spiceypy not installed")
    def test_spice_import(self):
        """Smoke test: SPICE wrapper can be imported."""
        assert SPICEWrapper is not None
    
    @pytest.mark.skipif(not SPICEYPY_AVAILABLE, reason="spiceypy not installed")
    def test_spice_initialization_no_kernels(self):
        """Test SPICE wrapper initialization without kernel loading."""
        wrapper = SPICEWrapper(
            kernel_dir=Path('./nonexistent'),
            auto_load_kernels=False,
            verbose=False
        )
        assert wrapper is not None
        assert wrapper._kernels_loaded is False
    
    @pytest.mark.skipif(not SPICEYPY_AVAILABLE, reason="spiceypy not installed")
    def test_jd_to_utc_conversion(self):
        """Test Julian Date to UTC string conversion."""
        # JD 2451545.0 = 2000-01-01 12:00:00 UTC
        jd = 2451545.0
        utc_str = SPICEWrapper._jd_to_utc_string(jd)
        
        # Should contain valid datetime info
        assert isinstance(utc_str, str)
        assert len(utc_str) > 0
        
        # Test a more recent date
        jd_recent = 2460000.0  # Around 2023
        utc_recent = SPICEWrapper._jd_to_utc_string(jd_recent)
        assert isinstance(utc_recent, str)
    
    @pytest.mark.skipif(not SPICEYPY_AVAILABLE, reason="spiceypy not installed")
    def test_spice_body_ids(self):
        """Test SPICE body ID mapping."""
        assert SPICEWrapper.BODY_IDS['EARTH'] == 399
        assert SPICEWrapper.BODY_IDS['MOON'] == 301
        assert SPICEWrapper.BODY_IDS['SUN'] == 10


# ============================================================================
# CR3BP Propagator Tests
# ============================================================================

class TestCR3BPConfig:
    """Test CR3BP configuration."""
    
    def test_default_config(self):
        """Test default CR3BP config values."""
        config = CR3BPConfig()
        
        assert config.mu == pytest.approx(0.01215, rel=1e-4)
        assert config.rotating_frame is False
        assert config.include_srp is False
        assert config.rtol == 1e-9
        assert config.atol == 1e-12
    
    def test_custom_config(self):
        """Test custom CR3BP config."""
        config = CR3BPConfig(
            mu=0.012,
            rotating_frame=True,
            include_srp=True
        )
        
        assert config.mu == 0.012
        assert config.rotating_frame is True
        assert config.include_srp is True


class TestCR3BPPropagator:
    """Test CR3BP propagator."""
    
    def test_propagator_initialization(self):
        """Test CR3BP propagator can be initialized."""
        config = CR3BPConfig(rotating_frame=False)
        prop = CR3BPPropagator(config)
        
        assert prop is not None
        assert prop.mu == pytest.approx(0.01215, rel=1e-4)
    
    def test_propagator_default_config(self):
        """Test CR3BP propagator with default config."""
        prop = CR3BPPropagator()
        
        assert prop.config is not None
        assert prop.mu > 0
    
    def test_earth_moon_parameters(self):
        """Test Earth-Moon system parameters."""
        prop = CR3BPPropagator()
        
        assert prop.MU_EARTH == pytest.approx(398600.4418, rel=1e-6)
        assert prop.MU_MOON == pytest.approx(4902.8005, rel=1e-4)
        assert prop.EARTH_MOON_DISTANCE == pytest.approx(384400.0, rel=1e-3)
    
    def test_accelerations_rotating_frame_at_origin(self):
        """Test CR3BP accelerations in rotating frame at origin."""
        config = CR3BPConfig(rotating_frame=True, mu=0.01215)
        prop = CR3BPPropagator(config)
        
        # State at origin (minimal velocity)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        accel = prop._accelerations(state, time_abs=0.0)
        
        # Should return 6-element array [vx, vy, vz, ax, ay, az]
        assert accel.shape == (6,)
        assert np.isfinite(accel).all()
    
    def test_accelerations_inertial_frame(self):
        """Test CR3BP accelerations in inertial frame."""
        config = CR3BPConfig(rotating_frame=False)
        prop = CR3BPPropagator(config)
        
        # State: spacecraft at (400 km altitude, 0, 0) with orbital velocity
        r_earth = 6371.0 + 400.0  # km (400 km altitude)
        v_orbit = np.sqrt(398600.4418 / r_earth)  # Orbital velocity
        
        state = np.array([r_earth, 0.0, 0.0, 0.0, v_orbit, 0.0])
        
        accel = prop._accelerations(state, time_abs=0.0)
        
        assert accel.shape == (6,)
        assert np.isfinite(accel).all()
        
        # Velocity components should match state
        assert accel[0] == pytest.approx(0.0, abs=1e-10)
        assert accel[1] == pytest.approx(v_orbit, rel=1e-6)
        assert accel[2] == pytest.approx(0.0, abs=1e-10)
    
    @pytest.mark.slow
    def test_propagation_short_leo(self):
        """
        Test short LEO propagation (1 hour).
        
        Validates that propagation runs without errors and produces reasonable state.
        """
        config = CR3BPConfig(rotating_frame=False, rtol=1e-8, atol=1e-11)
        prop = CR3BPPropagator(config)
        
        # Initial LEO state (400 km circular orbit)
        r_earth = 6371.0 + 400.0
        v_orbit = np.sqrt(398600.4418 / r_earth)
        state0 = np.array([r_earth, 0.0, 0.0, 0.0, v_orbit, 0.0])
        
        # Propagate for 1 hour
        t_eval = np.linspace(0, 3600, 100)
        
        sol = prop.propagate(state0, t_eval)
        
        assert sol is not None
        assert sol.status == 0  # Success
        assert len(sol.t) > 0
        
        # Check that position remains in reasonable range
        pos_final = sol.y[0:3, -1]
        r_final = np.linalg.norm(pos_final)
        
        # Distance from Earth should be close to initial orbital radius
        assert r_final == pytest.approx(r_earth, rel=0.1)  # Allow 10% variation
    
    def test_lagrange_point_locations(self):
        """Test Lagrange point computation."""
        prop = CR3BPPropagator()
        
        for point in range(1, 6):
            l_point = prop.lagrange_point(point)
            
            assert isinstance(l_point, np.ndarray)
            assert l_point.shape == (3,)
            assert np.isfinite(l_point).all()
        
        # L4 and L5 should be symmetric about x-axis
        l4 = prop.lagrange_point(4)
        l5 = prop.lagrange_point(5)
        
        assert l4[0] == pytest.approx(l5[0], rel=1e-10)
        assert l4[1] == pytest.approx(-l5[1], rel=1e-10)
        assert l4[2] == pytest.approx(l5[2], rel=1e-10)
    
    def test_invalid_lagrange_point(self):
        """Test error handling for invalid Lagrange point."""
        prop = CR3BPPropagator()
        
        with pytest.raises(ValueError):
            prop.lagrange_point(6)
        
        with pytest.raises(ValueError):
            prop.lagrange_point(0)


class TestCR3BPSolution:
    """Test CR3BP solution object."""
    
    @pytest.mark.slow
    def test_solution_position_extraction(self):
        """Test extracting position from solution."""
        config = CR3BPConfig(rotating_frame=False)
        prop = CR3BPPropagator(config)
        
        # Initial state
        r_earth = 6371.0 + 400.0
        v_orbit = np.sqrt(398600.4418 / r_earth)
        state0 = np.array([r_earth, 0.0, 0.0, 0.0, v_orbit, 0.0])
        
        # Propagate
        t_eval = np.linspace(0, 3600, 50)
        sol = prop.propagate(state0, t_eval)
        
        # Extract position at first time point
        pos = sol.get_position(t_eval[0])
        
        assert isinstance(pos, np.ndarray)
        if pos.ndim == 1:
            assert pos.shape == (3,)
        else:
            assert pos.shape[0] > 0
    
    @pytest.mark.slow
    def test_solution_velocity_extraction(self):
        """Test extracting velocity from solution."""
        config = CR3BPConfig(rotating_frame=False)
        prop = CR3BPPropagator(config)
        
        r_earth = 6371.0 + 400.0
        v_orbit = np.sqrt(398600.4418 / r_earth)
        state0 = np.array([r_earth, 0.0, 0.0, 0.0, v_orbit, 0.0])
        
        t_eval = np.linspace(0, 3600, 50)
        sol = prop.propagate(state0, t_eval)
        
        vel = sol.get_velocity(t_eval[0])
        
        assert isinstance(vel, np.ndarray)
        if vel.ndim == 1:
            assert vel.shape == (3,)


# ============================================================================
# Integration Tests
# ============================================================================

class TestCR3BPIntegration:
    """Integration tests for CR3BP with other modules."""
    
    @pytest.mark.slow
    def test_cr3bp_vs_earth_only(self):
        """
        Compare CR3BP to Earth-only propagation for LEO (deviation should be small).
        
        At LEO altitudes, CR3BP perturbation from Moon is small,
        so trajectory should differ slightly from pure Earth gravity.
        """
        config = CR3BPConfig(rotating_frame=False)
        prop = CR3BPPropagator(config)
        
        # LEO circular orbit
        r_earth = 6371.0 + 400.0
        v_orbit = np.sqrt(398600.4418 / r_earth)
        state0 = np.array([r_earth, 0.0, 0.0, 0.0, v_orbit, 0.0])
        
        # Short propagation (1 hour)
        t_eval = np.linspace(0, 3600, 100)
        sol = prop.propagate(state0, t_eval)
        
        # Ending position
        pos_end = sol.get_position(t_eval[-1])
        r_end = np.linalg.norm(pos_end)
        
        # Should remain in LEO (orbital radius ±10%)
        assert r_end > r_earth * 0.9
        assert r_end < r_earth * 1.1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])

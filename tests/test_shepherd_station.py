"""
Unit tests for shepherd station module.
"""

import numpy as np
import pytest

from dynamics.shepherd_station import (
    StationType,
    QuadrupoleLens,
    TrimCoil,
    ShepherdConfig,
    ShepherdStation,
    create_passive_shepherd,
    create_active_shepherd,
    create_anchor_station,
)


class TestQuadrupoleLens:
    """Tests for QuadrupoleLens dataclass."""
    
    def test_default_initialization(self):
        """Test default lens values."""
        lens = QuadrupoleLens()
        assert lens.length == 0.5
        assert lens.bore_radius == 0.1
        assert lens.gradient == 10.0
        assert lens.max_field == 1.5
    
    def test_lens_stiffness(self):
        """Test lens stiffness calculation."""
        lens = QuadrupoleLens(gradient=10.0, bore_radius=0.1)
        k = lens.lens_stiffness
        
        # Stiffness should be positive
        assert k > 0
    
    def test_lens_stiffness_zero_bore(self):
        """Test lens stiffness with zero bore radius."""
        lens = QuadrupoleLens(bore_radius=0.0)
        k = lens.lens_stiffness
        
        assert k == 0.0


class TestTrimCoil:
    """Tests for TrimCoil dataclass."""
    
    def test_default_initialization(self):
        """Test default coil values."""
        coil = TrimCoil()
        assert coil.n_turns == 100
        assert coil.radius == 0.15
        assert coil.max_current == 10.0
    
    def test_compute_magnetic_field(self):
        """Test magnetic field calculation."""
        coil = TrimCoil(n_turns=100, radius=0.15)
        B = coil.compute_magnetic_field(current=1.0, axial_position=0.0)
        
        # Field at center should be positive for positive current
        assert B > 0
    
    def test_compute_magnetic_field_zero_current(self):
        """Test that zero current gives zero field."""
        coil = TrimCoil()
        B = coil.compute_magnetic_field(current=0.0, axial_position=0.0)
        
        assert B == 0.0
    
    def test_compute_force_on_dipole(self):
        """Test force on dipole calculation."""
        coil = TrimCoil()
        F = coil.compute_force_on_dipole(
            dipole_moment=580.0,
            current=1.0,
            axial_velocity=1000.0
        )
        
        # Force should be finite
        assert np.isfinite(F)
    
    def test_power_consumption(self):
        """Test power consumption calculation."""
        coil = TrimCoil(resistance=2.0, max_current=5.0)
        P = coil.power_consumption
        
        expected = 2.0 * 5.0**2
        assert np.isclose(P, expected)


class TestShepherdConfig:
    """Tests for ShepherdConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default config values."""
        config = ShepherdConfig()
        assert config.station_type == StationType.PASSIVE
        assert config.capture_radius == 10.0
        assert np.allclose(config.position, np.zeros(3))
    
    def test_custom_position(self):
        """Test custom position."""
        pos = np.array([1000.0, 0.0, 0.0])
        config = ShepherdConfig(position=pos)
        
        assert np.allclose(config.position, pos)
    
    def test_active_station_has_trim_coil(self):
        """Test that active station gets trim coil."""
        config = ShepherdConfig(station_type=StationType.ACTIVE)
        
        assert config.trim_coil is not None
    
    def test_anchor_station_has_trim_coil(self):
        """Test that anchor station gets trim coil."""
        config = ShepherdConfig(station_type=StationType.ANCHOR)
        
        assert config.trim_coil is not None


class TestShepherdStation:
    """Tests for ShepherdStation class."""
    
    def test_initialization(self):
        """Test station initialization."""
        config = ShepherdConfig()
        station = ShepherdStation(config, station_id=1)
        
        assert station.station_id == 1
        assert station.config == config
    
    def test_position_property(self):
        """Test position property."""
        pos = np.array([1000.0, 0.0, 0.0])
        config = ShepherdConfig(position=pos)
        station = ShepherdStation(config)
        
        assert np.allclose(station.position, pos)
    
    def test_lens_stiffness_property(self):
        """Test lens stiffness property."""
        lens = QuadrupoleLens(gradient=20.0)
        config = ShepherdConfig(quadrupole=lens)
        station = ShepherdStation(config)
        
        assert station.lens_stiffness > 0
    
    def test_compute_passive_force_centered(self):
        """Test passive force when ball is centered."""
        config = ShepherdConfig()
        station = ShepherdStation(config)
        
        # Ball at station center
        pos = np.zeros(3)
        vel = np.array([1000.0, 0.0, 0.0])
        
        F = station.compute_passive_force(pos, vel)
        
        # At center, force should be zero
        assert np.allclose(F, 0.0)
    
    def test_compute_passive_force_off_center(self):
        """Test passive force when ball is off-center."""
        config = ShepherdConfig(capture_radius=10.0)
        station = ShepherdStation(config)
        
        # Ball off-center in x
        pos = np.array([1.0, 0.0, 0.0])
        vel = np.array([1000.0, 0.0, 0.0])
        
        F = station.compute_passive_force(pos, vel)
        
        # Force should be restoring (negative x)
        assert F[0] < 0
        # No axial force from quadrupole
        assert np.isclose(F[2], 0.0)
    
    def test_compute_passive_force_outside_capture(self):
        """Test passive force when ball is outside capture radius."""
        config = ShepherdConfig(capture_radius=1.0)
        station = ShepherdStation(config)
        
        # Ball outside capture radius
        pos = np.array([10.0, 0.0, 0.0])
        vel = np.array([1000.0, 0.0, 0.0])
        
        F = station.compute_passive_force(pos, vel)
        
        # No force outside capture radius
        assert np.allclose(F, 0.0)
    
    def test_compute_trim_force_no_coil(self):
        """Test trim force when no trim coil."""
        config = ShepherdConfig(station_type=StationType.PASSIVE)
        station = ShepherdStation(config)
        
        pos = np.zeros(3)
        vel = np.array([1000.0, 0.0, 0.0])
        
        F = station.compute_trim_force(pos, vel)
        
        # No trim coil, so no force
        assert np.allclose(F, 0.0)
    
    def test_compute_trim_force_with_coil(self):
        """Test trim force with active coil."""
        config = ShepherdConfig(station_type=StationType.ACTIVE)
        station = ShepherdStation(config)
        station.set_trim_current(1.0)
        
        pos = np.zeros(3)
        vel = np.array([1000.0, 0.0, 0.0])
        
        F = station.compute_trim_force(pos, vel)
        
        # Should have some axial force
        assert np.isfinite(F[2])
    
    def test_set_trim_current(self):
        """Test setting trim current."""
        config = ShepherdConfig(station_type=StationType.ACTIVE)
        station = ShepherdStation(config)
        
        station.set_trim_current(5.0)
        
        assert station._trim_current == 5.0
    
    def test_set_trim_current_clamping(self):
        """Test that trim current is clamped to max."""
        config = ShepherdConfig(station_type=StationType.ACTIVE)
        station = ShepherdStation(config)
        
        max_current = config.trim_coil.max_current
        station.set_trim_current(max_current + 10.0)
        
        # Should be clamped to max
        assert station._trim_current == max_current
    
    def test_compute_anchor_force(self):
        """Test anchor force calculation."""
        config = ShepherdConfig(station_type=StationType.ANCHOR)
        station = ShepherdStation(config)
        
        T = 1000.0  # N
        theta = np.pi / 6.0  # 30 degrees
        
        F = station.compute_anchor_force(T, theta)
        
        expected = T * np.sin(theta)
        assert np.isclose(F, expected)
    
    def test_anchor_force_non_anchor_station(self):
        """Test that non-anchor stations return zero anchor force."""
        config = ShepherdConfig(station_type=StationType.PASSIVE)
        station = ShepherdStation(config)
        
        F = station.compute_anchor_force(1000.0, np.pi / 6.0)
        
        assert F == 0.0
    
    def test_anchor_force_clamping(self):
        """Test that anchor force is clamped to max deflection."""
        config = ShepherdConfig(
            station_type=StationType.ANCHOR,
            max_deflection_angle=np.pi / 6.0  # 30 degrees
        )
        station = ShepherdStation(config)
        
        T = 1000.0
        theta = np.pi / 2.0  # 90 degrees (should be clamped)
        
        F = station.compute_anchor_force(T, theta)
        
        # Should be clamped to 30 degrees
        expected = T * np.sin(np.pi / 6.0)
        assert np.isclose(F, expected)
    
    def test_can_capture_inside(self):
        """Test capture detection inside radius."""
        config = ShepherdConfig(capture_radius=10.0)
        station = ShepherdStation(config)
        
        pos = np.array([5.0, 0.0, 0.0])
        
        assert station.can_capture(pos)
    
    def test_can_capture_outside(self):
        """Test capture detection outside radius."""
        config = ShepherdConfig(capture_radius=10.0)
        station = ShepherdStation(config)
        
        pos = np.array([15.0, 0.0, 0.0])
        
        assert not station.can_capture(pos)
    
    def test_get_status(self):
        """Test status report."""
        config = ShepherdConfig(station_type=StationType.ACTIVE)
        station = ShepherdStation(config, station_id=5)
        
        status = station.get_status()
        
        assert status['station_id'] == 5
        assert status['station_type'] == 'active'
        assert 'position' in status
        assert 'lens_stiffness' in status


class TestCreatePassiveShepherd:
    """Tests for create_passive_shepherd function."""
    
    def test_creation(self):
        """Test passive shepherd creation."""
        pos = np.array([1000.0, 0.0, 0.0])
        station = create_passive_shepherd(pos, station_id=1)
        
        assert station.station_id == 1
        assert station.config.station_type == StationType.PASSIVE
        assert np.allclose(station.position, pos)


class TestCreateActiveShepherd:
    """Tests for create_active_shepherd function."""
    
    def test_creation(self):
        """Test active shepherd creation."""
        pos = np.array([1000.0, 0.0, 0.0])
        station = create_active_shepherd(pos, station_id=2)
        
        assert station.station_id == 2
        assert station.config.station_type == StationType.ACTIVE
        assert station.config.trim_coil is not None


class TestCreateAnchorStation:
    """Tests for create_anchor_station function."""
    
    def test_creation(self):
        """Test anchor station creation."""
        pos = np.array([0.0, 0.0, 0.0])
        station = create_anchor_station(pos, station_id=3, max_deflection=45.0)
        
        assert station.station_id == 3
        assert station.config.station_type == StationType.ANCHOR
        assert station.config.max_deflection_angle == np.pi / 4.0  # 45 degrees

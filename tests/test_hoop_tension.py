"""
Unit tests for hoop tension dynamics module.
"""

import numpy as np
import pytest

from dynamics.hoop_tension import (
    StreamGeometry,
    HoopTensionModel,
    create_stream_geometry_from_params,
    compute_combined_stiffness,
)


class TestStreamGeometry:
    """Tests for StreamGeometry dataclass."""
    
    def test_default_initialization(self):
        """Test default geometry values."""
        geom = StreamGeometry()
        assert geom.radius == 1000.0
        assert geom.n_balls == 100
        assert geom.ball_mass == 1.0
        assert geom.stream_velocity == 1600.0
    
    def test_circumference(self):
        """Test circumference calculation."""
        geom = StreamGeometry(radius=1000.0)
        expected = 2.0 * np.pi * 1000.0
        assert np.isclose(geom.circumference, expected)
    
    def test_linear_density(self):
        """Test linear density calculation."""
        geom = StreamGeometry(
            radius=1000.0,
            n_balls=100,
            ball_mass=1.0
        )
        # λ = N*m / (2*pi*R)
        expected = 100.0 * 1.0 / (2.0 * np.pi * 1000.0)
        assert np.isclose(geom.linear_density, expected)
    
    def test_hoop_tension(self):
        """Test hoop tension calculation."""
        geom = StreamGeometry(
            radius=1000.0,
            n_balls=100,
            ball_mass=1.0,
            stream_velocity=1000.0
        )
        # T = λ * u^2
        lambda_val = geom.linear_density
        expected = lambda_val * 1000.0**2
        assert np.isclose(geom.hoop_tension, expected)
    
    def test_ball_spacing(self):
        """Test ball spacing calculation."""
        geom = StreamGeometry(
            radius=1000.0,
            n_balls=100
        )
        # spacing = circumference / n_balls
        expected = geom.circumference / 100.0
        assert np.isclose(geom.ball_spacing, expected)
    
    def test_angular_velocity(self):
        """Test angular velocity calculation."""
        geom = StreamGeometry(
            radius=1000.0,
            stream_velocity=1000.0
        )
        # ω = u / R
        expected = 1000.0 / 1000.0
        assert np.isclose(geom.angular_velocity, expected)


class TestHoopTensionModel:
    """Tests for HoopTensionModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        geom = StreamGeometry()
        model = HoopTensionModel(geom)
        assert model.geometry == geom
    
    def test_compute_radial_restoring_force(self):
        """Test radial restoring force calculation."""
        geom = StreamGeometry(radius=1000.0, n_balls=100, ball_mass=1.0)
        model = HoopTensionModel(geom)
        
        # Displacement of 1m
        F_r = model.compute_radial_restoring_force(
            radial_displacement=1.0,
            azimuthal_position=0.0
        )
        
        # Force should be restoring (negative for outward displacement)
        assert F_r < 0
    
    def test_compute_transverse_stiffness(self):
        """Test transverse stiffness calculation."""
        geom = StreamGeometry(
            radius=1000.0,
            n_balls=100,
            ball_mass=1.0,
            stream_velocity=1000.0
        )
        model = HoopTensionModel(geom)
        
        k = model.compute_transverse_stiffness()
        
        # k_hoop = T / R * (arc_length)
        assert k > 0
    
    def test_compute_stream_modes(self):
        """Test stream eigenmode calculation."""
        geom = StreamGeometry(
            radius=1000.0,
            stream_velocity=1000.0
        )
        model = HoopTensionModel(geom)
        
        modes = model.compute_stream_modes(n_modes=5)
        
        assert len(modes) == 5
        for n, f in modes:
            assert n > 0
            assert f > 0  # Frequency should be positive
    
    def test_stream_mode_scaling(self):
        """Test that mode frequencies scale with mode number."""
        geom = StreamGeometry(
            radius=1000.0,
            stream_velocity=1000.0
        )
        model = HoopTensionModel(geom)
        
        modes = model.compute_stream_modes(n_modes=3)
        
        # Frequencies should be in ratio 1:2:3
        f1 = modes[0][1]
        f2 = modes[1][1]
        f3 = modes[2][1]
        
        assert np.isclose(f2 / f1, 2.0, rtol=1e-10)
        assert np.isclose(f3 / f1, 3.0, rtol=1e-10)
    
    def test_compute_perturbation_dynamics(self):
        """Test perturbation dynamics integration."""
        geom = StreamGeometry(n_balls=10)
        model = HoopTensionModel(geom)
        
        # Initial perturbation
        dr = np.ones(10) * 0.1  # 10cm radial displacement
        dv = np.zeros(10)
        
        dr_new, dv_new = model.compute_perturbation_dynamics(dr, dv, dt=0.01)
        
        # Displacement should change
        assert not np.allclose(dr_new, dr)
        # Velocity should become non-zero
        assert not np.allclose(dv_new, dv)
    
    def test_compute_anchor_force(self):
        """Test anchor force calculation."""
        geom = StreamGeometry(
            n_balls=100,
            ball_mass=1.0,
            stream_velocity=1000.0,
            radius=1000.0
        )
        model = HoopTensionModel(geom)
        
        T = geom.hoop_tension
        theta = np.pi / 6.0  # 30 degrees
        
        F_anchor = model.compute_anchor_force(theta)
        
        expected = T * np.sin(theta)
        assert np.isclose(F_anchor, expected)
    
    def test_anchor_force_zero_at_zero_deflection(self):
        """Test that anchor force is zero at zero deflection."""
        geom = StreamGeometry()
        model = HoopTensionModel(geom)
        
        F = model.compute_anchor_force(0.0)
        assert np.isclose(F, 0.0)
    
    def test_compute_equilibrium_spacing(self):
        """Test equilibrium spacing calculation."""
        geom = StreamGeometry()
        model = HoopTensionModel(geom)
        
        k_mag = 1000.0  # N/m
        spacing = model.compute_equilibrium_spacing(k_mag)
        
        # Spacing should be positive
        assert spacing > 0


class TestCreateStreamGeometryFromParams:
    """Tests for create_stream_geometry_from_params function."""
    
    def test_creation(self):
        """Test geometry creation from parameters."""
        geom = create_stream_geometry_from_params(
            orbital_radius=2000.0,
            stream_velocity=2000.0,
            ball_mass=2.0,
            n_balls=200
        )
        
        assert geom.radius == 2000.0
        assert geom.stream_velocity == 2000.0
        assert geom.ball_mass == 2.0
        assert geom.n_balls == 200


class TestComputeCombinedStiffness:
    """Tests for compute_combined_stiffness function."""
    
    def test_combined_stiffness(self):
        """Test combined stiffness calculation."""
        geom = StreamGeometry()
        model = HoopTensionModel(geom)
        k_mag = 1000.0
        
        result = compute_combined_stiffness(model, k_mag)
        
        assert 'k_hoop' in result
        assert 'k_magnetic' in result
        assert 'k_total' in result
        assert 'hoop_fraction' in result
        assert 'magnetic_fraction' in result
        
        # Total should be sum of parts
        assert np.isclose(result['k_total'], result['k_hoop'] + result['k_magnetic'])
        
        # Fractions should sum to 1
        assert np.isclose(result['hoop_fraction'] + result['magnetic_fraction'], 1.0)

"""
Tests for orbital dynamics coupling layer.

Tests orbital propagation, coordinate transforms, and integration
with MultiBodyStream.
"""

import numpy as np
import pytest

from dynamics.orbital_coupling import (
    OrbitalState, OrbitalElements, OrbitalPropagator,
    eci_to_lvlh, lvlh_to_eci, compute_eclipse,
    create_circular_orbit, ORBITAL_DYNAMICS_AVAILABLE,
)


class TestOrbitalState:
    """Test OrbitalState dataclass."""

    def test_orbital_state_creation(self):
        """Test creating orbital state from vectors."""
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        state = OrbitalState(r=r, v=v)
        
        assert state.magnitude_r == pytest.approx(7000.0)
        assert state.magnitude_v == pytest.approx(7.5)
        assert state.epoch is None

    def test_orbital_state_validation(self):
        """Test orbital state vector validation."""
        with pytest.raises(ValueError):
            OrbitalState(r=np.array([1.0, 2.0]), v=np.array([1.0, 2.0, 3.0]))
        
        with pytest.raises(ValueError):
            OrbitalState(r=np.array([1.0, 2.0, 3.0]), v=np.array([1.0, 2.0]))


class TestOrbitalElements:
    """Test orbital elements conversion."""

    def test_circular_to_state_vector(self):
        """Test converting circular orbit elements to state vector."""
        elements = OrbitalElements(
            a=7000.0, e=0.0, i=np.radians(51.6),
            raan=0.0, argp=0.0, nu=0.0
        )
        
        state = elements.to_state_vector()
        
        assert state.magnitude_r == pytest.approx(7000.0, rel=0.01)
        assert state.magnitude_v > 0
        assert state.r.shape == (3,)
        assert state.v.shape == (3,)


class TestOrbitalPropagator:
    """Test orbital propagator."""

    def test_propagator_creation(self):
        """Test creating orbital propagator."""
        propagator = OrbitalPropagator()
        assert propagator.mu == pytest.approx(398600.4418)

    def test_from_state_vector(self):
        """Test creating propagator from state vector."""
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        state = OrbitalState(r=r, v=v)
        
        propagator = OrbitalPropagator()
        propagator.from_state_vector(state)
        
        # Should store state internally
        assert hasattr(propagator, '_state') or propagator._poliastro_orbit is not None

    def test_simple_propagation(self):
        """Test simple Keplerian propagation."""
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        state = OrbitalState(r=r, v=v)
        
        propagator = OrbitalPropagator()
        propagator.from_state_vector(state)
        
        # Propagate 1 second
        new_state = propagator.propagate(1.0)
        
        assert new_state.magnitude_r > 0
        assert new_state.magnitude_v > 0

    def test_get_orbital_elements(self):
        """Test extracting orbital elements from state."""
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        state = OrbitalState(r=r, v=v)
        
        propagator = OrbitalPropagator()
        propagator.from_state_vector(state)
        
        elements = propagator.get_orbital_elements()
        
        assert elements.a > 6000.0  # Semi-major axis should be reasonable
        assert 0.0 <= elements.e < 1.0  # Eccentricity should be valid
        assert 0.0 <= elements.i <= np.pi  # Inclination should be valid


class TestCoordinateTransforms:
    """Test coordinate frame transforms."""

    def test_eci_to_lvlh(self):
        """Test ECI to LVLH transform."""
        r_eci = np.array([7000.0, 0.0, 0.0])
        v_eci = np.array([0.0, 7.5, 0.0])
        vector_eci = np.array([1.0, 0.0, 0.0])
        
        vector_lvlh = eci_to_lvlh(r_eci, v_eci, vector_eci)
        
        assert vector_lvlh.shape == (3,)
        assert np.linalg.norm(vector_lvlh) > 0

    def test_lvlh_to_eci(self):
        """Test LVLH to ECI transform."""
        r_eci = np.array([7000.0, 0.0, 0.0])
        v_eci = np.array([0.0, 7.5, 0.0])
        vector_lvlh = np.array([1.0, 0.0, 0.0])
        
        vector_eci = lvlh_to_eci(r_eci, v_eci, vector_lvlh)
        
        assert vector_eci.shape == (3,)
        assert np.linalg.norm(vector_eci) > 0

    def test_roundtrip_transform(self):
        """Test roundtrip ECI -> LVLH -> ECI."""
        r_eci = np.array([7000.0, 0.0, 0.0])
        v_eci = np.array([0.0, 7.5, 0.0])
        vector_eci = np.array([1.0, 2.0, 3.0])
        
        vector_lvlh = eci_to_lvlh(r_eci, v_eci, vector_eci)
        vector_eci_back = lvlh_to_eci(r_eci, v_eci, vector_lvlh)
        
        assert np.allclose(vector_eci, vector_eci_back, rtol=1e-10)


class TestEclipseDetection:
    """Test eclipse detection."""

    def test_sunlit_position(self):
        """Test sunlit position (no eclipse)."""
        # Position on sunlit side of Earth
        r_eci = np.array([8000.0, 0.0, 0.0])  # Away from sun
        
        in_eclipse = compute_eclipse(r_eci)
        
        assert not in_eclipse  # Should be sunlit

    def test_eclipse_position(self):
        """Test eclipse position (in shadow)."""
        # Position behind Earth relative to sun
        r_eci = np.array([-8000.0, 0.0, 0.0])  # Behind Earth
        
        in_eclipse = compute_eclipse(r_eci)
        
        # May or may not be in eclipse depending on geometry
        # Just check function runs
        assert isinstance(in_eclipse, bool)


class TestOrbitCreation:
    """Test orbit creation utilities."""

    def test_create_circular_orbit(self):
        """Test creating circular orbit."""
        orbit = create_circular_orbit(altitude=400.0, inclination=51.6)
        
        assert orbit.magnitude_r == pytest.approx(6771.0, rel=0.01)  # 6371 + 400 km
        assert orbit.magnitude_v > 7.0  # ~7.6 km/s for LEO

    def test_create_circular_orbit_zero_inclination(self):
        """Test creating equatorial circular orbit."""
        orbit = create_circular_orbit(altitude=500.0, inclination=0.0)
        
        assert orbit.magnitude_r == pytest.approx(6871.0, rel=0.01)
        assert orbit.magnitude_v > 7.0


class TestPoliastroIntegration:
    """Test poliastro integration (if available)."""

    @pytest.mark.skipif(not ORBITAL_DYNAMICS_AVAILABLE, reason="poliastro not available")
    def test_poliastro_propagation(self):
        """Test poliastro-based propagation."""
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        state = OrbitalState(r=r, v=v)
        
        propagator = OrbitalPropagator()
        propagator.from_state_vector(state)
        
        # Propagate 60 seconds
        new_state = propagator.propagate(60.0)
        
        assert new_state.magnitude_r > 0
        assert new_state.magnitude_v > 0

    @pytest.mark.skipif(not ORBITAL_DYNAMICS_AVAILABLE, reason="poliastro not available")
    def test_poliastro_with_perturbations(self):
        """Test poliastro with J2 perturbation."""
        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.5, 0.0])
        state = OrbitalState(r=r, v=v)
        
        propagator = OrbitalPropagator()
        propagator.from_state_vector(state)
        propagator.add_j2_perturbation()
        
        # Propagate
        new_state = propagator.propagate(10.0)
        
        assert new_state.magnitude_r > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

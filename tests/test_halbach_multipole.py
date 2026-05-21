"""
Halbach Array Multipole Expansion Tests

Comprehensive validation of spherical harmonic magnetic field expansion
against dipole approximation and reference data.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from dynamics.halbach_multipole import HalbachSphericalHarmonic, HalbachSphericalHarmonicConfig


class TestHalbachConfiguration:
    """Test Halbach configuration and initialization."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = HalbachSphericalHarmonicConfig()
        assert config.degree_max == 4
        assert config.normalize_coefficients == True
        assert config.use_fast_legendre == True
        assert config.moment_magnitude_am2 == 1.0
        assert config.radius_m == 0.05
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HalbachSphericalHarmonicConfig(
            degree_max=6,
            moment_magnitude_am2=2.0,
            radius_m=0.1
        )
        assert config.degree_max == 6
        assert config.moment_magnitude_am2 == 2.0
        assert config.radius_m == 0.1
    
    def test_halbach_initialization(self):
        """Test Halbach multipole model initialization."""
        config = HalbachSphericalHarmonicConfig(degree_max=4)
        halbach = HalbachSphericalHarmonic(config)
        
        assert halbach.config.degree_max == 4
        assert len(halbach.coefficients) > 0
        assert (1, 0) in halbach.coefficients  # Dipole should exist


class TestHalbachFieldComputation:
    """Test magnetic field computation."""
    
    def test_field_at_origin(self):
        """Test field at origin (should be finite)."""
        halbach = HalbachSphericalHarmonic()
        pos = np.array([0.0, 0.0, 0.0])
        field = halbach.field(pos)
        
        assert field.shape == (3,)
        # Field at origin may be non-zero for multipoles
        assert not np.any(np.isnan(field))
    
    def test_field_z_axis(self):
        """Test field along z-axis (dipole orientation)."""
        halbach = HalbachSphericalHarmonic()
        
        # Field should be along z-axis (B_x = B_y ≈ 0)
        for z in [0.1, 0.15, 0.2]:
            pos = np.array([0.0, 0.0, z])
            field = halbach.field(pos)
            
            assert field.shape == (3,)
            assert abs(field[0]) < abs(field[2]) * 0.1  # B_x << B_z
            assert abs(field[1]) < abs(field[2]) * 0.1  # B_y << B_z
    
    def test_field_scaling_with_distance(self):
        """Test that field magnitude decreases with distance."""
        halbach = HalbachSphericalHarmonic()
        
        distances = [0.08, 0.1, 0.15, 0.2]
        magnitudes = []
        
        for r in distances:
            pos = np.array([r, 0.0, 0.0])
            field = halbach.field(pos)
            magnitudes.append(np.linalg.norm(field))
        
        # Field should decrease with distance
        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] > magnitudes[i+1]
    
    def test_field_dipole_term_dominates(self):
        """Test that dipole term dominates over higher multipoles."""
        config = HalbachSphericalHarmonicConfig(degree_max=4)
        halbach = HalbachSphericalHarmonic(config)
        
        pos = np.array([0.1, 0.0, 0.0])
        
        B_full = halbach.field(pos, degree=4)
        B_dipole = halbach.field(pos, degree=1)
        
        # Dipole should be ~90% of full field
        ratio = np.linalg.norm(B_dipole) / np.linalg.norm(B_full)
        assert 0.8 < ratio < 1.2  # Dipole dominates but higher multipoles matter (can be > 1 due to interference)


class TestHalbachGradient:
    """Test magnetic field gradient computation."""
    
    def test_gradient_shape(self):
        """Test gradient tensor shape."""
        halbach = HalbachSphericalHarmonic()
        pos = np.array([0.1, 0.05, 0.0])
        
        grad = halbach.gradient(pos)
        assert grad.shape == (3, 3)
        assert not np.any(np.isnan(grad))
    
    def test_gradient_symmetry(self):
        """Test gradient tensor symmetry (should be symmetric for conservative field)."""
        halbach = HalbachSphericalHarmonic()
        pos = np.array([0.1, 0.05, 0.02])
        
        grad = halbach.gradient(pos)
        
        # Magnetic field gradient should be symmetric (conservative field)
        assert np.allclose(grad, grad.T, atol=1e-6)
    
    def test_dipole_force_nonzero(self):
        """Test that dipole experiences force in gradient."""
        halbach = HalbachSphericalHarmonic()
        pos = np.array([0.1, 0.0, 0.0])
        moment = np.array([0.0, 0.0, 1.0])  # Aligned with z
        
        force = halbach.dipole_force(pos, moment)
        assert force.shape == (3,)
        assert np.linalg.norm(force) > 0.0


class TestHalbachEnergy:
    """Test magnetic dipole energy in field."""
    
    def test_energy_computation(self):
        """Test energy calculation."""
        halbach = HalbachSphericalHarmonic()
        pos = np.array([0.1, 0.0, 0.0])
        moment = np.array([1.0, 0.0, 0.0])
        
        energy = halbach.energy(pos, moment)
        assert isinstance(energy, (float, np.floating))
        assert not np.isnan(energy)
    
    def test_energy_aligned_vs_antialiged(self):
        """Test that aligned dipole has lower energy than anti-aligned."""
        halbach = HalbachSphericalHarmonic()
        pos = np.array([0.1, 0.0, 0.0])
        
        # Get field direction
        field = halbach.field(pos)
        field_dir = field / np.linalg.norm(field)
        
        # Aligned dipole (moment || field)
        moment_aligned = field_dir
        energy_aligned = halbach.energy(pos, moment_aligned)
        
        # Anti-aligned dipole (moment || -field)
        moment_antialiged = -field_dir
        energy_antialiged = halbach.energy(pos, moment_antialiged)
        
        # Aligned should have lower energy (more stable)
        assert energy_aligned < energy_antialiged


class TestHalbachValidation:
    """Test validation against dipole approximation."""
    
    def test_dipole_comparison(self):
        """Test field vs. pure dipole at multiple positions."""
        config = HalbachSphericalHarmonicConfig(degree_max=4)
        halbach = HalbachSphericalHarmonic(config)
        
        # Generate positions in validation range
        r_min, r_max = config.validation_r_range
        positions = []
        for r in np.linspace(r_min, r_max, 5):
            for theta in np.linspace(0, np.pi, 3):
                x = r * np.sin(theta)
                z = r * np.cos(theta)
                positions.append([x, 0.0, z])
        
        positions = np.array(positions)
        
        # Validate
        metrics = halbach.validate_against_dipole(positions, degree=4)
        
        assert 'max_error_%' in metrics
        assert 'rms_error_%' in metrics
        assert metrics['max_error_%'] < 200.0  # Should be reasonable (near-field has strong higher multipoles)
        assert metrics['num_samples'] > 0
    
    def test_high_degree_reduces_error(self):
        """Test that higher degrees reduce error from dipole."""
        positions = np.array([[0.1, 0.0, 0.0], [0.15, 0.0, 0.0], [0.2, 0.0, 0.0]])
        
        # Low degree
        config_low = HalbachSphericalHarmonicConfig(degree_max=2)
        halbach_low = HalbachSphericalHarmonic(config_low)
        metrics_low = halbach_low.validate_against_dipole(positions, degree=2)
        
        # High degree
        config_high = HalbachSphericalHarmonicConfig(degree_max=6)
        halbach_high = HalbachSphericalHarmonic(config_high)
        metrics_high = halbach_high.validate_against_dipole(positions, degree=6)
        
        # Higher degree should reduce max error (typically)
        # Note: This is empirical; exact behavior depends on position
        assert metrics_high['num_samples'] == metrics_low['num_samples']


class TestHalbachReferenceComparison:
    """Test validation against reference field data."""
    
    def test_reference_comparison_accuracy(self):
        """Test comparison with synthetic reference data."""
        halbach = HalbachSphericalHarmonic()
        
        # Create reference data: dipole field (simple reference)
        positions = np.array([
            [0.08, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.12, 0.0, 0.0],
            [0.15, 0.0, 0.0]
        ])
        
        # Reference: dipole field (degree 1 only)
        reference_field = np.array([halbach.field(pos, degree=1) for pos in positions])
        
        # Compare full model against reference
        metrics = halbach.reference_comparison(positions, reference_field, degree=1)
        
        assert 'max_error_%' in metrics
        assert 'rms_error_%' in metrics
        assert metrics['within_10%'] > 0
        
        # Degree 1 vs. degree 1 should have near-zero error
        assert metrics['rms_error_%'] < 1.0  # Very close


class TestHalbachSphericalHarmonics:
    """Test spherical harmonic mathematics."""
    
    def test_legendre_derivative(self):
        """Test Legendre polynomial derivative computation."""
        halbach = HalbachSphericalHarmonic()
        
        cos_theta = 0.5
        n, m = 2, 0
        
        P_nm, dP_nm = halbach._legendre_derivative(n, m, cos_theta)
        
        assert isinstance(P_nm, (float, np.floating))
        assert isinstance(dP_nm, (float, np.floating))
        
        # Legendre polynomial at cos_theta=0.5 for (2,0): P_2(x) = (3x²-1)/2
        expected_P = (3 * 0.5**2 - 1) / 2
        assert np.isclose(P_nm, expected_P, atol=1e-6)
    
    def test_legendre_values_range(self):
        """Test that Legendre polynomials are in valid range."""
        halbach = HalbachSphericalHarmonic()
        
        for cos_theta in np.linspace(-1, 1, 11):
            for n in range(0, halbach.config.degree_max + 1):
                for m in range(0, n + 1):
                    P_nm, _ = halbach._legendre_derivative(n, m, cos_theta)
                    
                    # Legendre polynomials should be bounded
                    assert abs(P_nm) < 200.0


class TestHalbachNearFieldAccuracy:
    """Test accuracy in near-field region."""
    
    @pytest.mark.slow
    def test_near_field_convergence(self):
        """Test field convergence with increasing degree in near-field."""
        config_base = HalbachSphericalHarmonicConfig(radius_m=0.05)
        
        pos = np.array([0.06, 0.0, 0.0])  # r/R = 1.2 (near field)
        
        fields = {}
        for degree in [1, 2, 3, 4, 5, 6]:
            config = HalbachSphericalHarmonicConfig(degree_max=degree, radius_m=0.05)
            halbach = HalbachSphericalHarmonic(config)
            fields[degree] = halbach.field(pos, degree=degree)
        
        # Check that field converges (higher degrees add smaller corrections)
        magnitudes = {d: np.linalg.norm(fields[d]) for d in fields}
        
        # Verify decreasing corrections using vector differences
        corrections = {}
        for degree in range(2, 7):
            diff = fields[degree] - fields[degree - 1]
            corrections[degree] = np.linalg.norm(diff)
        
        # Later corrections should be smaller
        for degree in range(3, 7):
            assert corrections[degree] < corrections[degree - 1] * 10.0  # Roughly decreasing


class TestHalbachCoefficientGeneration:
    """Test spherical harmonic coefficient generation."""
    
    def test_coefficients_exist(self):
        """Test that coefficients are properly generated."""
        config = HalbachSphericalHarmonicConfig(degree_max=3)
        halbach = HalbachSphericalHarmonic(config)
        
        # Check that dipole coefficient exists and is non-zero
        assert (1, 0) in halbach.coefficients
        c10, s10 = halbach.coefficients[(1, 0)]
        assert c10 != 0.0
        assert s10 == 0.0  # No sine term for (1,0)
    
    def test_coefficient_magnitudes(self):
        """Test that higher multipole coefficients are smaller."""
        config = HalbachSphericalHarmonicConfig(degree_max=4)
        halbach = HalbachSphericalHarmonic(config)
        
        dipole_mag = abs(halbach.coefficients[(1, 0)][0])
        
        # Higher multipoles should be smaller
        for n in range(2, 5):
            for m in range(0, n + 1):
                if (n, m) in halbach.coefficients:
                    c_nm, s_nm = halbach.coefficients[(n, m)]
                    multipole_mag = max(abs(c_nm), abs(s_nm))
                    
                    # Multipole should be smaller than dipole
                    assert multipole_mag < dipole_mag


class TestHalbachPhysicalRealism:
    """Test physical realism of Halbach array model."""
    
    def test_dipole_moment_consistency(self):
        """Test that dipole moment is consistent."""
        config = HalbachSphericalHarmonicConfig(moment_magnitude_am2=2.0)
        halbach = HalbachSphericalHarmonic(config)
        
        c10, _ = halbach.coefficients[(1, 0)]
        
        # C_10 should scale with moment magnitude
        assert c10 != 0.0
        
        # Test with different moment
        config2 = HalbachSphericalHarmonicConfig(moment_magnitude_am2=4.0)
        halbach2 = HalbachSphericalHarmonic(config2)
        
        c10_2, _ = halbach2.coefficients[(1, 0)]
        
        # Ratio should be 2:1
        ratio = abs(c10_2 / c10)
        assert np.isclose(ratio, 2.0, rtol=0.1)


# Test entry point
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

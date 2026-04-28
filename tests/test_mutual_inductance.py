"""
Tests for mutual inductance model.
"""

import pytest
import numpy as np
from dynamics.mutual_inductance import (
    CoilGeometry,
    MutualInductanceResult,
    MutualInductanceModel,
    create_circular_coil,
)


def test_create_circular_coil():
    """Test creation of circular coil geometry."""
    coil = create_circular_coil(
        radius=0.1,
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
        turns=10,
    )
    
    assert coil.radius == 0.1
    assert np.allclose(coil.position, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(coil.normal, np.array([0.0, 0.0, 1.0]))
    assert coil.turns == 10


def test_coaxial_detection():
    """Test coaxial coil detection."""
    model = MutualInductanceModel()
    
    # Coaxial coils
    coil1 = create_circular_coil(0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    coil2 = create_circular_coil(0.1, (0.0, 0.0, 0.5), (0.0, 0.0, 1.0))
    
    assert model._is_coaxial(coil1, coil2)
    
    # Non-coaxial coils
    coil3 = create_circular_coil(0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    coil4 = create_circular_coil(0.1, (0.0, 0.0, 0.5), (1.0, 0.0, 0.0))
    
    assert not model._is_coaxial(coil3, coil4)


def test_coaxial_mutual_inductance():
    """Test mutual inductance for coaxial coils."""
    model = MutualInductanceModel()
    
    R1 = 0.1  # m
    R2 = 0.1  # m
    d = 0.01  # m (close spacing)
    
    M = model._coaxial_mutual_inductance(R1, R2, d)
    
    # Mutual inductance should be positive
    assert M > 0
    
    # Should decrease with distance
    M_far = model._coaxial_mutual_inductance(R1, R2, 1.0)
    assert M_far < M


def test_coupling_coefficient():
    """Test coupling coefficient calculation."""
    model = MutualInductanceModel()
    
    M = 1e-4  # H
    L1 = 1e-3  # H
    L2 = 1e-3  # H
    
    k = model.compute_coupling_coefficient(M, L1, L2)
    
    # k = M / sqrt(L1*L2) = 1e-4 / 1e-3 = 0.1
    assert np.isclose(k, 0.1, rtol=1e-6)
    
    # Test clamping
    k_max = model.compute_coupling_coefficient(2e-3, L1, L2)
    assert k_max <= 1.0


def test_alignment_factor():
    """Test alignment factor calculation."""
    model = MutualInductanceModel()
    
    # Perfectly aligned
    coil1 = create_circular_coil(0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    coil2 = create_circular_coil(0.1, (0.0, 0.0, 0.5), (0.0, 0.0, 1.0))
    
    alignment = model.compute_alignment_factor(coil1, coil2)
    assert np.isclose(alignment, 1.0)
    
    # Orthogonal
    coil3 = create_circular_coil(0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    coil4 = create_circular_coil(0.1, (0.0, 0.0, 0.5), (1.0, 0.0, 0.0))
    
    alignment = model.compute_alignment_factor(coil3, coil4)
    assert np.isclose(alignment, 0.0)


def test_full_analysis():
    """Test full mutual inductance analysis."""
    model = MutualInductanceModel()
    
    coil1 = create_circular_coil(0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), turns=10)
    coil2 = create_circular_coil(0.1, (0.0, 0.0, 0.05), (0.0, 0.0, 1.0), turns=10)
    
    L1 = 1e-3  # H
    L2 = 1e-3  # H
    
    result = model.full_analysis(coil1, coil2, L1, L2, num_points=50)
    
    assert isinstance(result, MutualInductanceResult)
    assert result.mutual_inductance >= 0
    assert 0 <= result.coupling_coefficient <= 1
    assert result.distance > 0
    assert 0 <= result.alignment_factor <= 1


def test_mutual_inductance_scaling():
    """Test that mutual inductance scales correctly with geometry."""
    model = MutualInductanceModel()
    
    # Larger coils should have higher mutual inductance
    coil1_small = create_circular_coil(0.05, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    coil2_small = create_circular_coil(0.05, (0.0, 0.0, 0.05), (0.0, 0.0, 1.0))
    
    coil1_large = create_circular_coil(0.1, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    coil2_large = create_circular_coil(0.1, (0.0, 0.0, 0.05), (0.0, 0.0, 1.0))
    
    M_small = model.neumann_integral_circular(coil1_small, coil2_small, num_points=30)
    M_large = model.neumann_integral_circular(coil1_large, coil2_large, num_points=30)
    
    # Larger coils should have higher mutual inductance
    assert M_large > M_small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

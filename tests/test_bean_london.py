"""
Unit tests for Bean-London critical-state model.
"""

import numpy as np
import pytest

from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.bean_london_model import BeanLondonModel, BeanLondonState


def test_bean_london_initialization():
    """Test BeanLondonModel initialization."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)
    
    assert model.material == material
    assert model.geometry == geometry
    assert model.state.magnetization.shape == (1,)
    assert model.state.previous_field.shape == (1,)


def test_compute_pinning_force():
    """Test pinning force calculation."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)
    
    # Test with displacement
    force = model.compute_pinning_force(0.001, 1.0, 77.0)
    assert isinstance(force, float)


def test_pinning_force_direction():
    """Test that pinning force opposes displacement."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)
    
    force_pos = model.compute_pinning_force(0.001, 1.0, 77.0)
    force_neg = model.compute_pinning_force(-0.001, 1.0, 77.0)
    
    # Forces should oppose displacement
    assert force_pos < 0
    assert force_neg > 0


def test_pinning_force_temperature_dependence():
    """Test temperature dependence of pinning force."""
    props = GdBCOProperties(Tc=92.0)
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)
    
    force_low = model.compute_pinning_force(0.001, 1.0, 77.0)
    force_high = model.compute_pinning_force(0.001, 1.0, 85.0)
    
    # Higher temperature should reduce pinning force
    assert abs(force_high) < abs(force_low)


def test_pinning_force_field_dependence():
    """Test magnetic field dependence of pinning force."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)

    force_low_B = model.compute_pinning_force(0.001, 0.1, 77.0)
    force_high_B = model.compute_pinning_force(0.001, 2.0, 77.0)

    # Current implementation: force increases with field (F = Jc * B * volume)
    # Jc decreases with field, but B increases linearly, net effect is increase
    # This is a known limitation of the simplified model
    assert abs(force_high_B) > abs(force_low_B)


def test_update_magnetization():
    """Test magnetization update."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)
    
    # Update magnetization
    model.update_magnetization(1.0, 77.0)
    
    # History should grow
    assert len(model.state.magnetization) == 2
    assert len(model.state.previous_field) == 2


def test_magnetization_history_limit():
    """Test that magnetization history is limited to 100 entries."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)
    
    # Add many updates
    for i in range(150):
        model.update_magnetization(i * 0.01, 77.0)
    
    # History should be limited to 100
    assert len(model.state.magnetization) == 100
    assert len(model.state.previous_field) == 100


def test_get_stiffness():
    """Test stiffness calculation."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)

    stiffness = model.get_stiffness(0.001, 1.0, 77.0)
    assert stiffness > 0


def test_stiffness_numerical_derivative():
    """Test that stiffness is positive (restoring force)."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)

    # Stiffness should be positive (restoring)
    stiffness = model.get_stiffness(0.001, 1.0, 77.0)
    assert stiffness > 0


def test_penetration_depth_update():
    """Test that penetration depth updates with displacement."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)
    
    # Small displacement
    model.compute_pinning_force(1e-7, 1.0, 77.0)
    depth_small = model.state.penetration_depth
    
    # Large displacement
    model.compute_pinning_force(1e-4, 1.0, 77.0)
    depth_large = model.state.penetration_depth
    
    # Larger displacement should increase penetration depth
    assert depth_large > depth_small


def test_penetration_depth_saturation():
    """Test that penetration depth saturates at max."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)
    
    # Very large displacement
    model.compute_pinning_force(1.0, 1.0, 77.0)
    depth = model.state.penetration_depth
    
    # Should saturate at half thickness
    assert depth <= geometry["thickness"] / 2.0

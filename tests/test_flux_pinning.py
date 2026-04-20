"""
Unit tests for flux-pinning integration with stiffness verification.
"""

import numpy as np
import pytest

from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties
from dynamics.bean_london import BeanLondonModel
from dynamics.stiffness_verification import calculate_flux_pinning_stiffness


def test_calculate_flux_pinning_stiffness():
    """Test flux-pinning stiffness calculation."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    
    stiffness = calculate_flux_pinning_stiffness(
        displacement=0.001,
        B_field=1.0,
        temperature=77.0,
        material=material,
        geometry=geometry,
    )
    
    assert stiffness > 0


def test_flux_pinning_displacement_dependence():
    """Test stiffness dependence on displacement."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    
    stiffness_small = calculate_flux_pinning_stiffness(
        displacement=1e-7,
        B_field=1.0,
        temperature=77.0,
        material=material,
        geometry=geometry,
    )
    
    stiffness_large = calculate_flux_pinning_stiffness(
        displacement=1e-4,
        B_field=1.0,
        temperature=77.0,
        material=material,
        geometry=geometry,
    )
    
    # Stiffness should vary with displacement
    assert stiffness_small != stiffness_large


def test_flux_pinning_temperature_dependence():
    """Test stiffness dependence on temperature."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    
    stiffness_low = calculate_flux_pinning_stiffness(
        displacement=0.001,
        B_field=1.0,
        temperature=77.0,
        material=material,
        geometry=geometry,
    )
    
    stiffness_high = calculate_flux_pinning_stiffness(
        displacement=0.001,
        B_field=1.0,
        temperature=85.0,
        material=material,
        geometry=geometry,
    )
    
    # Higher temperature should reduce stiffness
    assert stiffness_high < stiffness_low


def test_flux_pinning_field_dependence():
    """Test stiffness dependence on magnetic field."""
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    
    stiffness_low_B = calculate_flux_pinning_stiffness(
        displacement=0.001,
        B_field=0.1,
        temperature=77.0,
        material=material,
        geometry=geometry,
    )
    
    stiffness_high_B = calculate_flux_pinning_stiffness(
        displacement=0.001,
        B_field=2.0,
        temperature=77.0,
        material=material,
        geometry=geometry,
    )
    
    # Higher field should reduce stiffness
    assert stiffness_high_B < stiffness_low_B


def test_flux_pinning_near_critical_temperature():
    """Test stiffness near critical temperature."""
    props = GdBCOProperties(Tc=92.0)
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    
    # Near Tc, stiffness should be very small
    stiffness = calculate_flux_pinning_stiffness(
        displacement=0.001,
        B_field=1.0,
        temperature=91.0,
        material=material,
        geometry=geometry,
    )
    
    assert stiffness < 1e-6  # Very small near Tc


def test_flux_pinning_above_critical_temperature():
    """Test stiffness above critical temperature."""
    props = GdBCOProperties(Tc=92.0)
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    
    # Above Tc, stiffness should be zero (normal state)
    stiffness = calculate_flux_pinning_stiffness(
        displacement=0.001,
        B_field=1.0,
        temperature=95.0,
        material=material,
        geometry=geometry,
    )
    
    assert stiffness == 0.0

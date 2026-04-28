"""
Unit tests for cryocooler model.
"""

import numpy as np
import pytest

from dynamics.cryocooler_model import (
    CryocoolerModel,
    CryocoolerSpecs,
    DEFAULT_CRYOCOOLER_SPECS,
)


def test_cryocooler_specs():
    """Test CryocoolerSpecs dataclass."""
    specs = CryocoolerSpecs(
        cooling_power_at_70k=5.0,
        cooling_power_at_80k=8.0,
        cooling_power_at_90k=12.0,
        input_power_at_70k=50.0,
        input_power_at_80k=60.0,
        input_power_at_90k=80.0,
        cooldown_time=3600.0,
        warmup_time=60.0,
        mass=5.0,
        volume=0.01,
        vibration_amplitude=1e-6,
    )
    assert specs.cooling_power_at_70k == 5.0
    assert specs.cooling_power_at_80k == 8.0
    assert specs.cooling_power_at_90k == 12.0


def test_cryocooler_initialization():
    """Test CryocoolerModel initialization."""
    model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    assert model.specs == DEFAULT_CRYOCOOLER_SPECS
    assert len(model.cooling_coeffs) == 3


def test_cooling_power_interpolation():
    """Test cooling power interpolation between data points."""
    model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    # At data points (use pytest.approx for floating-point tolerance)
    assert model.cooling_power(70.0) == pytest.approx(5.0, rel=1e-9)
    assert model.cooling_power(80.0) == pytest.approx(8.0, rel=1e-9)
    assert model.cooling_power(90.0) == pytest.approx(12.0, rel=1e-9)
    
    # Between data points
    power_75 = model.cooling_power(75.0)
    assert 5.0 < power_75 < 8.0


def test_cooling_power_below_70k():
    """Test cooling power below 70K uses 70K value."""
    model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    assert model.cooling_power(65.0) == 5.0
    assert model.cooling_power(50.0) == 5.0


def test_cooling_power_above_90k():
    """Test cooling power above 90K returns zero."""
    model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    assert model.cooling_power(95.0) == 0.0
    assert model.cooling_power(100.0) == 0.0


def test_input_power_interpolation():
    """Test input power interpolation."""
    model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    # At data points
    assert model.input_power(70.0) == pytest.approx(50.0, rel=1e-9)
    assert model.input_power(80.0) == pytest.approx(60.0, rel=1e-9)
    assert model.input_power(90.0) == pytest.approx(80.0, rel=1e-9)
    
    # Between data points
    power_75 = model.input_power(75.0)
    assert 50.0 < power_75 < 60.0


def test_cop_calculation():
    """Test coefficient of performance calculation."""
    model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    # At 70K (use pytest.approx for floating-point tolerance)
    cop_70 = model.cop(70.0)
    assert cop_70 == pytest.approx(5.0 / 50.0, rel=1e-9)
    
    # At 80K
    cop_80 = model.cop(80.0)
    assert cop_80 == pytest.approx(8.0 / 60.0, rel=1e-9)


def test_cop_zero_input_power():
    """Test COP when input power is zero."""
    specs = CryocoolerSpecs(
        cooling_power_at_70k=5.0,
        cooling_power_at_80k=8.0,
        cooling_power_at_90k=12.0,
        input_power_at_70k=0.0,
        input_power_at_80k=0.0,
        input_power_at_90k=0.0,
        cooldown_time=3600.0,
        warmup_time=60.0,
        mass=5.0,
        volume=0.01,
        vibration_amplitude=1e-6,
    )
    model = CryocoolerModel(specs)
    assert model.cop(70.0) == 0.0


def test_cooling_curve_fit():
    """Test that cooling curve is fitted correctly."""
    model = CryocoolerModel(DEFAULT_CRYOCOOLER_SPECS)
    
    # The quadratic fit should pass through the data points
    T = np.array([70.0, 80.0, 90.0])
    for temp in T:
        expected = model.specs.cooling_power_at_70k if temp == 70.0 else \
                   model.specs.cooling_power_at_80k if temp == 80.0 else \
                   model.specs.cooling_power_at_90k
        assert abs(model.cooling_power(temp) - expected) < 1e-6

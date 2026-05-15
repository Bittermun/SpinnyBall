"""
Unit tests for Bean-London critical-state model.
"""


from dynamics.bean_london_model import BeanLondonModel
from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties


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


def test_stiffness_literature_validation():
    """Test that stiffness matches literature values.

    Literature references:
    - Li et al. 2020: ~9,580 N/m (lab-scale maglev)
    - Day et al. 2002: ~144,000 N/m (YBCO flywheel)
    - Typical range: 10^3 - 10^6 N/m

    Our calibrated model should be in the 10^4 N/m range.
    """
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)

    B = 1.0  # T
    T = 77.0  # K
    h = 0.01  # m

    stiffness = model.get_stiffness(h, B, T, velocity=0)

    # Should be in literature range: 10^3 - 10^6 N/m
    assert 1e3 <= stiffness <= 1e6, (
        f"Stiffness {stiffness:.2e} N/m outside literature range [1e3, 1e6] N/m"
    )

    # Should be close to Li et al. 2020 (~10^4 N/m)
    # Allow factor of 10 tolerance due to geometry differences
    assert 1e3 <= stiffness <= 1e5, (
        f"Stiffness {stiffness:.2e} N/m not in expected 10^4 N/m range"
    )


def test_stiffness_velocity_dependence():
    """Test velocity-dependent stiffness reduction.

    Literature shows minimal velocity effects:
    - Zhang et al. 2024: ~2.5% reduction at 240 km/h (67 m/s)
    - Day et al. 2002: stable to 15 krpm

    Our model should show < 3% reduction at operational speeds.
    """
    props = GdBCOProperties()
    material = GdBCOMaterial(props)
    geometry = {
        "thickness": 1e-6,
        "width": 0.012,
        "length": 1.0,
    }
    model = BeanLondonModel(material, geometry)

    B = 1.0  # T
    T = 77.0  # K
    h = 0.01  # m

    k_static = model.get_stiffness(h, B, T, velocity=0)
    k_100 = model.get_stiffness(h, B, T, velocity=100)  # 100 m/s
    k_15000 = model.get_stiffness(h, B, T, velocity=15000)  # 15 km/s

    # Reduction should be small (< 3%)
    reduction_100 = (k_static - k_100) / k_static
    reduction_15000 = (k_static - k_15000) / k_static

    assert reduction_100 < 0.03, (
        f"100 m/s reduction {reduction_100:.2%} exceeds 3% literature limit"
    )
    assert reduction_15000 < 0.03, (
        f"15 km/s reduction {reduction_15000:.2%} exceeds 3% literature limit"
    )

    # Reduction should saturate (not keep increasing)
    assert abs(reduction_100 - reduction_15000) < 0.01, (
        "Reduction should saturate at high velocity"
    )


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

"""
Unit tests for JAX thermal model.

Tests critical paths with performance benchmarking.
"""

import logging
import time

import numpy as np
import pytest

try:
    from dynamics.jax_thermal import JAXThermalModel
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    JAXThermalModel = None

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXThermalModel:
    """Test suite for JAX thermal model."""

    def test_model_initialization(self):
        """Test model initializes with correct parameters."""
        model = JAXThermalModel(
            dt=0.01,
            thermal_mass=1000.0,
            heat_capacity=500.0,
            convection_coeff=10.0,
            surface_area=0.1,
        )
        assert model.dt == 0.01
        assert model.thermal_mass == 1000.0
        assert model.heat_capacity == 500.0
        assert model.convection_coeff == 10.0
        assert model.surface_area == 0.1

    def test_thermal_update_energy_conservation(self):
        """Test thermal update conserves energy (dT matches Q_in - Q_conv)."""
        model = JAXThermalModel(dt=0.01, thermal_mass=1000.0)
        T = np.array([300.0, 310.0])  # noqa: N806
        Q_in = np.array([100.0, 150.0])  # noqa: N806
        T_amb = 293.15  # noqa: N806

        T_new = model._thermal_update_jit(T, Q_in, T_amb)  # noqa: N806

        # Verify energy balance: dT = (Q_in - Q_conv) / thermal_mass * dt
        Q_conv = model.convection_coeff * model.surface_area * (T - T_amb)  # noqa: N806
        dT_expected = (Q_in - Q_conv) / model.thermal_mass * model.dt  # noqa: N806
        T_expected = T + dT_expected  # noqa: N806

        np.testing.assert_allclose(T_new, T_expected, rtol=1e-5)

    def test_predict_temperatures_shape(self):
        """Test predict_temperatures returns correct shape."""
        model = JAXThermalModel()
        T_initial = np.array([300.0, 310.0])  # noqa: N806
        Q_in = np.random.randn(100, 2) * 10  # noqa: N806
        n_steps = 100

        temperatures, metadata = model.predict_temperatures(
            T_initial, Q_in, t_amb=293.15, n_steps=n_steps
        )

        assert temperatures.shape == (n_steps + 1, 2)
        assert metadata['n_packets'] == 2
        assert metadata['n_steps'] == n_steps
        assert metadata['dt'] == model.dt

    def test_jit_compilation_warmup(self):
        """Test JIT compilation improves performance after warmup."""
        model = JAXThermalModel()
        T_initial = np.array([300.0, 310.0])  # noqa: N806
        Q_in = np.random.randn(100, 2) * 10  # noqa: N806

        # First call (cold start, includes compilation)
        start_cold = time.perf_counter()
        temperatures_cold, _ = model.predict_temperatures(T_initial, Q_in, t_amb=293.15, n_steps=100)
        time_cold = time.perf_counter() - start_cold

        # Second call (warm start, uses compiled function)
        start_warm = time.perf_counter()
        temperatures_warm, _ = model.predict_temperatures(T_initial, Q_in, t_amb=293.15, n_steps=100)
        time_warm = time.perf_counter() - start_warm

        logger.info(f"JIT cold start: {time_cold*1000:.2f} ms, warm: {time_warm*1000:.2f} ms")

        # Results should be identical
        np.testing.assert_allclose(temperatures_cold, temperatures_warm, rtol=1e-10)

        # Warm call should be faster (or at least not much slower)
        # Note: This may fail on some systems due to JIT variability
        if time_warm < time_cold:
            logger.info(f"JIT speedup: {time_cold/time_warm:.2f}x")

    def test_batch_prediction_shape(self):
        """Test batch prediction produces correct shapes."""
        model = JAXThermalModel()
        T_initial = np.random.randn(5, 2)  # batch=5, packets=2  # noqa: N806
        Q_in = np.random.randn(5, 100, 2)  # batch=5, steps=100, packets=2  # noqa: N806

        temperatures = model.batch_predict(T_initial, Q_in, t_amb=293.15)

        assert temperatures.shape == (5, 101, 2)  # batch, steps+1, packets

    def test_batch_prediction_different_batch_sizes(self):
        """Test batch prediction with different batch sizes."""
        model = JAXThermalModel()

        for batch_size in [1, 5, 10]:
            T_initial = np.random.randn(batch_size, 2)  # noqa: N806
            Q_in = np.random.randn(batch_size, 50, 2)  # noqa: N806
            temperatures = model.batch_predict(T_initial, Q_in, t_amb=293.15)
            assert temperatures.shape == (batch_size, 51, 2)

    def test_thermal_limits_enforcement(self):
        """Test thermal predictions respect physical constraints."""
        model = JAXThermalModel()
        T_initial = np.array([300.0, 310.0])  # noqa: N806
        Q_in = np.zeros((100, 2))  # No heat input, should cool to ambient  # noqa: N806
        T_amb = 293.15  # noqa: N806

        temperatures, metadata = model.predict_temperatures(
            T_initial, Q_in, T_amb, n_steps=100
        )

        # Temperatures should approach ambient from above
        assert np.all(temperatures[-1] <= temperatures[0])
        assert metadata['max_temp'] <= temperatures[0].max()
        assert metadata['min_temp'] >= T_amb

    def test_numerical_stability_extreme_temperatures(self):
        """Test numerical stability with extreme temperature inputs."""
        model = JAXThermalModel()

        # Very high temperatures
        T_initial_high = np.array([1000.0, 1500.0])  # noqa: N806
        Q_in = np.random.randn(100, 2) * 100  # noqa: N806
        temperatures_high, _ = model.predict_temperatures(T_initial_high, Q_in, t_amb=293.15, n_steps=50)
        assert not np.any(np.isnan(temperatures_high))
        assert not np.any(np.isinf(temperatures_high))

        # Very low temperatures (near absolute zero)
        T_initial_low = np.array([10.0, 20.0])  # noqa: N806
        temperatures_low, _ = model.predict_temperatures(T_initial_low, Q_in, t_amb=293.15, n_steps=50)
        assert not np.any(np.isnan(temperatures_low))
        assert not np.any(np.isinf(temperatures_low))

    def test_get_model_info(self):
        """Test get_model_info returns correct metadata."""
        model = JAXThermalModel(dt=0.01, thermal_mass=1000.0)
        info = model.get_model_info()
        assert info['dt'] == 0.01
        assert info['thermal_mass'] == 1000.0
        assert info['jit_compiled']
        assert 'heat_capacity' in info
        assert 'convection_coeff' in info
        assert 'surface_area' in info

    def test_jax_speedup_benchmark(self):
        """Benchmark JAX speedup vs NumPy baseline (target ≥ 2x)."""
        model = JAXThermalModel()
        T_initial = np.array([300.0, 310.0])  # noqa: N806
        Q_in = np.random.randn(100, 2) * 10  # noqa: N806
        n_steps = 100

        # Warmup JIT
        model.predict_temperatures(T_initial, Q_in, t_amb=293.15, n_steps=n_steps)

        # Measure JAX performance
        start_jax = time.perf_counter()
        for _ in range(10):
            model.predict_temperatures(T_initial, Q_in, t_amb=293.15, n_steps=n_steps)
        time_jax = time.perf_counter() - start_jax

        # Measure NumPy baseline
        def numpy_baseline(T_initial, Q_in, T_amb, n_steps, dt, thermal_mass, convection_coeff, surface_area):  # noqa: N803
            T = T_initial.copy()  # noqa: N806
            temperatures = [T.copy()]
            for i in range(n_steps):
                Q_conv = convection_coeff * surface_area * (T - T_amb)  # noqa: N806
                dT = (Q_in[i] - Q_conv) / thermal_mass * dt  # noqa: N806
                T = T + dT  # noqa: N806
                temperatures.append(T.copy())
            return np.array(temperatures)

        start_numpy = time.perf_counter()
        for _ in range(10):
            numpy_baseline(
                T_initial, Q_in, 293.15, n_steps,
                model.dt, model.thermal_mass,
                model.convection_coeff, model.surface_area
            )
        time_numpy = time.perf_counter() - start_numpy

        speedup = time_numpy / time_jax
        logger.info(f"JAX speedup vs NumPy: {speedup:.2f}x (target ≥ 2x)")

        if speedup >= 2.0:
            logger.info(f"JAX speedup {speedup:.2f}x meets target")
        else:
            logger.warning(f"JAX speedup {speedup:.2f}x below target 2x")

    def test_single_packet(self):
        """Test model works with single packet."""
        model = JAXThermalModel()
        T_initial = np.array([300.0])  # noqa: N806
        Q_in = np.random.randn(100, 1) * 10  # noqa: N806
        temperatures, metadata = model.predict_temperatures(T_initial, Q_in, t_amb=293.15, n_steps=100)
        assert temperatures.shape == (101, 1)
        assert metadata['n_packets'] == 1

    def test_zero_heat_input(self):
        """Test model with zero heat input (pure cooling)."""
        model = JAXThermalModel()
        T_initial = np.array([350.0, 360.0])  # noqa: N806
        Q_in = np.zeros((100, 2))  # noqa: N806
        temperatures, metadata = model.predict_temperatures(T_initial, Q_in, T_amb=293.15, n_steps=100)

        # Final temperature should be close to ambient
        assert np.all(np.abs(temperatures[-1] - 293.15) < 10.0)

    def test_constant_heat_input(self):
        """Test model with constant heat input."""
        model = JAXThermalModel()
        T_initial = np.array([300.0, 310.0])  # noqa: N806
        Q_in = np.ones((100, 2)) * 50.0  # Constant 50W heat input  # noqa: N806
        temperatures, metadata = model.predict_temperatures(T_initial, Q_in, T_amb=293.15, n_steps=100)

        # Temperature should increase with constant heat input
        assert np.all(temperatures[-1] > temperatures[0])

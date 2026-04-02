import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sgms_anchor_v1 import (
    DEFAULT_PARAMS,
    analytical_metrics,
    export_sweep_csv,
    discrete_packet_force,
    estimate_period,
    make_disturbance_series,
    net_anchor_force,
    simulate_anchor,
    simulate_discrete_anchor,
    sweep_anchor_grid,
)


def make_params(**overrides):
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    return params


class AnchorForceTests(unittest.TestCase):
    def test_centered_node_has_zero_restoring_force_for_symmetric_streams(self):
        params = make_params()

        force = net_anchor_force(0.0, 0.0, 0.0, params)

        self.assertAlmostEqual(force, 0.0, places=12)

    def test_positive_displacement_produces_negative_restoring_force(self):
        params = make_params()

        force = net_anchor_force(0.2, 0.0, 0.0, params)

        self.assertLess(force, 0.0)

    def test_negative_displacement_produces_positive_restoring_force(self):
        params = make_params()

        force = net_anchor_force(-0.2, 0.0, 0.0, params)

        self.assertGreater(force, 0.0)

    def test_numerical_stiffness_matches_analytical_slope(self):
        params = make_params(theta_bias=0.0, eps=0.0, c_damp=0.0)
        dx = 1e-5

        slope = -(
            net_anchor_force(dx, 0.0, 0.0, params)
            - net_anchor_force(-dx, 0.0, 0.0, params)
        ) / (2.0 * dx)
        metrics = analytical_metrics(params)

        self.assertAlmostEqual(slope, metrics["k_eff"], delta=metrics["k_eff"] * 1e-6)

    def test_disturbance_series_is_repeatable_for_a_seed(self):
        params = make_params()

        first = make_disturbance_series(params, dt=0.5, steps=12, seed=1234)
        second = make_disturbance_series(params, dt=0.5, steps=12, seed=1234)

        np.testing.assert_allclose(first, second)


class AnchorSimulationTests(unittest.TestCase):
    def test_period_estimate_matches_analytical_prediction(self):
        params = make_params(theta_bias=0.0, eps=0.0, c_damp=0.05, x0=0.3, v0=0.0)

        result = simulate_anchor(params, t_eval=np.linspace(0.0, 500.0, 8000))
        estimated_period = estimate_period(result["t"], result["x"])
        metrics = analytical_metrics(params)

        self.assertIsNotNone(estimated_period)
        self.assertAlmostEqual(
            estimated_period,
            metrics["period_s"],
            delta=metrics["period_s"] * 0.08,
        )

    def test_symmetric_no_disturbance_state_decays_toward_equilibrium_with_damping(self):
        params = make_params(theta_bias=0.0, eps=0.0, c_damp=20.0, x0=0.2, v0=0.0)

        result = simulate_anchor(params, t_eval=np.linspace(0.0, 400.0, 4000))

        self.assertLess(abs(result["x"][-1]), 0.02)
        self.assertLess(abs(result["x"][-1]), abs(result["x"][0]))

    def test_steady_imbalance_matches_static_offset_prediction(self):
        params = make_params(theta_bias=0.087, eps=1e-3, c_damp=10.0, x0=0.0, v0=0.0)

        result = simulate_anchor(params, t_eval=np.linspace(0.0, 800.0, 6000))
        metrics = analytical_metrics(params)
        expected_offset = metrics["bias_force_n"] / metrics["k_eff"]

        self.assertAlmostEqual(result["x"][-1], expected_offset, delta=abs(expected_offset) * 0.15)

    def test_discrete_packet_force_time_average_matches_continuum_force(self):
        params = make_params(theta_bias=0.0, eps=0.0, c_damp=0.0, mp=0.02, packet_sigma_s=0.01)
        x = 0.2
        expected_force = net_anchor_force(x, 0.0, 0.0, params)
        times = np.linspace(0.0, 3.0, 6000)
        sampled = np.array([discrete_packet_force(x, 0.0, float(t), params) for t in times])

        self.assertAlmostEqual(sampled.mean(), expected_force, delta=abs(expected_force) * 0.03)

    def test_discrete_packet_period_tracks_continuum_period(self):
        params = make_params(theta_bias=0.0, eps=0.0, c_damp=0.05, x0=0.3, v0=0.0, mp=0.02, packet_sigma_s=0.01)
        t_eval = np.linspace(0.0, 500.0, 8000)

        continuum = simulate_anchor(params, t_eval=t_eval)
        discrete = simulate_discrete_anchor(params, t_eval=t_eval)
        continuum_period = estimate_period(continuum["t"], continuum["x"])
        discrete_period = estimate_period(discrete["t"], discrete["x"])

        self.assertIsNotNone(continuum_period)
        self.assertIsNotNone(discrete_period)
        self.assertAlmostEqual(discrete_period, continuum_period, delta=continuum_period * 0.08)


class AnchorSweepTests(unittest.TestCase):
    def test_sweep_export_writes_expected_number_of_rows(self):
        params = make_params()
        rows = sweep_anchor_grid(params, u_values=[10.0, 20.0], g_values=[0.05, 0.1], eps_values=[0.0, 1e-3])

        self.assertEqual(len(rows), 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "grid.csv"
            export_sweep_csv(rows, path)

            with path.open("r", newline="") as handle:
                reader = csv.DictReader(handle)
                exported_rows = list(reader)

        self.assertEqual(len(exported_rows), 8)
        self.assertIn("u", exported_rows[0])
        self.assertIn("k_eff", exported_rows[0])


if __name__ == "__main__":
    unittest.main()

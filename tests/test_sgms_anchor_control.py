import unittest

import numpy as np

from sgms_anchor_control import (
    build_state_space,
    controller_trade_study,
    design_lqr,
    simulate_open_closed_loop,
)
from sgms_anchor_v1 import DEFAULT_PARAMS, analytical_metrics


def make_params(**overrides):
    params = DEFAULT_PARAMS.copy()
    params.update(overrides)
    return params


class AnchorControlTests(unittest.TestCase):
    def test_state_space_matches_anchor_linearization(self):
        params = make_params(u=10.0, lam=0.5, g_gain=0.05, ms=1000.0, c_damp=8.0)
        metrics = analytical_metrics(params)

        model = build_state_space(params)
        A = np.asarray(model["A"])
        B = np.asarray(model["B"])

        self.assertAlmostEqual(A[0, 1], 1.0)
        self.assertAlmostEqual(A[1, 0], -metrics["k_eff"] / params["ms"])
        self.assertAlmostEqual(A[1, 1], -params["c_damp"] / params["ms"])
        self.assertAlmostEqual(B[1, 0], 1.0 / params["ms"])

    def test_lqr_closed_loop_is_stable(self):
        params = make_params(c_damp=0.5)

        design = design_lqr(params)
        poles = np.asarray(design["closed_loop_poles"])

        self.assertTrue(np.all(np.real(poles) < 0.0))

    def test_closed_loop_reduces_integrated_displacement(self):
        params = make_params(c_damp=0.5, x0=0.3, v0=0.0)

        result = simulate_open_closed_loop(params, t_eval=np.linspace(0.0, 300.0, 3000))
        open_area = np.trapezoid(np.abs(result["open_x"]), result["t"])
        closed_area = np.trapezoid(np.abs(result["closed_x"]), result["t"])

        self.assertLess(closed_area, open_area * 0.5)

    def test_controller_trade_study_returns_expected_rows(self):
        params = make_params(c_damp=0.5, x0=0.3, v0=0.0)

        study = controller_trade_study(
            params,
            controllers=("open", "p", "lqr"),
            t_eval=np.linspace(0.0, 120.0, 1200),
            p_gain_scale=0.5,
        )

        rows = {row["controller"]: row for row in study["rows"]}
        self.assertEqual(set(rows), {"open", "p", "lqr"})
        self.assertLess(rows["lqr"]["area_abs_x"], rows["open"]["area_abs_x"])


if __name__ == "__main__":
    unittest.main()

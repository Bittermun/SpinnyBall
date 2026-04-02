import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sgms_anchor_sensitivity import (
    DEFAULT_PROBLEM,
    evaluate_parameter_vector,
    export_sobol_indices_csv,
    run_sobol_sensitivity,
    sample_anchor_problem,
)
from sgms_anchor_v1 import DEFAULT_PARAMS, analytical_metrics


class AnchorSensitivityTests(unittest.TestCase):
    def test_evaluate_parameter_vector_matches_analytical_metrics(self):
        params = DEFAULT_PARAMS.copy()
        vector = np.array([12.0, 0.08, 2e-4, 0.75])

        outputs = evaluate_parameter_vector(vector, base_params=params)
        local = params.copy()
        local.update({"u": 12.0, "g_gain": 0.08, "eps": 2e-4, "lam": 0.75})
        metrics = analytical_metrics(local)

        self.assertAlmostEqual(outputs["k_eff"], metrics["k_eff"])
        self.assertAlmostEqual(outputs["period_s"], metrics["period_s"])
        self.assertAlmostEqual(outputs["static_offset_m"], metrics["static_offset_m"])

    def test_sample_anchor_problem_shape_matches_sobol_formula(self):
        samples = sample_anchor_problem(DEFAULT_PROBLEM, N=16, calc_second_order=False, seed=7)

        self.assertEqual(samples.shape, (16 * (DEFAULT_PROBLEM["num_vars"] + 2), DEFAULT_PROBLEM["num_vars"]))

    def test_k_eff_is_insensitive_to_eps_in_sobol_indices(self):
        result = run_sobol_sensitivity(
            N=128,
            outputs=("k_eff",),
            calc_second_order=False,
            seed=5,
        )
        st = result["indices"]["k_eff"]["ST"]
        names = result["problem"]["names"]
        index = {name: i for i, name in enumerate(names)}

        self.assertLess(abs(st[index["eps"]]), 0.05)
        self.assertGreater(st[index["u"]], 0.2)
        self.assertGreater(st[index["g_gain"]], 0.1)
        self.assertGreater(st[index["lam"]], 0.1)

    def test_export_sobol_indices_csv_writes_rows(self):
        result = run_sobol_sensitivity(
            N=32,
            outputs=("k_eff", "static_offset_m"),
            calc_second_order=False,
            seed=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sobol.csv"
            export_sobol_indices_csv(result["indices"], result["problem"]["names"], path)

            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2 * DEFAULT_PROBLEM["num_vars"])
        self.assertIn("output", rows[0])
        self.assertIn("ST", rows[0])


if __name__ == "__main__":
    unittest.main()

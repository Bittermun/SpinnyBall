import json
import tempfile
import unittest
from pathlib import Path

from sgms_anchor_pipeline import load_experiment_config, run_experiment_suite


class AnchorPipelineTests(unittest.TestCase):
    def test_load_experiment_config_reads_named_experiments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "experiments.json"
            path.write_text(
                json.dumps(
                    {
                        "defaults": {"params": {"u": 10.0}},
                        "experiments": [{"name": "baseline", "params": {"eps": 1e-4}}],
                    }
                ),
                encoding="utf-8",
            )

            config = load_experiment_config(path)

        self.assertIn("defaults", config)
        self.assertEqual(config["experiments"][0]["name"], "baseline")

    def test_run_experiment_suite_writes_manifest_and_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "experiments.json"
            config_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "params": {
                                "u": 10.0,
                                "lam": 0.5,
                                "g_gain": 0.05,
                                "ms": 1000.0,
                                "eps": 1e-4,
                                "c_damp": 4.0,
                                "t_max": 80.0,
                                "x0": 0.1,
                                "v0": 0.0,
                            },
                            "trade_study": {
                                "controllers": ["open", "p", "lqr"],
                                "t_max": 100.0,
                                "num_points": 800,
                                "p_gain_scale": 0.5,
                            },
                            "robustness": {
                                "base_disturbance": {"start": 20.0, "end": 35.0, "force": 0.02},
                                "scenarios": [
                                    {"name": "nominal", "params": {}},
                                    {"name": "imbalance", "params": {"eps": 5e-4}},
                                ],
                                "num_points": 800,
                                "t_max": 100.0,
                            },
                            "sensitivity": {
                                "N": 64,
                                "outputs": ["k_eff", "static_offset_m"],
                                "calc_second_order": False,
                                "seed": 9,
                            },
                        },
                        "experiments": [
                            {
                                "name": "baseline",
                                "params": {"disturbance_theta_std": 0.0},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_root = Path(tmpdir) / "artifacts"

            manifest = run_experiment_suite(config_path, output_root=output_root, run_label="testrun")

            manifest_path = output_root / "testrun" / "manifest.json"
            summary_path = output_root / "testrun" / "baseline" / "summary.json"
            trade_path = output_root / "testrun" / "baseline" / "metrics" / "controller_trade_study.csv"
            robustness_path = output_root / "testrun" / "baseline" / "metrics" / "robustness_summary.csv"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertTrue(trade_path.exists())
            self.assertTrue(robustness_path.exists())
            self.assertEqual(manifest["run_label"], "testrun")
            self.assertEqual(len(manifest["experiments"]), 1)


if __name__ == "__main__":
    unittest.main()

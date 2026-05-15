import json
import tempfile
import unittest
from pathlib import Path

from sgms_anchor_pipeline import run_experiment_suite
from sgms_anchor_report import build_report_payload, write_report_html


class AnchorReportTests(unittest.TestCase):
    def test_build_report_payload_contains_experiment_summaries(self):
        manifest = {
            "run_label": "demo",
            "experiments": [
                {
                    "name": "baseline",
                    "slug": "baseline",
                    "summary_path": "baseline/summary.json",
                    "files": [],
                }
            ],
        }
        summaries = {
            "baseline": {
                "name": "baseline",
                "anchor": {"continuum_metrics": {"k_eff": 2.5}},
                "trade_study": {"rows": [{"controller": "lqr", "area_abs_x": 10.0}]},
            }
        }

        payload = build_report_payload(manifest, summaries)

        self.assertEqual(payload["run_label"], "demo")
        self.assertEqual(payload["experiments"][0]["name"], "baseline")
        self.assertEqual(payload["experiments"][0]["anchor"]["continuum_metrics"]["k_eff"], 2.5)

    def test_write_report_html_embeds_payload_json(self):
        payload = {"run_label": "demo", "experiments": [{"name": "baseline"}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            write_report_html(payload, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("SGMS Anchor Report", text)
        self.assertIn('"run_label": "demo"', text)
        self.assertIn("baseline", text)

    def test_pipeline_writes_report_html(self):
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
                                "eps": 0.0001,
                                "c_damp": 4.0,
                                "t_max": 60.0,
                                "x0": 0.1,
                                "v0": 0.0
                            },
                            "trade_study": {
                                "controllers": ["open", "lqr"],
                                "t_max": 60.0,
                                "num_points": 600,
                                "disturbance": {"start": 10.0, "end": 20.0, "force": 0.01}
                            },
                            "robustness": {
                                "controller": "lqr",
                                "t_max": 60.0,
                                "num_points": 600,
                                "base_disturbance": {"start": 10.0, "end": 20.0, "force": 0.01},
                                "scenarios": [{"name": "nominal", "params": {}}]
                            },
                            "sensitivity": {
                                "N": 32,
                                "outputs": ["k_eff"],
                                "calc_second_order": False,
                                "seed": 1
                            }
                        },
                        "experiments": [{"name": "baseline"}]
                    }
                ),
                encoding="utf-8",
            )
            output_root = Path(tmpdir) / "artifacts"

            manifest = run_experiment_suite(config_path, output_root=output_root, run_label="reportcase")
            report_path = output_root / "reportcase" / "report.html"

            self.assertTrue(report_path.exists())
            self.assertEqual(manifest["report_path"], str(report_path))


if __name__ == "__main__":
    unittest.main()

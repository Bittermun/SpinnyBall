import json
import tempfile
import unittest
from pathlib import Path

from sgms_anchor_dashboard import build_dashboard_payload, write_dashboard_html
from sgms_anchor_pipeline import run_experiment_suite


class AnchorDashboardTests(unittest.TestCase):
    def test_build_dashboard_payload_exposes_profiles_and_experiments(self):
        manifest = {
            "run_label": "demo",
            "experiments": [{"name": "paper-baseline", "slug": "paper-baseline"}],
        }
        summaries = {
            "paper-baseline": {
                "name": "paper-baseline",
                "params": {"u": 10.0, "g_gain": 0.05, "eps": 0.0001, "c_damp": 4.0, "lam": 0.5, "ms": 1000.0, "theta_bias": 0.087},
                "profile": {"name": "paper-baseline", "category": "paper", "notes": ["paper profile"]},
                "trade_study": {"rows": [{"controller": "lqr", "peak_abs_x_m": 0.1}]}
            }
        }

        payload = build_dashboard_payload(manifest, summaries)

        self.assertEqual(payload["run_label"], "demo")
        self.assertEqual(payload["experiments"][0]["profile"]["name"], "paper-baseline")
        self.assertEqual(payload["experiments"][0]["params"]["u"], 10.0)

    def test_write_dashboard_html_embeds_slider_ui(self):
        payload = {"run_label": "demo", "experiments": [{"name": "paper-baseline", "params": {"u": 10.0, "lam": 0.5, "g_gain": 0.05, "eps": 0.0001, "c_damp": 4.0, "ms": 1000.0, "theta_bias": 0.087}}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dashboard.html"
            write_dashboard_html(payload, path)
            text = path.read_text(encoding="utf-8")

        self.assertIn("SGMS Anchor Dashboard", text)
        self.assertIn("uSlider", text)
        self.assertIn('"run_label": "demo"', text)

    def test_pipeline_writes_dashboard_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profiles_path = root / "profiles.json"
            profiles_path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "name": "paper-baseline",
                                "category": "paper",
                                "params": {"u": 10.0, "lam": 0.5, "g_gain": 0.05, "ms": 1000.0, "eps": 0.0001, "c_damp": 4.0, "theta_bias": 0.087, "t_max": 40.0, "x0": 0.1, "v0": 0.0},
                                "provenance": {"u": "memo"},
                                "notes": ["paper profile"]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "experiments.json"
            config_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "profiles_path": str(profiles_path),
                            "trade_study": {"controllers": ["open", "lqr"], "t_max": 40.0, "num_points": 400},
                            "robustness": {"controller": "lqr", "t_max": 40.0, "num_points": 400, "scenarios": [{"name": "nominal", "params": {}}]},
                            "sensitivity": {"N": 16, "outputs": ["k_eff"], "calc_second_order": False, "seed": 1}
                        },
                        "experiments": [{"name": "paper-baseline", "profile": "paper-baseline"}]
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_experiment_suite(config_path, output_root=root / "artifacts", run_label="dashboardrun")
            dashboard_data = root / "artifacts" / "dashboardrun" / "dashboard_data.json"
            dashboard_html = root / "artifacts" / "dashboardrun" / "dashboard.html"

            self.assertTrue(dashboard_data.exists())
            self.assertTrue(dashboard_html.exists())
            self.assertEqual(manifest["dashboard_path"], str(dashboard_html))


if __name__ == "__main__":
    unittest.main()

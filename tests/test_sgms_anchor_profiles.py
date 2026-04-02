import json
import tempfile
import unittest
from pathlib import Path

from sgms_anchor_pipeline import run_experiment_suite
from sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params


class AnchorProfileTests(unittest.TestCase):
    def test_load_anchor_profiles_reads_named_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.json"
            path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "name": "paper-baseline",
                                "category": "paper",
                                "params": {"u": 10.0},
                                "provenance": {"u": "memo"}
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            data = load_anchor_profiles(path)

        self.assertEqual(data["profiles"][0]["name"], "paper-baseline")

    def test_resolve_profile_params_merges_profile_and_overrides(self):
        profiles = {
            "profiles": [
                {
                    "name": "paper-baseline",
                    "category": "paper",
                    "params": {"u": 10.0, "g_gain": 0.05},
                    "provenance": {"u": "memo", "g_gain": "memo"}
                }
            ]
        }

        resolved = resolve_profile_params(
            profiles,
            "paper-baseline",
            overrides={"g_gain": 0.08, "eps": 1e-4},
        )

        self.assertEqual(resolved["params"]["u"], 10.0)
        self.assertEqual(resolved["params"]["g_gain"], 0.08)
        self.assertEqual(resolved["params"]["eps"], 1e-4)
        self.assertEqual(resolved["profile"]["category"], "paper")

    def test_pipeline_profile_resolution_writes_profile_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_path = Path(tmpdir) / "profiles.json"
            profiles_path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "name": "paper-baseline",
                                "category": "paper",
                                "params": {
                                    "u": 10.0,
                                    "lam": 0.5,
                                    "g_gain": 0.05,
                                    "ms": 1000.0,
                                    "eps": 0.0001,
                                    "c_damp": 4.0,
                                    "t_max": 40.0,
                                    "x0": 0.1,
                                    "v0": 0.0
                                },
                                "provenance": {
                                    "u": "memo-baseline",
                                    "lam": "reduced-order assumption"
                                },
                                "notes": ["paper profile"]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path = Path(tmpdir) / "experiments.json"
            config_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "profiles_path": str(profiles_path),
                            "trade_study": {
                                "controllers": ["open", "lqr"],
                                "t_max": 40.0,
                                "num_points": 400
                            },
                            "robustness": {
                                "controller": "lqr",
                                "t_max": 40.0,
                                "num_points": 400,
                                "scenarios": [{"name": "nominal", "params": {}}]
                            },
                            "sensitivity": {
                                "N": 16,
                                "outputs": ["k_eff"],
                                "calc_second_order": False,
                                "seed": 1
                            }
                        },
                        "experiments": [
                            {
                                "name": "paper-run",
                                "profile": "paper-baseline",
                                "params": {"eps": 0.0002}
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_root = Path(tmpdir) / "artifacts"

            manifest = run_experiment_suite(config_path, output_root=output_root, run_label="profilerun")
            summary = json.loads((output_root / "profilerun" / "paper-run" / "summary.json").read_text(encoding="utf-8"))
            profile_csv = output_root / "profilerun" / "profile_summary.csv"

            self.assertTrue(profile_csv.exists())
            self.assertEqual(summary["profile"]["name"], "paper-baseline")
            self.assertEqual(summary["profile"]["category"], "paper")
            self.assertEqual(summary["params"]["eps"], 0.0002)
            self.assertEqual(manifest["experiments"][0]["name"], "paper-run")


if __name__ == "__main__":
    unittest.main()

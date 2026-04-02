import json
import tempfile
import unittest
from pathlib import Path

from sgms_anchor_calibration import load_anchor_calibration, resolve_calibrated_params
from sgms_anchor_claims import load_anchor_claims, resolve_claim_context
from sgms_anchor_pipeline import run_experiment_suite


class AnchorCalibrationTests(unittest.TestCase):
    def test_resolve_calibrated_params_applies_defaults_and_tags(self):
        calibration = {
            "defaults": {
                "c_damp": {
                    "value": 4.0,
                    "status": "placeholder",
                    "source": "reduced-order assumption"
                }
            },
            "profiles": {
                "paper-baseline": {
                    "u": {
                        "value": 10.0,
                        "status": "memo-baseline",
                        "source": "master memo"
                    }
                }
            }
        }

        resolved = resolve_calibrated_params(
            calibration,
            {"lam": 0.5},
            profile_name="paper-baseline",
        )

        self.assertEqual(resolved["params"]["c_damp"], 4.0)
        self.assertEqual(resolved["params"]["u"], 10.0)
        self.assertEqual(resolved["provenance"]["u"]["status"], "memo-baseline")
        self.assertEqual(resolved["provenance"]["c_damp"]["status"], "placeholder")

    def test_resolve_claim_context_returns_phase_decision(self):
        claims = {
            "phase_decision": {
                "high_fidelity_required": False,
                "decision": "defer",
                "claim_level": "L1-reduced-order"
            },
            "profiles": {
                "paper-baseline": {
                    "claim_level": "L1-reduced-order",
                    "intended_use": "paper"
                }
            }
        }

        context = resolve_claim_context(claims, profile_name="paper-baseline")

        self.assertFalse(context["phase_decision"]["high_fidelity_required"])
        self.assertEqual(context["profile_claim"]["claim_level"], "L1-reduced-order")

    def test_pipeline_writes_calibration_and_claim_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profiles_path = root / "profiles.json"
            calibration_path = root / "calibration.json"
            claims_path = root / "claims.json"
            config_path = root / "experiments.json"

            profiles_path.write_text(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "name": "paper-baseline",
                                "category": "paper",
                                "params": {"u": 10.0, "lam": 0.5, "g_gain": 0.05, "ms": 1000.0, "eps": 0.0001, "t_max": 40.0, "x0": 0.1, "v0": 0.0},
                                "provenance": {"u": "memo"},
                                "notes": ["paper profile"]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            calibration_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "c_damp": {"value": 4.0, "status": "placeholder", "source": "reduced-order assumption"},
                            "theta_bias": {"value": 0.087, "status": "memo-baseline", "source": "master memo"}
                        }
                    }
                ),
                encoding="utf-8",
            )
            claims_path.write_text(
                json.dumps(
                    {
                        "phase_decision": {
                            "high_fidelity_required": False,
                            "decision": "defer",
                            "claim_level": "L1-reduced-order"
                        },
                        "profiles": {
                            "paper-baseline": {
                                "claim_level": "L1-reduced-order",
                                "intended_use": "paper"
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "profiles_path": str(profiles_path),
                            "calibration_path": str(calibration_path),
                            "claims_path": str(claims_path),
                            "trade_study": {"controllers": ["open", "lqr"], "t_max": 40.0, "num_points": 400},
                            "robustness": {"controller": "lqr", "t_max": 40.0, "num_points": 400, "scenarios": [{"name": "nominal", "params": {}}]},
                            "sensitivity": {"N": 16, "outputs": ["k_eff"], "calc_second_order": False, "seed": 1}
                        },
                        "experiments": [{"name": "paper-run", "profile": "paper-baseline"}]
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_experiment_suite(config_path, output_root=root / "artifacts", run_label="claimrun")
            summary = json.loads((root / "artifacts" / "claimrun" / "paper-run" / "summary.json").read_text(encoding="utf-8"))

            self.assertIn("calibration", summary)
            self.assertIn("claim_context", summary)
            self.assertEqual(summary["claim_context"]["phase_decision"]["decision"], "defer")
            self.assertEqual(manifest["validation_decision"]["decision"], "defer")


if __name__ == "__main__":
    unittest.main()

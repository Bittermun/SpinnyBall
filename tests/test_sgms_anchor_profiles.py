import json
import tempfile
import unittest
from pathlib import Path

from sgms_anchor_pipeline import run_experiment_suite
from sgms_anchor_profiles import (
    load_anchor_profiles, 
    resolve_profile_params, 
    load_material_catalog, 
    load_geometry_catalog, 
    load_environment_catalog,
    _validate_material_profile,
    _validate_geometry_profile,
    _validate_environment_profile
)


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

    def test_material_profile_resolution(self):
        """Test that material profiles are resolved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create material catalog
            material_path = Path(tmpdir) / "materials.json"
            material_path.write_text(
                json.dumps({
                    "material_profiles": {
                        "test_material": {
                            "name": "Test Material",
                            "k_fp_range": [100000, 150000],
                            "damping_ratio": 0.06,
                            "source": "test"
                        }
                    }
                }),
                encoding="utf-8"
            )
            
            # Create profile with material reference
            profiles = {
                "profiles": [
                    {
                        "name": "test-profile",
                        "category": "test",
                        "material_profile": "test_material",
                        "params": {"u": 10.0}
                    }
                ]
            }
            
            resolved = resolve_profile_params(
                profiles,
                "test-profile",
                material_catalog_path=material_path
            )
            
            self.assertIsNotNone(resolved["profile"]["material_profile"])
            self.assertEqual(resolved["profile"]["material_profile"]["name"], "Test Material")
            self.assertEqual(resolved["profile"]["material_profile"]["k_fp_range"], [100000, 150000])

    def test_material_profile_missing_catalog(self):
        """Test that missing material catalog returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            material_path = Path(tmpdir) / "nonexistent.json"
            catalog = load_material_catalog(material_path)
            self.assertEqual(catalog, {"material_profiles": {}})

    def test_material_profile_unknown_reference(self):
        """Test that unknown material profile raises KeyError."""
        profiles = {
            "profiles": [
                {
                    "name": "test-profile",
                    "category": "test",
                    "material_profile": "unknown_material",
                    "params": {"u": 10.0}
                }
            ]
        }
        
        with self.assertRaises(KeyError) as context:
            resolve_profile_params(profiles, "test-profile")
        
        self.assertIn("Unknown material profile", str(context.exception))

    def test_geometry_profile_resolution(self):
        """Test that geometry profiles are resolved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create geometry catalog
            geometry_path = Path(tmpdir) / "geometry.json"
            geometry_path.write_text(
                json.dumps({
                    "geometry_profiles": {
                        "test_geometry": {
                            "name": "Test Geometry",
                            "shape": "sphere",
                            "mass": 1.0,
                            "radius": 0.05,
                            "inertia_type": "sphere",
                            "description": "test"
                        }
                    }
                }),
                encoding="utf-8"
            )
            
            # Create profile with geometry reference
            profiles = {
                "profiles": [
                    {
                        "name": "test-profile",
                        "category": "test",
                        "geometry_profile": "test_geometry",
                        "params": {"u": 10.0}
                    }
                ]
            }
            
            resolved = resolve_profile_params(
                profiles,
                "test-profile",
                geometry_catalog_path=geometry_path
            )
            
            self.assertIsNotNone(resolved["profile"]["geometry_profile"])
            self.assertEqual(resolved["profile"]["geometry_profile"]["name"], "Test Geometry")
            self.assertEqual(resolved["profile"]["geometry_profile"]["shape"], "sphere")

    def test_geometry_profile_missing_catalog(self):
        """Test that missing geometry catalog returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            geometry_path = Path(tmpdir) / "nonexistent.json"
            catalog = load_geometry_catalog(geometry_path)
            self.assertEqual(catalog, {"geometry_profiles": {}})

    def test_geometry_profile_unknown_reference(self):
        """Test that unknown geometry profile raises KeyError."""
        profiles = {
            "profiles": [
                {
                    "name": "test-profile",
                    "category": "test",
                    "geometry_profile": "unknown_geometry",
                    "params": {"u": 10.0}
                }
            ]
        }
        
        with self.assertRaises(KeyError) as context:
            resolve_profile_params(profiles, "test-profile")
        
        self.assertIn("Unknown geometry profile", str(context.exception))

    def test_environment_profile_resolution(self):
        """Test that environment profiles are resolved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create environment catalog
            environment_path = Path(tmpdir) / "environment.json"
            environment_path.write_text(
                json.dumps({
                    "environment_profiles": {
                        "test_environment": {
                            "name": "Test Environment",
                            "temperature": 77.0,
                            "B_field": 1.0,
                            "radiation_flux": 0.0,
                            "gravity": 9.81,
                            "description": "test"
                        }
                    }
                }),
                encoding="utf-8"
            )
            
            # Create profile with environment reference
            profiles = {
                "profiles": [
                    {
                        "name": "test-profile",
                        "category": "test",
                        "environment_profile": "test_environment",
                        "params": {"u": 10.0}
                    }
                ]
            }
            
            resolved = resolve_profile_params(
                profiles,
                "test-profile",
                environment_catalog_path=environment_path
            )
            
            self.assertIsNotNone(resolved["profile"]["environment_profile"])
            self.assertEqual(resolved["profile"]["environment_profile"]["name"], "Test Environment")
            self.assertEqual(resolved["params"]["temperature"], 77.0)
            self.assertEqual(resolved["params"]["B_field"], 1.0)

    def test_environment_profile_missing_catalog(self):
        """Test that missing environment catalog returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            environment_path = Path(tmpdir) / "nonexistent.json"
            catalog = load_environment_catalog(environment_path)
            self.assertEqual(catalog, {"environment_profiles": {}})

    def test_environment_profile_unknown_reference(self):
        """Test that unknown environment profile raises KeyError."""
        profiles = {
            "profiles": [
                {
                    "name": "test-profile",
                    "category": "test",
                    "environment_profile": "unknown_environment",
                    "params": {"u": 10.0}
                }
            ]
        }
        
        with self.assertRaises(KeyError) as context:
            resolve_profile_params(profiles, "test-profile")
        
        self.assertIn("Unknown environment profile", str(context.exception))

    def test_material_profile_validation_missing_name(self):
        """Test that material profile without name raises ValueError."""
        with self.assertRaises(ValueError) as context:
            _validate_material_profile({"k_fp_range": [100000, 150000]})
        self.assertIn("name", str(context.exception))

    def test_geometry_profile_validation_missing_required_fields(self):
        """Test that geometry profile without required fields raises ValueError."""
        with self.assertRaises(ValueError) as context:
            _validate_geometry_profile({"name": "test"})
        self.assertIn("shape", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            _validate_geometry_profile({"name": "test", "shape": "sphere"})
        self.assertIn("mass", str(context.exception))

    def test_environment_profile_validation_missing_required_fields(self):
        """Test that environment profile without required fields raises ValueError."""
        with self.assertRaises(ValueError) as context:
            _validate_environment_profile({"name": "test"})
        self.assertIn("temperature", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            _validate_environment_profile({"name": "test", "temperature": 77.0})
        self.assertIn("B_field", str(context.exception))

    def test_material_profile_validation_invalid_types(self):
        """Test that material profile with invalid types raises ValueError."""
        with self.assertRaises(ValueError) as context:
            _validate_material_profile({"name": "test", "k_fp_range": [100000, "invalid"]})
        self.assertIn("must be numbers", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            _validate_material_profile({"name": "test", "k_fp_range": [-10000, 100000]})
        self.assertIn("non-negative", str(context.exception))

    def test_geometry_profile_validation_invalid_types(self):
        """Test that geometry profile with invalid types raises ValueError."""
        with self.assertRaises(ValueError) as context:
            _validate_geometry_profile({"name": "test", "shape": "sphere", "mass": "invalid", "radius": 0.1})
        self.assertIn("must be a number", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            _validate_geometry_profile({"name": "test", "shape": "sphere", "mass": -1.0, "radius": 0.1})
        self.assertIn("must be positive", str(context.exception))

    def test_environment_profile_validation_invalid_types(self):
        """Test that environment profile with invalid types raises ValueError."""
        with self.assertRaises(ValueError) as context:
            _validate_environment_profile({"name": "test", "temperature": "invalid", "B_field": 1.0})
        self.assertIn("must be a number", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            _validate_environment_profile({"name": "test", "temperature": -10.0, "B_field": 1.0})
        self.assertIn("non-negative", str(context.exception))

    def test_resolve_profile_with_invalid_material_profile(self):
        """Test that resolving profile with invalid material raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            material_path = Path(tmpdir) / "materials.json"
            material_path.write_text(
                json.dumps({
                    "material_profiles": {
                        "invalid_material": {
                            "k_fp_range": [100000, 150000]  # Missing 'name' field
                        }
                    }
                }),
                encoding="utf-8"
            )
            
            profiles = {
                "profiles": [
                    {
                        "name": "test-profile",
                        "category": "test",
                        "material_profile": "invalid_material",
                        "params": {"u": 10.0}
                    }
                ]
            }
            
            with self.assertRaises(ValueError) as context:
                resolve_profile_params(profiles, "test-profile", material_catalog_path=material_path)
            self.assertIn("name", str(context.exception))

    def test_resolve_profile_with_invalid_geometry_profile(self):
        """Test that resolving profile with invalid geometry raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            geometry_path = Path(tmpdir) / "geometry.json"
            geometry_path.write_text(
                json.dumps({
                    "geometry_profiles": {
                        "invalid_geometry": {
                            "name": "test",
                            "shape": "sphere"  # Missing 'mass' and 'radius'
                        }
                    }
                }),
                encoding="utf-8"
            )
            
            profiles = {
                "profiles": [
                    {
                        "name": "test-profile",
                        "category": "test",
                        "geometry_profile": "invalid_geometry",
                        "params": {"u": 10.0}
                    }
                ]
            }
            
            with self.assertRaises(ValueError) as context:
                resolve_profile_params(profiles, "test-profile", geometry_catalog_path=geometry_path)
            self.assertIn("mass", str(context.exception))

    def test_all_three_profiles_resolution(self):
        """Test that material, geometry, and environment profiles resolve together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create all three catalogs
            material_path = Path(tmpdir) / "materials.json"
            material_path.write_text(
                json.dumps({
                    "material_profiles": {
                        "test_mat": {"name": "Test Mat", "k_fp_range": [100000, 150000]}
                    }
                }),
                encoding="utf-8"
            )
            
            geometry_path = Path(tmpdir) / "geometry.json"
            geometry_path.write_text(
                json.dumps({
                    "geometry_profiles": {
                        "test_geo": {"name": "Test Geo", "shape": "sphere", "mass": 1.0, "radius": 0.05}
                    }
                }),
                encoding="utf-8"
            )
            
            environment_path = Path(tmpdir) / "environment.json"
            environment_path.write_text(
                json.dumps({
                    "environment_profiles": {
                        "test_env": {"name": "Test Env", "temperature": 77.0, "B_field": 1.0}
                    }
                }),
                encoding="utf-8"
            )
            
            # Create profile with all three references
            profiles = {
                "profiles": [
                    {
                        "name": "test-profile",
                        "category": "test",
                        "material_profile": "test_mat",
                        "geometry_profile": "test_geo",
                        "environment_profile": "test_env",
                        "params": {"u": 10.0}
                    }
                ]
            }
            
            resolved = resolve_profile_params(
                profiles,
                "test-profile",
                material_catalog_path=material_path,
                geometry_catalog_path=geometry_path,
                environment_catalog_path=environment_path
            )
            
            # Verify all three profiles resolved
            self.assertIsNotNone(resolved["profile"]["material_profile"])
            self.assertEqual(resolved["profile"]["material_profile"]["name"], "Test Mat")
            self.assertIsNotNone(resolved["profile"]["geometry_profile"])
            self.assertEqual(resolved["profile"]["geometry_profile"]["name"], "Test Geo")
            self.assertIsNotNone(resolved["profile"]["environment_profile"])
            self.assertEqual(resolved["profile"]["environment_profile"]["name"], "Test Env")
            
            # Verify environment params were applied
            self.assertEqual(resolved["params"]["temperature"], 77.0)
            self.assertEqual(resolved["params"]["B_field"], 1.0)

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

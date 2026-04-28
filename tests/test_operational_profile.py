"""
Tests for operational profile with paper targets.
"""

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from sgms_anchor_profiles import load_anchor_profiles, resolve_profile_params
from sgms_anchor_v1 import DEFAULT_PARAMS, analytical_metrics, simulate_anchor


class TestOperationalProfile:
    """Test operational profile with paper targets."""

    def test_operational_profile_exists(self):
        """Test that operational profile is defined in anchor_profiles.json."""
        profiles_path = Path("anchor_profiles.json")
        data = load_anchor_profiles(profiles_path)
        
        profile_names = [p["name"] for p in data["profiles"]]
        assert "operational" in profile_names, "Operational profile not found"

    def test_operational_profile_has_paper_targets(self):
        """Test that operational profile has correct paper target parameters."""
        profiles_path = Path("anchor_profiles.json")
        data = load_anchor_profiles(profiles_path)
        
        operational_profile = next(p for p in data["profiles"] if p["name"] == "operational")
        params = operational_profile["params"]
        
        # Paper targets from documentation
        assert params["u"] == 1600.0, f"Stream velocity should be 1600 m/s, got {params['u']}"
        assert params["mp"] == 8.0, f"Mass should be 8.0 kg, got {params['mp']}"
        assert params["lam"] == 16.6667, f"Linear density should be 16.6667 kg/m, got {params['lam']}"
        assert params["k_fp"] == 6000.0, f"Flux-pinning stiffness should be 6000 N/m, got {params['k_fp']}"
        # g_gain tuned for k_eff ≈ 6000 N/m at u=1600 m/s
        assert params["g_gain"] == 0.00014, f"Control gain should be 0.00014, got {params['g_gain']}"

    def test_resolve_operational_profile(self):
        """Test resolving operational profile parameters."""
        profiles_path = Path("anchor_profiles.json")
        data = load_anchor_profiles(profiles_path)
        
        resolved = resolve_profile_params(data, "operational")
        params = resolved["params"]
        profile_meta = resolved["profile"]
        
        # Check that profile metadata includes new profile types
        assert profile_meta.get("material_profile") is not None, "Material profile should be resolved"
        assert profile_meta.get("geometry_profile") is not None, "Geometry profile should be resolved"
        assert profile_meta.get("environment_profile") is not None, "Environment profile should be resolved"
        
        # Check that operational values override defaults
        assert params["u"] == 1600.0
        assert params["mp"] == 8.0
        assert params["k_fp"] == 6000.0
        
        # Check that defaults are still present where not overridden
        assert params["ms"] == 1000.0  # From DEFAULT_PARAMS
        assert params["theta_bias"] == 0.087  # From DEFAULT_PARAMS

    def test_operational_profile_analytical_metrics(self):
        """Test that analytical_metrics works with operational parameters."""
        profiles_path = Path("anchor_profiles.json")
        data = load_anchor_profiles(profiles_path)
        
        resolved = resolve_profile_params(data, "operational")
        params = resolved["params"]
        
        # Compute analytical metrics
        metrics = analytical_metrics(params)
        
        # Check that metrics are computed without errors
        assert "k_eff" in metrics
        assert "force_per_stream_n" in metrics
        assert "period_s" in metrics
        
        # Check that k_eff is reasonable (should be > 0)
        assert metrics["k_eff"] > 0, "Effective stiffness should be positive"
        
        # With operational parameters, k_eff should be significantly higher than test values
        # (u=1600 m/s, k_fp=6000 N/m vs u=10 m/s, k_fp=0)
        assert metrics["k_eff"] > 1000, f"k_eff should be > 1000 N/m with operational params, got {metrics['k_eff']}"

    def test_operational_profile_simulation(self):
        """Test that simulation runs with operational parameters."""
        profiles_path = Path("anchor_profiles.json")
        data = load_anchor_profiles(profiles_path)
        
        resolved = resolve_profile_params(data, "operational")
        params = resolved["params"]
        
        # Reduce t_max for faster test
        params["t_max"] = 10.0
        
        # Run simulation
        t_eval = np.linspace(0.0, params["t_max"], 200)
        result = simulate_anchor(params, t_eval=t_eval, seed=42)
        
        # Check that simulation completed
        assert result is not None
        assert "t" in result
        assert "x" in result
        assert len(result["t"]) > 0
        
        # Check that metrics are computed
        assert "metrics" in result
        assert result["metrics"]["k_eff"] > 0

    def test_mass_sweep_in_sensitivity_analysis(self):
        """Test that mass (mp) is included in sensitivity analysis problem."""
        from sgms_anchor_sensitivity import DEFAULT_PROBLEM
        
        # Check that mp is in the problem
        assert "mp" in DEFAULT_PROBLEM["names"], "mp should be in sensitivity analysis parameters"
        
        # Check that bounds are reasonable
        mp_index = DEFAULT_PROBLEM["names"].index("mp")
        mp_bounds = DEFAULT_PROBLEM["bounds"][mp_index]
        assert mp_bounds[0] == 0.05, "Lower bound should be 0.05 kg"
        assert mp_bounds[1] == 8.0, "Upper bound should be 8.0 kg"

    def test_sensitivity_analysis_with_mass(self):
        """Test that sensitivity analysis can evaluate mass parameter."""
        from sgms_anchor_sensitivity import evaluate_parameter_vector, DEFAULT_PROBLEM
        
        # Create a test parameter vector (midpoint of bounds)
        mid_vector = np.array([
            (5.0 + 1600.0) / 2,  # u
            (0.02 + 0.2) / 2,    # g_gain
            (0.0 + 1e-3) / 2,    # eps
            (0.1 + 20.0) / 2,    # lam
            (0.05 + 8.0) / 2,    # mp
        ])
        
        # Evaluate
        metrics = evaluate_parameter_vector(mid_vector)
        
        # Check that metrics are computed
        assert "k_eff" in metrics
        assert "force_per_stream_n" in metrics
        
        # Check that mass parameter was used (mp should be in params)
        # This is implicit - if no error occurred, mp was accepted

    def test_operational_vs_baseline_comparison(self):
        """Test that operational profile produces different results than baseline."""
        profiles_path = Path("anchor_profiles.json")
        data = load_anchor_profiles(profiles_path)
        
        # Get baseline parameters
        baseline = resolve_profile_params(data, "paper-baseline")
        baseline_params = baseline["params"]
        
        # Get operational parameters
        operational = resolve_profile_params(data, "operational")
        operational_params = operational["params"]
        
        # Compute metrics for both
        baseline_metrics = analytical_metrics(baseline_params)
        operational_metrics = analytical_metrics(operational_params)
        
        # Operational should have higher k_eff (due to k_fp=6000 vs 0)
        assert operational_metrics["k_eff"] > baseline_metrics["k_eff"], \
            "Operational profile should have higher effective stiffness"
        
        # Operational should have higher force per stream (due to u=1600 vs 10)
        assert operational_metrics["force_per_stream_n"] > baseline_metrics["force_per_stream_n"], \
            "Operational profile should have higher force per stream"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

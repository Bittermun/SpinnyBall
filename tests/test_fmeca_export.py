"""
Tests for FMECA JSON export functionality.
"""

import pytest
from sgms_anchor_pipeline import export_fmeca_json


class TestFMECAExport:
    """Test FMECA risk matrix export."""

    def test_fmeca_export_structure(self):
        """Test that FMECA export has correct structure."""
        results = {
            "anchor": {
                "continuum_metrics": {
                    "omega_initial": 1.0,
                    "omega_final": 0.95,
                    "eta_ind": 0.85,
                    "temperature_max": 350.0,
                    "stress_max": 0.5e9,
                    "max_displacement": 0.05,
                }
            },
            "trade_study": {
                "rows": [
                    {"max_latency_ms": 5.0},
                    {"max_latency_ms": 8.0},
                ]
            }
        }

        fmeca = export_fmeca_json(results)

        # Check structure
        assert "schema_version" in fmeca
        assert "generated_at" in fmeca
        assert "failure_modes" in fmeca
        assert "kill_criteria" in fmeca

        # Check failure modes
        assert "FM-01" in fmeca["failure_modes"]
        assert "FM-06" in fmeca["failure_modes"]
        assert "FM-09" in fmeca["failure_modes"]
        assert "FM-12" in fmeca["failure_modes"]
        assert "FM-15" in fmeca["failure_modes"]

    def test_fmeca_mode_fields(self):
        """Test that each failure mode has required fields."""
        results = {
            "anchor": {
                "continuum_metrics": {
                    "omega_initial": 1.0,
                    "omega_final": 0.95,
                    "eta_ind": 0.85,
                    "temperature_max": 350.0,
                    "stress_max": 0.5e9,
                }
            },
            "trade_study": {"rows": [{"max_latency_ms": 5.0}]}
        }

        fmeca = export_fmeca_json(results)

        for mode_id, mode_data in fmeca["failure_modes"].items():
            assert "mode" in mode_data
            assert "description" in mode_data
            assert "severity" in mode_data
            assert "probability" in mode_data
            assert "risk" in mode_data
            assert "status" in mode_data

            # Check severity is between 1-10
            assert 1 <= mode_data["severity"] <= 10

            # Check probability is between 0-1
            assert 0 <= mode_data["probability"] <= 1

            # Check status is valid
            assert mode_data["status"] in ["PASS", "FAIL", "WARNING"]

    def test_kill_criteria_flags(self):
        """Test that kill criteria flags are computed correctly."""
        results = {
            "anchor": {
                "continuum_metrics": {
                    "omega_initial": 1.0,
                    "omega_final": 0.90,  # 10% decay - should trigger
                    "eta_ind": 0.80,  # Below threshold - should trigger
                    "temperature_max": 460.0,  # Above limit - should trigger
                    "stress_max": 0.9e9,  # Above limit - should trigger
                    "max_displacement": 0.15,  # Above limit - should trigger
                }
            },
            "trade_study": {"rows": [{"max_latency_ms": 5.0}]}
        }

        fmeca = export_fmeca_json(results)

        kill_criteria = fmeca["kill_criteria"]

        # Check individual flags
        assert kill_criteria["energy_dissipation_exceeded"] == True
        assert kill_criteria["misalignment_exceeded"] == True
        assert kill_criteria["induction_failed"] == True
        assert kill_criteria["thermal_limit_exceeded"] == True
        assert kill_criteria["stress_limit_exceeded"] == True
        assert kill_criteria["any_kill_criteria"] == True

    def test_passing_case(self):
        """Test FMECA with all passing metrics."""
        results = {
            "anchor": {
                "continuum_metrics": {
                    "omega_initial": 1.0,
                    "omega_final": 0.99,  # Only 1% decay - pass
                    "eta_ind": 0.90,  # Above threshold - pass
                    "temperature_max": 350.0,  # Below limit - pass
                    "stress_max": 0.5e9,  # Below limit - pass
                    "max_displacement": 0.05,  # Below limit - pass
                }
            },
            "trade_study": {"rows": [{"max_latency_ms": 5.0}]}
        }

        fmeca = export_fmeca_json(results)

        kill_criteria = fmeca["kill_criteria"]

        # All should pass
        assert kill_criteria["energy_dissipation_exceeded"] == False
        assert kill_criteria["misalignment_exceeded"] == False
        assert kill_criteria["induction_failed"] == False
        assert kill_criteria["thermal_limit_exceeded"] == False
        assert kill_criteria["stress_limit_exceeded"] == False
        assert kill_criteria["any_kill_criteria"] == False

        # Check failure mode statuses
        assert fmeca["failure_modes"]["FM-01"]["status"] == "PASS"
        assert fmeca["failure_modes"]["FM-06"]["status"] == "PASS"
        assert fmeca["failure_modes"]["FM-09"]["status"] == "PASS"
        assert fmeca["failure_modes"]["FM-12"]["status"] == "PASS"
        assert fmeca["failure_modes"]["FM-15"]["status"] == "PASS"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

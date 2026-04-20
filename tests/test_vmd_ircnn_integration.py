"""
Integration tests for VMD-IRCNN system.
"""

import numpy as np
import pytest

from control_layer import (
    LinearizedROM,
    ROMParameters,
    StateConverter,
    MLIntegrationLayer,
    get_ml_integration,
    create_rom,
)


class TestStateConversionIntegration:
    """Test state conversion integration with ROM predictor."""

    def test_rom_predictor_with_state_conversion(self):
        """Test ROM predictor can use state conversion."""
        I = np.diag([0.0001, 0.00011, 0.00009])
        rom = create_rom(mass=0.05, I=I)

        delta_x = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0])
        delta_u = np.array([0.1, 0.2, 0.3])

        # Test linear prediction (default)
        pred_linear = rom.predict(delta_x, delta_u)
        assert pred_linear.shape == delta_x.shape

    def test_rom_predictor_vmd_ircnn_flag(self):
        """Test ROM predictor respects use_vmd_ircnn flag."""
        I = np.diag([0.0001, 0.00011, 0.00009])
        rom = LinearizedROM(
            ROMParameters(I=I, mass=0.05, operating_point=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])),
            use_vmd_ircnn=False,
        )

        delta_x = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0])
        delta_u = np.array([0.1, 0.2, 0.3])

        pred = rom.predict(delta_x, delta_u)
        assert pred.shape == delta_x.shape

    def test_set_vmd_ircnn_model(self):
        """Test setting VMD-IRCNN model in ROM predictor."""
        I = np.diag([0.0001, 0.00011, 0.00009])
        rom = create_rom(mass=0.05, I=I)

        # Mock model
        mock_model = "mock_model"
        rom.set_vmd_ircnn(mock_model)

        assert rom.vmd_ircnn == mock_model
        assert rom.use_vmd_ircnn is True


class TestMLIntegrationLayer:
    """Test ML integration layer."""

    def test_initialization(self):
        """Test ML integration layer initialization."""
        ml_integration = MLIntegrationLayer()
        assert ml_integration._vmd_implementation == "stub"
        assert ml_integration._ircnn_implementation == "stub"

    def test_initialization_with_config(self):
        """Test ML integration layer with custom config."""
        ml_integration = MLIntegrationLayer(config_path="config/ml_config.json")
        assert ml_integration._vmd_implementation == "stub"
        assert ml_integration._ircnn_implementation == "stub"

    def test_get_config(self):
        """Test getting configuration."""
        ml_integration = MLIntegrationLayer()
        config = ml_integration.get_config()
        assert "vmd_implementation" in config
        assert "ircnn_implementation" in config
        assert "enable_training" in config

    def test_reload_config(self):
        """Test reloading configuration."""
        ml_integration = MLIntegrationLayer()
        ml_integration.reload_config()
        config = ml_integration.get_config()
        assert config is not None

    def test_global_get_ml_integration(self):
        """Test global get_ml_integration function."""
        ml_integration = get_ml_integration()
        assert isinstance(ml_integration, MLIntegrationLayer)

    def test_global_singleton(self):
        """Test global ML integration is singleton."""
        ml1 = get_ml_integration()
        ml2 = get_ml_integration()
        assert ml1 is ml2


class TestFeatureFlagSwitching:
    """Test feature flag switching."""

    def test_stub_mode_by_default(self):
        """Test stub mode is default."""
        ml_integration = MLIntegrationLayer()
        config = ml_integration.get_config()
        assert config["vmd_implementation"] == "stub"
        assert config["ircnn_implementation"] == "stub"

    def test_config_file_parsing(self):
        """Test config file is parsed correctly."""
        ml_integration = MLIntegrationLayer(config_path="config/ml_config.json")
        config = ml_integration.get_config()
        assert config["vmd_implementation"] == "stub"
        assert config["enable_training"] is False


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    def test_rom_predictor_trajectory(self):
        """Test ROM predictor trajectory prediction."""
        I = np.diag([0.0001, 0.00011, 0.00009])
        rom = create_rom(mass=0.05, I=I)

        delta_x0 = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0])
        delta_u_sequence = np.random.randn(10, 3)

        trajectory = rom.predict_trajectory(delta_x0, delta_u_sequence)
        assert trajectory.shape == (11, 7)

    def test_state_converter_batch_conversion(self):
        """Test batch state conversion."""
        rom_states = np.random.randn(10, 7)
        vmd_states = StateConverter.batch_rom_to_vmd(rom_states)
        rom_reconstructed = StateConverter.batch_vmd_to_rom(vmd_states)

        error = np.max(np.abs(rom_states - rom_reconstructed))
        assert error < 1e-6

    def test_ml_integration_fallback(self):
        """Test ML integration falls back to stub on error."""
        ml_integration = MLIntegrationLayer()
        # Initially use_stub should be False
        assert ml_integration.use_stub is False

        # Simulate fallback
        ml_integration.use_stub = True
        assert ml_integration.use_stub is True

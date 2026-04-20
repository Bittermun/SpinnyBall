"""
Unit tests for state converter.
"""

import numpy as np
import pytest

from control_layer.state_converter import StateConverter


class TestStateConverter:
    """Test state converter implementation."""

    def test_rom_to_vmd_shape(self):
        """Test ROM to VMD conversion preserves shape."""
        rom_state = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0])
        vmd_state = StateConverter.rom_to_vmd(rom_state)
        assert vmd_state.shape == rom_state.shape

    def test_vmd_to_rom_shape(self):
        """Test VMD to ROM conversion preserves shape."""
        vmd_state = np.array([0.9, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        rom_state = StateConverter.vmd_to_rom(vmd_state)
        assert rom_state.shape == vmd_state.shape

    def test_rom_to_vmd_ordering(self):
        """Test ROM to VMD conversion reorders quaternion components."""
        rom_state = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0])
        vmd_state = StateConverter.rom_to_vmd(rom_state)
        
        # ROM: [qx, qy, qz, qw, ωx, ωy, ωz]
        # VMD: [qw, qx, qy, qz, ωx, ωy, ωz]
        assert vmd_state[0] == rom_state[3]  # qw
        assert vmd_state[1] == rom_state[0]  # qx
        assert vmd_state[2] == rom_state[1]  # qy
        assert vmd_state[3] == rom_state[2]  # qz
        assert np.array_equal(vmd_state[4:], rom_state[4:])  # omega

    def test_vmd_to_rom_ordering(self):
        """Test VMD to ROM conversion reorders quaternion components."""
        vmd_state = np.array([0.9, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        rom_state = StateConverter.vmd_to_rom(vmd_state)
        
        # VMD: [qw, qx, qy, qz, ωx, ωy, ωz]
        # ROM: [qx, qy, qz, qw, ωx, ωy, ωz]
        assert rom_state[0] == vmd_state[1]  # qx
        assert rom_state[1] == vmd_state[2]  # qy
        assert rom_state[2] == vmd_state[3]  # qz
        assert rom_state[3] == vmd_state[0]  # qw
        assert np.array_equal(rom_state[4:], vmd_state[4:])  # omega

    def test_round_trip_conversion(self):
        """Test round-trip conversion preserves state (error < 1e-6)."""
        rom_state = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0])
        vmd_state = StateConverter.rom_to_vmd(rom_state)
        rom_reconstructed = StateConverter.vmd_to_rom(vmd_state)
        
        error = np.max(np.abs(rom_state - rom_reconstructed))
        assert error < 1e-6, f"Round-trip conversion error {error:.2e} exceeds 1e-6"

    def test_validate_conversion_error(self):
        """Test validate_conversion_error method."""
        rom_state = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0])
        error = StateConverter.validate_conversion_error(rom_state)
        assert error < 1e-6

    def test_batch_rom_to_vmd_shape(self):
        """Test batch ROM to VMD conversion preserves shape."""
        rom_states = np.random.randn(10, 7)
        vmd_states = StateConverter.batch_rom_to_vmd(rom_states)
        assert vmd_states.shape == rom_states.shape

    def test_batch_vmd_to_rom_shape(self):
        """Test batch VMD to ROM conversion preserves shape."""
        vmd_states = np.random.randn(10, 7)
        rom_states = StateConverter.batch_vmd_to_rom(vmd_states)
        assert rom_states.shape == vmd_states.shape

    def test_batch_round_trip_conversion(self):
        """Test batch round-trip conversion preserves states (error < 1e-6)."""
        rom_states = np.random.randn(10, 7)
        vmd_states = StateConverter.batch_rom_to_vmd(rom_states)
        rom_reconstructed = StateConverter.batch_vmd_to_rom(vmd_states)
        
        error = np.max(np.abs(rom_states - rom_reconstructed))
        assert error < 1e-6, f"Batch round-trip conversion error {error:.2e} exceeds 1e-6"

    def test_different_batch_sizes(self):
        """Test batch conversion with different batch sizes."""
        for batch_size in [1, 5, 10, 20]:
            rom_states = np.random.randn(batch_size, 7)
            vmd_states = StateConverter.batch_rom_to_vmd(rom_states)
            rom_reconstructed = StateConverter.batch_vmd_to_rom(vmd_states)
            
            error = np.max(np.abs(rom_states - rom_reconstructed))
            assert error < 1e-6, f"Batch size {batch_size} error {error:.2e} exceeds 1e-6"

    def test_zero_state(self):
        """Test conversion with zero state."""
        rom_state = np.zeros(7)
        vmd_state = StateConverter.rom_to_vmd(rom_state)
        rom_reconstructed = StateConverter.vmd_to_rom(vmd_state)
        
        np.testing.assert_array_almost_equal(rom_state, rom_reconstructed)

    def test_large_values(self):
        """Test conversion with large values."""
        rom_state = np.array([1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3])
        vmd_state = StateConverter.rom_to_vmd(rom_state)
        rom_reconstructed = StateConverter.vmd_to_rom(vmd_state)
        
        np.testing.assert_array_almost_equal(rom_state, rom_reconstructed)

    def test_negative_values(self):
        """Test conversion with negative values."""
        rom_state = np.array([-0.1, -0.2, -0.3, -0.9, -1.0, -2.0, -3.0])
        vmd_state = StateConverter.rom_to_vmd(rom_state)
        rom_reconstructed = StateConverter.vmd_to_rom(vmd_state)
        
        np.testing.assert_array_almost_equal(rom_state, rom_reconstructed)

    def test_quaternion_normalization_preserved(self):
        """Test that quaternion normalization is preserved through conversion."""
        # Create normalized quaternion
        q = np.array([0.1, 0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        
        rom_state = np.concatenate([q[1:], [q[0]], [1.0, 2.0, 3.0]])
        vmd_state = StateConverter.rom_to_vmd(rom_state)
        rom_reconstructed = StateConverter.vmd_to_rom(vmd_state)
        
        # Check quaternion is still normalized
        q_reconstructed = np.array([rom_reconstructed[3], rom_reconstructed[0], rom_reconstructed[1], rom_reconstructed[2]])
        norm = np.linalg.norm(q_reconstructed)
        assert abs(norm - 1.0) < 1e-6, f"Quaternion norm {norm:.2e} not preserved"

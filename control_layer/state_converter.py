"""
State converter for ROM ↔ VMD-IRCNN compatibility.

Converts between ROM state representation (linearized deviations)
and VMD-IRCNN state representation (full quaternions).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StateConverter:
    """
    Convert between ROM and VMD-IRCNN state representations.

    ROM: Uses linearized state deviations [δqx, δqy, δqz, δqw, δωx, δωy, δωz]
    VMD-IRCNN: Uses full quaternion [qw, qx, qy, qz] + angular velocity [ωx, ωy, ωz]
    """

    @staticmethod
    def rom_to_vmd(rom_state: np.ndarray) -> np.ndarray:
        """
        Convert ROM state to VMD-IRCNN state.

        Args:
            rom_state: ROM state [qx, qy, qz, qw, ωx, ωy, ωz] (scalar-last)

        Returns:
            vmd_state: VMD-IRCNN state [qw, qx, qy, qz, ωx, ωy, ωz] (scalar-first)
        """
        qx, qy, qz, qw = rom_state[:4]
        omega = rom_state[4:]
        return np.concatenate([[qw], [qx], [qy], [qz], omega])

    @staticmethod
    def vmd_to_rom(vmd_state: np.ndarray) -> np.ndarray:
        """
        Convert VMD-IRCNN state to ROM state.

        Args:
            vmd_state: VMD-IRCNN state [qw, qx, qy, qz, ωx, ωy, ωz] (scalar-first)

        Returns:
            rom_state: ROM state [qx, qy, qz, qw, ωx, ωy, ωz] (scalar-last)
        """
        qw, qx, qy, qz = vmd_state[:4]
        omega = vmd_state[4:]
        return np.concatenate([[qx], [qy], [qz], [qw], omega])

    @staticmethod
    def batch_rom_to_vmd(rom_states: np.ndarray) -> np.ndarray:
        """
        Batch convert ROM states to VMD-IRCNN states.

        Args:
            rom_states: ROM states [batch × 7]

        Returns:
            vmd_states: VMD-IRCNN states [batch × 7]
        """
        qx = rom_states[:, 0]
        qy = rom_states[:, 1]
        qz = rom_states[:, 2]
        qw = rom_states[:, 3]
        omega = rom_states[:, 4:]
        
        vmd_states = np.stack([qw, qx, qy, qz], axis=1)
        vmd_states = np.concatenate([vmd_states, omega], axis=1)
        return vmd_states

    @staticmethod
    def batch_vmd_to_rom(vmd_states: np.ndarray) -> np.ndarray:
        """
        Batch convert VMD-IRCNN states to ROM states.

        Args:
            vmd_states: VMD-IRCNN states [batch × 7]

        Returns:
            rom_states: ROM states [batch × 7]
        """
        qw = vmd_states[:, 0]
        qx = vmd_states[:, 1]
        qy = vmd_states[:, 2]
        qz = vmd_states[:, 3]
        omega = vmd_states[:, 4:]
        
        rom_states = np.stack([qx, qy, qz, qw], axis=1)
        rom_states = np.concatenate([rom_states, omega], axis=1)
        return rom_states

    @staticmethod
    def validate_conversion_error(rom_state: np.ndarray, tolerance: float = 1e-6) -> float:
        """
        Validate round-trip conversion error.

        Args:
            rom_state: Original ROM state
            tolerance: Maximum acceptable error

        Returns:
            Conversion error (max absolute difference)
        """
        # ROM -> VMD -> ROM
        vmd_state = StateConverter.rom_to_vmd(rom_state)
        rom_reconstructed = StateConverter.vmd_to_rom(vmd_state)
        
        error = np.max(np.abs(rom_state - rom_reconstructed))
        return error

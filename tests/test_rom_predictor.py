"""
Unit tests for ROM predictor.
"""

from __future__ import annotations

import numpy as np
import pytest

from control_layer.rom_predictor import (
    LinearizedROM,
    ROMParameters,
    create_rom,
    SYMPY_AVAILABLE,
)


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="sympy not available")
class TestROMParameters:
    """Test ROMParameters dataclass."""
    
    def test_initialization(self):
        """ROMParameters initializes correctly."""
        I = np.diag([0.01, 0.02, 0.03])
        mass = 0.05
        operating_point = np.zeros(7)
        
        params = ROMParameters(I=I, mass=mass, operating_point=operating_point)
        
        assert np.allclose(params.I, I)
        assert params.mass == mass
        assert np.allclose(params.operating_point, operating_point)


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="sympy not available")
class TestLinearizedROM:
    """Test LinearizedROM class."""
    
    def test_initialization(self):
        """LinearizedROM initializes correctly."""
        I = np.diag([0.01, 0.02, 0.03])
        mass = 0.05
        operating_point = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        params = ROMParameters(I=I, mass=mass, operating_point=operating_point)
        rom = LinearizedROM(params)
        
        assert rom.params == params
        assert rom.A is not None
        assert rom.B is not None
        assert rom.A.shape == (7, 7)
        assert rom.B.shape == (7, 3)
    
    def test_predict_single_step(self):
        """Test single-step prediction."""
        I = np.diag([0.01, 0.02, 0.03])
        mass = 0.05
        operating_point = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        params = ROMParameters(I=I, mass=mass, operating_point=operating_point)
        rom = LinearizedROM(params)
        
        delta_x = np.random.randn(7)
        delta_u = np.random.randn(3)
        
        delta_x_next = rom.predict(delta_x, delta_u, dt=0.01)
        
        assert delta_x_next.shape == (7,)
        assert not np.allclose(delta_x_next, delta_x)  # Should change
    
    def test_predict_trajectory(self):
        """Test trajectory prediction."""
        I = np.diag([0.01, 0.02, 0.03])
        mass = 0.05
        operating_point = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        
        params = ROMParameters(I=I, mass=mass, operating_point=operating_point)
        rom = LinearizedROM(params)
        
        delta_x0 = np.random.randn(7)
        delta_u_sequence = np.random.randn(10, 3)
        
        trajectory = rom.predict_trajectory(delta_x0, delta_u_sequence, dt=0.01)
        
        assert trajectory.shape == (11, 7)  # 10 steps + initial
        assert np.allclose(trajectory[0, :], delta_x0)


@pytest.mark.skipif(not SYMPY_AVAILABLE, reason="sympy not available")
class TestCreateROM:
    """Test ROM factory function."""
    
    def test_create_rom_default_operating_point(self):
        """Create ROM with default operating point."""
        I = np.diag([0.01, 0.02, 0.03])
        mass = 0.05
        
        rom = create_rom(mass, I)
        
        assert rom is not None
        assert rom.A is not None
        assert rom.B is not None
        # Default operating point should be identity quaternion + zero spin
        expected_op = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        assert np.allclose(rom.params.operating_point, expected_op)
    
    def test_create_rom_custom_operating_point(self):
        """Create ROM with custom operating point."""
        I = np.diag([0.01, 0.02, 0.03])
        mass = 0.05
        operating_point = np.array([0.1, 0.2, 0.3, 0.9, 10.0, 5.0, 2.0])
        
        rom = create_rom(mass, I, operating_point=operating_point)
        
        assert np.allclose(rom.params.operating_point, operating_point)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

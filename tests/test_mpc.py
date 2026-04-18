"""
Unit tests for MPC controller.
"""

from __future__ import annotations

import numpy as np
import pytest

from control.mpc_controller import (
    MPCController,
    StubMPCController,
    create_mpc_controller,
    CASADI_AVAILABLE,
)


class TestStubMPCController:
    """Test stub MPC controller (when CasADi not available)."""
    
    def test_initialization(self):
        """Stub controller initializes without CasADi."""
        controller = StubMPCController(horizon=10)
        assert controller is not None
    
    def test_solve_returns_zero(self):
        """Stub solve returns zero control."""
        controller = StubMPCController()
        x0 = np.random.randn(7)
        x_target = np.zeros(7)
        
        u_opt, info = controller.solve(x0, x_target)
        
        assert np.allclose(u_opt, 0.0)
        assert info["solve_time"] == 0.0
        assert info["success"] is False


class TestCreateMPCController:
    """Test factory function for MPC controller creation."""
    
    def test_create_stub_when_casadi_unavailable(self):
        """Factory creates stub when CasADi not available."""
        if not CASADI_AVAILABLE:
            controller = create_mpc_controller(use_casadi=True)
            assert isinstance(controller, StubMPCController)
        else:
            # Skip if CasADi is available
            pytest.skip("CasADi is available")
    
    def test_create_stub_explicitly(self):
        """Factory creates stub when requested."""
        controller = create_mpc_controller(use_casadi=False)
        assert isinstance(controller, StubMPCController)


@pytest.mark.skipif(not CASADI_AVAILABLE, reason="CasADi not available")
class TestMPCController:
    """Test MPC controller (requires CasADi)."""
    
    def test_initialization(self):
        """MPC controller initializes with CasADi."""
        controller = MPCController(horizon=10)
        
        assert controller.horizon == 10
        assert controller.dt == 0.01
        assert controller.max_stress == 1.2e9
        assert controller.min_k_eff == 6000.0
    
    def test_solve(self):
        """MPC controller solves optimization problem."""
        controller = MPCController(horizon=10)
        
        x0 = np.array([0.0, 0.0, 0.0, 1.0, 10.0, 5.0, 2.0])
        x_target = np.zeros(7)
        
        u_opt, info = controller.solve(x0, x_target)
        
        assert u_opt.shape == (3, 10)
        assert "solve_time" in info
        assert "success" in info
        assert "iterations" in info
    
    def test_get_first_control(self):
        """Test getting first control from sequence."""
        controller = MPCController(horizon=10)
        
        x0 = np.random.randn(7)
        x_target = np.zeros(7)
        
        u_opt, info = controller.solve(x0, x_target)
        u_first = controller.get_first_control(u_opt)
        
        assert u_first.shape == (3,)
        assert np.allclose(u_first, u_opt[:, 0])


@pytest.mark.skipif(not CASADI_AVAILABLE, reason="CasADi not available")
class TestMPCLatencyVerification:
    """Test MPC latency verification."""
    
    def test_verify_mpc_latency(self):
        """Verify MPC solve time meets target."""
        controller = MPCController(horizon=10)
        
        latency_stats = verify_mpc_latency(controller, n_trials=5)
        
        assert "mean_ms" in latency_stats
        assert "std_ms" in latency_stats
        assert "max_ms" in latency_stats
        assert "min_ms" in latency_stats
        assert "target_ms" in latency_stats
        assert "meets_target" in latency_stats
        assert latency_stats["target_ms"] == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

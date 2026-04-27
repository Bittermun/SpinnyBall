"""
Unit tests for Monte-Carlo cascade runner and pass/fail gates.
"""

from __future__ import annotations

import numpy as np
import pytest

from monte_carlo.cascade_runner import (
    CascadeRunner,
    MonteCarloConfig,
    Perturbation,
    PerturbationType,
    RealizationResult,
    create_default_perturbations,
)
from monte_carlo.pass_fail_gates import (
    PassFailGate,
    GateStatus,
    GateResult,
    GateSet,
    InductionEfficiencyGate,
    StressGate,
    StiffnessGate,
    CascadeProbabilityGate,
    create_default_gate_set,
    evaluate_monte_carlo_gates,
)


class TestPerturbation:
    """Test Perturbation dataclass."""
    
    def test_initialization(self):
        """Perturbation initializes correctly."""
        perturbation = Perturbation(
            type=PerturbationType.DEBRIS_IMPACT,
            magnitude=0.1,
            direction=np.array([1.0, 0.0, 0.0]),
            probability=0.5,
        )
        
        assert perturbation.type == PerturbationType.DEBRIS_IMPACT
        assert perturbation.magnitude == 0.1
        assert np.allclose(perturbation.direction, np.array([1.0, 0.0, 0.0]))
        assert perturbation.probability == 0.5


class TestMonteCarloConfig:
    """Test MonteCarloConfig dataclass."""
    
    def test_initialization(self):
        """MonteCarloConfig initializes correctly."""
        config = MonteCarloConfig(
            n_realizations=100,
            time_horizon=5.0,
            dt=0.01,
            random_seed=42,
        )
        
        assert config.n_realizations == 100
        assert config.time_horizon == 5.0
        assert config.dt == 0.01
        assert config.random_seed == 42
        assert len(config.perturbations) == 0


class TestCascadeRunner:
    """Test CascadeRunner class."""
    
    def test_initialization(self):
        """CascadeRunner initializes correctly."""
        config = MonteCarloConfig(n_realizations=10, random_seed=42)
        runner = CascadeRunner(config)
        
        assert runner.config == config
        assert "eta_ind" in runner.config.pass_fail_gates
        assert "stress" in runner.config.pass_fail_gates
        assert "k_eff" in runner.config.pass_fail_gates
    
    def test_apply_perturbation_debris(self):
        """Test applying debris impact perturbation."""
        from dynamics.multi_body import Packet
        from dynamics.rigid_body import RigidBody
        
        config = MonteCarloConfig(n_realizations=10)
        runner = CascadeRunner(config)
        
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        body = RigidBody(mass, I)
        packet = Packet(id=0, body=body)
        
        perturbation = Perturbation(
            type=PerturbationType.DEBRIS_IMPACT,
            magnitude=0.1,
            direction=np.array([1.0, 0.0, 0.0]),
            probability=1.0,
        )
        
        initial_velocity = packet.body.velocity.copy()
        runner.apply_perturbation(packet, perturbation)
        
        # Velocity should change
        assert not np.allclose(packet.body.velocity, initial_velocity)
    
    def test_apply_perturbation_thermal(self):
        """Test applying thermal transient perturbation."""
        from dynamics.multi_body import Packet
        from dynamics.rigid_body import RigidBody
        
        config = MonteCarloConfig(n_realizations=10)
        runner = CascadeRunner(config)
        
        mass = 0.05
        I = np.diag([0.0001, 0.00011, 0.00009])
        body = RigidBody(mass, I)
        packet = Packet(id=0, body=body)
        
        perturbation = Perturbation(
            type=PerturbationType.THERMAL_TRANSIENT,
            magnitude=0.2,
            probability=1.0,
        )
        
        initial_eta = packet.eta_ind
        runner.apply_perturbation(packet, perturbation)
        
        # eta_ind should decrease
        assert packet.eta_ind < initial_eta


class TestCreateDefaultPerturbations:
    """Test default perturbation creation."""
    
    def test_create_default_perturbations(self):
        """Default perturbations created correctly."""
        perturbations = create_default_perturbations()
        
        assert len(perturbations) == 3
        perturbation_types = [p.type for p in perturbations]
        assert PerturbationType.DEBRIS_IMPACT in perturbation_types
        assert PerturbationType.THERMAL_TRANSIENT in perturbation_types
        assert PerturbationType.MAGNETIC_NOISE in perturbation_types


class TestPassFailGate:
    """Test PassFailGate class."""
    
    def test_initialization(self):
        """PassFailGate initializes correctly."""
        gate = PassFailGate(
            name="test_gate",
            threshold=1.0,
            comparison=">=",
        )
        
        assert gate.name == "test_gate"
        assert gate.threshold == 1.0
        assert gate.comparison == ">="
    
    def test_evaluate_pass(self):
        """Test gate evaluation with passing value."""
        gate = PassFailGate(name="test", threshold=1.0, comparison=">=")
        result = gate.evaluate(2.0)
        
        assert result.status == GateStatus.PASS
        assert result.value == 2.0
        assert result.threshold == 1.0
    
    def test_evaluate_fail(self):
        """Test gate evaluation with failing value."""
        gate = PassFailGate(name="test", threshold=1.0, comparison=">=")
        result = gate.evaluate(0.5)
        
        assert result.status == GateStatus.FAIL
        assert result.value == 0.5
    
    def test_evaluate_warning(self):
        """Test gate evaluation with warning threshold (near-fail condition)."""
        gate = PassFailGate(
            name="test",
            threshold=1.0,
            comparison=">=",
            warning_threshold=1.1,  # Warning when close to threshold
        )
        result = gate.evaluate(1.05)  # Just above threshold but below warning threshold
        
        assert result.status == GateStatus.WARNING


class TestSpecificGates:
    """Test specific gate implementations."""
    
    def test_induction_efficiency_gate(self):
        """Test InductionEfficiencyGate."""
        gate = InductionEfficiencyGate()
        
        result_pass = gate.evaluate(0.9)
        assert result_pass.status == GateStatus.PASS
        
        result_fail = gate.evaluate(0.8)
        assert result_fail.status == GateStatus.FAIL
    
    def test_stress_gate(self):
        """Test StressGate."""
        gate = StressGate()
        
        result_pass = gate.evaluate(0.5e9)  # 500 MPa
        assert result_pass.status == GateStatus.PASS
        
        result_fail = gate.evaluate(1.0e9)  # 1 GPa > 800 MPa limit
        assert result_fail.status == GateStatus.FAIL
    
    def test_stiffness_gate(self):
        """Test StiffnessGate."""
        gate = StiffnessGate()
        
        result_pass = gate.evaluate(7000.0)
        assert result_pass.status == GateStatus.PASS
        
        result_fail = gate.evaluate(5000.0)
        assert result_fail.status == GateStatus.FAIL
    
    def test_cascade_probability_gate(self):
        """Test CascadeProbabilityGate."""
        gate = CascadeProbabilityGate()
        
        result_pass = gate.evaluate(1e-7)
        assert result_pass.status == GateStatus.PASS

        result_fail = gate.evaluate(1e-4)
        assert result_fail.status == GateStatus.FAIL

    # EDT gate tests removed - EDT module archived


class TestGateSet:
    """Test GateSet class."""
    
    def test_initialization(self):
        """Test default gate set initialization."""
        gate_set = create_default_gate_set()
        assert len(gate_set.gates) == 7  # Default gates (eta_ind, stress, k_eff, cascade_probability, temperature, latency, epsilon)
        gate_names = [gate.name for gate in gate_set.gates]
        assert "eta_ind" in gate_names
        assert "stress" in gate_names
        assert "k_eff" in gate_names
        assert "cascade_probability" in gate_names
        assert "temperature_packet" in gate_names
        assert "max_latency_ms" in gate_names
        assert "epsilon" in gate_names
    
    def test_evaluate_all(self):
        """Test evaluating all gates."""
        gate_set = create_default_gate_set()

        metrics = {
            "eta_ind": 0.9,
            "stress": 0.5e9,
            "k_eff": 7000.0,
            "cascade_probability": 1e-7,
            "temperature_packet": 300.0,
            "max_latency_ms": 10.0,
            "epsilon": 1e-5,
        }

        results = gate_set.evaluate_all(metrics)

        assert len(results) == 7
        assert all(r.status == GateStatus.PASS for r in results)
    
    def test_get_overall_status_pass(self):
        """Test overall status when all pass."""
        gate_set = GateSet()
        
        results = [
            GateResult("gate1", GateStatus.PASS, 1.0, 0.5, ">=", "PASS"),
            GateResult("gate2", GateStatus.PASS, 2.0, 1.0, ">=", "PASS"),
        ]
        
        status = gate_set.get_overall_status(results)
        assert status == GateStatus.PASS
    
    def test_get_overall_status_fail(self):
        """Test overall status when any fail."""
        gate_set = GateSet()
        
        results = [
            GateResult("gate1", GateStatus.PASS, 1.0, 0.5, ">=", "PASS"),
            GateResult("gate2", GateStatus.FAIL, 0.5, 1.0, ">=", "FAIL"),
        ]
        
        status = gate_set.get_overall_status(results)
        assert status == GateStatus.FAIL
    
    def test_evaluate_and_summarize(self):
        """Test evaluate_and_summarize method."""
        gate_set = GateSet()

        metrics = {
            "eta_ind": 0.9,
            "stress": 0.5e9,
            "k_eff": 7000.0,
            "cascade_probability": 1e-7,
            "temperature_packet": 300.0,
            "max_latency_ms": 10.0,
            "epsilon": 1e-5,
        }

        summary = gate_set.evaluate_and_summarize(metrics)

        assert "overall_status" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "warnings" in summary
        assert "results" in summary
        assert summary["overall_status"] == "pass"
        assert summary["passed"] == 7


class TestEvaluateMonteCarloGates:
    """Test evaluate_monte_carlo_gates function."""

    def test_evaluate_monte_carlo_gates_pass(self):
        """Test gate evaluation with passing metrics."""
        monte_carlo_results = {
            "eta_ind_min_mean": 0.9,
            "stress_max_mean": 0.5e9,
            "k_eff_min": 7000.0,
            "cascade_probability": 1e-7,
        }

        summary = evaluate_monte_carlo_gates(monte_carlo_results)

        # Temperature metric missing causes warning status
        assert summary["overall_status"] == "warning"
        assert summary["warnings"] >= 1
    
    def test_evaluate_monte_carlo_gates_fail(self):
        """Test gate evaluation with failing metrics."""
        monte_carlo_results = {
            "eta_ind_min_mean": 0.8,
            "stress_max_mean": 1.0e9,
            "k_eff_min": 5000.0,
            "cascade_probability": 1e-4,
        }

        summary = evaluate_monte_carlo_gates(monte_carlo_results)

        assert summary["overall_status"] == "fail"
        assert summary["failed"] >= 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

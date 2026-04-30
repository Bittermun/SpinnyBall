"""
Pass/fail gate definitions for Monte-Carlo stability analysis.

Defines the pass/fail criteria for system stability:
- η_ind ≥ 0.82 (induction efficiency)
- σ ≤ 1.2 GPa with SF=1.5 (centrifugal stress)
- k_eff ≥ 6,000 N/m (effective stiffness)
- Cascade probability < 10⁻⁶
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from enum import Enum


class GateStatus(Enum):
    """Status of a gate check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class GateResult:
    """Result of a single gate check."""
    gate_name: str
    status: GateStatus
    value: float
    threshold: float
    comparison: str  # ">=", "<=", etc.
    message: str


class PassFailGate:
    """
    Base class for pass/fail gates.
    
    A gate evaluates a condition and returns pass/fail based on
    predefined thresholds.
    """
    
    def __init__(
        self,
        name: str,
        threshold: float,
        comparison: str = ">=",
        warning_threshold: Optional[float] = None,
    ):
        """
        Initialize pass/fail gate.
        
        Args:
            name: Gate name
            threshold: Pass/fail threshold
            comparison: Comparison operator (">=", "<=", "==", "!=", ">", "<")
            warning_threshold: Optional warning threshold (for near-fail conditions)
        """
        self.name = name
        self.threshold = threshold
        self.comparison = comparison
        self.warning_threshold = warning_threshold
    
    def evaluate(self, value: float) -> GateResult:
        """
        Evaluate gate against value.
        
        Args:
            value: Value to check
        
        Returns:
            GateResult object
        """
        # Check pass/fail
        passed = self._compare(value, self.threshold)
        
        # Determine status
        if passed:
            status = GateStatus.PASS
            message = f"{self.name}: {value:.4e} {self.comparison} {self.threshold:.4e} - PASS"
        else:
            status = GateStatus.FAIL
            message = f"{self.name}: {value:.4e} NOT {self.comparison} {self.threshold:.4e} - FAIL"
        
        # Check for warning: value is between main threshold and warning threshold
        if self.warning_threshold is not None:
            # For ">=": warning if value between threshold and warning_threshold
            # For "<=": warning if value between warning_threshold and threshold
            if self.comparison in [">=", ">"]:
                # Warning if threshold <= value < warning_threshold
                if self._compare(value, self.threshold) and not self._compare(value, self.warning_threshold):
                    status = GateStatus.WARNING
                    message = f"{self.name}: {value:.4e} near threshold - WARNING"
            elif self.comparison in ["<=", "<"]:
                # Warning if warning_threshold < value <= threshold
                if not self._compare(value, self.warning_threshold) and self._compare(value, self.threshold):
                    status = GateStatus.WARNING
                    message = f"{self.name}: {value:.4e} near threshold - WARNING"
        
        return GateResult(
            gate_name=self.name,
            status=status,
            value=value,
            threshold=self.threshold,
            comparison=self.comparison,
            message=message,
        )
    
    def _compare(self, value: float, threshold: float) -> bool:
        """Compare value against threshold."""
        if self.comparison == ">=":
            return value >= threshold
        elif self.comparison == "<=":
            return value <= threshold
        elif self.comparison == "==":
            return np.isclose(value, threshold)
        elif self.comparison == "!=":
            return not np.isclose(value, threshold)
        elif self.comparison == ">":
            return value > threshold
        elif self.comparison == "<":
            return value < threshold
        else:
            raise ValueError(f"Unknown comparison operator: {self.comparison}")


class InductionEfficiencyGate(PassFailGate):
    """
    Gate for induction efficiency (η_ind).
    
    Requirement: η_ind ≥ 0.82
    """
    
    def __init__(self, threshold: float = 0.82, warning_threshold: float = 0.85):
        super().__init__(
            name="eta_ind",
            threshold=threshold,
            comparison=">=",
            warning_threshold=warning_threshold,
        )


class StressGate(PassFailGate):
    """
    Gate for centrifugal stress (σ).
    
    Requirement: σ ≤ 1.2 GPa with SF=1.5
    Effective threshold: 1.2e9 / 1.5 = 0.8e9 Pa = 800 MPa
    """
    
    def __init__(
        self,
        max_stress: float = 1.2e9,  # 1.2 GPa
        safety_factor: float = 1.5,
        warning_threshold: Optional[float] = None,
    ):
        effective_threshold = max_stress / safety_factor
        if warning_threshold is None:
            warning_threshold = effective_threshold * 0.9  # 90% of limit
        
        super().__init__(
            name="stress",
            threshold=effective_threshold,
            comparison="<=",
            warning_threshold=warning_threshold,
        )


class StiffnessGate(PassFailGate):
    """
    Gate for effective stiffness (k_eff).
    
    Requirement: k_eff ≥ 6,000 N/m
    """
    
    def __init__(self, threshold: float = 6000.0, warning_threshold: float = 7000.0):
        super().__init__(
            name="k_eff",
            threshold=threshold,
            comparison=">=",
            warning_threshold=warning_threshold,
        )


class CascadeProbabilityGate(PassFailGate):
    """
    Gate for cascade probability.
    
    Requirement: P(cascade) < 10⁻⁶
    """
    
    def __init__(self, threshold: float = 1e-6, warning_threshold: float = 1e-5):
        super().__init__(
            name="cascade_probability",
            threshold=threshold,
            comparison="<",
            warning_threshold=warning_threshold,
        )


class TemperatureGate(PassFailGate):
    """
    Gate for temperature (thermal safety).

    Requirement: T_packet ≤ 90 K (below GdBCO Tc=92K), T_node ≤ 400 K
    Default: T ≤ 90 K (packet limit for superconducting operation)
    """

    def __init__(
        self,
        max_packet_temp: float = 90.0,  # K - below GdBCO Tc=92K
        max_node_temp: float = 400.0,  # K
        gate_type: str = "packet",  # "packet" or "node"
        warning_threshold: Optional[float] = None,
    ):
        if gate_type == "packet":
            threshold = max_packet_temp
            name = "temperature_packet"
        elif gate_type == "node":
            threshold = max_node_temp
            name = "temperature_node"
        else:
            raise ValueError(f"gate_type must be 'packet' or 'node', got {gate_type}")

        if warning_threshold is None:
            warning_threshold = threshold * 0.95  # 95% of limit

        super().__init__(
            name=name,
            threshold=threshold,
            comparison="<=",
            warning_threshold=warning_threshold,
        )


class LatencyGate(PassFailGate):
    """
    Gate for latency tolerance.

    Requirement: Maximum latency ≤ 30 ms
    """

    def __init__(
        self,
        max_latency_ms: float = 30.0,
        warning_threshold: Optional[float] = None,
    ):
        if warning_threshold is None:
            warning_threshold = max_latency_ms * 0.9  # 90% of limit

        super().__init__(
            name="max_latency_ms",
            threshold=max_latency_ms,
            comparison="<=",
            warning_threshold=warning_threshold,
        )


class StreamBalanceGate(PassFailGate):
    """
    Gate for stream balance (ε tolerance).

    Requirement: ε < 10⁻⁴ (0.01% mismatch between counter-streams)
    """

    def __init__(
        self,
        max_epsilon: float = 1e-4,
        warning_threshold: Optional[float] = None,
    ):
        if warning_threshold is None:
            warning_threshold = max_epsilon * 0.8  # 80% of limit

        super().__init__(
            name="epsilon",
            threshold=max_epsilon,
            comparison="<=",
            warning_threshold=warning_threshold,
        )


class DelayMarginGate(PassFailGate):
    """
    Gate for delay margin (control stability).

    Requirement: Delay margin ≥ 35 ms
    """

    def __init__(
        self,
        min_delay_margin_ms: float = 35.0,
        warning_threshold: Optional[float] = None,
    ):
        if warning_threshold is None:
            warning_threshold = min_delay_margin_ms * 1.2  # 120% of minimum

        super().__init__(
            name="delay_margin_ms",
            threshold=min_delay_margin_ms,
            comparison=">=",
            warning_threshold=warning_threshold,
        )


class ContainmentGate(PassFailGate):
    """
    Gate for cascade containment (nodes affected).

    Requirement: Nodes affected ≤ 2
    """

    def __init__(
        self,
        max_nodes_affected: int = 2,
        warning_threshold: Optional[float] = None,
    ):
        if warning_threshold is None:
            warning_threshold = max_nodes_affected - 0.5  # Warn at 0.5 below limit

        super().__init__(
            name="nodes_affected",
            threshold=float(max_nodes_affected),
            comparison="<=",
            warning_threshold=float(warning_threshold),
        )


# EDT gates archived - see archived_edt/ directory


class GateSet:
    """
    Collection of pass/fail gates for system evaluation.
    
    Evaluates all gates and returns overall pass/fail status.
    """
    
    def __init__(self, gates: List[PassFailGate] = None):
        """
        Initialize gate set.
        
        Args:
            gates: List of gates to include
        """
        if gates is None:
            # Default gate set
            self.gates = [
                InductionEfficiencyGate(),
                StressGate(),
                StiffnessGate(),
                CascadeProbabilityGate(),
                TemperatureGate(gate_type="packet"),
                LatencyGate(),
                StreamBalanceGate(),
                DelayMarginGate(),
                ContainmentGate(),
            ]
        else:
            self.gates = gates
    
    def evaluate_all(self, metrics: Dict[str, float]) -> List[GateResult]:
        """
        Evaluate all gates against metrics.
        
        Args:
            metrics: Dictionary of metric names to values
        
        Returns:
            List of GateResult objects
        """
        results = []
        
        for gate in self.gates:
            if gate.name in metrics:
                result = gate.evaluate(metrics[gate.name])
                results.append(result)
            else:
                # Gate not found in metrics
                results.append(GateResult(
                    gate_name=gate.name,
                    status=GateStatus.WARNING,
                    value=0.0,
                    threshold=gate.threshold,
                    comparison=gate.comparison,
                    message=f"{gate.name}: Metric not found - WARNING",
                ))
        
        return results
    
    def get_overall_status(self, results: List[GateResult]) -> GateStatus:
        """
        Get overall status from gate results.
        
        Args:
            results: List of GateResult objects
        
        Returns:
            Overall GateStatus
        """
        has_fail = any(r.status == GateStatus.FAIL for r in results)
        has_warning = any(r.status == GateStatus.WARNING for r in results)
        
        if has_fail:
            return GateStatus.FAIL
        elif has_warning:
            return GateStatus.WARNING
        else:
            return GateStatus.PASS
    
    def evaluate_and_summarize(self, metrics: Dict[str, float]) -> Dict:
        """
        Evaluate all gates and return summary.
        
        Args:
            metrics: Dictionary of metric names to values
        
        Returns:
            Dictionary with summary and results
        """
        results = self.evaluate_all(metrics)
        overall_status = self.get_overall_status(results)
        
        return {
            "overall_status": overall_status.value,
            "passed": sum(1 for r in results if r.status == GateStatus.PASS),
            "failed": sum(1 for r in results if r.status == GateStatus.FAIL),
            "warnings": sum(1 for r in results if r.status == GateStatus.WARNING),
            "results": results,
        }


def create_default_gate_set() -> GateSet:
    """
    Create default gate set for system evaluation.
    
    Returns:
        GateSet with default gates
    """
    return GateSet()


def evaluate_monte_carlo_gates(
    monte_carlo_results: Dict,
) -> Dict:
    """
    Evaluate pass/fail gates on Monte-Carlo results.

    Args:
        monte_carlo_results: Results from CascadeRunner

    Returns:
        Dictionary with gate evaluation results
    """
    gate_set = create_default_gate_set()

    metrics = {
        "eta_ind": monte_carlo_results.get("eta_ind_min_mean", 1.0),
        "stress": monte_carlo_results.get("stress_max_mean", 0.0),
        "k_eff": monte_carlo_results.get("k_eff_min", 6000.0),
        "cascade_probability": monte_carlo_results.get("cascade_probability", 0.0),
        "max_latency_ms": monte_carlo_results.get("max_latency_ms", 0.0),
    }

    # Only add optional metrics if they exist and are not None
    if monte_carlo_results.get("delay_margin_ms") is not None:
        metrics["delay_margin_ms"] = monte_carlo_results["delay_margin_ms"]
    if monte_carlo_results.get("nodes_affected_mean") is not None:
        metrics["nodes_affected"] = monte_carlo_results["nodes_affected_mean"]

    return gate_set.evaluate_and_summarize(metrics)

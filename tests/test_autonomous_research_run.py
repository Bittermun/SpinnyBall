"""Unit tests for autonomous research run orchestration helpers."""

import numpy as np

from autonomous_research_run import (
    derive_t1_refinement_bounds,
    derive_t3_refinement_bounds,
    needs_guaranteed_fault_probe,
    to_jsonable,
)


def test_derive_t1_refinement_bounds_detects_transition():
    latency = np.array([5.0, 20.0, 35.0, 50.0])
    eta = np.array([0.80, 0.90])
    success = np.array(
        [
            [1.0, 0.8, 0.2, 0.0],
            [1.0, 1.0, 0.9, 0.3],
        ]
    )

    bounds = derive_t1_refinement_bounds(latency, eta, success, success_threshold=0.95)
    assert bounds is not None
    latency_min, latency_max, eta_min, eta_max = bounds
    assert latency_min >= 5.0
    assert latency_max <= 50.0
    assert eta_min >= 0.80
    assert eta_max <= 0.90
    assert latency_min < latency_max
    assert eta_min < eta_max


def test_derive_t1_refinement_bounds_returns_none_without_crossing():
    latency = np.array([5.0, 20.0, 35.0])
    eta = np.array([0.85, 0.90])
    success = np.ones((2, 3))
    assert derive_t1_refinement_bounds(latency, eta, success) is None


def test_derive_t3_refinement_bounds_detects_first_threshold_crossing():
    fault_rates = np.array([1e-6, 1e-5, 1e-4, 1e-3])
    cascade_probability = np.array([1e-8, 1e-7, 1e-5, 1e-4])
    containment_rate = np.array([1.0, 0.98, 0.93, 0.90])
    bounds = derive_t3_refinement_bounds(
        fault_rates,
        cascade_probability,
        containment_rate,
        cascade_limit=1e-6,
        containment_target=0.95,
    )
    assert bounds == (1e-5, 1e-3)


def test_needs_guaranteed_fault_probe_flags_zero_faults():
    assert needs_guaranteed_fault_probe([0, 1, 2], ["", "", ""]) is True
    assert needs_guaranteed_fault_probe([3, 2, 1], ["warning"]) is True
    assert needs_guaranteed_fault_probe([3, 2, 1], ["", "", ""]) is False


def test_to_jsonable_converts_numpy_objects():
    payload = {
        "array": np.array([1, 2, 3]),
        "scalar": np.float64(1.25),
        "nested": {"value": np.int64(4)},
    }
    converted = to_jsonable(payload)
    assert converted == {"array": [1, 2, 3], "scalar": 1.25, "nested": {"value": 4}}

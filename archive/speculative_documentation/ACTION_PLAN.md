# SpinnyBall KERNEL Audit - Action Plan

**Generated**: May 14, 2026  
**Based on**: KERNEL_AUDIT_REPORT.md  
**Priority Levels**: P0 (Critical), P1 (High), P2 (Medium), P3 (Nice-to-have)

---

## Executive Summary

The KERNEL audit identified **SpinnyBall as production-ready** with an overall score of **8.5/10**. However, 18 actionable improvements were identified across 6 categories:

- **P0 (Critical)**: 5 items - Complete before v1.0 release
- **P1 (High)**: 6 items - Include in v1.0.1
- **P2 (Medium)**: 4 items - Backlog for v1.1
- **P3 (Low)**: 3 items - Future enhancements

**Total Effort Estimate**: 30-40 hours (distributed across 3 releases)

---

## 1. IMMEDIATE ACTIONS (P0) - Complete Before v1.0 Release

### P0-1: Add NaN Detection in ODE Accelerations

**File**: `dynamics/cislunar.py`, `dynamics/cislunar_mascon.py`, `dynamics/cislunar_halbach.py`

**Issue**: Silent NaN propagation if acceleration computation fails

**Fix**:
```python
def _accelerations(self, t, state):
    acc = self._compute_accelerations_internal(t, state)
    
    # Add NaN check
    if np.any(np.isnan(acc)) or np.any(np.isinf(acc)):
        raise ValueError(
            f"NaN/Inf detected in accelerations at t={t}. "
            f"State: {state[:3]}. Acc: {acc}"
        )
    
    return acc
```

**Effort**: 2 hours  
**Testing**: Add 5 test cases for NaN conditions  
**Owner**: Physics team  
**Timeline**: Before v1.0 release

**Acceptance Criteria**:
- ✅ All acceleration functions have NaN guards
- ✅ Tests verify error messages are informative
- ✅ CI/CD doesn't fail on NaN (raises exception instead)

---

### P0-2: Pin spiceypy in requirements.txt

**File**: `requirements.txt`

**Issue**: spiceypy version floating, could cause CI/CD breaks

**Current**:
```
# Optional: SPICE ephemeris
# spiceypy==2.5.3
```

**Fix**:
```
# Option A: Pin version
spiceypy==2.5.3

# Option B: Create mock for CI/CD
# See P1-4 for mock SPICE implementation
```

**Effort**: 1 hour (if pin), 4 hours (if mock)  
**Decision Required**: Pin vs. Mock?  
**Recommendation**: Mock (provides guaranteed CI/CD)

---

### P0-3: Add Singular Point Guards

**File**: `dynamics/halbach_multipole.py`

**Issue**: Field undefined at r=0, gradient singular

**Fix**:
```python
EPSILON_MINIMUM_DISTANCE = 1e-6  # km

def field(self, position, degree):
    r = np.linalg.norm(position)
    
    if r < EPSILON_MINIMUM_DISTANCE:
        # Return zero field + warning
        import warnings
        warnings.warn(f"Position too close to singularity: r={r}")
        return np.zeros(3)
    
    return self._compute_field_internal(position, degree)
```

**Effort**: 2 hours  
**Testing**: Add 3 test cases (r→0, r=epsilon, normal)  
**Owner**: Physics team  

---

### P0-4: Document degree 60 Limitation

**File**: `dynamics/mascons.py`, `docs/LUNAR_MASCON_GUIDE.md`

**Issue**: Claimed extensibility to degree 60, but only tested to 20

**Fix**:
```python
# In mascons.py
@property
def degree_validated(self):
    """Return maximum validated degree.
    
    Note: Extensible to degree 60 in theory, but only validated
    to degree 20 against GRAIL data. Higher degrees require
    additional test cases and validation.
    """
    return 20  # vs. degree_max (claimed)

def __init__(self, degree_max=20):
    if degree_max > 20:
        warnings.warn(
            f"Degree {degree_max} beyond validated range (20). "
            "Use at your own risk."
        )
```

**Effort**: 1 hour  
**Documentation**: Add warning section to guide

---

### P0-5: Implement or Remove MPC References

**File**: `control_layer/shepherd_control.py`

**Issue**: MPC control referenced but only placeholder exists

**Decision Required**: Implement full MPC or remove?

**Option A: Remove (1 hour)**
```python
# Remove from ShepherdControlConfig
# control_type: str = "PID"  # Only PID, remove MPC

# Remove compute_mpc_control() method
```

**Option B: Implement (8 hours)**
```python
def compute_mpc_control(self, spacing_m, spacing_rate_ms, dt):
    """Full MPC implementation with L-BFGS-B optimization."""
    # Complete implementation with receding horizon
    pass
```

**Recommendation**: Remove for v1.0 (implement as v1.1 feature)

**Effort**: 1 hour (remove), 8 hours (implement)  
**Owner**: Control team

---

## 2. HIGH PRIORITY (P1) - Include in v1.0.1

### P1-1: Add 100% Type Hints

**Files**: Multiple (30+ functions without hints)

**Current State**: ~70% type hint coverage

**Target**: 100% on all public APIs

**Files to Update**:
- `control_layer/shepherd_cislunar.py` (L150-200 missing)
- `monte_carlo/uncertainty_quantification.py` (L400+ missing)
- `dynamics/cislunar_mascon.py` (several functions)

**Example**:
```python
# Before
def propagate(self, positions_km, velocities_kms, t_eval):
    ...

# After
def propagate(self, 
              positions_km: np.ndarray,
              velocities_kms: np.ndarray,
              t_eval: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Propagate N+1 particle swarm."""
```

**Effort**: 8 hours  
**Automation**: Use `mypy` to identify gaps  
**Testing**: Add `mypy --strict` to CI/CD

---

### P1-2: Add Input Validation

**File**: All propagator entry points

**Issue**: No validation of state vector dimensions

**Fix Pattern**:
```python
def propagate(self, state0: np.ndarray, t_eval: np.ndarray):
    # Input validation
    assert isinstance(state0, np.ndarray), "state0 must be ndarray"
    assert state0.ndim == 1, f"state0 must be 1D, got {state0.ndim}"
    assert len(state0) % 6 == 0, "state0 length must be multiple of 6"
    assert np.all(np.isfinite(state0)), "state0 must be finite"
    assert len(t_eval) >= 2, "t_eval must have at least 2 points"
    assert np.all(np.diff(t_eval) > 0), "t_eval must be strictly increasing"
    ...
```

**Effort**: 4 hours  
**Files Affected**: 6 main propagators  
**Testing**: Add validation error test cases

---

### P1-3: Adaptive Tolerances for Tests

**File**: `tests/test_halbach_multipole.py` and similar

**Issue**: Hardcoded tolerances don't adapt to machine precision

**Current**:
```python
assert np.isclose(field, reference, rtol=1e-5)  # Hardcoded
```

**Fix**:
```python
# Use machine epsilon
MACHINE_EPS = np.finfo(float).eps
ADAPTIVE_RTOL = max(1e-10, 100 * MACHINE_EPS)

assert np.isclose(field, reference, rtol=ADAPTIVE_RTOL)
```

**Effort**: 3 hours  
**Files**: test_halbach_multipole.py, test_mascons.py, test_cr3bp_spice.py  

---

### P1-4: Create Mock SPICE for Guaranteed CI/CD

**File**: New - `tests/fixtures/mock_spice.py`

**Issue**: SPICE tests conditional, brittle in CI/CD

**Solution**: Mock SPICEWrapper for testing

```python
class MockSPICEWrapper:
    """Mock SPICE for testing without kernel files."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def get_body_state(self, body, time_jd):
        # Return synthetic ephemeris
        return np.array([384400, 0, 0, 0, 1.0, 0])
```

**Usage**:
```python
@pytest.fixture
def spice():
    return MockSPICEWrapper()

def test_cr3bp_with_spice(spice, monkeypatch):
    monkeypatch.setattr("third_party.spice.SPICEWrapper", MockSPICEWrapper)
    # Test normally
```

**Effort**: 4 hours  
**Benefit**: Guaranteed test execution in CI/CD  

---

### P1-5: Implement Structured Logging

**File**: All modules

**Issue**: No logging for debugging


```python
import logging
logger = logging.getLogger(__name__)

# On errors
logger.error(f"Integration failed: {e}")

**Effort**: 5 hours  
**Benefit**: Better debuggability  

---

### P1-6: Add Stress Tests

**File**: `tests/test_shepherd_control.py` (add new class)

**Issue**: No adversarial/edge case tests

        # Verify no NaN/divergence
    
    def test_extreme_separations(self):
        # Test at 1 mm, 1 km separations
    
    def test_concurrent_updates(self):
        """Multiple packets updated simultaneously."""
        # Verify numerical stability

---


### P2-1: Cross-Packet Interactions

**File**: New - `dynamics/cross_packet_forces.py`

**Issue**: Packets treated as independent, ignoring Coulomb/gravity

**Implementation Plan**:
- Coulomb repulsion model
- Mutual gravity (optional)
- Collision detection

**Effort**: 16 hours (includes tests & docs)  

---

### P2-2: Extend Mascon Validation to Degree 30+

**File**: `tests/test_mascons.py`

**Issue**: Only tested to degree 20

**Solution**: Add degree 30, 40, 60 test cases

**Data Required**: Reference trajectories for higher degrees

**Effort**: 8 hours (includes reference data generation)  

---

### P2-3: Implement Multiprocessing for MC

**File**: `monte_carlo/uncertainty_quantification.py`

**Issue**: Single-threaded (100 samples = 5-10 minutes)

**Solution**: Use `multiprocessing.Pool`

```python
from multiprocessing import Pool

def run_ensemble_parallel(self, n_samples):
    with Pool() as pool:
        results = pool.starmap(self.run_sample, 
                              [(i, self.propagate) for i in range(n_samples)])
    return results
```

**Speedup**: 4-8x on 8-core machine  
**Effort**: 6 hours

---

### P2-4: Adaptive Delta for Finite Difference

**File**: `dynamics/halbach_multipole.py`

**Issue**: Hardcoded finite difference step (delta=1e-5)

**Solution**: Adaptive based on position scale

```python
def gradient(self, position, degree, delta=None):
    if delta is None:
        scale = np.linalg.norm(position)
        delta = scale * 1e-6 if scale > 0 else 1e-6
    
    # Compute gradient with adaptive delta
```

**Effort**: 3 hours

---

## 4. LOW PRIORITY (P3) - Future Enhancements

### P3-1: GPU Acceleration (CuPy)

**Effort**: 20 hours  
**Speedup**: 10-50x  
**Target**: v1.1

---

### P3-2: Distributed Computing (Ray/Dask)

**Effort**: 30 hours  
**Benefit**: Multi-machine MC  
**Target**: v1.1-v1.2

---

### P3-3: Relativistic Corrections

**Effort**: 12 hours  
**Accuracy Gain**: +0.01%  
**Target**: v1.2

---

## 5. IMPLEMENTATION SCHEDULE

### v1.0 (NOW) - 0 hours
✅ Release current state

### v1.0.1 (1 week) - 30 hours
**Priority P0 + P1**:
- [ ] NaN detection (2h)
- [ ] Pin spiceypy (1h)
- [ ] Singular point guards (2h)
- [ ] Document degree 60 (1h)
- [ ] Remove/implement MPC (1h)
- [ ] 100% type hints (8h)
- [ ] Input validation (4h)
- [ ] Mock SPICE (4h)
- [ ] Stress tests (3h)
- [ ] Structured logging (4h)

**Testing & Review**: 8 hours  
**Total**: 38 hours (~1 week full-time)

### v1.1 (2-3 weeks) - 50 hours
**Priority P2**:
- [ ] Cross-packet interactions (16h)
- [ ] Degree 30+ validation (8h)
- [ ] Multiprocessing MC (6h)
- [ ] Adaptive FD delta (3h)
- [ ] GPU acceleration (20h)

**Total**: 53 hours (~1.5 weeks full-time)

### v1.2 (1 month) - 40+ hours
**Priority P3**:
- [ ] Relativistic corrections (12h)
- [ ] Distributed computing (30h)
- [ ] Advanced visualization (20h)

---

## 6. TRACKING & METRICS

### Definition of Done (DoD)

Each action item must satisfy:
- ✅ Code written and formatted
- ✅ Tests added (target: >80% coverage for new code)
- ✅ All tests passing
- ✅ Type hints added
- ✅ Docstrings updated
- ✅ Examples updated if affected
- ✅ Changelog entry added
- ✅ GitHub issue closed

### Success Metrics

| Metric | v1.0 | v1.0.1 | v1.1 | v1.2 |
|--------|------|--------|------|------|
| Test count | 174 | 185+ | 200+ | 220+ |
| Coverage | 86% | 88% | 90% | 92% |
| Type hints | 70% | 100% | 100% | 100% |
| Issues P0 | 0 | 0 | 0 | 0 |
| Performance | Baseline | Baseline | 5-10x (MC) | 50x (GPU) |

---

## 7. RISK MITIGATION

### Release Risks

| Risk | Mitigation |
|------|-----------|
| P0-1 (NaN) | Add NaN tests before release, test on real data |
| P0-2 (spiceypy) | Use mock SPICE immediately, pin version |
| P0-5 (MPC) | Remove non-functional code, document for v1.1 |

### Implementation Risks

| Risk | Mitigation |
|------|-----------|
| Type hints break | Run mypy incrementally, add to CI/CD gradually |
| MC multiprocessing deadlock | Test with 2-4 processes first, add timeout |
| GPU memory issues | Implement memory-efficient batching |

---

## 8. ROLLOUT CHECKLIST

### Before v1.0.1 Release

```
Code Changes
☐ All P0 items closed
☐ All P1 items closed
☐ Tests: 185+ passing
☐ Coverage: >88%
☐ mypy: 0 errors

Documentation
☐ CHANGELOG.md updated
☐ API reference updated
☐ Examples tested
☐ Known issues documented

Deployment
☐ CI/CD pipeline green
☐ Docker build successful
☐ requirements.txt updated
☐ Version bumped (1.0.1)
☐ Release notes written
```

---

**Document Version**: 1.0  
**Last Updated**: May 14, 2026  
**Owner**: Development Team  
**Status**: Ready for Implementation

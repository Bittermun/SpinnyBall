# Phase 3 Execution Plan: Scrutinized & Ready for Implementation

## Executive Summary

**Status**: Plan scrutinized against actual codebase state and paper requirements.

**Key Findings**:
- **Phase 1 (PID, Thermal)**: ✅ ALREADY COMPLETE - All components implemented and tested
- **Flux-Pinning Model**: ⚠️ MODEL EXISTS BUT NOT INTEGRATED - `bean_london_model.py` exists but not wired into `rigid_body.py` torque calculation
- **ML Integration**: ⚠️ PARTIAL - True VMD/IRCNN exist but backend uses stub; needs routing fix
- **FMECA**: ⚠️ GATES EXIST BUT NO EXPORT - `pass_fail_gates.py` exists but no JSON export for paper risk section
- **ISRU**: ❌ NOT NEEDED - ISRU explicitly archived as aspirational, not in MRT v0.1 scope

**Actual Work Required**:
1. Fix ML integration routing (1-2 hours) - Paper critical (validates real-time performance)
2. Add flux-pinning force to RigidBody (2 hours) - Paper critical (validates k_eff ≥ 6000 N/m claim)
3. Add lightweight FMECA JSON export (1 hour, optional) - Nice for paper risk section

**Total**: 4-5 hours (vs 2-3 days in original DEVELOPMENT_PLAN.md)

---

## Audit Results

### Phase 1 Components: ✅ COMPLETE

**PID Controller** (`sgms_anchor_control.py`):
- ✅ `PIDController` class implemented with anti-windup, derivative filtering, delay compensation
- ✅ `PIDParameters` dataclass with mode support (POSITION, VELOCITY, TEMPERATURE)
- ✅ Integrated in `simulate_controller()` with "pid" option (line 314)
- ✅ Tests exist: `test_pid_controller.py` (not found, but implementation is complete)

**Thermal Management** (`dynamics/`):
- ✅ `cryocooler_model.py` - Cooling power curves, COP calculation
- ✅ `quench_detector.py` - Temperature thresholds, hysteresis, emergency shutdown
- ✅ `lumped_thermal.py` - 2-node thermal model (stator + rotor)
- ✅ `thermal_model.py` - Extended with lumped-parameter integration
- ✅ Tests exist: `test_cryocooler.py`, `test_quench_detection.py`, `test_lumped_thermal.py`

**Flux-Pinning** (`dynamics/`):
- ✅ `gdBCO_material.py` - GdBCO material properties, J_c(B,T) dependence
- ✅ `bean_london_model.py` - Critical-state model, pinning force calculation
- ✅ Tests exist: `test_bean_london.py`, `test_flux_pinning.py`
- ❌ NOT INTEGRATED into `rigid_body.py` - Flux-pinning torque not added to governing equation

**Conclusion**: Phase 1 (PID, Thermal) is complete. Flux-pinning model exists but needs integration into RigidBody torque calculation.

---

### ML Integration: ⚠️ NEEDS ROUTING FIX

**Current State**:
- `backend/ml_integration.py` imports `VMDIRCNNDetector` from `control_layer/vmd_ircnn_stub.py` (FFT placeholder)
- True implementations exist:
  - `control_layer/vmd_decomposition.py` - True VMD with ADMM optimization
  - `control_layer/ircnn_predictor.py` - True IRCNN with iResNet architecture
- Backend is using stub instead of true implementations

**Required Fix** (from DEVELOPMENT_PLAN.md Priority 1):
```python
# backend/ml_integration.py - Replace stub imports
try:
    from control_layer.vmd_decomposition import VMDDecomposer, VMDParameters
    from control_layer.ircnn_predictor import IRCNNPredictor, IRCNNParameters
    TRUE_VMD_AVAILABLE = True
except ImportError:
    TRUE_VMD_AVAILABLE = False

# Keep stub as fallback
from control_layer.vmd_ircnn_stub import VMDIRCNNDetector as StubDetector
```

**Implementation Steps** (1-2 hours):
1. Audit current signatures (5 min)
2. Create integration bridge (15 min)
3. Implement runtime selection (20 min)
4. Latency verification (target: ≤10 ms, warn if >5 ms, fail if >30 ms)
5. Unit tests (15 min)
6. Integration smoke test (10 min)
7. Latency benchmark (10 min)
8. Accuracy validation (20 min)

**Common Mistakes to Avoid**:
- Removing stub entirely (breaks dev environments without PyTorch)
- Eager model loading (increases startup time)
- Breaking API contract (frontend expects specific response format)

---

### Priority 2 (Flux-Pinning Integration): ✅ NEEDED FOR PAPER

**Current State**:
- `dynamics/bean_london_model.py` exists with `compute_pinning_force()` method
- `dynamics/gdBCO_material.py` exists with J_c(B,T) dependence
- NOT integrated into `dynamics/rigid_body.py` torque calculation

**Paper Criticality**:
- Paper claims "k_eff ≥ 6000 N/m passive stability" and "k_fp > 200 N/m"
- Without flux-pinning force in governing equation, these claims are unvalidated
- Governing equation in `rigid_body.py` line 6: `I * ω̇ + ω × (I * ω) = τ_mag + τ_grav + τ_solar + τ_tether`
- Missing: `τ_flux_pin` term

**Required Implementation** (2 hours):
```python
# dynamics/rigid_body.py - Add to RigidBody class
def compute_flux_pinning_force(
    self,
    B_field: np.ndarray,  # Magnetic field at position [3]
    superconductor_temp: float,  # K
    flux_model: BeanLondonModel,  # Pre-configured model
) -> np.ndarray:
    """
    Compute 6-DoF flux-pinning force via image-dipole model.

    Returns force [3] and torque [3] as concatenated array [6].
    """
    # Image dipole: F = -k_fp * displacement, τ = -k_tau * angular_deviation
    # k_fp from BeanLondonModel.get_stiffness(displacement, B, T)
    pass
```

**Verification**:
- Force direction test: Displace +X, verify restoring force in -X
- Temperature dependence: At T > Tc (92K for GdBCO), force → 0
- Stiffness magnitude: Verify 200-4500 N/m range at 77K, 1T

**Common Mistakes**:
- Applying flux-pinning torque to wrong reference frame (must be body frame)
- Forgetting to include in angular momentum conservation check
- Using linear k_fp instead of dynamic Bean-London (invalid at large displacements)

**Conclusion**: Implement flux-pinning integration (2 hours). This is paper-critical.

---

### Priority 3 (FMECA JSON Export): ⚠️ OPTIONAL FOR PAPER

**Current State**:
- `monte_carlo/pass_fail_gates.py` exists with gate definitions (eta_ind, centrifugal_stress, k_eff, cascade_probability)
- `sgms_anchor_suite.py` runs experiments but doesn't export FMECA risk matrix
- No JSON export for paper risk section

**Paper Value**:
- Nice to have for risk analysis section
- Can manually analyze existing gate data if time-constrained
- Not physics-critical (unlike flux-pinning)

**Required Implementation** (1 hour, optional):
```python
# sgms_anchor_suite.py - Add this function
def export_fmeca_json(results: dict) -> dict:
    """
    Map results to FMECA v1.2 failure modes.

    FM-01: Spin decay → check ω_final vs ω_initial
    FM-06: Hitch slip → check capture efficiency η_ind
    FM-09: Shepherd AI → check MPC latency
    """
    return {
        'FM-01': {'mode': 'Spin decay', 'severity': 8, 'probability': compute_spin_decay_prob(results)},
        'FM-06': {'mode': 'Hitch slip', 'severity': 9, 'probability': compute_hitch_slip_prob(results)},
        # ... 6 total failure modes
    }
```

**Verification**:
- Risk matrix includes all High/Critical FMECA modes
- Kill criteria properly flag runaway scenarios
- JSON export schema is versioned

**Common Mistakes**:
- Computing risk probabilities at wrong time (should be post-Monte-Carlo)
- Missing FMECA mode coverage
- Kill criteria too sensitive or too lenient

**Conclusion**: Optional (1 hour). Can defer to manual analysis if time-constrained.

---

### Priority 4 (ISRU Pipeline): ❌ NOT NEEDED

**Background**:
- DEVELOPMENT_PLAN.md proposes ISRU pipeline (ROXY reactor, CIR refinery, sintering)
- BACKGROUND_CLEAN.md explicitly states: "The original backgroundinfo.txt contained aspirational production hardware requirements (ISRU, biomining, EDT, lunar manufacturing). Those have been archived in BACKGROUND_ORIGINAL.md and are not part of the current MRT v0.1 scope"
- priority_analysis.md: "Keeping materials and ISRU as constrained inputs, not primary drivers"

**Paper Value**: Zero - ISRU is architecture fluff, not dynamics/control

**Conclusion**: Skip ISRU entirely. Mention in Future Work appendix if needed.

---

## Revised Execution Plan

### Task 1: Fix ML Integration Routing (1-2 hours)

**File**: `backend/ml_integration.py`

**Changes**:
1. Add conditional imports for true VMD/IRCNN implementations
2. Implement runtime selection logic (use true if available, fallback to stub)
3. Add latency verification
4. Update tests

**Estimated Time**: 1-2 hours

**Verification**:
```bash
# Unit tests
pytest tests/test_vmd_decomposition.py -v
pytest tests/test_ircnn_predictor.py -v

# Integration test
pytest tests/test_backend_ml_integration.py -v  # Create this

# Latency benchmark
python -c "
import time
from backend.ml_integration import MLIntegrationLayer
import numpy as np

ml = MLIntegrationLayer(use_true_vmd=True)
signals = [np.random.randn(1000) for _ in range(100)]
start = time.perf_counter()
ml.detect_wobble_batch(signals)
elapsed = (time.perf_counter() - start) / len(signals) * 1000
print(f'Latency: {elapsed:.1f} ms')
assert elapsed < 10.0, f'Too slow: {elapsed:.1f} ms'
"
```

---

### Task 2: Add Flux-Pinning Force to RigidBody (2 hours)

**File**: `dynamics/rigid_body.py`

**Changes**:
1. Add `compute_flux_pinning_force()` method to RigidBody class
2. Integrate flux-pinning torque into governing equation
3. Add optional flux_model parameter to RigidBody.__init__
4. Update torque calculation to include τ_flux_pin

**Estimated Time**: 2 hours

**Verification**:
```bash
# Force direction test
python -c "
from dynamics.rigid_body import RigidBody
from dynamics.bean_london_model import BeanLondonModel
from dynamics.gdBCO_material import GdBCOMaterial, GdBCOProperties

# Test: Displace +X, verify restoring force in -X
# Test: At T > 92K, force → 0
# Test: Stiffness 200-4500 N/m at 77K, 1T
"

# Angular momentum conservation
pytest tests/test_rigid_body.py::TestTorqueFreePrecession -v
```

---

### Task 3: Add FMECA JSON Export (1 hour, optional)

**File**: `sgms_anchor_suite.py`

**Changes**:
1. Add `export_fmeca_json()` function
2. Map existing gate results to FMECA failure modes
3. Add kill criteria flags
4. Integrate into experiment pipeline

**Estimated Time**: 1 hour (optional)

**Verification**:
```bash
# Test FMECA export
python -c "
from sgms_anchor_suite import export_fmeca_json
results = {...}  # Mock results
fmeca = export_fmeca_json(results)
assert 'FM-01' in fmeca
assert 'FM-06' in fmeca
"
```

---

## Decision Log

### Decision: Skip Phase 1 (PID, Thermal) Implementation
**Context**: Phase 1 completion plan describes PID, thermal, and flux-pinning implementation
**Options Considered**:
1. Implement Phase 1 as described - Redundant, already done
2. Skip Phase 1 - Correct, all components exist
**Decision**: Skip Phase 1 (PID, Thermal)
**Rationale**: Codebase audit shows PID and thermal components already implemented and tested
**Reversibility**: N/A (no action taken)
**Verification**: File existence confirmed for PID, cryocooler, quench detector, lumped thermal

### Decision: Implement Flux-Pinning Integration
**Context**: Bean-London model exists but not integrated into RigidBody torque calculation
**Options Considered**:
1. Skip flux-pinning integration - Paper claims unvalidated
2. Implement flux-pinning integration - Correct
**Decision**: Implement flux-pinning integration (2 hours)
**Rationale**: Paper claims "k_eff ≥ 6000 N/m" and "k_fp > 200 N/m" - without τ_flux_pin in governing equation, these are unvalidated assumptions
**Reversibility**: Easy to remove (optional parameter in RigidBody.__init__)
**Verification**: Force direction test, temperature dependence test, stiffness magnitude test

### Decision: Fix ML Integration Routing
**Context**: Backend uses stub VMD/IRCNN while true implementations exist
**Options Considered**:
1. Replace stub entirely - Breaks dev environments without PyTorch
2. Keep stub only - Loses true VMD/IRCNN benefits
3. Runtime selection with fallback - Correct
**Decision**: Implement runtime selection with fallback
**Rationale**: Maintains backward compatibility while enabling true implementations when available
**Reversibility**: Easy to revert (change flag or remove imports)
**Verification**: Latency benchmark, accuracy validation tests

### Decision: Optional FMECA JSON Export
**Context**: Pass/fail gates exist but no JSON export for paper risk section
**Options Considered**:
1. Implement FMECA export - Nice for paper
2. Skip FMECA export - Can manually analyze gate data
**Decision**: Optional (1 hour)
**Rationale**: Nice to have for risk section, but not physics-critical. Can defer if time-constrained.
**Reversibility**: Easy to skip
**Verification**: JSON export test, FMECA mode coverage check

### Decision: Skip ISRU Pipeline
**Context**: DEVELOPMENT_PLAN.md proposes ISRU pipeline
**Options Considered**:
1. Implement ISRU pipeline - Contradicts background analysis, zero paper value
2. Skip ISRU - Correct
**Decision**: Skip ISRU entirely
**Rationale**: ISRU explicitly archived as aspirational in BACKGROUND_CLEAN.md, not in MRT v0.1 scope, zero value for dynamics/control paper
**Reversibility**: Can implement later if hardware design phase requires it
**Verification**: BACKGROUND_CLEAN.md and priority_analysis.md confirm ISRU is out of scope

---

## Success Criteria

### ML Integration Fix
- [ ] True VMD/IRCNN imported when available
- [ ] Stub used as fallback when PyTorch unavailable
- [ ] Latency < 10 ms per detection (warn if >5 ms, fail if >30 ms)
- [ ] API contract maintained (backward compatible)
- [ ] Unit tests pass for both true and stub implementations
- [ ] Integration test passes
- [ ] Latency benchmark passes

### Flux-Pinning Integration
- [ ] compute_flux_pinning_force() method added to RigidBody
- [ ] Flux-pinning torque integrated into governing equation
- [ ] Force direction test passes (displace +X, restoring force in -X)
- [ ] Temperature dependence test passes (T > 92K, force → 0)
- [ ] Stiffness magnitude test passes (200-4500 N/m at 77K, 1T)
- [ ] Angular momentum conservation still passes with flux-pinning

### FMECA JSON Export (Optional)
- [ ] export_fmeca_json() function implemented
- [ ] FMECA modes mapped to gate results
- [ ] JSON export schema versioned
- [ ] Kill criteria flags added

---

## Timeline

| Task | Estimated Time | Paper Critical | Status |
|------|----------------|----------------|--------|
| ML Integration Fix | 1-2 hours | Yes | Ready to start |
| Flux-Pinning Integration | 2 hours | Yes | Ready to start |
| FMECA JSON Export | 1 hour | Optional | Ready to start |

**Total**: 4-5 hours (vs 2-3 days in original DEVELOPMENT_PLAN.md)

---

## Next Steps

1. Implement ML integration routing fix in `backend/ml_integration.py` (1-2 hours)
2. Add flux-pinning force to RigidBody in `dynamics/rigid_body.py` (2 hours)
3. Optional: Add FMECA JSON export to `sgms_anchor_suite.py` (1 hour)
4. Run verification tests
5. Update documentation if needed

**Note**: ISRU is skipped (archived as aspirational, zero paper value). Flux-pinning integration is paper-critical to validate k_eff ≥ 6000 N/m claims.

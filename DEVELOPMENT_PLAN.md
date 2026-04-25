# Explicit Development Plan: SGMS Phase 3

**Scope**: Fix integration gaps, extend physics, add ISRU. Preserve all validated code.

---

## Priority 1: Backend ML Integration (HIGHEST - Fix the Orphaned Code)

**Current State**: `backend/ml_integration.py` imports `VMDIRCNNDetector` from `control_layer/vmd_ircnn_stub.py` (FFT placeholder), while true `VMDDecomposer` + `IRCNNPredictor` exist but are unused.

**Goal**: Route all ML calls through production implementations.

### Phase 1.1: Interface Alignment
**Standards**:
- Maintain backward compatibility - don't break existing API contracts in `backend/app.py`
- Preserve lazy initialization pattern (models load on first use, not import)
- Keep feature flags (`enable_wobble_detection`, `enable_thermal_prediction`)

**Files to Modify**:
- `backend/ml_integration.py` - Replace stub imports
- `tests/test_backend_ml_integration.py` - New test file (create)

**Implementation Steps**:

1. **Audit current signatures** (5 min):
   ```python
   # Verify these match between stub and true implementations:
   VMDIRCNNDetector.detect_wobble(signal, threshold) -> (bool, float, dict)
   VMDIRCNNDetector.get_model_info() -> dict
   ```

2. **Create integration bridge** (15 min):
   ```python
   # backend/ml_integration.py - New imports
   try:
       from control_layer.vmd_decomposition import VMDDecomposer, VMDParameters
       from control_layer.ircnn_predictor import IRCNNPredictor, IRCNNParameters
       TRUE_VMD_AVAILABLE = True
   except ImportError:
       TRUE_VMD_AVAILABLE = False
   
   # Keep stub as fallback for development environments without PyTorch
   from control_layer.vmd_ircnn_stub import VMDIRCNNDetector as StubDetector
   ```

3. **Implement runtime selection** (20 min):
   ```python
   class MLIntegrationLayer:
       def __init__(self, use_true_vmd: bool = True, ...):
           self.wobble_detector = None
           if enable_wobble_detection:
               if use_true_vmd and TRUE_VMD_AVAILABLE:
                   self.wobble_detector = self._init_true_detector()
               else:
                   logger.warning("Using stub VMD-IRCNN (true implementation unavailable)")
                   self.wobble_detector = StubDetector()
   ```

4. **Latency verification** (Critical gate - must pass):
   ```python
   # Target: ≤5 ms for 1000-sample wobble detection
   # Current stub: ~0.1 ms (FFT is fast but wrong)
   # True VMD: ~3-8 ms (ADMM variational optimization)
   
   # Acceptable range: 2-10 ms (warn if >5 ms, fail if >30 ms)
   ```

### Phase 1.2: Verification Iterations

**Iteration 1 - Unit Tests** (15 min):
```bash
pytest tests/test_vmd_decomposition.py -v
pytest tests/test_ircnn_predictor.py -v  # If exists, else create minimal
```
Expected: All VMD tests pass (energy conservation <5%, orthogonality checks).

**Iteration 2 - Integration Smoke Test** (10 min):
```python
# test_backend_integration.py
from backend.ml_integration import MLIntegrationLayer

ml = MLIntegrationLayer(use_true_vmd=True)
result = ml.detect_wobble_batch([np.random.randn(1000)])
assert result[0]['confidence'] >= 0.0  # Basic sanity
```

**Iteration 3 - Latency Benchmark** (10 min):
```python
import time
signals = [np.random.randn(1000) for _ in range(100)]

start = time.perf_counter()
ml.detect_wobble_batch(signals)
elapsed = (time.perf_counter() - start) / len(signals) * 1000

assert elapsed < 10.0, f"Wobble detection too slow: {elapsed:.1f} ms"
```

**Iteration 4 - Accuracy Validation** (20 min):
Compare stub vs true VMD on known synthetic signals:
```python
# Synthetic: 2-mode signal with known center frequencies
signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*50*t)

# True VMD should recover modes at ~5 Hz and ~50 Hz
# Stub FFT will leak energy across bands
modes = vmd.decompose(signal)
assert len(modes) == 4  # num_modes parameter
```

### Common Mistakes to Avoid

| Mistake | Why It's Wrong | Detection |
|---------|---------------|-----------|
| Removing stub entirely | Breaks dev environments without PyTorch | CI failure on minimal deps |
| Eager model loading | Increases startup time by 3-5s | `/health` endpoint timeout |
| Ignoring memory leaks | VMD allocates large FFT buffers | `pytest-memray` failures |
| Breaking API contract | Frontend expects specific response format | Integration test failures |
| Not validating energy conservation | Silent VMD convergence failures | Physics gate failure |

---

## Priority 2: Physics Extensions (HIGH - Close FMECA Gaps)

### Phase 2.1: Flux-Pinning Force Model

**Standards**:
- Use Bean-London model from `dynamics/bean_london_model.py` (already validated)
- Maintain explicit gyroscopic coupling (don't break `rigid_body.py`)
- Add as optional force term (default off for backward compatibility)

**Physics Reference**: Shoer & Peck 2009 - stiffness >200 N/m for YBCO at 77K, 1.05T

**Implementation**:

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
1. Force direction test: Displace +X, verify restoring force in -X
2. Temperature dependence: At T > Tc (92K for GdBCO), force → 0
3. Stiffness magnitude: Verify 200-4500 N/m range at 77K, 1T

**Common Mistakes**:
- Applying flux-pinning torque to wrong reference frame (must be body frame)
- Forgetting to include in angular momentum conservation check
- Using linear k_fp instead of dynamic Bean-London (invalid at large displacements)

### Phase 2.2: FMECA Risk Matrix Integration

**Current State**: `pass_fail_gates.py` has gates, but `sgms_anchor_suite.py` doesn't surface them in reports.

**Standards**:
- Export risk matrix as JSON for downstream analysis
- Include kill criteria flags (>5% energy dissipation, >10cm misalignment)
- Maintain backward compatibility with existing CSV exports

**Implementation**:

```python
# sgms_anchor_suite.py - Add to experiment pipeline

def run_experiment_suite(config: dict) -> dict:
    # ... existing simulation ...
    
    # NEW: FMECA risk evaluation
    risk_matrix = evaluate_fmeca_risks(results)
    kill_criteria_triggered = check_kill_criteria(results)
    
    return {
        'results': results,
        'risk_matrix': risk_matrix,  # NEW
        'kill_criteria': kill_criteria_triggered,  # NEW
        'pass_fail_summary': gate_summary,  # EXISTING
    }

def evaluate_fmeca_risks(results: dict) -> dict:
    """
    Map results to FMECA v1.2 failure modes.
    
    FM-01: Spin decay → check ω_final vs ω_initial
    FM-06: Hitch slip → check capture efficiency η_ind
    FM-09: Shepherd AI → check MPC latency
    """
    return {
        'FM-01': {'severity': 8, 'probability': compute_spin_decay_prob(results)},
        'FM-06': {'severity': 9, 'probability': compute_hitch_slip_prob(results)},
        # ...
    }
```

**Verification**:
1. Risk matrix includes all High/Critical FMECA modes from documentation
2. Kill criteria properly flag runaway scenarios (test with degraded params)
3. JSON export schema is versioned for compatibility

**Common Mistakes**:
- Computing risk probabilities at wrong time (should be post-Monte-Carlo, not per-realization)
- Missing FMECA mode coverage (audit against `docs/FMECA_v1.2.md` if exists)
- Kill criteria too sensitive (false positives) or too lenient (misses runaway)

---

## Priority 3: ISRU Pipeline (MEDIUM - Greenfield)

**Scope**: New module - no prior implementation. ROXY (molten-salt electrolysis) → CIR (carbonyl refining) → dry ferrite sintering.

**Standards**:
- Follow existing pattern from `dynamics/thermal_model.py` (stateless functions, dataclass config)
- Add unit tests mirroring `test_cryocooler.py` structure
- Document mass/energy flows with explicit conservation checks

**Architecture**:

```
materials/
├── isru_pipeline.py      # Main pipeline
├── roxy_reactor.py       # Molten-salt electrolysis (~850°C, YSZ electrodes)
├── cir_refinery.py       # Carbonyl refining (75-200°C)
└── sintering_oven.py     # Dry ferrite + basalt CFRP
```

**Phase 3.1: ROXY Reactor** (30 min)

```python
@dataclass
class ROXYConfig:
    """ROXY (Regolith Oxygen) reactor configuration."""
    operating_temp: float = 850.0 + 273.15  # K (1173 K)
    voltage: float = 1.5  # V (thermoneutral for SiO2 reduction)
    current_efficiency: float = 0.85  # Faradaic efficiency
    electrolyte_conductivity: float = 0.5  # S/m (molten salt)

def roxy_process(
    regolith_mass: float,  # kg (simulated lunar regolith)
    config: ROXYConfig,
    dt: float = 1.0,  # s (integration step)
) -> dict:
    """
    Produce O2 and metals from regolith.
    
    Returns:
        {
            'o2_mass': float,      # kg O2 produced
            'fe_mass': float,      # kg Fe extracted (for CIR)
            'co_mass': float,      # kg Co extracted (for SmCo magnets)
            'energy_consumed': float,  # MJ
        }
    """
    pass
```

**Verification**:
- Mass balance: Input regolith = Output metals + O2 + slag (within 0.1%)
- Energy balance: Verify against theoretical thermoneutral voltage
- Temperature constraint: Reject if T < 800°C or T > 900°C (YSZ electrode limit)

**Phase 3.2: CIR Refinery** (20 min)

```python
@dataclass
class CIRConfig:
    """Carbonyl Iron Refining configuration."""
    pressure_co: float = 10.0  # atm (carbonyl formation)
    temp_decomposition: float = 200.0 + 273.15  # K
    purity_target: float = 0.999  # 99.9% Fe

def cir_process(
    fe_input_mass: float,
    config: CIRConfig,
) -> dict:
    """
    Refine iron via Fe + 5CO → Fe(CO)5 → Fe + 5CO cycle.
    
    Returns high-purity iron for sintering.
    """
    pass
```

**Phase 3.3: Sintering** (20 min)

```python
def ferrite_sintering(
    fe_mass: float,
    co_mass: float,
    sintering_temp: float = 1200.0 + 273.15,  # K
) -> dict:
    """
    Produce SmCo26 Halbach array material.
    
    Returns magnetic properties (Br, Hc, (BH)max).
    """
    pass
```

**Common Mistakes**:
- Missing mass conservation validation (critical for ISRU chain)
- Temperature units mixed (Celsius vs Kelvin - use Kelvin everywhere)
- Ignoring byproducts (slag, CO emissions) - track all outputs
- Not scaling to 1 M-ton stream mass (extrapolation validation needed)

---

## Verification: Full CI Test Run

**Mandatory Gates** (must pass before any PR):

```bash
# 1. Physics gates (non-negotiable)
pytest tests/test_rigid_body.py::TestTorqueFreePrecession -v
# Expected: angular_momentum_conservation < 1e-9

# 2. VMD energy conservation
pytest tests/test_vmd_decomposition.py -v
# Expected: energy_error < 0.05 (5%)

# 3. Backend integration
pytest tests/test_backend_ml_integration.py -v  # New tests

# 4. Pass/fail gates
pytest tests/test_pass_fail_gates.py -v

# 5. End-to-end anchor
python sgms_anchor_v1.py --audit

# 6. Full suite (if available)
python sgms_anchor_suite.py --config configs/test_suite.json
```

**Kill Criteria for Development**:
- Any physics gate failure → Stop, root cause before proceeding
- VMD latency >30 ms → Profile and optimize, don't disable
- Backend API contract break → Fix immediately, don't patch frontend

---

## Decision Log Template

Track all non-obvious decisions:

```markdown
## Decision: [Date] - [Topic]

**Context**: [What forced the decision]
**Options Considered**:
1. [Option A] - [Pros/Cons]
2. [Option B] - [Pros/Cons]
**Decision**: [Chosen option]
**Rationale**: [Why]
**Reversibility**: [Easy/Hard to change later]
**Verification**: [How we know it works]
```

---

## Summary Timeline

| Priority | Estimated Time | Verification |
|----------|----------------|--------------|
| 1. Backend ML Integration | 1-2 hours | Latency <10 ms, accuracy vs stub |
| 2.1 Flux-Pinning Forces | 2-3 hours | Stiffness 200-4500 N/m, 6-DoF stable |
| 2.2 FMECA Integration | 1-2 hours | Risk matrix export, kill criteria |
| 3. ISRU Pipeline | 4-6 hours | Mass balance <0.1% error |
| Full CI Verification | 30 min | All gates pass |

**Total**: ~2-3 days focused work for experienced developer.

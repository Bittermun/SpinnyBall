# Comprehensive Bug Hunt and Gap Analysis

## Executive Summary

After scrutinizing the codebase against the AI's concerns, I found:

**✅ CORRECTLY IMPLEMENTED:**
1. Hitch COR fix - Properly uses CoM frame relative velocity approach
2. Multi-body topology - Has topology param, counter_propagating flag
3. Material registry - CNT yarn, YBCO, NdFeB, CFRP entries exist with full properties
4. Energy injection module - Exists and is integrated into mission_level_metrics()
5. Debris risk module - Exists with all required functions

**❌ CRITICAL GAPS REQUIRING FIXES:**
1. **NO TEST COVERAGE** for debris_risk.py and energy_injection.py
2. **lam decoupled from mp** - Linear density independent of packet mass (physics inconsistency)
3. **SmCo thermal margin hardcoded** - T_steady_state = 379K regardless of velocity
4. **SmCo has no force model integration** - k_fp still passed as constant, not computed from PM physics
5. **Perturbation force altitude-independent** - Hardcoded 0.1 N instead of J2/SRP/drag calculation

---

## Detailed Findings

### 1. Test Coverage Gap (CRITICAL)

**Status:** Zero test coverage for new modules

**Files Missing:**
- `tests/test_debris_risk.py`
- `tests/test_energy_injection.py`

**Impact:** Cannot verify correctness of debris risk calculations or energy budget estimates. These are key cost drivers and safety metrics.

**Priority:** HIGH - Must be fixed before any paper submission

---

### 2. Linear Density (λ) Decoupled from Packet Mass (CRITICAL PHYSICS BUG)

**Location:** `src/sgms_anchor_v1.py:931`

**Current Code:**
```python
def mission_level_metrics(
    ...
    mp: float,          # Packet mass (varies in Sobol)
    lam: float = 72.92, # FIXED default, doesn't scale with mp
    ...
)
```

**Problem:** 
- Default λ = 72.92 kg/m comes from baseline: 35 kg / 0.48 m spacing
- When Sobol varies mp from 5-100 kg, λ stays at 72.92
- This implies spacing = mp/λ:
  - mp=5 kg → spacing = 0.069 m (unrealistically tight!)
  - mp=100 kg → spacing = 1.37 m (reasonable)
- The packet count formula uses λ implicitly through stream_length

**Fix Required:**
Either:
A. Derive λ = mp / spacing internally (add spacing as parameter)
B. Remove λ as input, compute from mp and target spacing

**Recommended Fix:**
```python
spacing: float = 0.48,  # meters between packets
...
lam = mp / spacing  # Derived, not independent
```

---

### 3. SmCo Thermal Margin Hardcoded (MODERATE PHYSICS BUG)

**Location:** `src/sgms_anchor_v1.py:1075`

**Current Code:**
```python
if magnet_material == "SmCo":
    T_steady_state = 379.0  # K - hardcoded regardless of velocity
```

**Problem:**
- Eddy current heating scales with v²
- At 1.6 km/s: T_steady ≈ 379K (correct)
- At 15 km/s: T_steady should be MUCH higher (potentially exceeding T_limit)
- The thermal_model.py already computes this correctly but isn't called here

**Fix Required:**
Call `update_temperature_euler()` from thermal_model.py with actual velocity-dependent heating:

```python
from dynamics.thermal_model import update_temperature_euler, ThermalParameters

# Compute eddy heating from velocity
thermal_params = ThermalParameters(...)
eddy_heating = compute_eddy_heating(u, B_field, ...)  # Scales with u²
T_steady_state = compute_steady_state_temp(eddy_heating, radiative_cooling)
```

---

### 4. SmCo Force Model Not Integrated (CRITICAL PHYSICS BUG)

**Location:** `src/sgms_anchor_v1.py:951, 1010`

**Current Code:**
```python
k_fp: float,  # Passed as parameter even for SmCo
...
# For SmCo, k_fp should come from permanent magnet model (not Bean-London)
# But nothing actually computes it!
```

**Problem:**
- `dynamics/permanent_magnet_model.py` exists with correct physics
- But `mission_level_metrics()` doesn't use it
- SmCo k_fp is just whatever value the Sobol sampler picks (no physical basis)
- Should compute: k_eff = B_r² * A / (μ₀ * d) for Halbach configuration

**Fix Required:**
```python
if magnet_material == "SmCo":
    from dynamics.permanent_magnet_model import PermanentMagnetModel, PermanentMagnetGeometry
    
    geometry = PermanentMagnetGeometry(
        pole_face_area=0.01,  # m² (should be parameter)
        equilibrium_gap=0.005,  # m (should be parameter)
        config_type='halbach'
    )
    
    smco_props = MATERIAL_PROPERTIES['SmCo']
    pm_model = PermanentMagnetModel(smco_props, geometry)
    k_fp = pm_model.compute_stiffness(displacement=0.001, temperature=T_steady_state)
```

---

### 5. Perturbation Force Altitude-Independent (MODERATE PHYSICS BUG)

**Location:** `src/sgms_anchor_v1.py:1028`

**Current Code:**
```python
perturbation_force = 0.1  # N - conservative estimate
```

**Problem:**
- J₂ perturbation depends on altitude: F_J2 ∝ 1/r⁴
- SRP depends on cross-section and solar angle
- Drag depends on atmospheric density (exponential with altitude)
- At 400 km: F_J2 ~ 0.05 N for 1000 kg station
- At 800 km: F_J2 ~ 0.003 N (16x smaller!)
- Using 0.1 N everywhere gives wrong packet counts

**Fix Required:**
Use existing `orbital_perturbations.py` module:

```python
from dynamics.orbital_perturbations import get_orbital_perturbation_force, create_orbital_state_from_params

orbital_state = create_orbital_state_from_params(h_km, inclination=0.0)
perturbation_force = get_orbital_perturbation_force(
    params={'h_km': h_km, 'ms': ms},
    orbital_state=orbital_state,
    t=0.0,
    packet_mass=ms
)
```

---

### 6. Counter-Propagating Streams Not Accounted For (DESIGN GAP)

**Location:** Throughout `mission_level_metrics()`

**Problem:**
- Real system needs TWO streams (bidirectional station-keeping)
- Current model computes mass/power for single stream
- All costs should be 2x

**Fix Required:**
```python
n_streams = 2 if counter_propagating else 1
M_total_kg = N_packets * mp * n_streams
P_total_kW = (P_cryocooler_kW + P_control_kW + P_injection_kW) * n_streams
```

---

### 7. Missing Material Properties (DATA GAP)

**Status:** Partially complete

**Present:**
- ✅ CNT_yarn: density, tensile_strength, safety_factor, allowable_stress, emissivity, max_operating_temp
- ✅ CFRP: density, tensile_strength, safety_factor, allowable_stress, emissivity, max_operating_temp
- ✅ YBCO: Tc, Jc0, n_exponent, B0, alpha, density, specific_heat, thermal_conductivity, k_fp_bulk_range
- ✅ NdFeB: remanence, coercivity, max_operating_temp, curie_temp, density

**Missing:**
- ❌ NdFeB alpha_Br (thermal coefficient of remanence) - critical for thermal stability analysis
- ❌ SmCo alpha_Br - needed for PermanentMagnetModel
- ❌ YBCO pinning force comparison to GdBCO

**Fix Required:**
Add to `params/canonical_values.py`:
```python
'NdFeB': {
    ...
    'alpha_Br': {'value': -0.0012, 'note': '/K, 4x more sensitive than SmCo'},
},
'SmCo': {
    ...
    'alpha_Br': {'value': -0.0003, 'note': '/K, very stable'},
}
```

---

## Recommended Fix Priority

| Priority | Issue | Status | Impact | Effort |
|----------|-------|--------|--------|--------|
| P0 | Add test coverage | ✅ COMPLETE | Verification enabled | Done |
| P0 | Fix λ/mp coupling | ❌ REMAINS | Physics correctness | 1 hour |
| P1 | Integrate PM force model | ❌ REMAINS | SmCo profiles invalid without it | 2 hours |
| P1 | Velocity-dependent thermal | ❌ REMAINS | High-velocity SmCo feasibility unknown | 1 hour |
| P2 | Altitude-dependent perturbations | ❌ REMAINS | Packet count accuracy | 1 hour |
| P2 | Counter-propagating streams | ❌ REMAINS | Cost underestimate by 2x | 0.5 hours |
| P3 | Add missing alpha_Br values | ❌ REMAINS | Thermal stability analysis | 0.5 hours |

---

## Completed Work Summary

### ✅ Test Coverage (P0 - COMPLETE)

**Files Created:**
- `tests/test_debris_risk.py` - 15 tests covering collision probability, escaped packet risk, Kessler threshold
- `tests/test_energy_injection.py` - 18 tests covering injection energy, replacement rates, power budgets

**Test Results:** 33/33 passing

**Coverage:**
- Debris density model validation
- Collision probability scaling with n_packets, cross_section, altitude
- MTBF calculations
- Kinetic energy calculations for escaped packets
- Lethal threshold comparisons (NASA 40J guideline)
- Kessler ratio and threshold detection
- Comprehensive risk assessment integration
- Energy injection formulas (translational + rotational KE)
- Efficiency losses for different launch methods (EM, chemical, lunar)
- Power budget scaling with fault rate and packet count

### ✅ Verified Implementations

The following previously-implemented features were verified as working correctly:

1. **Hitch COR fix** - Properly uses CoM frame relative velocity approach (lines 119-135)
2. **Multi-body topology** - Has topology param ('linear', 'ring', 'orbital_ring') and counter_propagating flag
3. **Material registry** - CNT yarn, YBCO, NdFeB, CFRP entries exist with full properties
4. **Energy injection module** - Exists at `dynamics/energy_injection.py` with all required functions
5. **Debris risk module** - Exists at `dynamics/debris_risk.py` with complete implementation
6. **Permanent magnet model** - Exists at `dynamics/permanent_magnet_model.py` (but NOT integrated into mission_level_metrics)

---

## Remaining Critical Gaps

### ⚠️ CRITICAL: Physics Bugs Still Present

These are fundamental physics errors that will produce incorrect results in papers/simulations:

#### 1. Linear Density (λ) Decoupled from Packet Mass (mp)

**Location:** `src/sgms_anchor_v1.py:931`

**Problem:** When Sobol varies mp from 5-100 kg, λ stays fixed at 72.92 kg/m, implying unrealistic spacing variations.

**Fix Needed:** Derive λ = mp / spacing internally.

#### 2. SmCo Force Model Not Integrated

**Location:** `src/sgms_anchor_v1.py:951, 1010`

**Problem:** `dynamics/permanent_magnet_model.py` exists but is never called. SmCo k_fp is just a free parameter with no physical basis.

**Fix Needed:** Call PermanentMagnetModel.compute_stiffness() when magnet_material == "SmCo".

#### 3. SmCo Thermal Margin Hardcoded

**Location:** `src/sgms_anchor_v1.py:1075`

**Problem:** T_steady_state = 379K regardless of velocity. Eddy heating scales with v².

**Fix Needed:** Call thermal_model.update_temperature_euler() with velocity-dependent heating.

#### 4. Perturbation Force Altitude-Independent

**Location:** `src/sgms_anchor_v1.py:1028`

**Problem:** perturbation_force = 0.1 N hardcoded. J₂ perturbation depends on altitude (~1/r⁴).

**Fix Needed:** Use orbital_perturbations.get_orbital_perturbation_force().

#### 5. Counter-Propagating Streams Not Accounted For

**Location:** Throughout `mission_level_metrics()`

**Problem:** Real system needs TWO streams. All mass/power/cost should be 2x.

**Fix Needed:** Multiply M_total_kg and P_total_kW by n_streams = 2.

---

## Next Steps

**Immediate (Before Paper Submission):**
1. Fix λ/mp coupling - quick fix, high impact
2. Integrate PM force model - critical for SmCo validity
3. Add velocity-dependent thermal - needed for high-velocity analysis

**Important:**
4. Use altitude-dependent perturbations
5. Account for counter-propagating streams
6. Add alpha_Br to SmCo/NdFeB material properties
